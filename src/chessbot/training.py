from array import array
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import chess
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.chessbot.model import (
    NextMoveLSTM,
    build_vocab,
    compute_topk,
    encode_tokens,
    side_to_move_id_from_context_len,
    winner_to_id,
)
from src.chessbot.phase import PHASE_MIDDLEGAME, PHASE_OPENING, PHASE_ENDGAME, PHASE_UNKNOWN, phase_to_id
from src.chessbot.phase import classify_board_phase


class MoveDataset(Dataset):
    def __init__(self, rows: List[Dict], vocab: Dict[str, int]) -> None:
        self.rows = rows
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        context_ids = encode_tokens(row["context"], self.vocab)
        label = self.vocab.get(row["next_move"], self.vocab["<UNK>"])
        winner = winner_to_id(row.get("winner_side", "?"))
        phase = phase_to_id(row.get("phase", PHASE_UNKNOWN))
        side_to_move = side_to_move_id_from_context_len(len(row.get("context", [])))
        return context_ids, label, winner, phase, side_to_move


class IndexedJsonlDataset(Dataset):
    """Map-style dataset backed by JSONL files via precomputed line offsets."""

    def __init__(self, paths: List[str], path_ids: List[int], offsets: List[int], vocab: Dict[str, int]) -> None:
        self.paths = paths
        self.path_ids = path_ids
        self.offsets = offsets
        self.vocab = vocab
        self._handle_cache: Dict[int, object] = {}

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self):
        state = self.__dict__.copy()
        # File handles cannot be pickled for DataLoader workers.
        state["_handle_cache"] = {}
        return state

    def _handle_for(self, path_id: int):
        h = self._handle_cache.get(path_id)
        if h is None:
            h = open(self.paths[path_id], "rb")
            self._handle_cache[path_id] = h
        return h

    def __getitem__(self, idx: int):
        path_id = self.path_ids[idx]
        offset = self.offsets[idx]
        h = self._handle_for(path_id)
        h.seek(offset)
        line = h.readline()
        row = json.loads(line.decode("utf-8"))
        context_ids = encode_tokens(row["context"], self.vocab)
        label = self.vocab.get(row["next_move"], self.vocab["<UNK>"])
        winner = winner_to_id(row.get("winner_side", "?"))
        phase = phase_to_id(row.get("phase", PHASE_UNKNOWN))
        side_to_move = side_to_move_id_from_context_len(len(row.get("context", [])))
        return context_ids, label, winner, phase, side_to_move


def _encode_rollout_targets(row: Dict, vocab: Dict[str, int], rollout_horizon: int) -> Tuple[List[int], List[int]]:
    horizon = max(1, int(rollout_horizon))
    target_tokens = list(row.get("target", []) or [])
    unk = vocab["<UNK>"]
    out_ids: List[int] = []
    out_mask: List[int] = []
    for i in range(horizon):
        if i < len(target_tokens):
            out_ids.append(vocab.get(target_tokens[i], unk))
            out_mask.append(1)
        else:
            out_ids.append(0)  # PAD id
            out_mask.append(0)
    return out_ids, out_mask


class IndexedJsonlRolloutDataset(Dataset):
    """Map-style dataset backed by JSONL files with fixed-horizon rollout targets."""

    def __init__(
        self,
        paths: List[str],
        path_ids: List[int],
        offsets: List[int],
        vocab: Dict[str, int],
        rollout_horizon: int,
    ) -> None:
        self.paths = paths
        self.path_ids = path_ids
        self.offsets = offsets
        self.vocab = vocab
        self.rollout_horizon = int(max(1, rollout_horizon))
        self._handle_cache: Dict[int, object] = {}

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle_cache"] = {}
        return state

    def _handle_for(self, path_id: int):
        h = self._handle_cache.get(path_id)
        if h is None:
            h = open(self.paths[path_id], "rb")
            self._handle_cache[path_id] = h
        return h

    def __getitem__(self, idx: int):
        path_id = self.path_ids[idx]
        offset = self.offsets[idx]
        h = self._handle_for(path_id)
        h.seek(offset)
        line = h.readline()
        row = json.loads(line.decode("utf-8"))
        context_ids = encode_tokens(row["context"], self.vocab)
        rollout_ids, rollout_mask = _encode_rollout_targets(row, self.vocab, self.rollout_horizon)
        label = int(rollout_ids[0]) if rollout_ids else self.vocab["<UNK>"]
        winner = winner_to_id(row.get("winner_side", "?"))
        phase = phase_to_id(row.get("phase", PHASE_UNKNOWN))
        side_to_move = side_to_move_id_from_context_len(len(row.get("context", [])))
        return context_ids, label, winner, phase, side_to_move, rollout_ids, rollout_mask


def _moves_from_row(row: Dict) -> List[str]:
    moves = row.get("moves")
    if isinstance(moves, list):
        return list(moves)
    moves_uci = row.get("moves_uci")
    if isinstance(moves_uci, list):
        return list(moves_uci)
    return []


def _sniff_jsonl_schema(path: str) -> str:
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line.decode("utf-8"))
            if "context" in row:
                return "spliced"
            if "moves" in row or "moves_uci" in row:
                return "game"
            return "unknown"
    return "empty"


def _sniff_paths_schema(paths: List[str]) -> str:
    schema = None
    for p in paths:
        cur = _sniff_jsonl_schema(os.fspath(p))
        if cur == "empty":
            continue
        if cur == "unknown":
            raise RuntimeError(f"Unsupported JSONL row schema in {p}")
        if schema is None:
            schema = cur
        elif schema != cur:
            raise RuntimeError(f"Mixed dataset schemas across inputs: {schema} vs {cur} (path {p})")
    return schema or "spliced"


@dataclass
class RuntimeSpliceConfig:
    min_context: int = 8
    min_target: int = 1
    max_samples_per_game: int = 0
    seed: int = 7


def _runtime_splice_indices_for_moves(
    moves: List[str],
    cfg: RuntimeSpliceConfig,
    game_id: str = "",
) -> List[int]:
    n = len(moves)
    start_i = int(cfg.min_context) - 1
    end_i = n - int(cfg.min_target) - 1
    if end_i < start_i:
        return []
    idxs = list(range(start_i, end_i + 1))
    if int(cfg.max_samples_per_game) > 0 and len(idxs) > int(cfg.max_samples_per_game):
        rnd = random.Random(f"{int(cfg.seed)}:{game_id or 'unknown'}")
        rnd.shuffle(idxs)
        idxs = idxs[: int(cfg.max_samples_per_game)]
    return idxs


def _phase_id_from_moves_prefix(moves: List[str], splice_i: int) -> int:
    board = chess.Board()
    upto = min(len(moves), int(splice_i) + 1)
    for j in range(upto):
        uci = str(moves[j])
        try:
            mv = chess.Move.from_uci(uci)
        except Exception:
            return phase_to_id(PHASE_UNKNOWN)
        if mv not in board.legal_moves:
            return phase_to_id(PHASE_UNKNOWN)
        board.push(mv)
    phase_info = classify_board_phase(board, ply=upto)
    return phase_to_id(phase_info.get("phase", PHASE_UNKNOWN))


def _phase_ids_by_ply_prefix(moves: List[str]) -> List[int]:
    """Return phase-id per ply index using one incremental replay pass."""
    out: List[int] = []
    board = chess.Board()
    board_ok = True
    for ply_idx, uci in enumerate(moves, start=1):
        if board_ok:
            try:
                mv = chess.Move.from_uci(str(uci))
            except Exception:
                board_ok = False
            else:
                if mv in board.legal_moves:
                    board.push(mv)
                else:
                    board_ok = False
        if board_ok:
            info = classify_board_phase(board, ply=ply_idx)
            out.append(phase_to_id(info.get("phase", PHASE_UNKNOWN)))
        else:
            out.append(phase_to_id(PHASE_UNKNOWN))
    return out


def _index_game_jsonl_paths(
    paths: List[str],
    runtime_cfg: RuntimeSpliceConfig,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[List[str], array, array, array, array, Dict[str, int], int, Dict[str, int], int]:
    path_strs = [os.fspath(p) for p in paths]
    path_ids = array("I")
    offsets = array("Q")
    splice_indices = array("I")
    sample_phase_ids = array("B")
    game_rows_by_file: Dict[str, int] = {}
    sample_rows_by_file: Dict[str, int] = {}
    total_game_rows = 0
    total_sample_rows = 0
    for path_id, path in enumerate(path_strs):
        game_count = 0
        sample_count = 0
        file_size_bytes = 0
        try:
            file_size_bytes = int(os.path.getsize(path))
        except OSError:
            file_size_bytes = 0
        last_progress_game_count = 0
        last_progress_bytes = 0
        if progress_cb is not None:
            progress_cb(
                {
                    "stage": "indexing",
                    "path": path,
                    "path_id": int(path_id),
                    "file_size_bytes": int(file_size_bytes),
                    "file_bytes_read": 0,
                    "file_game_rows": 0,
                    "file_sample_rows": 0,
                    "total_game_rows": int(total_game_rows),
                    "total_sample_rows": int(total_sample_rows),
                    "done": False,
                }
            )
        with open(path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line.decode("utf-8"))
                moves = _moves_from_row(row)
                game_id = str(row.get("game_id", ""))
                splices = _runtime_splice_indices_for_moves(moves, runtime_cfg, game_id=game_id)
                phase_ids_for_game = _phase_ids_by_ply_prefix(moves) if splices else []
                game_count += 1
                total_game_rows += 1
                sample_count += len(splices)
                total_sample_rows += len(splices)
                for splice_i in splices:
                    path_ids.append(int(path_id))
                    offsets.append(int(offset))
                    splice_indices.append(int(splice_i))
                    if splice_i < len(phase_ids_for_game):
                        sample_phase_ids.append(int(phase_ids_for_game[splice_i]))
                    else:
                        sample_phase_ids.append(int(phase_to_id(PHASE_UNKNOWN)))
                if progress_cb is not None:
                    bytes_read = int(offset + len(line))
                    if (
                        (game_count - last_progress_game_count) >= 1000
                        or (bytes_read - last_progress_bytes) >= (8 * 1024 * 1024)
                    ):
                        last_progress_game_count = int(game_count)
                        last_progress_bytes = int(bytes_read)
                        progress_cb(
                            {
                                "stage": "indexing",
                                "path": path,
                                "path_id": int(path_id),
                                "file_size_bytes": int(file_size_bytes),
                                "file_bytes_read": bytes_read,
                                "file_game_rows": int(game_count),
                                "file_sample_rows": int(sample_count),
                                "total_game_rows": int(total_game_rows),
                                "total_sample_rows": int(total_sample_rows),
                                "done": False,
                            }
                        )
        game_rows_by_file[path] = game_count
        sample_rows_by_file[path] = sample_count
        if progress_cb is not None:
            progress_cb(
                {
                    "stage": "indexing",
                    "path": path,
                    "path_id": int(path_id),
                    "file_size_bytes": int(file_size_bytes),
                    "file_bytes_read": int(file_size_bytes),
                    "file_game_rows": int(game_count),
                    "file_sample_rows": int(sample_count),
                    "total_game_rows": int(total_game_rows),
                    "total_sample_rows": int(total_sample_rows),
                    "done": True,
                }
            )
    return (
        path_strs,
        path_ids,
        offsets,
        splice_indices,
        sample_phase_ids,
        game_rows_by_file,
        total_game_rows,
        sample_rows_by_file,
        total_sample_rows,
    )


def _index_game_jsonl_paths_from_runtime_cache(
    paths: List[str],
    runtime_cfg: RuntimeSpliceConfig,
    expected_split: Optional[str] = None,
) -> Tuple[Optional[Tuple[List[str], array, array, array, array, Dict[str, int], int, Dict[str, int], int]], str]:
    """Load precomputed runtime splice indexes from runtime_splice_cache.

    Returns (index_tuple, reason). index_tuple is None when cache cannot be used.
    """
    path_strs = [os.fspath(p) for p in paths]
    resolved_to_global: Dict[str, int] = {str(Path(p).resolve()): i for i, p in enumerate(path_strs)}

    def _extract_dataset_token(path_obj: Path) -> str:
        for part in path_obj.parts:
            if part.endswith("_game"):
                return part
        return ""

    alias_to_global: Dict[str, int] = {}
    alias_ambiguous: set[str] = set()
    for i, p in enumerate(path_strs):
        p_obj = Path(p).resolve()
        dataset_token = _extract_dataset_token(p_obj)
        file_name = p_obj.name
        if dataset_token and file_name:
            alias = f"{dataset_token}::{file_name}"
            if alias in alias_to_global and alias_to_global[alias] != i:
                alias_ambiguous.add(alias)
            else:
                alias_to_global[alias] = i
    split_expected = str(expected_split or "").strip().lower()
    if split_expected not in {"", "train", "val", "test"}:
        return None, f"unsupported_expected_split:{split_expected}"

    final_path_ids = array("I")
    final_offsets = array("Q")
    final_splice_indices = array("I")
    final_sample_phase_ids = array("B")
    game_rows_by_file: Dict[str, int] = {p: 0 for p in path_strs}
    sample_rows_by_file: Dict[str, int] = {p: 0 for p in path_strs}
    total_game_rows = 0
    total_sample_rows = 0

    def _load_arr(path: Path, typecode: str):
        arr_obj = array(typecode)
        with path.open("rb") as f:
            arr_obj.frombytes(f.read())
        return arr_obj

    def _cache_cfg_matches(manifest_cfg: Dict[str, Any]) -> bool:
        for key in ("min_context", "min_target", "max_samples_per_game", "seed"):
            try:
                if int(manifest_cfg.get(key, -10**9)) != int(getattr(runtime_cfg, key)):
                    return False
            except Exception:
                return False
        return True

    for raw_path in path_strs:
        p_resolved = Path(raw_path).resolve()
        split = split_expected or p_resolved.stem.strip().lower()
        if split not in {"train", "val", "test"}:
            return None, f"cannot_infer_split:{raw_path}"

        cache_root = p_resolved.parent / "runtime_splice_cache"
        manifest_path = cache_root / "manifest.json"
        if not manifest_path.is_file():
            return None, f"cache_manifest_missing:{manifest_path}"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return None, f"cache_manifest_invalid:{exc}"
        if str(manifest.get("kind", "")) != "runtime_splice_cache":
            return None, f"cache_manifest_kind_invalid:{manifest.get('kind')}"
        manifest_cfg = manifest.get("config")
        if not isinstance(manifest_cfg, dict) or not _cache_cfg_matches(manifest_cfg):
            return None, "cache_config_mismatch"

        split_dir = cache_root / split
        required = {
            "paths": split_dir / "paths.json",
            "path_ids": split_dir / "path_ids.u32.bin",
            "offsets": split_dir / "offsets.u64.bin",
            "splice_indices": split_dir / "splice_indices.u32.bin",
            "sample_phase_ids": split_dir / "sample_phase_ids.u8.bin",
        }
        for fp in required.values():
            if not fp.is_file():
                return None, f"cache_file_missing:{fp}"

        try:
            cache_paths_raw = json.loads(required["paths"].read_text(encoding="utf-8"))
        except Exception as exc:
            return None, f"cache_paths_invalid:{exc}"
        if not isinstance(cache_paths_raw, list) or not cache_paths_raw:
            return None, "cache_paths_empty"
        cache_paths_resolved = [str(Path(str(x)).resolve()) for x in cache_paths_raw]

        local_path_ids = _load_arr(required["path_ids"], "I")
        local_offsets = _load_arr(required["offsets"], "Q")
        local_splice_indices = _load_arr(required["splice_indices"], "I")
        local_sample_phase_ids = _load_arr(required["sample_phase_ids"], "B")
        n_rows = len(local_offsets)
        if not (n_rows == len(local_path_ids) == len(local_splice_indices) == len(local_sample_phase_ids)):
            return None, "cache_row_count_mismatch"

        local_to_global: List[int] = []
        for cp in cache_paths_resolved:
            gid = resolved_to_global.get(cp)
            if gid is None:
                cp_obj = Path(cp)
                alias = f"{_extract_dataset_token(cp_obj)}::{cp_obj.name}"
                if alias in alias_ambiguous:
                    return None, f"cache_path_alias_ambiguous:{cp}"
                gid = alias_to_global.get(alias)
                if gid is None:
                    return None, f"cache_path_not_in_input:{cp}"
            local_to_global.append(int(gid))

        for i in range(n_rows):
            local_pid = int(local_path_ids[i])
            if local_pid < 0 or local_pid >= len(local_to_global):
                return None, "cache_path_id_out_of_range"
            global_pid = int(local_to_global[local_pid])
            final_path_ids.append(global_pid)
            final_offsets.append(int(local_offsets[i]))
            final_splice_indices.append(int(local_splice_indices[i]))
            final_sample_phase_ids.append(int(local_sample_phase_ids[i]))
            sample_rows_by_file[path_strs[global_pid]] += 1
            total_sample_rows += 1

        split_meta = (manifest.get("splits") or {}).get(split) or {}
        if len(local_to_global) == 1:
            gp = path_strs[int(local_to_global[0])]
            try:
                games = int(split_meta.get("game_rows_total", 0) or 0)
            except Exception:
                games = 0
            if games > 0:
                game_rows_by_file[gp] += games
                total_game_rows += games

    out = (
        path_strs,
        final_path_ids,
        final_offsets,
        final_splice_indices,
        final_sample_phase_ids,
        game_rows_by_file,
        total_game_rows,
        sample_rows_by_file,
        total_sample_rows,
    )
    return out, "loaded_runtime_splice_cache"


def _index_game_jsonl_paths_cached_or_runtime(
    paths: List[str],
    runtime_cfg: RuntimeSpliceConfig,
    expected_split: Optional[str] = None,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[Tuple[List[str], array, array, array, array, Dict[str, int], int, Dict[str, int], int], bool, str]:
    cache_out, reason = _index_game_jsonl_paths_from_runtime_cache(
        paths=paths,
        runtime_cfg=runtime_cfg,
        expected_split=expected_split,
    )
    if cache_out is not None:
        return cache_out, True, reason
    runtime_out = _index_game_jsonl_paths(paths=paths, runtime_cfg=runtime_cfg, progress_cb=progress_cb)
    return runtime_out, False, reason


def _cache_load_reason_label(*, used_cache: bool, reason: str) -> str:
    if bool(used_cache):
        return "hit"
    return str(reason or "runtime_index_fallback")


def _is_primary_process(*, distributed_enabled: bool, distributed_rank: int) -> bool:
    return (not bool(distributed_enabled)) or int(distributed_rank) == 0


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _cpu_cloned_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    base = _unwrap_model(model)
    return {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}


class IndexedJsonlGameSpliceDataset(Dataset):
    """Runtime-splicing map-style dataset backed by game-level JSONL files."""

    def __init__(
        self,
        paths: List[str],
        path_ids,
        offsets,
        splice_indices,
        sample_phase_ids,
        vocab: Dict[str, int],
    ) -> None:
        self.paths = paths
        self.path_ids = path_ids
        self.offsets = offsets
        self.splice_indices = splice_indices
        self.sample_phase_ids = sample_phase_ids
        self.vocab = vocab
        self._handle_cache: Dict[int, object] = {}

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle_cache"] = {}
        return state

    def _handle_for(self, path_id: int):
        h = self._handle_cache.get(path_id)
        if h is None:
            h = open(self.paths[path_id], "rb")
            self._handle_cache[path_id] = h
        return h

    def __getitem__(self, idx: int):
        path_id = self.path_ids[idx]
        offset = self.offsets[idx]
        splice_i = self.splice_indices[idx]
        h = self._handle_for(path_id)
        h.seek(offset)
        row = json.loads(h.readline().decode("utf-8"))
        moves = _moves_from_row(row)
        context = moves[: splice_i + 1]
        target = moves[splice_i + 1 :]
        if not target:
            raise IndexError("runtime splice produced empty target; index/build mismatch")
        context_ids = encode_tokens(context, self.vocab)
        label = self.vocab.get(target[0], self.vocab["<UNK>"])
        winner = winner_to_id(row.get("winner_side", "?"))
        phase = int(self.sample_phase_ids[idx]) if self.sample_phase_ids is not None else _phase_id_from_moves_prefix(moves, splice_i)
        side_to_move = side_to_move_id_from_context_len(len(context))
        return context_ids, label, winner, phase, side_to_move


class IndexedJsonlGameRolloutDataset(Dataset):
    """Runtime-splicing map-style dataset with fixed-horizon rollout targets."""

    def __init__(
        self,
        paths: List[str],
        path_ids,
        offsets,
        splice_indices,
        sample_phase_ids,
        vocab: Dict[str, int],
        rollout_horizon: int,
    ) -> None:
        self.paths = paths
        self.path_ids = path_ids
        self.offsets = offsets
        self.splice_indices = splice_indices
        self.sample_phase_ids = sample_phase_ids
        self.vocab = vocab
        self.rollout_horizon = int(max(1, rollout_horizon))
        self._handle_cache: Dict[int, object] = {}

    def __len__(self) -> int:
        return len(self.offsets)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle_cache"] = {}
        return state

    def _handle_for(self, path_id: int):
        h = self._handle_cache.get(path_id)
        if h is None:
            h = open(self.paths[path_id], "rb")
            self._handle_cache[path_id] = h
        return h

    def __getitem__(self, idx: int):
        path_id = self.path_ids[idx]
        offset = self.offsets[idx]
        splice_i = self.splice_indices[idx]
        h = self._handle_for(path_id)
        h.seek(offset)
        row = json.loads(h.readline().decode("utf-8"))
        moves = _moves_from_row(row)
        context = moves[: splice_i + 1]
        target = moves[splice_i + 1 :]
        if not target:
            raise IndexError("runtime splice produced empty target; index/build mismatch")
        context_ids = encode_tokens(context, self.vocab)
        pseudo_row = {"target": target}
        rollout_ids, rollout_mask = _encode_rollout_targets(pseudo_row, self.vocab, self.rollout_horizon)
        label = int(rollout_ids[0]) if rollout_ids else self.vocab["<UNK>"]
        winner = winner_to_id(row.get("winner_side", "?"))
        phase = int(self.sample_phase_ids[idx]) if self.sample_phase_ids is not None else _phase_id_from_moves_prefix(moves, splice_i)
        side_to_move = side_to_move_id_from_context_len(len(context))
        return context_ids, label, winner, phase, side_to_move, rollout_ids, rollout_mask


def _build_vocab_and_count_rows_from_train_paths(train_paths: List[str]) -> Tuple[Dict[str, int], Dict[str, int], int]:
    vocab = {"<PAD>": 0, "<UNK>": 1}
    rows_by_file: Dict[str, int] = {}
    total_rows = 0
    for path in train_paths:
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                count += 1
                total_rows += 1
                for tok in row.get("context", []):
                    if tok not in vocab:
                        vocab[tok] = len(vocab)
                for tok in row.get("target", []):
                    if tok not in vocab:
                        vocab[tok] = len(vocab)
        rows_by_file[path] = count
    return vocab, rows_by_file, total_rows


def _build_vocab_and_count_rows_from_train_game_paths(
    train_paths: List[str],
    runtime_cfg: RuntimeSpliceConfig,
) -> Tuple[Dict[str, int], Dict[str, int], int, Dict[str, int], int]:
    vocab = {"<PAD>": 0, "<UNK>": 1}
    game_rows_by_file: Dict[str, int] = {}
    sample_rows_by_file: Dict[str, int] = {}
    total_game_rows = 0
    total_sample_rows = 0
    for path in train_paths:
        game_count = 0
        sample_count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                game_count += 1
                total_game_rows += 1
                moves = _moves_from_row(row)
                for tok in moves:
                    if tok not in vocab:
                        vocab[tok] = len(vocab)
                splices = _runtime_splice_indices_for_moves(
                    moves=moves,
                    cfg=runtime_cfg,
                    game_id=str(row.get("game_id", "")),
                )
                sample_count += len(splices)
                total_sample_rows += len(splices)
        game_rows_by_file[path] = game_count
        sample_rows_by_file[path] = sample_count
    return vocab, sample_rows_by_file, total_sample_rows, game_rows_by_file, total_game_rows


def _count_rows_in_jsonl_paths(paths: List[str]) -> Tuple[Dict[str, int], int]:
    rows_by_file: Dict[str, int] = {}
    total_rows = 0
    for path in paths:
        count = 0
        with open(path, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                count += 1
        rows_by_file[path] = count
        total_rows += count
    return rows_by_file, total_rows


def _count_rows_in_game_jsonl_paths_runtime_splice(
    paths: List[str],
    runtime_cfg: RuntimeSpliceConfig,
) -> Tuple[Dict[str, int], int, Dict[str, int], int]:
    sample_rows_by_file: Dict[str, int] = {}
    game_rows_by_file: Dict[str, int] = {}
    total_sample_rows = 0
    total_game_rows = 0
    for path in paths:
        game_count = 0
        sample_count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                game_count += 1
                total_game_rows += 1
                moves = _moves_from_row(row)
                sample_count += len(
                    _runtime_splice_indices_for_moves(
                        moves=moves,
                        cfg=runtime_cfg,
                        game_id=str(row.get("game_id", "")),
                    )
                )
        game_rows_by_file[path] = game_count
        sample_rows_by_file[path] = sample_count
        total_sample_rows += sample_count
    return sample_rows_by_file, total_sample_rows, game_rows_by_file, total_game_rows


def _runtime_index_memory_bytes(path_ids, offsets, splice_indices, sample_phase_ids) -> int:
    total = 0
    for arr_obj in (path_ids, offsets, splice_indices, sample_phase_ids):
        try:
            total += int(getattr(arr_obj, "buffer_info")()[1] * arr_obj.itemsize)
        except Exception:
            try:
                total += len(arr_obj) * 8
            except Exception:
                pass
    return total


def _index_jsonl_paths(paths: List[str]) -> Tuple[List[str], List[int], List[int]]:
    path_strs = [os.fspath(p) for p in paths]
    path_ids: List[int] = []
    offsets: List[int] = []
    for path_id, path in enumerate(path_strs):
        with open(path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                path_ids.append(path_id)
                offsets.append(offset)
    return path_strs, path_ids, offsets


def _sample_subset_indices(total: int, keep: int, seed: int) -> List[int]:
    total = int(total)
    keep = int(keep)
    if keep <= 0 or total <= 0 or keep >= total:
        return list(range(max(total, 0)))
    rnd = random.Random(int(seed))
    idxs = rnd.sample(range(total), keep)
    idxs.sort()
    return idxs


def collate_train(batch: List[Tuple[List[int], int, int, int, int]]):
    lengths = torch.tensor([len(x[0]) for x in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    tokens = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
    winners = torch.tensor([x[2] for x in batch], dtype=torch.long)
    phases = torch.tensor([x[3] for x in batch], dtype=torch.long)
    side_to_moves = torch.tensor([x[4] for x in batch], dtype=torch.long)

    for i, (ctx, _, _, _, _) in enumerate(batch):
        tokens[i, : len(ctx)] = torch.tensor(ctx, dtype=torch.long)
    return tokens, lengths, labels, winners, phases, side_to_moves


def collate_train_rollout(batch: List[Tuple[List[int], int, int, int, int, List[int], List[int]]]):
    lengths = torch.tensor([len(x[0]) for x in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    tokens = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
    winners = torch.tensor([x[2] for x in batch], dtype=torch.long)
    phases = torch.tensor([x[3] for x in batch], dtype=torch.long)
    side_to_moves = torch.tensor([x[4] for x in batch], dtype=torch.long)
    rollout_horizon = len(batch[0][5]) if batch else 1
    rollout_targets = torch.zeros((len(batch), rollout_horizon), dtype=torch.long)
    rollout_mask = torch.zeros((len(batch), rollout_horizon), dtype=torch.bool)

    for i, (ctx, _, _, _, _, target_ids, target_mask) in enumerate(batch):
        tokens[i, : len(ctx)] = torch.tensor(ctx, dtype=torch.long)
        rollout_targets[i] = torch.tensor(target_ids, dtype=torch.long)
        rollout_mask[i] = torch.tensor(target_mask, dtype=torch.bool)
    return tokens, lengths, labels, winners, phases, side_to_moves, rollout_targets, rollout_mask


def _build_rollout_step_weights(rollout_horizon: int, decay: float) -> List[float]:
    horizon = max(1, int(rollout_horizon))
    d = float(decay)
    if d <= 0:
        d = 1.0
    weights = [1.0]
    for _ in range(1, horizon):
        weights.append(weights[-1] * d)
    return weights


def _weighted_rollout_closeness(
    step_matches: List[bool],
    weights: List[float],
    closeness_horizon: int,
) -> float:
    h = max(0, min(int(closeness_horizon), len(step_matches), len(weights)))
    if h <= 0:
        return 0.0
    denom = float(sum(weights[:h])) or 1.0
    num = 0.0
    for i in range(h):
        if step_matches[i]:
            num += float(weights[i])
    return num / denom


def _prefix_match_len(step_matches: List[bool], closeness_horizon: int) -> int:
    h = max(0, min(int(closeness_horizon), len(step_matches)))
    n = 0
    for i in range(h):
        if not step_matches[i]:
            break
        n += 1
    return n


def _board_from_context_safe(context_ids: List[int], inv_vocab: Dict[int, str]) -> Tuple[chess.Board, bool]:
    board = chess.Board()
    for tok_id in context_ids:
        uci = inv_vocab.get(int(tok_id), "")
        try:
            mv = chess.Move.from_uci(uci)
        except Exception:
            return board, False
        if mv not in board.legal_moves:
            return board, False
        board.push(mv)
    return board, True


def evaluate_loader_multistep(
    model,
    loader,
    device,
    criterion,
    winner_weight: float,
    phase_weight_vector: torch.Tensor,
    rollout_horizon: int,
    closeness_horizon: int,
    rollout_step_weights: List[float],
    inv_vocab: Dict[int, str],
) -> Dict[str, float]:
    model.eval()
    horizon = max(1, int(rollout_horizon))
    close_h = max(1, min(int(closeness_horizon), horizon))
    step_correct = [0.0 for _ in range(horizon)]
    step_total = [0.0 for _ in range(horizon)]
    top5_hits_step1 = 0.0
    top5_total_step1 = 0.0
    total_loss = 0.0
    total_weight = 0.0
    n_rows = 0
    prefix_total = 0.0
    closeness_total = 0.0
    legal_num = 0.0
    legal_den = 0.0

    with torch.no_grad():
        for batch in loader:
            (
                tokens,
                lengths,
                labels,
                winners,
                phases,
                side_to_moves,
                rollout_targets,
                rollout_mask,
            ) = batch
            tokens = tokens.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            winners = winners.to(device, non_blocking=True)
            phases = phases.to(device, non_blocking=True)
            side_to_moves = side_to_moves.to(device, non_blocking=True)
            rollout_targets = rollout_targets.to(device, non_blocking=True)
            rollout_mask = rollout_mask.to(device, non_blocking=True)
            base_example_weights = _example_loss_weights(
                winners=winners,
                phases=phases,
                winner_weight=winner_weight,
                phase_weight_vector=phase_weight_vector,
            )
            current_tokens = tokens
            current_lengths = lengths

            bs = tokens.size(0)
            n_rows += bs
            pred_ids_by_step: List[List[int]] = []
            valid_mask_cpu = rollout_mask.detach().cpu()
            target_cpu = rollout_targets.detach().cpu()
            tokens_cpu = tokens.detach().cpu()
            lengths_cpu = lengths.detach().cpu()

            for step_idx in range(horizon):
                logits = model(current_tokens, current_lengths, winners, phases, side_to_moves)
                losses = criterion(logits, rollout_targets[:, step_idx])
                valid_step = rollout_mask[:, step_idx].float()
                step_w = float(rollout_step_weights[step_idx] if step_idx < len(rollout_step_weights) else 1.0)
                weighted = losses * base_example_weights * valid_step * step_w
                total_loss += float(weighted.sum().item())
                total_weight += float((base_example_weights * valid_step * step_w).sum().item())

                pred_ids = logits.argmax(dim=1)
                pred_ids_by_step.append(pred_ids.detach().cpu().tolist())
                matches = ((pred_ids == rollout_targets[:, step_idx]) & rollout_mask[:, step_idx]).float()
                step_correct[step_idx] += float(matches.sum().item())
                step_total[step_idx] += float(valid_step.sum().item())

                if step_idx == 0:
                    top5 = logits.topk(min(5, logits.shape[-1]), dim=1).indices
                    top5_hits_step1 += float(
                        (((top5 == labels.unsqueeze(1)).any(dim=1)) & rollout_mask[:, 0]).float().sum().item()
                    )
                    top5_total_step1 += float(valid_step.sum().item())

                # Teacher forcing: append ground-truth token where present.
                append_tok = torch.where(
                    rollout_mask[:, step_idx], rollout_targets[:, step_idx], torch.zeros_like(rollout_targets[:, step_idx])
                )
                current_tokens = torch.cat([current_tokens, append_tok.unsqueeze(1)], dim=1)
                current_lengths = current_lengths + rollout_mask[:, step_idx].long()

            # Per-sample closeness/prefix/legal metrics over first closeness horizon.
            # Build boards from original contexts and advance with ground-truth rollout.
            for row_idx in range(bs):
                row_context_ids = tokens_cpu[row_idx, : int(lengths_cpu[row_idx].item())].tolist()
                board, board_ok = _board_from_context_safe(row_context_ids, inv_vocab)
                step_matches: List[bool] = []
                for step_idx in range(close_h):
                    if not bool(valid_mask_cpu[row_idx, step_idx].item()):
                        break
                    pred_id = int(pred_ids_by_step[step_idx][row_idx])
                    gt_id = int(target_cpu[row_idx, step_idx].item())
                    step_matches.append(pred_id == gt_id)

                    if board_ok:
                        pred_uci = inv_vocab.get(pred_id, "")
                        try:
                            pred_mv = chess.Move.from_uci(pred_uci)
                        except Exception:
                            legal_den += 1.0
                        else:
                            legal_den += 1.0
                            if pred_mv in board.legal_moves:
                                legal_num += 1.0

                        # Advance board with ground-truth token for teacher-forced legality context.
                        gt_uci = inv_vocab.get(gt_id, "")
                        try:
                            gt_mv = chess.Move.from_uci(gt_uci)
                        except Exception:
                            board_ok = False
                        else:
                            if gt_mv in board.legal_moves:
                                board.push(gt_mv)
                            else:
                                board_ok = False
                prefix_total += float(_prefix_match_len(step_matches, close_h))
                closeness_total += float(
                    _weighted_rollout_closeness(step_matches, rollout_step_weights, close_h)
                )

    out: Dict[str, float] = {
        "val_loss": (total_loss / total_weight) if total_weight > 0 else 0.0,
        "top1": (step_correct[0] / step_total[0]) if step_total[0] > 0 else 0.0,
        "top5": (top5_hits_step1 / top5_total_step1) if top5_total_step1 > 0 else 0.0,
        "rollout_prefix_match_len_avg": (prefix_total / n_rows) if n_rows else 0.0,
        "rollout_legal_rate": (legal_num / legal_den) if legal_den > 0 else 0.0,
        "rollout_legal_rate_denominator": legal_den,
        "rollout_weighted_continuation_score": (closeness_total / n_rows) if n_rows else 0.0,
    }
    for idx in range(horizon):
        key = f"rollout_step{idx + 1}_acc"
        out[key] = (step_correct[idx] / step_total[idx]) if step_total[idx] > 0 else 0.0
    return out


def _build_phase_weight_vector(
    device: torch.device, phase_weights: Optional[Dict[str, float]] = None
) -> torch.Tensor:
    weights = {
        PHASE_UNKNOWN: 1.0,
        PHASE_OPENING: 1.0,
        PHASE_MIDDLEGAME: 1.0,
        PHASE_ENDGAME: 1.0,
    }
    if phase_weights:
        for key, value in phase_weights.items():
            try:
                weights[str(key).strip().lower()] = float(value)
            except Exception:
                continue
    return torch.tensor(
        [
            weights[PHASE_UNKNOWN],
            weights[PHASE_OPENING],
            weights[PHASE_MIDDLEGAME],
            weights[PHASE_ENDGAME],
        ],
        dtype=torch.float32,
        device=device,
    )


def _example_loss_weights(
    winners: torch.Tensor,
    phases: torch.Tensor,
    winner_weight: float,
    phase_weight_vector: torch.Tensor,
) -> torch.Tensor:
    winner_mask = ((winners == 0) | (winners == 1)).float()
    winner_weights = 1.0 + winner_mask * (winner_weight - 1.0)
    phase_ids = phases.clamp(min=0, max=int(phase_weight_vector.numel() - 1))
    phase_weights = phase_weight_vector[phase_ids]
    return winner_weights * phase_weights


def _iter_sparsity_parameters(model: nn.Module, include_bias: bool = False):
    base = _unwrap_model(model)
    for name, param in base.named_parameters():
        if not param.requires_grad:
            continue
        is_bias = name.endswith(".bias") or name == "bias"
        if not include_bias and is_bias:
            continue
        if param.dim() < 2 and not is_bias:
            # Skip scalar/1D tensors by default; these are typically embeddings norms/bias-like stats.
            continue
        yield name, param


def _sparsity_l1_penalty(model: nn.Module, include_bias: bool = False) -> torch.Tensor:
    total_abs = None
    total_count = 0
    for _, param in _iter_sparsity_parameters(model, include_bias=include_bias):
        cur_abs = param.abs().sum()
        total_abs = cur_abs if total_abs is None else (total_abs + cur_abs)
        total_count += int(param.numel())
    if total_abs is None or total_count <= 0:
        return torch.tensor(0.0, device=next(_unwrap_model(model).parameters()).device)
    return total_abs / float(total_count)


def _compute_model_sparsity_stats(model: nn.Module, include_bias: bool = False) -> Dict[str, float]:
    zero = 0
    total = 0
    with torch.no_grad():
        for _, param in _iter_sparsity_parameters(model, include_bias=include_bias):
            total += int(param.numel())
            zero += int((param == 0).sum().item())
    frac = (float(zero) / float(total)) if total > 0 else 0.0
    return {"tracked_params": int(total), "zero_params": int(zero), "zero_fraction": float(frac)}


def _metric_value(row: Dict, metric_name: str) -> float:
    if metric_name not in ("val_loss", "top1"):
        raise ValueError(f"Unsupported metric: {metric_name}")
    return float(row.get(metric_name, 0.0))


def _metric_improved(metric_name: str, current: float, best: Optional[float], min_delta: float = 0.0) -> bool:
    if best is None:
        return True
    if metric_name == "val_loss":
        return current < (best - float(min_delta))
    if metric_name == "top1":
        return current > (best + float(min_delta))
    raise ValueError(f"Unsupported metric: {metric_name}")


def evaluate_loader(
    model,
    loader,
    device,
    topks=(1, 5),
    criterion=None,
    winner_weight: float = 1.0,
    phase_weight_vector: Optional[torch.Tensor] = None,
):
    model.eval()
    totals = {k: 0.0 for k in topks}
    total_loss = 0.0
    n = 0
    if phase_weight_vector is None:
        phase_weight_vector = _build_phase_weight_vector(device=device, phase_weights=None)
    with torch.no_grad():
        for tokens, lengths, labels, winners, phases, side_to_moves in loader:
            tokens = tokens.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            winners = winners.to(device, non_blocking=True)
            phases = phases.to(device, non_blocking=True)
            side_to_moves = side_to_moves.to(device, non_blocking=True)
            logits = model(tokens, lengths, winners, phases, side_to_moves)
            if criterion is not None:
                losses = criterion(logits, labels)
                weights = _example_loss_weights(
                    winners=winners,
                    phases=phases,
                    winner_weight=winner_weight,
                    phase_weight_vector=phase_weight_vector,
                )
                batch_loss = (losses * weights).mean().item()
            batch_metrics = compute_topk(logits, labels, topks)
            bs = labels.size(0)
            n += bs
            if criterion is not None:
                total_loss += batch_loss * bs
            for k in topks:
                totals[k] += batch_metrics[k] * bs
    out = {f"top{k}": (totals[k] / n if n else 0.0) for k in topks}
    if criterion is not None:
        out["val_loss"] = total_loss / n if n else 0.0
    return out


def _print_epoch_progress(epoch: int, epochs: int, batch_idx: int, total_batches: int, running_loss: float, seen: int) -> None:
    if total_batches <= 0:
        return
    width = 28
    frac = min(1.0, max(0.0, batch_idx / total_batches))
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    avg_loss = running_loss / max(seen, 1)
    msg = (
        f"\r[train] epoch {epoch}/{epochs} "
        f"[{bar}] {batch_idx}/{total_batches} "
        f"loss={avg_loss:.4f}"
    )
    sys.stdout.write(msg)
    sys.stdout.flush()


def _resolve_amp_autocast_dtype(amp_dtype: str, *, use_amp: bool, device: torch.device) -> Optional[torch.dtype]:
    if not use_amp or device.type != "cuda":
        return None
    mode = str(amp_dtype or "auto").strip().lower()
    if mode in {"", "auto"}:
        return None
    if mode == "fp16":
        return torch.float16
    if mode == "bf16":
        supports_bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        if not supports_bf16:
            raise RuntimeError("amp_dtype=bf16 requested but CUDA bf16 is not supported on this runtime")
        return torch.bfloat16
    raise ValueError(f"Unsupported amp_dtype: {amp_dtype}")


def train_next_move_model(
    train_rows: List[Dict],
    val_rows: List[Dict],
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    embed_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    winner_weight: float,
    use_winner: bool,
    phase_weights: Optional[Dict[str, float]] = None,
    device_str: str = "auto",
    num_workers: int = 0,
    pin_memory: bool = True,
    amp: bool = False,
    amp_dtype: str = "auto",
    restore_best: bool = True,
    use_phase_feature: bool = True,
    phase_embed_dim: int = 8,
    use_side_to_move_feature: bool = True,
    side_to_move_embed_dim: int = 4,
    lr_scheduler: str = "plateau",
    lr_scheduler_metric: str = "val_loss",
    lr_plateau_factor: float = 0.5,
    lr_plateau_patience: int = 3,
    lr_plateau_threshold: float = 1e-4,
    lr_plateau_min_lr: float = 0.0,
    early_stopping_patience: int = 0,
    early_stopping_metric: str = "val_loss",
    early_stopping_min_delta: float = 0.0,
    verbose: bool = True,
    show_progress: bool = True,
):
    random.seed(seed)
    torch.manual_seed(seed)

    vocab = build_vocab(train_rows)
    train_ds = MoveDataset(train_rows, vocab)
    val_ds = MoveDataset(val_rows, vocab)
    use_cuda = torch.cuda.is_available()
    if device_str == "auto":
        device = torch.device("cuda" if use_cuda else "cpu")
    else:
        device = torch.device(device_str)
    if device.type == "cuda" and not use_cuda:
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    pin_memory = bool(pin_memory and device.type == "cuda")
    train_loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_train,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 1
        train_loader_kwargs["persistent_workers"] = False
    train_loader = DataLoader(train_ds, shuffle=True, **train_loader_kwargs)
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collate_train,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = NextMoveLSTM(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_winner=use_winner,
        use_phase=use_phase_feature,
        phase_embed_dim=phase_embed_dim,
        use_side_to_move=use_side_to_move_feature,
        side_to_move_embed_dim=side_to_move_embed_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    has_val_rows = len(val_rows) > 0
    scheduler_kind = str(lr_scheduler or "none").strip().lower()
    scheduler_metric = str(lr_scheduler_metric or "val_loss").strip().lower()
    scheduler = None
    if scheduler_kind not in ("none", "plateau"):
        raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler}")
    if scheduler_metric not in ("val_loss", "top1"):
        raise ValueError(f"Unsupported lr_scheduler_metric: {lr_scheduler_metric}")
    if scheduler_kind == "plateau" and has_val_rows:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=("min" if scheduler_metric == "val_loss" else "max"),
            factor=float(lr_plateau_factor),
            patience=max(0, int(lr_plateau_patience)),
            threshold=float(lr_plateau_threshold),
            threshold_mode="abs",
            min_lr=float(lr_plateau_min_lr),
        )
    criterion = nn.CrossEntropyLoss(reduction="none")
    use_amp = bool(amp and device.type == "cuda")
    amp_autocast_dtype = _resolve_amp_autocast_dtype(amp_dtype, use_amp=use_amp, device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    phase_weight_vector = _build_phase_weight_vector(device=device, phase_weights=phase_weights)
    best_state_dict = None
    best_epoch = None
    best_val_loss = None
    best_top1 = None
    early_stop_metric_name = str(early_stopping_metric or "val_loss").strip().lower()
    if early_stop_metric_name not in ("val_loss", "top1"):
        raise ValueError(f"Unsupported early_stopping_metric: {early_stopping_metric}")
    early_stop_enabled = bool(int(early_stopping_patience) > 0 and has_val_rows)
    early_stop_best_metric = None
    early_stop_bad_epochs = 0
    early_stop_info = {
        "enabled": bool(int(early_stopping_patience) > 0),
        "used": False,
        "metric": early_stop_metric_name,
        "patience": int(max(0, int(early_stopping_patience))),
        "min_delta": float(early_stopping_min_delta),
        "stopped_epoch": None,
        "best_metric": None,
        "bad_epochs": 0,
    }
    if verbose:
        print(
            {
                "train_setup": {
                    "train_rows": len(train_rows),
                    "val_rows": len(val_rows),
                    "vocab_size": len(vocab),
                    "device_selected": str(device),
                    "amp_enabled": use_amp,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "num_workers": num_workers,
                    "pin_memory_effective": pin_memory,
                    "winner_weight": winner_weight,
                    "phase_weights": {
                        "unknown": float(phase_weight_vector[0].item()),
                        "opening": float(phase_weight_vector[1].item()),
                        "middlegame": float(phase_weight_vector[2].item()),
                        "endgame": float(phase_weight_vector[3].item()),
                    },
                    "use_winner": use_winner,
                    "use_phase_feature": bool(use_phase_feature),
                    "phase_embed_dim": int(phase_embed_dim),
                    "use_side_to_move_feature": bool(use_side_to_move_feature),
                    "side_to_move_embed_dim": int(side_to_move_embed_dim),
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "restore_best": bool(restore_best),
                    "restore_best_has_val": bool(has_val_rows),
                    "lr_scheduler": {
                        "kind": scheduler_kind,
                        "metric": scheduler_metric,
                        "enabled": bool(scheduler is not None),
                        "factor": float(lr_plateau_factor),
                        "patience": int(max(0, int(lr_plateau_patience))),
                        "threshold": float(lr_plateau_threshold),
                        "min_lr": float(lr_plateau_min_lr),
                    },
                    "early_stopping": {
                        "enabled": early_stop_enabled,
                        "metric": early_stop_metric_name,
                        "patience": int(max(0, int(early_stopping_patience))),
                        "min_delta": float(early_stopping_min_delta),
                    },
                }
            }
        )

    history: List[Dict] = []
    for epoch in range(1, epochs + 1):
        if verbose:
            print(f"[train] epoch {epoch}/{epochs} start")
        model.train()
        running_loss = 0.0
        seen = 0
        total_batches = len(train_loader)
        for batch_idx, (tokens, lengths, labels, winners, phases, side_to_moves) in enumerate(train_loader, start=1):
            tokens = tokens.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            winners = winners.to(device, non_blocking=True)
            phases = phases.to(device, non_blocking=True)
            side_to_moves = side_to_moves.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_autocast_dtype):
                logits = model(tokens, lengths, winners, phases, side_to_moves)
                losses = criterion(logits, labels)
                weights = _example_loss_weights(
                    winners=winners,
                    phases=phases,
                    winner_weight=winner_weight,
                    phase_weight_vector=phase_weight_vector,
                )
                loss = (losses * weights).mean()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = labels.size(0)
            seen += bs
            running_loss += loss.item() * bs
            if verbose and show_progress:
                _print_epoch_progress(epoch, epochs, batch_idx, total_batches, running_loss, seen)
        if verbose and show_progress and total_batches > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()

        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate_loader(
            model,
            val_loader,
            device,
            topks=(1, 5),
            criterion=criterion,
            winner_weight=winner_weight,
            phase_weight_vector=phase_weight_vector,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "device": str(device),
            "amp": use_amp,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **val_metrics,
        }
        history.append(row)
        if verbose:
            print(
                {
                    "epoch": epoch,
                    "train_loss": round(float(train_loss), 6),
                    "val_loss": round(float(row.get("val_loss", 0.0)), 6),
                    "top1": round(float(row.get("top1", 0.0)), 6),
                    "top5": round(float(row.get("top5", 0.0)), 6),
                }
            )

        if scheduler is not None:
            before_lr = float(optimizer.param_groups[0]["lr"])
            sched_value = _metric_value(row, scheduler_metric)
            scheduler.step(sched_value)
            after_lr = float(optimizer.param_groups[0]["lr"])
            if verbose and after_lr != before_lr:
                print(
                    {
                        "lr_scheduler_step": {
                            "epoch": epoch,
                            "metric": scheduler_metric,
                            "metric_value": round(sched_value, 6),
                            "lr_before": before_lr,
                            "lr_after": after_lr,
                        }
                    }
                )

        if restore_best and has_val_rows:
            cur_val_loss = float(row.get("val_loss", 0.0))
            cur_top1 = float(row.get("top1", 0.0))
            is_better = (
                best_val_loss is None
                or cur_val_loss < best_val_loss
                or (cur_val_loss == best_val_loss and (best_top1 is None or cur_top1 > best_top1))
            )
            if is_better:
                best_val_loss = cur_val_loss
                best_top1 = cur_top1
                best_epoch = epoch
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if verbose:
                    print(
                        {
                            "best_checkpoint_update": {
                                "epoch": epoch,
                                "val_loss": round(cur_val_loss, 6),
                                "top1": round(cur_top1, 6),
                            }
                        }
                    )

        if early_stop_enabled:
            cur_metric = _metric_value(row, early_stop_metric_name)
            if _metric_improved(
                early_stop_metric_name,
                cur_metric,
                early_stop_best_metric,
                min_delta=float(early_stopping_min_delta),
            ):
                early_stop_best_metric = cur_metric
                early_stop_bad_epochs = 0
            else:
                early_stop_bad_epochs += 1
                if early_stop_bad_epochs >= int(early_stopping_patience):
                    early_stop_info.update(
                        {
                            "used": True,
                            "stopped_epoch": int(epoch),
                            "best_metric": float(early_stop_best_metric) if early_stop_best_metric is not None else None,
                            "bad_epochs": int(early_stop_bad_epochs),
                        }
                    )
                    if verbose:
                        print({"early_stopping_triggered": early_stop_info})
                    break

    if early_stop_enabled and not early_stop_info["used"]:
        early_stop_info.update(
            {
                "best_metric": float(early_stop_best_metric) if early_stop_best_metric is not None else None,
                "bad_epochs": int(early_stop_bad_epochs),
            }
        )

    best_checkpoint_info = {
        "enabled": bool(restore_best),
        "used": False,
        "metric": "val_loss",
        "best_epoch": None,
        "best_val_loss": None,
    }
    if restore_best and has_val_rows and best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        best_checkpoint_info.update(
            {
                "used": True,
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val_loss),
            }
        )
        if verbose:
            print({"best_checkpoint_restored": best_checkpoint_info})
    elif verbose:
        print({"best_checkpoint_restored": best_checkpoint_info})

    artifact = {
        "artifact_format_version": 2,
        "model_family": "next_move_lstm",
        "training_objective": "single_step_next_move",
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "config": {
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "use_winner": use_winner,
            "use_phase": bool(use_phase_feature),
            "phase_embed_dim": int(phase_embed_dim),
            "use_side_to_move": bool(use_side_to_move_feature),
            "side_to_move_embed_dim": int(side_to_move_embed_dim),
        },
        "runtime": {
            "device": str(device),
            "amp": use_amp,
            "best_checkpoint": best_checkpoint_info,
            "early_stopping": early_stop_info,
            "lr_scheduler": {
                "kind": scheduler_kind,
                "metric": scheduler_metric,
                "enabled": bool(scheduler is not None),
                "factor": float(lr_plateau_factor),
                "patience": int(max(0, int(lr_plateau_patience))),
                "threshold": float(lr_plateau_threshold),
                "min_lr": float(lr_plateau_min_lr),
                "final_lr": float(optimizer.param_groups[0]["lr"]),
            },
            "phase_weights": {
                "unknown": float(phase_weight_vector[0].item()),
                "opening": float(phase_weight_vector[1].item()),
                "middlegame": float(phase_weight_vector[2].item()),
                "endgame": float(phase_weight_vector[3].item()),
            },
            "training_objective": "single_step_next_move",
        },
    }
    return artifact, history


def _train_next_move_model_from_jsonl_paths_multistep(
    train_paths: List[str],
    val_paths: List[str],
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    embed_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    winner_weight: float,
    use_winner: bool,
    phase_weights: Optional[Dict[str, float]] = None,
    device_str: str = "auto",
    num_workers: int = 0,
    pin_memory: bool = True,
    amp: bool = False,
    amp_dtype: str = "auto",
    restore_best: bool = True,
    use_phase_feature: bool = True,
    phase_embed_dim: int = 8,
    use_side_to_move_feature: bool = True,
    side_to_move_embed_dim: int = 4,
    lr_scheduler: str = "plateau",
    lr_scheduler_metric: str = "val_loss",
    lr_plateau_factor: float = 0.5,
    lr_plateau_patience: int = 3,
    lr_plateau_threshold: float = 1e-4,
    lr_plateau_min_lr: float = 0.0,
    early_stopping_patience: int = 0,
    early_stopping_metric: str = "val_loss",
    early_stopping_min_delta: float = 0.0,
    verbose: bool = True,
    show_progress: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    rollout_horizon: int = 8,
    closeness_horizon: int = 4,
    rollout_loss_decay: float = 0.7,
    runtime_min_context: int = 8,
    runtime_min_target: int = 1,
    runtime_max_samples_per_game: int = 0,
    require_runtime_splice_cache: bool = False,
    distributed_enabled: bool = False,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
):
    is_primary = _is_primary_process(distributed_enabled=distributed_enabled, distributed_rank=distributed_rank)
    sparsity_mode_norm = "off"
    sparsity_l1_lambda_value = 0.0
    sparsity_enabled = False
    sparsity_include_bias = False
    random.seed(seed)
    torch.manual_seed(seed)
    train_paths = [os.fspath(p) for p in train_paths]
    val_paths = [os.fspath(p) for p in val_paths]
    rollout_horizon = max(1, int(rollout_horizon))
    closeness_horizon = max(1, min(int(closeness_horizon), rollout_horizon))
    rollout_step_weights = _build_rollout_step_weights(rollout_horizon, rollout_loss_decay)
    runtime_cfg = RuntimeSpliceConfig(
        min_context=int(runtime_min_context),
        min_target=int(runtime_min_target),
        max_samples_per_game=int(runtime_max_samples_per_game),
        seed=int(seed),
    )
    schema_kind = _sniff_paths_schema(train_paths + val_paths)
    train_game_rows_by_file: Dict[str, int] = {}
    val_game_rows_by_file: Dict[str, int] = {}
    if schema_kind == "game":
        if bool(require_runtime_splice_cache):
            train_cache_out, _train_cache_reason = _index_game_jsonl_paths_from_runtime_cache(
                train_paths,
                runtime_cfg,
                expected_split="train",
            )
            val_cache_out, _val_cache_reason = _index_game_jsonl_paths_from_runtime_cache(
                val_paths,
                runtime_cfg,
                expected_split="val",
            )
            if train_cache_out is None or val_cache_out is None:
                raise RuntimeError(
                    "Runtime splice cache required but unavailable "
                    f"(train={_train_cache_reason}, val={_val_cache_reason})"
                )
            train_index = train_cache_out
            val_index = val_cache_out
            train_cache_used = True
            val_cache_used = True
        else:
            train_index, train_cache_used, _train_cache_reason = _index_game_jsonl_paths_cached_or_runtime(
                train_paths,
                runtime_cfg,
                expected_split="train",
            )
            val_index, val_cache_used, _val_cache_reason = _index_game_jsonl_paths_cached_or_runtime(
                val_paths,
                runtime_cfg,
                expected_split="val",
            )
        (
            vocab,
            train_rows_by_file,
            train_rows_total,
            train_game_rows_by_file,
            train_games_total,
        ) = _build_vocab_and_count_rows_from_train_game_paths(train_paths, runtime_cfg)
        (
            val_rows_by_file,
            val_rows_total,
            val_game_rows_by_file,
            val_games_total,
        ) = _count_rows_in_game_jsonl_paths_runtime_splice(val_paths, runtime_cfg)
        cache_load_reason_by_split = {
            "train": _cache_load_reason_label(used_cache=train_cache_used, reason=_train_cache_reason),
            "val": _cache_load_reason_label(used_cache=val_cache_used, reason=_val_cache_reason),
        }
        train_ds = IndexedJsonlGameRolloutDataset(
            train_index[0], train_index[1], train_index[2], train_index[3], train_index[4], vocab=vocab, rollout_horizon=rollout_horizon
        )
        val_ds = IndexedJsonlGameRolloutDataset(
            val_index[0], val_index[1], val_index[2], val_index[3], val_index[4], vocab=vocab, rollout_horizon=rollout_horizon
        )
        train_runtime_index_bytes = _runtime_index_memory_bytes(train_index[1], train_index[2], train_index[3], train_index[4])
        val_runtime_index_bytes = _runtime_index_memory_bytes(val_index[1], val_index[2], val_index[3], val_index[4])
        if train_cache_used and val_cache_used:
            data_loading_mode = "indexed_game_jsonl_runtime_splice_cache"
        elif train_cache_used or val_cache_used:
            data_loading_mode = "indexed_game_jsonl_runtime_splice_hybrid"
        else:
            data_loading_mode = "indexed_game_jsonl_runtime_splice"
    else:
        vocab, train_rows_by_file, train_rows_total = _build_vocab_and_count_rows_from_train_paths(train_paths)
        val_rows_by_file, val_rows_total = _count_rows_in_jsonl_paths(val_paths)
        train_games_total = None
        val_games_total = None
        train_ds = IndexedJsonlRolloutDataset(*_index_jsonl_paths(train_paths), vocab=vocab, rollout_horizon=rollout_horizon)
        val_ds = IndexedJsonlRolloutDataset(*_index_jsonl_paths(val_paths), vocab=vocab, rollout_horizon=rollout_horizon)
        data_loading_mode = "indexed_jsonl_on_demand"
        train_runtime_index_bytes = None
        val_runtime_index_bytes = None
        cache_load_reason_by_split = None
    if train_rows_total <= 0:
        raise RuntimeError("No training rows found")
    inv_vocab = {idx: tok for tok, idx in vocab.items()}

    use_cuda = torch.cuda.is_available()
    if device_str == "auto":
        device = torch.device("cuda" if use_cuda else "cpu")
    else:
        device = torch.device(device_str)
    if device.type == "cuda" and not use_cuda:
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    if bool(distributed_enabled) and int(distributed_world_size) <= 1:
        raise RuntimeError("distributed_enabled requires distributed_world_size > 1")
    pin_memory = bool(pin_memory and device.type == "cuda")
    train_sampler = None
    if bool(distributed_enabled):
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=int(distributed_world_size),
            rank=int(distributed_rank),
            shuffle=True,
            seed=int(seed),
        )
    train_loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_train_rollout,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 1
        train_loader_kwargs["persistent_workers"] = False
    train_loader = DataLoader(
        train_ds,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collate_train_rollout,
        num_workers=0,
        pin_memory=pin_memory,
    )

    model = NextMoveLSTM(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_winner=use_winner,
        use_phase=use_phase_feature,
        phase_embed_dim=phase_embed_dim,
        use_side_to_move=use_side_to_move_feature,
        side_to_move_embed_dim=side_to_move_embed_dim,
    ).to(device)
    train_model: nn.Module = model
    if bool(distributed_enabled):
        ddp_kwargs: Dict[str, Any] = {}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [int(device.index) if device.index is not None else 0]
            ddp_kwargs["output_device"] = int(device.index) if device.index is not None else 0
        train_model = DDP(model, **ddp_kwargs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    has_val_rows = len(val_ds) > 0
    scheduler_kind = str(lr_scheduler or "none").strip().lower()
    scheduler_metric = str(lr_scheduler_metric or "val_loss").strip().lower()
    scheduler = None
    if scheduler_kind not in ("none", "plateau"):
        raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler}")
    if scheduler_metric not in ("val_loss", "top1"):
        raise ValueError(f"Unsupported lr_scheduler_metric: {lr_scheduler_metric}")
    if scheduler_kind == "plateau" and has_val_rows:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=("min" if scheduler_metric == "val_loss" else "max"),
            factor=float(lr_plateau_factor),
            patience=max(0, int(lr_plateau_patience)),
            threshold=float(lr_plateau_threshold),
            threshold_mode="abs",
            min_lr=float(lr_plateau_min_lr),
        )
    criterion = nn.CrossEntropyLoss(reduction="none")
    use_amp = bool(amp and device.type == "cuda")
    amp_autocast_dtype = _resolve_amp_autocast_dtype(amp_dtype, use_amp=use_amp, device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    phase_weight_vector = _build_phase_weight_vector(device=device, phase_weights=phase_weights)
    best_state_dict = None
    best_epoch = None
    best_val_loss = None
    best_top1 = None
    early_stop_metric_name = str(early_stopping_metric or "val_loss").strip().lower()
    if early_stop_metric_name not in ("val_loss", "top1"):
        raise ValueError(f"Unsupported early_stopping_metric: {early_stopping_metric}")
    early_stop_enabled = bool(int(early_stopping_patience) > 0 and has_val_rows)
    early_stop_best_metric = None
    early_stop_bad_epochs = 0
    early_stop_info = {
        "enabled": bool(int(early_stopping_patience) > 0),
        "used": False,
        "metric": early_stop_metric_name,
        "patience": int(max(0, int(early_stopping_patience))),
        "min_delta": float(early_stopping_min_delta),
        "stopped_epoch": None,
        "best_metric": None,
        "bad_epochs": 0,
    }
    if verbose and is_primary:
        print(
            {
                "train_setup": {
                    "train_rows": len(train_ds),
                    "val_rows": len(val_ds),
                    "vocab_size": len(vocab),
                    "device_selected": str(device),
                    "amp_enabled": use_amp,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "num_workers": num_workers,
                    "pin_memory_effective": pin_memory,
                    "winner_weight": winner_weight,
                    "phase_weights": {
                        "unknown": float(phase_weight_vector[0].item()),
                        "opening": float(phase_weight_vector[1].item()),
                        "middlegame": float(phase_weight_vector[2].item()),
                        "endgame": float(phase_weight_vector[3].item()),
                    },
                    "use_winner": use_winner,
                    "use_phase_feature": bool(use_phase_feature),
                    "phase_embed_dim": int(phase_embed_dim),
                    "use_side_to_move_feature": bool(use_side_to_move_feature),
                    "side_to_move_embed_dim": int(side_to_move_embed_dim),
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "restore_best": bool(restore_best),
                    "restore_best_has_val": bool(has_val_rows),
                    "lr_scheduler": {
                        "kind": scheduler_kind,
                        "metric": scheduler_metric,
                        "enabled": bool(scheduler is not None),
                        "factor": float(lr_plateau_factor),
                        "patience": int(max(0, int(lr_plateau_patience))),
                        "threshold": float(lr_plateau_threshold),
                        "min_lr": float(lr_plateau_min_lr),
                    },
                    "early_stopping": {
                        "enabled": early_stop_enabled,
                        "metric": early_stop_metric_name,
                        "patience": int(max(0, int(early_stopping_patience))),
                        "min_delta": float(early_stopping_min_delta),
                    },
                    "data_loading": data_loading_mode,
                    "dataset_schema": schema_kind,
                    "cache_load_reason_by_split": cache_load_reason_by_split,
                    "training_objective": "multistep_teacher_forced_recursive",
                    "rollout_horizon": int(rollout_horizon),
                    "closeness_horizon": int(closeness_horizon),
                    "rollout_loss_decay": float(rollout_loss_decay),
                    "rollout_loss_weights": [float(x) for x in rollout_step_weights],
                    "distributed": {
                        "enabled": bool(distributed_enabled),
                        "world_size": int(distributed_world_size),
                        "rank": int(distributed_rank),
                    },
                    "sparsity": {
                        "mode": str(sparsity_mode_norm),
                        "enabled": bool(sparsity_enabled),
                        "l1_lambda": float(sparsity_l1_lambda_value),
                        "include_bias": bool(sparsity_include_bias),
                    },
                }
            }
        )
    if progress_callback is not None and is_primary:
        progress_callback(
            {
                "event": "train_setup",
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "train_rows": int(len(train_ds)),
                "val_rows": int(len(val_ds)),
                "device_selected": str(device),
                "amp_enabled": bool(use_amp),
                "num_workers": int(num_workers),
                "pin_memory": bool(pin_memory),
                "data_loading": data_loading_mode,
                "dataset_schema": schema_kind,
                "cache_load_reason_by_split": cache_load_reason_by_split,
                "training_objective": "multistep_teacher_forced_recursive",
                "rollout_horizon": int(rollout_horizon),
                "closeness_horizon": int(closeness_horizon),
                "distributed": {
                    "enabled": bool(distributed_enabled),
                    "world_size": int(distributed_world_size),
                    "rank": int(distributed_rank),
                },
                "sparsity": {
                    "mode": str(sparsity_mode_norm),
                    "enabled": bool(sparsity_enabled),
                    "l1_lambda": float(sparsity_l1_lambda_value),
                    "include_bias": bool(sparsity_include_bias),
                },
            }
        )

    history: List[Dict] = []
    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(int(epoch))
        if verbose and is_primary:
            print(f"[train] epoch {epoch}/{epochs} start")
        if progress_callback is not None and is_primary:
            progress_callback(
                {
                    "event": "epoch_start",
                    "epoch": int(epoch),
                    "epochs": int(epochs),
                    "train_batches_total": int(len(train_loader)),
                }
            )

        train_model.train()
        running_loss = 0.0
        running_weight = 0.0
        total_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader, start=1):
            (
                tokens,
                lengths,
                labels,
                winners,
                phases,
                side_to_moves,
                rollout_targets,
                rollout_mask,
            ) = batch
            tokens = tokens.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            winners = winners.to(device, non_blocking=True)
            phases = phases.to(device, non_blocking=True)
            side_to_moves = side_to_moves.to(device, non_blocking=True)
            rollout_targets = rollout_targets.to(device, non_blocking=True)
            rollout_mask = rollout_mask.to(device, non_blocking=True)
            example_weights = _example_loss_weights(
                winners=winners,
                phases=phases,
                winner_weight=winner_weight,
                phase_weight_vector=phase_weight_vector,
            )

            optimizer.zero_grad(set_to_none=True)
            current_tokens = tokens
            current_lengths = lengths
            total_batch_loss = None
            total_batch_weight = None
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_autocast_dtype):
                for step_idx in range(rollout_horizon):
                    logits = train_model(current_tokens, current_lengths, winners, phases, side_to_moves)
                    losses = criterion(logits, rollout_targets[:, step_idx])
                    valid_step = rollout_mask[:, step_idx].float()
                    step_weight = float(rollout_step_weights[step_idx])
                    contrib = losses * example_weights * valid_step * step_weight
                    contrib_weight = example_weights * valid_step * step_weight
                    if total_batch_loss is None:
                        total_batch_loss = contrib.sum()
                        total_batch_weight = contrib_weight.sum()
                    else:
                        total_batch_loss = total_batch_loss + contrib.sum()
                        total_batch_weight = total_batch_weight + contrib_weight.sum()

                    append_tok = torch.where(
                        rollout_mask[:, step_idx], rollout_targets[:, step_idx], torch.zeros_like(rollout_targets[:, step_idx])
                    )
                    current_tokens = torch.cat([current_tokens, append_tok.unsqueeze(1)], dim=1)
                    current_lengths = current_lengths + rollout_mask[:, step_idx].long()
                if total_batch_loss is None or total_batch_weight is None or float(total_batch_weight.item()) <= 0.0:
                    loss = torch.zeros((), device=device, dtype=torch.float32)
                else:
                    loss = total_batch_loss / total_batch_weight.clamp_min(1e-12)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_weight_val = float(total_batch_weight.item()) if total_batch_weight is not None else 0.0
            running_weight += batch_weight_val
            running_loss += float(loss.item()) * batch_weight_val
            if verbose and show_progress and is_primary:
                # Reuse progress helper; "seen" is weighted effective count in multistep mode.
                _print_epoch_progress(epoch, epochs, batch_idx, total_batches, running_loss, int(max(running_weight, 1)))
        if verbose and show_progress and total_batches > 0 and is_primary:
            sys.stdout.write("\n")
            sys.stdout.flush()

        train_loss = running_loss / max(running_weight, 1e-12)
        val_metrics = evaluate_loader_multistep(
            model=train_model,
            loader=val_loader,
            device=device,
            criterion=criterion,
            winner_weight=winner_weight,
            phase_weight_vector=phase_weight_vector,
            rollout_horizon=rollout_horizon,
            closeness_horizon=closeness_horizon,
            rollout_step_weights=rollout_step_weights,
            inv_vocab=inv_vocab,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "device": str(device),
            "amp": use_amp,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **val_metrics,
        }
        history.append(row)
        if progress_callback is not None and is_primary:
            progress_callback(
                {
                    "event": "epoch_end",
                    "epoch": int(epoch),
                    "epochs": int(epochs),
                    "metrics": {
                        "train_loss": float(train_loss),
                        "val_loss": float(row.get("val_loss", 0.0)),
                        "top1": float(row.get("top1", 0.0)),
                        "top5": float(row.get("top5", 0.0)),
                        "lr": float(row.get("lr", optimizer.param_groups[0]["lr"])),
                        "rollout_step2_acc": float(row.get("rollout_step2_acc", 0.0)),
                        "rollout_step4_acc": float(row.get("rollout_step4_acc", 0.0)),
                        "rollout_step8_acc": float(row.get("rollout_step8_acc", 0.0)),
                        "rollout_prefix_match_len_avg": float(row.get("rollout_prefix_match_len_avg", 0.0)),
                        "rollout_legal_rate": float(row.get("rollout_legal_rate", 0.0)),
                        "rollout_weighted_continuation_score": float(
                            row.get("rollout_weighted_continuation_score", 0.0)
                        ),
                    },
                }
            )
        if verbose and is_primary:
            print(
                {
                    "epoch": epoch,
                    "train_loss": round(float(train_loss), 6),
                    "val_loss": round(float(row.get("val_loss", 0.0)), 6),
                    "top1": round(float(row.get("top1", 0.0)), 6),
                    "top5": round(float(row.get("top5", 0.0)), 6),
                    "rollout_step4_acc": round(float(row.get("rollout_step4_acc", 0.0)), 6),
                    "rollout_prefix_match_len_avg": round(float(row.get("rollout_prefix_match_len_avg", 0.0)), 6),
                    "rollout_legal_rate": round(float(row.get("rollout_legal_rate", 0.0)), 6),
                    "rollout_weighted_continuation_score": round(
                        float(row.get("rollout_weighted_continuation_score", 0.0)), 6
                    ),
                }
            )

        if scheduler is not None:
            before_lr = float(optimizer.param_groups[0]["lr"])
            sched_value = _metric_value(row, scheduler_metric)
            scheduler.step(sched_value)
            after_lr = float(optimizer.param_groups[0]["lr"])
            if verbose and after_lr != before_lr and is_primary:
                print(
                    {
                        "lr_scheduler_step": {
                            "epoch": epoch,
                            "metric": scheduler_metric,
                            "metric_value": round(sched_value, 6),
                            "lr_before": before_lr,
                            "lr_after": after_lr,
                        }
                    }
                )

        if restore_best and has_val_rows:
            cur_val_loss = float(row.get("val_loss", 0.0))
            cur_top1 = float(row.get("top1", 0.0))
            is_better = (
                best_val_loss is None
                or cur_val_loss < best_val_loss
                or (cur_val_loss == best_val_loss and (best_top1 is None or cur_top1 > best_top1))
            )
            if is_better:
                best_val_loss = cur_val_loss
                best_top1 = cur_top1
                best_epoch = epoch
                best_state_dict = _cpu_cloned_state_dict(train_model)
                if verbose and is_primary:
                    print(
                        {
                            "best_checkpoint_update": {
                                "epoch": epoch,
                                "val_loss": round(cur_val_loss, 6),
                                "top1": round(cur_top1, 6),
                            }
                        }
                    )

        if early_stop_enabled:
            cur_metric = _metric_value(row, early_stop_metric_name)
            if _metric_improved(
                early_stop_metric_name,
                cur_metric,
                early_stop_best_metric,
                min_delta=float(early_stopping_min_delta),
            ):
                early_stop_best_metric = cur_metric
                early_stop_bad_epochs = 0
            else:
                early_stop_bad_epochs += 1
                if early_stop_bad_epochs >= int(early_stopping_patience):
                    early_stop_info.update(
                        {
                            "used": True,
                            "stopped_epoch": int(epoch),
                            "best_metric": float(early_stop_best_metric) if early_stop_best_metric is not None else None,
                            "bad_epochs": int(early_stop_bad_epochs),
                        }
                    )
                    if verbose and is_primary:
                        print({"early_stopping_triggered": early_stop_info})
                    if progress_callback is not None and is_primary:
                        progress_callback(
                            {
                                "event": "early_stopping_triggered",
                                "epoch": int(epoch),
                                "epochs": int(epochs),
                                "metric": early_stop_metric_name,
                                "bad_epochs": int(early_stop_bad_epochs),
                                "best_metric": (
                                    float(early_stop_best_metric) if early_stop_best_metric is not None else None
                                ),
                            }
                        )
                    break

    if early_stop_enabled and not early_stop_info["used"]:
        early_stop_info.update(
            {
                "best_metric": float(early_stop_best_metric) if early_stop_best_metric is not None else None,
                "bad_epochs": int(early_stop_bad_epochs),
            }
        )

    best_checkpoint_info = {
        "enabled": bool(restore_best),
        "used": False,
        "metric": "val_loss",
        "best_epoch": None,
        "best_val_loss": None,
    }
    if restore_best and has_val_rows and best_state_dict is not None:
        _unwrap_model(train_model).load_state_dict(best_state_dict)
        best_checkpoint_info.update(
            {
                "used": True,
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val_loss),
            }
        )
        if verbose and is_primary:
            print({"best_checkpoint_restored": best_checkpoint_info})
    elif verbose and is_primary:
        print({"best_checkpoint_restored": best_checkpoint_info})

    sparsity_stats = _compute_model_sparsity_stats(
        train_model,
        include_bias=bool(sparsity_include_bias),
    )

    artifact = {
        "artifact_format_version": 2,
        "model_family": "next_move_lstm",
        "training_objective": "multistep_teacher_forced_recursive",
        "state_dict": _unwrap_model(train_model).state_dict(),
        "vocab": vocab,
        "config": {
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "use_winner": use_winner,
            "use_phase": bool(use_phase_feature),
            "phase_embed_dim": int(phase_embed_dim),
            "use_side_to_move": bool(use_side_to_move_feature),
            "side_to_move_embed_dim": int(side_to_move_embed_dim),
        },
        "runtime": {
            "device": str(device),
            "amp": use_amp,
            "best_checkpoint": best_checkpoint_info,
            "early_stopping": early_stop_info,
            "lr_scheduler": {
                "kind": scheduler_kind,
                "metric": scheduler_metric,
                "enabled": bool(scheduler is not None),
                "factor": float(lr_plateau_factor),
                "patience": int(max(0, int(lr_plateau_patience))),
                "threshold": float(lr_plateau_threshold),
                "min_lr": float(lr_plateau_min_lr),
                "final_lr": float(optimizer.param_groups[0]["lr"]),
            },
            "phase_weights": {
                "unknown": float(phase_weight_vector[0].item()),
                "opening": float(phase_weight_vector[1].item()),
                "middlegame": float(phase_weight_vector[2].item()),
                "endgame": float(phase_weight_vector[3].item()),
            },
            "training_objective": "multistep_teacher_forced_recursive",
            "rollout_horizon": int(rollout_horizon),
            "closeness_horizon": int(closeness_horizon),
            "rollout_loss_decay": float(rollout_loss_decay),
            "rollout_loss_weights": [float(x) for x in rollout_step_weights],
            "distributed": {
                "enabled": bool(distributed_enabled),
                "world_size": int(distributed_world_size),
                "rank": int(distributed_rank),
            },
        },
    }
    dataset_info = {
        "train_rows": train_rows_total,
        "val_rows": val_rows_total,
        "train_rows_by_file": train_rows_by_file,
        "val_rows_by_file": val_rows_by_file,
        "train_index_rows": len(train_ds),
        "val_index_rows": len(val_ds),
        "vocab_size": len(vocab),
        "data_loading": data_loading_mode,
        "dataset_schema": schema_kind,
        "training_objective": "multistep_teacher_forced_recursive",
        "rollout_horizon": int(rollout_horizon),
        "closeness_horizon": int(closeness_horizon),
        "rollout_loss_decay": float(rollout_loss_decay),
        "distributed": {
            "enabled": bool(distributed_enabled),
            "world_size": int(distributed_world_size),
            "rank": int(distributed_rank),
        },
    }
    if schema_kind == "game":
        dataset_info.update(
            {
                "train_games": int(train_games_total or 0),
                "val_games": int(val_games_total or 0),
                "train_games_by_file": train_game_rows_by_file,
                "val_games_by_file": val_game_rows_by_file,
                "runtime_splice": {
                    "min_context": int(runtime_cfg.min_context),
                    "min_target": int(runtime_cfg.min_target),
                    "max_samples_per_game": int(runtime_cfg.max_samples_per_game),
                },
                "cache_load_reason_by_split": cache_load_reason_by_split,
                "runtime_splice_index_bytes_train": int(train_runtime_index_bytes or 0),
                "runtime_splice_index_bytes_val": int(val_runtime_index_bytes or 0),
            }
        )
    if progress_callback is not None and is_primary:
        progress_callback(
            {
                "event": "train_complete",
                "epochs_completed": int(len(history)),
                "epochs_requested": int(epochs),
                "best_checkpoint": best_checkpoint_info,
                "early_stopping": early_stop_info,
                "training_objective": "multistep_teacher_forced_recursive",
                "rollout_horizon": int(rollout_horizon),
                "closeness_horizon": int(closeness_horizon),
            }
        )
    return artifact, history, dataset_info


def train_next_move_model_from_jsonl_paths(
    train_paths: List[str],
    val_paths: List[str],
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    embed_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    winner_weight: float,
    use_winner: bool,
    phase_weights: Optional[Dict[str, float]] = None,
    device_str: str = "auto",
    num_workers: int = 0,
    pin_memory: bool = True,
    amp: bool = False,
    amp_dtype: str = "auto",
    restore_best: bool = True,
    use_phase_feature: bool = True,
    phase_embed_dim: int = 8,
    use_side_to_move_feature: bool = True,
    side_to_move_embed_dim: int = 4,
    lr_scheduler: str = "plateau",
    lr_scheduler_metric: str = "val_loss",
    lr_plateau_factor: float = 0.5,
    lr_plateau_patience: int = 3,
    lr_plateau_threshold: float = 1e-4,
    lr_plateau_min_lr: float = 0.0,
    early_stopping_patience: int = 0,
    early_stopping_metric: str = "val_loss",
    early_stopping_min_delta: float = 0.0,
    verbose: bool = True,
    show_progress: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    rollout_horizon: int = 1,
    closeness_horizon: int = 4,
    rollout_loss_decay: float = 0.7,
    runtime_min_context: int = 8,
    runtime_min_target: int = 1,
    runtime_max_samples_per_game: int = 0,
    require_runtime_splice_cache: bool = False,
    max_train_rows: int = 0,
    max_val_rows: int = 0,
    max_total_rows: int = 0,
    best_checkpoint_out: str = "",
    epoch_checkpoint_dir: str = "",
    distributed_enabled: bool = False,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
    sparsity_mode: str = "off",
    sparsity_l1_lambda: float = 0.0,
    sparsity_include_bias: bool = False,
):
    sparsity_mode_norm = str(sparsity_mode or "off").strip().lower()
    if sparsity_mode_norm not in {"off", "l1"}:
        raise ValueError(f"Unsupported sparsity_mode: {sparsity_mode}")
    sparsity_l1_lambda_value = max(0.0, float(sparsity_l1_lambda))
    sparsity_enabled = bool(sparsity_mode_norm != "off" and sparsity_l1_lambda_value > 0.0)
    if int(rollout_horizon) > 1:
        if sparsity_enabled:
            raise ValueError("sparsity_mode is currently supported only for rollout_horizon=1")
        return _train_next_move_model_from_jsonl_paths_multistep(
            train_paths=train_paths,
            val_paths=val_paths,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            winner_weight=winner_weight,
            use_winner=use_winner,
            phase_weights=phase_weights,
            device_str=device_str,
            num_workers=num_workers,
            pin_memory=pin_memory,
            amp=amp,
            amp_dtype=amp_dtype,
            restore_best=restore_best,
            use_phase_feature=use_phase_feature,
            phase_embed_dim=phase_embed_dim,
            use_side_to_move_feature=use_side_to_move_feature,
            side_to_move_embed_dim=side_to_move_embed_dim,
            lr_scheduler=lr_scheduler,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_plateau_factor=lr_plateau_factor,
            lr_plateau_patience=lr_plateau_patience,
            lr_plateau_threshold=lr_plateau_threshold,
            lr_plateau_min_lr=lr_plateau_min_lr,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            early_stopping_min_delta=early_stopping_min_delta,
            verbose=verbose,
            show_progress=show_progress,
            progress_callback=progress_callback,
            rollout_horizon=int(rollout_horizon),
            closeness_horizon=int(closeness_horizon),
            rollout_loss_decay=float(rollout_loss_decay),
            runtime_min_context=int(runtime_min_context),
            runtime_min_target=int(runtime_min_target),
            runtime_max_samples_per_game=int(runtime_max_samples_per_game),
            require_runtime_splice_cache=bool(require_runtime_splice_cache),
            distributed_enabled=bool(distributed_enabled),
            distributed_rank=int(distributed_rank),
            distributed_world_size=int(distributed_world_size),
        )
    is_primary = _is_primary_process(distributed_enabled=distributed_enabled, distributed_rank=distributed_rank)
    random.seed(seed)
    torch.manual_seed(seed)

    train_paths = [os.fspath(p) for p in train_paths]
    val_paths = [os.fspath(p) for p in val_paths]
    runtime_cfg = RuntimeSpliceConfig(
        min_context=int(runtime_min_context),
        min_target=int(runtime_min_target),
        max_samples_per_game=int(runtime_max_samples_per_game),
        seed=int(seed),
    )
    schema_kind = _sniff_paths_schema(train_paths + val_paths)
    train_game_rows_by_file: Dict[str, int] = {}
    val_game_rows_by_file: Dict[str, int] = {}
    if schema_kind == "game":
        if bool(require_runtime_splice_cache):
            train_cache_out, _train_cache_reason = _index_game_jsonl_paths_from_runtime_cache(
                train_paths,
                runtime_cfg,
                expected_split="train",
            )
            val_cache_out, _val_cache_reason = _index_game_jsonl_paths_from_runtime_cache(
                val_paths,
                runtime_cfg,
                expected_split="val",
            )
            if train_cache_out is None or val_cache_out is None:
                raise RuntimeError(
                    "Runtime splice cache required but unavailable "
                    f"(train={_train_cache_reason}, val={_val_cache_reason})"
                )
            train_index = train_cache_out
            val_index = val_cache_out
            train_cache_used = True
            val_cache_used = True
        else:
            train_index, train_cache_used, _train_cache_reason = _index_game_jsonl_paths_cached_or_runtime(
                train_paths,
                runtime_cfg,
                expected_split="train",
            )
            val_index, val_cache_used, _val_cache_reason = _index_game_jsonl_paths_cached_or_runtime(
                val_paths,
                runtime_cfg,
                expected_split="val",
            )
        (
            vocab,
            train_rows_by_file,
            train_rows_total,
            train_game_rows_by_file,
            train_games_total,
        ) = _build_vocab_and_count_rows_from_train_game_paths(train_paths, runtime_cfg)
        (
            val_rows_by_file,
            val_rows_total,
            val_game_rows_by_file,
            val_games_total,
        ) = _count_rows_in_game_jsonl_paths_runtime_splice(val_paths, runtime_cfg)
        cache_load_reason_by_split = {
            "train": _cache_load_reason_label(used_cache=train_cache_used, reason=_train_cache_reason),
            "val": _cache_load_reason_label(used_cache=val_cache_used, reason=_val_cache_reason),
        }
        train_ds = IndexedJsonlGameSpliceDataset(
            train_index[0], train_index[1], train_index[2], train_index[3], train_index[4], vocab=vocab
        )
        val_ds = IndexedJsonlGameSpliceDataset(
            val_index[0], val_index[1], val_index[2], val_index[3], val_index[4], vocab=vocab
        )
        train_runtime_index_bytes = _runtime_index_memory_bytes(train_index[1], train_index[2], train_index[3], train_index[4])
        val_runtime_index_bytes = _runtime_index_memory_bytes(val_index[1], val_index[2], val_index[3], val_index[4])
        if train_cache_used and val_cache_used:
            data_loading_mode = "indexed_game_jsonl_runtime_splice_cache"
        elif train_cache_used or val_cache_used:
            data_loading_mode = "indexed_game_jsonl_runtime_splice_hybrid"
        else:
            data_loading_mode = "indexed_game_jsonl_runtime_splice"
    else:
        # Stream train files once to build vocabulary and exact row counts.
        vocab, train_rows_by_file, train_rows_total = _build_vocab_and_count_rows_from_train_paths(train_paths)
        # Count validation rows without loading them.
        val_rows_by_file, val_rows_total = _count_rows_in_jsonl_paths(val_paths)
        train_games_total = None
        val_games_total = None
        train_ds = IndexedJsonlDataset(*_index_jsonl_paths(train_paths), vocab=vocab)
        val_ds = IndexedJsonlDataset(*_index_jsonl_paths(val_paths), vocab=vocab)
        data_loading_mode = "indexed_jsonl_on_demand"
        train_runtime_index_bytes = None
        val_runtime_index_bytes = None
        cache_load_reason_by_split = None

    if int(max_total_rows) > 0 and int(max_train_rows) <= 0 and int(max_val_rows) <= 0:
        total_rows_source = int(max(0, train_rows_total)) + int(max(0, val_rows_total))
        if total_rows_source > 0:
            if int(train_rows_total) > 0 and int(val_rows_total) > 0:
                train_share = float(train_rows_total) / float(total_rows_source)
                computed_train = int(round(float(max_total_rows) * train_share))
                computed_train = max(1, min(int(train_rows_total), computed_train))
                computed_val = int(max_total_rows) - computed_train
                computed_val = max(1, min(int(val_rows_total), computed_val))
                max_train_rows = int(computed_train)
                max_val_rows = int(computed_val)
            elif int(train_rows_total) > 0:
                max_train_rows = min(int(max_total_rows), int(train_rows_total))
            elif int(val_rows_total) > 0:
                max_val_rows = min(int(max_total_rows), int(val_rows_total))

    train_rows_source_total = int(train_rows_total)
    val_rows_source_total = int(val_rows_total)
    train_subset_applied = False
    val_subset_applied = False
    if int(max_train_rows) > 0 and int(max_train_rows) < int(len(train_ds)):
        keep_train = int(max_train_rows)
        train_ds = torch.utils.data.Subset(
            train_ds,
            _sample_subset_indices(total=len(train_ds), keep=keep_train, seed=int(seed) + 101),
        )
        train_rows_total = int(len(train_ds))
        train_subset_applied = True
    if int(max_val_rows) > 0 and int(max_val_rows) < int(len(val_ds)):
        keep_val = int(max_val_rows)
        val_ds = torch.utils.data.Subset(
            val_ds,
            _sample_subset_indices(total=len(val_ds), keep=keep_val, seed=int(seed) + 202),
        )
        val_rows_total = int(len(val_ds))
        val_subset_applied = True

    if train_rows_total <= 0:
        raise RuntimeError("No training rows found")

    use_cuda = torch.cuda.is_available()
    if device_str == "auto":
        device = torch.device("cuda" if use_cuda else "cpu")
    else:
        device = torch.device(device_str)
    if device.type == "cuda" and not use_cuda:
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    if bool(distributed_enabled) and int(distributed_world_size) <= 1:
        raise RuntimeError("distributed_enabled requires distributed_world_size > 1")
    pin_memory = bool(pin_memory and device.type == "cuda")
    train_sampler = None
    if bool(distributed_enabled):
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=int(distributed_world_size),
            rank=int(distributed_rank),
            shuffle=True,
            seed=int(seed),
        )
    train_loader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": collate_train,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 1
        train_loader_kwargs["persistent_workers"] = False
    train_loader = DataLoader(
        train_ds,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **train_loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        batch_size=batch_size,
        collate_fn=collate_train,
        num_workers=0,
        pin_memory=pin_memory,
    )
    model = NextMoveLSTM(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_winner=use_winner,
        use_phase=use_phase_feature,
        phase_embed_dim=phase_embed_dim,
        use_side_to_move=use_side_to_move_feature,
        side_to_move_embed_dim=side_to_move_embed_dim,
    ).to(device)
    train_model: nn.Module = model
    if bool(distributed_enabled):
        ddp_kwargs: Dict[str, Any] = {}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [int(device.index) if device.index is not None else 0]
            ddp_kwargs["output_device"] = int(device.index) if device.index is not None else 0
        train_model = DDP(model, **ddp_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    has_val_rows = len(val_ds) > 0
    scheduler_kind = str(lr_scheduler or "none").strip().lower()
    scheduler_metric = str(lr_scheduler_metric or "val_loss").strip().lower()
    scheduler = None
    if scheduler_kind not in ("none", "plateau"):
        raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler}")
    if scheduler_metric not in ("val_loss", "top1"):
        raise ValueError(f"Unsupported lr_scheduler_metric: {lr_scheduler_metric}")
    if scheduler_kind == "plateau" and has_val_rows:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=("min" if scheduler_metric == "val_loss" else "max"),
            factor=float(lr_plateau_factor),
            patience=max(0, int(lr_plateau_patience)),
            threshold=float(lr_plateau_threshold),
            threshold_mode="abs",
            min_lr=float(lr_plateau_min_lr),
        )
    criterion = nn.CrossEntropyLoss(reduction="none")
    use_amp = bool(amp and device.type == "cuda")
    amp_autocast_dtype = _resolve_amp_autocast_dtype(amp_dtype, use_amp=use_amp, device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    phase_weight_vector = _build_phase_weight_vector(device=device, phase_weights=phase_weights)
    best_state_dict = None
    best_epoch = None
    best_val_loss = None
    best_top1 = None
    early_stop_metric_name = str(early_stopping_metric or "val_loss").strip().lower()
    if early_stop_metric_name not in ("val_loss", "top1"):
        raise ValueError(f"Unsupported early_stopping_metric: {early_stopping_metric}")
    early_stop_enabled = bool(int(early_stopping_patience) > 0 and has_val_rows)
    early_stop_best_metric = None
    early_stop_bad_epochs = 0
    early_stop_info = {
        "enabled": bool(int(early_stopping_patience) > 0),
        "used": False,
        "metric": early_stop_metric_name,
        "patience": int(max(0, int(early_stopping_patience))),
        "min_delta": float(early_stopping_min_delta),
        "stopped_epoch": None,
        "best_metric": None,
        "bad_epochs": 0,
    }
    if verbose:
        print(
            {
                "train_setup": {
                    "train_rows": len(train_ds),
                    "val_rows": len(val_ds),
                    "vocab_size": len(vocab),
                    "device_selected": str(device),
                    "amp_enabled": use_amp,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "lr": lr,
                    "num_workers": num_workers,
                    "pin_memory_effective": pin_memory,
                    "winner_weight": winner_weight,
                    "phase_weights": {
                        "unknown": float(phase_weight_vector[0].item()),
                        "opening": float(phase_weight_vector[1].item()),
                        "middlegame": float(phase_weight_vector[2].item()),
                        "endgame": float(phase_weight_vector[3].item()),
                    },
                    "use_winner": use_winner,
                    "use_phase_feature": bool(use_phase_feature),
                    "phase_embed_dim": int(phase_embed_dim),
                    "use_side_to_move_feature": bool(use_side_to_move_feature),
                    "side_to_move_embed_dim": int(side_to_move_embed_dim),
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "restore_best": bool(restore_best),
                    "restore_best_has_val": bool(has_val_rows),
                    "lr_scheduler": {
                        "kind": scheduler_kind,
                        "metric": scheduler_metric,
                        "enabled": bool(scheduler is not None),
                        "factor": float(lr_plateau_factor),
                        "patience": int(max(0, int(lr_plateau_patience))),
                        "threshold": float(lr_plateau_threshold),
                        "min_lr": float(lr_plateau_min_lr),
                    },
                    "early_stopping": {
                        "enabled": early_stop_enabled,
                        "metric": early_stop_metric_name,
                        "patience": int(max(0, int(early_stopping_patience))),
                        "min_delta": float(early_stopping_min_delta),
                    },
                    "data_loading": data_loading_mode,
                    "dataset_schema": schema_kind,
                    "cache_load_reason_by_split": cache_load_reason_by_split,
                    "subset_sampling": {
                        "max_total_rows": int(max_total_rows),
                        "max_train_rows": int(max_train_rows),
                        "max_val_rows": int(max_val_rows),
                        "train_rows_source": int(train_rows_source_total),
                        "val_rows_source": int(val_rows_source_total),
                        "train_subset_applied": bool(train_subset_applied),
                        "val_subset_applied": bool(val_subset_applied),
                    },
                    "distributed": {
                        "enabled": bool(distributed_enabled),
                        "world_size": int(distributed_world_size),
                        "rank": int(distributed_rank),
                    },
                }
            }
        )
    if progress_callback is not None and is_primary:
        progress_callback(
            {
                "event": "train_setup",
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "train_rows": int(len(train_ds)),
                "val_rows": int(len(val_ds)),
                "device_selected": str(device),
                "amp_enabled": bool(use_amp),
                "num_workers": int(num_workers),
                "pin_memory": bool(pin_memory),
                "data_loading": data_loading_mode,
                "dataset_schema": schema_kind,
                "cache_load_reason_by_split": cache_load_reason_by_split,
                "subset_sampling": {
                    "max_total_rows": int(max_total_rows),
                    "max_train_rows": int(max_train_rows),
                    "max_val_rows": int(max_val_rows),
                    "train_rows_source": int(train_rows_source_total),
                    "val_rows_source": int(val_rows_source_total),
                    "train_subset_applied": bool(train_subset_applied),
                    "val_subset_applied": bool(val_subset_applied),
                },
                "distributed": {
                    "enabled": bool(distributed_enabled),
                    "world_size": int(distributed_world_size),
                    "rank": int(distributed_rank),
                },
            }
        )

    best_checkpoint_out_path = str(best_checkpoint_out or "").strip()
    epoch_checkpoint_dir_path = str(epoch_checkpoint_dir or "").strip()
    if best_checkpoint_out_path and is_primary:
        Path(best_checkpoint_out_path).parent.mkdir(parents=True, exist_ok=True)
    if epoch_checkpoint_dir_path and is_primary:
        Path(epoch_checkpoint_dir_path).mkdir(parents=True, exist_ok=True)

    def _build_checkpoint_artifact(
        model_state_dict: Dict[str, torch.Tensor],
        *,
        current_epoch: int,
        latest_row: Dict[str, Any],
        checkpoint_kind: str,
    ) -> Dict[str, Any]:
        return {
            "artifact_format_version": 2,
            "model_family": "next_move_lstm",
            "training_objective": "single_step_next_move",
            "state_dict": model_state_dict,
            "vocab": vocab,
            "config": {
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "use_winner": use_winner,
                "use_phase": bool(use_phase_feature),
                "phase_embed_dim": int(phase_embed_dim),
                "use_side_to_move": bool(use_side_to_move_feature),
                "side_to_move_embed_dim": int(side_to_move_embed_dim),
            },
            "runtime": {
                "device": str(device),
                "amp": use_amp,
                "checkpoint_kind": str(checkpoint_kind),
                "checkpoint_epoch": int(current_epoch),
                "checkpoint_metrics": {
                    "train_loss": float(latest_row.get("train_loss", 0.0)),
                    "val_loss": float(latest_row.get("val_loss", 0.0)),
                    "top1": float(latest_row.get("top1", 0.0)),
                    "top5": float(latest_row.get("top5", 0.0)),
                    "lr": float(latest_row.get("lr", optimizer.param_groups[0]["lr"])),
                },
                "phase_weights": {
                    "unknown": float(phase_weight_vector[0].item()),
                    "opening": float(phase_weight_vector[1].item()),
                    "middlegame": float(phase_weight_vector[2].item()),
                    "endgame": float(phase_weight_vector[3].item()),
                },
                "training_objective": "single_step_next_move",
                "sparsity": {
                    "mode": str(sparsity_mode_norm),
                    "enabled": bool(sparsity_enabled),
                    "l1_lambda": float(sparsity_l1_lambda_value),
                    "include_bias": bool(sparsity_include_bias),
                },
            },
        }

    history: List[Dict] = []
    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(int(epoch))
        if verbose and is_primary:
            print(f"[train] epoch {epoch}/{epochs} start")
        if progress_callback is not None and is_primary:
            progress_callback(
                {
                    "event": "epoch_start",
                    "epoch": int(epoch),
                    "epochs": int(epochs),
                    "train_batches_total": int(len(train_loader)),
                }
            )
        train_model.train()
        running_loss = 0.0
        running_l1_penalty = 0.0
        seen = 0
        total_batches = len(train_loader)
        for batch_idx, (tokens, lengths, labels, winners, phases, side_to_moves) in enumerate(train_loader, start=1):
            tokens = tokens.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            winners = winners.to(device, non_blocking=True)
            phases = phases.to(device, non_blocking=True)
            side_to_moves = side_to_moves.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_autocast_dtype):
                logits = train_model(tokens, lengths, winners, phases, side_to_moves)
                losses = criterion(logits, labels)
                weights = _example_loss_weights(
                    winners=winners,
                    phases=phases,
                    winner_weight=winner_weight,
                    phase_weight_vector=phase_weight_vector,
                )
                loss = (losses * weights).mean()
                l1_penalty_value = 0.0
                if sparsity_enabled:
                    l1_penalty = _sparsity_l1_penalty(
                        train_model,
                        include_bias=bool(sparsity_include_bias),
                    )
                    l1_penalty_value = float(l1_penalty.detach().item())
                    loss = loss + (sparsity_l1_lambda_value * l1_penalty)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = labels.size(0)
            seen += bs
            running_loss += loss.item() * bs
            running_l1_penalty += l1_penalty_value * bs
            if verbose and show_progress and is_primary:
                _print_epoch_progress(epoch, epochs, batch_idx, total_batches, running_loss, seen)
        if verbose and show_progress and total_batches > 0 and is_primary:
            sys.stdout.write("\n")
            sys.stdout.flush()

        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate_loader(
            train_model,
            val_loader,
            device,
            topks=(1, 5),
            criterion=criterion,
            winner_weight=winner_weight,
            phase_weight_vector=phase_weight_vector,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_l1_penalty": (running_l1_penalty / max(seen, 1)),
            "device": str(device),
            "amp": use_amp,
            "lr": float(optimizer.param_groups[0]["lr"]),
            **val_metrics,
        }
        history.append(row)
        if progress_callback is not None and is_primary:
            progress_callback(
                {
                    "event": "epoch_end",
                    "epoch": int(epoch),
                    "epochs": int(epochs),
                    "metrics": {
                        "train_loss": float(train_loss),
                        "train_l1_penalty": float(row.get("train_l1_penalty", 0.0)),
                        "val_loss": float(row.get("val_loss", 0.0)),
                        "top1": float(row.get("top1", 0.0)),
                        "top5": float(row.get("top5", 0.0)),
                        "lr": float(row.get("lr", optimizer.param_groups[0]["lr"])),
                    },
                }
            )
        if verbose and is_primary:
            print(
                {
                    "epoch": epoch,
                    "train_loss": round(float(train_loss), 6),
                    "train_l1_penalty": round(float(row.get("train_l1_penalty", 0.0)), 6),
                    "val_loss": round(float(row.get("val_loss", 0.0)), 6),
                    "top1": round(float(row.get("top1", 0.0)), 6),
                    "top5": round(float(row.get("top5", 0.0)), 6),
                }
            )

        if scheduler is not None:
            before_lr = float(optimizer.param_groups[0]["lr"])
            sched_value = _metric_value(row, scheduler_metric)
            scheduler.step(sched_value)
            after_lr = float(optimizer.param_groups[0]["lr"])
            if verbose and after_lr != before_lr and is_primary:
                print(
                    {
                        "lr_scheduler_step": {
                            "epoch": epoch,
                            "metric": scheduler_metric,
                            "metric_value": round(sched_value, 6),
                            "lr_before": before_lr,
                            "lr_after": after_lr,
                        }
                    }
                )

        if restore_best and has_val_rows:
            cur_val_loss = float(row.get("val_loss", 0.0))
            cur_top1 = float(row.get("top1", 0.0))
            is_better = (
                best_val_loss is None
                or cur_val_loss < best_val_loss
                or (cur_val_loss == best_val_loss and (best_top1 is None or cur_top1 > best_top1))
            )
            if is_better:
                best_val_loss = cur_val_loss
                best_top1 = cur_top1
                best_epoch = epoch
                best_state_dict = _cpu_cloned_state_dict(train_model)
                if verbose and is_primary:
                    print(
                        {
                            "best_checkpoint_update": {
                                "epoch": epoch,
                                "val_loss": round(cur_val_loss, 6),
                                "top1": round(cur_top1, 6),
                            }
                        }
                    )
                if best_checkpoint_out_path and is_primary:
                    best_artifact = _build_checkpoint_artifact(
                        _cpu_cloned_state_dict(train_model),
                        current_epoch=int(epoch),
                        latest_row=row,
                        checkpoint_kind="best_so_far",
                    )
                    torch.save(best_artifact, best_checkpoint_out_path)
                    if verbose and is_primary:
                        print({"best_checkpoint_saved_to_disk": {"epoch": int(epoch), "path": best_checkpoint_out_path}})

        if epoch_checkpoint_dir_path and is_primary:
            epoch_ckpt_path = os.path.join(epoch_checkpoint_dir_path, f"model_epoch_{int(epoch):02d}.pt")
            epoch_artifact = _build_checkpoint_artifact(
                _cpu_cloned_state_dict(train_model),
                current_epoch=int(epoch),
                latest_row=row,
                checkpoint_kind="epoch_end",
            )
            torch.save(epoch_artifact, epoch_ckpt_path)
            if verbose and is_primary:
                print({"epoch_checkpoint_saved_to_disk": {"epoch": int(epoch), "path": epoch_ckpt_path}})

        if early_stop_enabled:
            cur_metric = _metric_value(row, early_stop_metric_name)
            if _metric_improved(
                early_stop_metric_name,
                cur_metric,
                early_stop_best_metric,
                min_delta=float(early_stopping_min_delta),
            ):
                early_stop_best_metric = cur_metric
                early_stop_bad_epochs = 0
            else:
                early_stop_bad_epochs += 1
                if early_stop_bad_epochs >= int(early_stopping_patience):
                    early_stop_info.update(
                        {
                            "used": True,
                            "stopped_epoch": int(epoch),
                            "best_metric": float(early_stop_best_metric) if early_stop_best_metric is not None else None,
                            "bad_epochs": int(early_stop_bad_epochs),
                        }
                    )
                    if verbose and is_primary:
                        print({"early_stopping_triggered": early_stop_info})
                    if progress_callback is not None and is_primary:
                        progress_callback(
                            {
                                "event": "early_stopping_triggered",
                                "epoch": int(epoch),
                                "epochs": int(epochs),
                                "metric": early_stop_metric_name,
                                "bad_epochs": int(early_stop_bad_epochs),
                                "best_metric": (
                                    float(early_stop_best_metric) if early_stop_best_metric is not None else None
                                ),
                            }
                        )
                    break

    if early_stop_enabled and not early_stop_info["used"]:
        early_stop_info.update(
            {
                "best_metric": float(early_stop_best_metric) if early_stop_best_metric is not None else None,
                "bad_epochs": int(early_stop_bad_epochs),
            }
        )

    best_checkpoint_info = {
        "enabled": bool(restore_best),
        "used": False,
        "metric": "val_loss",
        "best_epoch": None,
        "best_val_loss": None,
    }
    if restore_best and has_val_rows and best_state_dict is not None:
        _unwrap_model(train_model).load_state_dict(best_state_dict)
        best_checkpoint_info.update(
            {
                "used": True,
                "best_epoch": int(best_epoch),
                "best_val_loss": float(best_val_loss),
            }
        )
        if verbose and is_primary:
            print({"best_checkpoint_restored": best_checkpoint_info})
    elif verbose and is_primary:
        print({"best_checkpoint_restored": best_checkpoint_info})

    sparsity_stats = _compute_model_sparsity_stats(
        train_model,
        include_bias=bool(sparsity_include_bias),
    )

    artifact = {
        "artifact_format_version": 2,
        "model_family": "next_move_lstm",
        "training_objective": "single_step_next_move",
        "state_dict": _unwrap_model(train_model).state_dict(),
        "vocab": vocab,
        "config": {
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "use_winner": use_winner,
            "use_phase": bool(use_phase_feature),
            "phase_embed_dim": int(phase_embed_dim),
            "use_side_to_move": bool(use_side_to_move_feature),
            "side_to_move_embed_dim": int(side_to_move_embed_dim),
        },
        "runtime": {
            "device": str(device),
            "amp": use_amp,
            "best_checkpoint": best_checkpoint_info,
            "early_stopping": early_stop_info,
            "lr_scheduler": {
                "kind": scheduler_kind,
                "metric": scheduler_metric,
                "enabled": bool(scheduler is not None),
                "factor": float(lr_plateau_factor),
                "patience": int(max(0, int(lr_plateau_patience))),
                "threshold": float(lr_plateau_threshold),
                "min_lr": float(lr_plateau_min_lr),
                "final_lr": float(optimizer.param_groups[0]["lr"]),
            },
            "phase_weights": {
                "unknown": float(phase_weight_vector[0].item()),
                "opening": float(phase_weight_vector[1].item()),
                "middlegame": float(phase_weight_vector[2].item()),
                "endgame": float(phase_weight_vector[3].item()),
            },
            "training_objective": "single_step_next_move",
            "distributed": {
                "enabled": bool(distributed_enabled),
                "world_size": int(distributed_world_size),
                "rank": int(distributed_rank),
            },
            "sparsity": {
                "mode": str(sparsity_mode_norm),
                "enabled": bool(sparsity_enabled),
                "l1_lambda": float(sparsity_l1_lambda_value),
                "include_bias": bool(sparsity_include_bias),
                **sparsity_stats,
            },
        },
    }

    dataset_info = {
        "train_rows": train_rows_total,
        "val_rows": val_rows_total,
        "train_rows_source": int(train_rows_source_total),
        "val_rows_source": int(val_rows_source_total),
        "train_rows_by_file": train_rows_by_file,
        "val_rows_by_file": val_rows_by_file,
        "train_index_rows": len(train_ds),
        "val_index_rows": len(val_ds),
        "vocab_size": len(vocab),
        "data_loading": data_loading_mode,
        "dataset_schema": schema_kind,
        "distributed": {
            "enabled": bool(distributed_enabled),
            "world_size": int(distributed_world_size),
            "rank": int(distributed_rank),
        },
        "subset_sampling": {
            "max_total_rows": int(max_total_rows),
            "max_train_rows": int(max_train_rows),
            "max_val_rows": int(max_val_rows),
            "train_subset_applied": bool(train_subset_applied),
            "val_subset_applied": bool(val_subset_applied),
        },
        "sparsity": {
            "mode": str(sparsity_mode_norm),
            "enabled": bool(sparsity_enabled),
            "l1_lambda": float(sparsity_l1_lambda_value),
            "include_bias": bool(sparsity_include_bias),
            **sparsity_stats,
        },
    }
    if schema_kind == "game":
        dataset_info.update(
            {
                "train_games": int(train_games_total or 0),
                "val_games": int(val_games_total or 0),
                "train_games_by_file": train_game_rows_by_file,
                "val_games_by_file": val_game_rows_by_file,
                "runtime_splice": {
                    "min_context": int(runtime_cfg.min_context),
                    "min_target": int(runtime_cfg.min_target),
                    "max_samples_per_game": int(runtime_cfg.max_samples_per_game),
                },
                "cache_load_reason_by_split": cache_load_reason_by_split,
                "runtime_splice_index_bytes_train": int(train_runtime_index_bytes or 0),
                "runtime_splice_index_bytes_val": int(val_runtime_index_bytes or 0),
            }
        )
    if progress_callback is not None and is_primary:
        progress_callback(
            {
                "event": "train_complete",
                "epochs_completed": int(len(history)),
                "epochs_requested": int(epochs),
                "best_checkpoint": best_checkpoint_info,
                "early_stopping": early_stop_info,
                "training_objective": "single_step_next_move",
            }
        )
    return artifact, history, dataset_info
