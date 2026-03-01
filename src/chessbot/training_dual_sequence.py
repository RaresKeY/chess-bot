from array import array
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import chess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.chessbot.model import NextMoveSeqLSTM, encode_tokens, side_to_move_id_from_context_len
from src.chessbot.phase import PHASE_ENDGAME, classify_context_phase, board_from_context

MODEL_SIDE_WHITE = "white"
MODEL_SIDE_BLACK = "black"

WINNER_SIDE_WHITE = "W"
WINNER_SIDE_BLACK = "B"


def model_side_to_winner_side(model_side: str) -> str:
    side = str(model_side or "").strip().lower()
    if side == MODEL_SIDE_WHITE:
        return WINNER_SIDE_WHITE
    if side == MODEL_SIDE_BLACK:
        return WINNER_SIDE_BLACK
    raise ValueError(f"Unsupported model_side: {model_side}")


def _moves_from_row(row: Dict[str, Any]) -> List[str]:
    moves = row.get("moves")
    if isinstance(moves, list):
        return [str(x) for x in moves]
    moves_uci = row.get("moves_uci")
    if isinstance(moves_uci, list):
        return [str(x) for x in moves_uci]
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


def _runtime_splice_indices_for_moves(
    moves: List[str],
    *,
    min_context: int,
    min_target: int,
    max_samples_per_game: int,
    seed: int,
    game_id: str = "",
) -> List[int]:
    n = len(moves)
    start_i = int(min_context) - 1
    end_i = n - int(min_target) - 1
    if end_i < start_i:
        return []
    idxs = list(range(start_i, end_i + 1))
    cap = int(max_samples_per_game)
    if cap > 0 and len(idxs) > cap:
        rnd = random.Random(f"{int(seed)}:{game_id or 'unknown'}")
        rnd.shuffle(idxs)
        idxs = idxs[:cap]
    return idxs


def _winner_side_from_row(row: Dict[str, Any]) -> str:
    side = str(row.get("winner_side", "?")).strip().upper()
    if side in {WINNER_SIDE_WHITE, WINNER_SIDE_BLACK, "D", "?"}:
        return side
    return "?"


def _encode_fixed_horizon_targets(
    target_tokens: List[str], vocab: Dict[str, int], horizon: int
) -> Tuple[List[int], List[int]]:
    h = max(1, int(horizon))
    out_ids: List[int] = []
    out_mask: List[int] = []
    unk = int(vocab["<UNK>"])
    for i in range(h):
        if i < len(target_tokens):
            out_ids.append(int(vocab.get(target_tokens[i], unk)))
            out_mask.append(1)
        else:
            out_ids.append(0)
            out_mask.append(0)
    return out_ids, out_mask


def _compute_mate_bias_weights(
    *,
    context: List[str],
    target: List[str],
    horizon: int,
    mask: List[int],
    phase_name_hint: str,
    enabled: bool,
    mate_in_x: int,
    mate_weight: float,
) -> List[float]:
    h = max(1, int(horizon))
    out = [1.0 for _ in range(h)]
    if not enabled:
        return out
    if int(mate_in_x) <= 0 or float(mate_weight) <= 1.0:
        return out

    phase_name = str(phase_name_hint or "").strip().lower()
    if phase_name != PHASE_ENDGAME:
        inferred = str(classify_context_phase(context).get("phase", "")).strip().lower()
        if inferred != PHASE_ENDGAME:
            return out

    board, ok = board_from_context(context)
    if not ok:
        return out

    mate_ply = 0
    for i in range(min(len(target), h)):
        try:
            mv = chess.Move.from_uci(target[i])
        except Exception:
            break
        if mv not in board.legal_moves:
            break
        board.push(mv)
        if board.is_checkmate():
            mate_ply = i + 1
            break
    if mate_ply <= 0 or mate_ply > int(mate_in_x):
        return out

    mw = float(mate_weight)
    upto = min(mate_ply, h)
    for i in range(upto):
        if int(mask[i]) == 1:
            out[i] = mw
    return out


def _index_paths_for_model_side(
    *,
    paths: List[str],
    model_side: str,
    schema: str,
    build_vocab: bool,
    runtime_min_context: int,
    runtime_min_target: int,
    runtime_max_samples_per_game: int,
    seed: int,
) -> Tuple[
    Dict[str, int],
    List[str],
    array,
    array,
    array,
    Dict[str, int],
    int,
    Dict[str, int],
    int,
]:
    winner_side_need = model_side_to_winner_side(model_side)
    vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    path_strs = [os.fspath(p) for p in paths]
    path_ids = array("I")
    offsets = array("Q")
    splice_indices = array("I")

    rows_by_file: Dict[str, int] = {}
    total_rows = 0
    dropped_by_file: Dict[str, int] = {}
    dropped_rows_total = 0

    for path_id, path in enumerate(path_strs):
        kept = 0
        dropped = 0
        with open(path, "rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                row = json.loads(line.decode("utf-8"))
                winner_side = _winner_side_from_row(row)
                if winner_side != winner_side_need:
                    dropped += 1
                    dropped_rows_total += 1
                    continue
                if schema == "spliced":
                    context = list(row.get("context", []) or [])
                    target = list(row.get("target", []) or [])
                    if not context or not target:
                        dropped += 1
                        dropped_rows_total += 1
                        continue
                    path_ids.append(int(path_id))
                    offsets.append(int(offset))
                    splice_indices.append(0)
                    kept += 1
                    total_rows += 1
                    if build_vocab:
                        for tok in context:
                            if tok not in vocab:
                                vocab[tok] = len(vocab)
                        for tok in target:
                            if tok not in vocab:
                                vocab[tok] = len(vocab)
                    continue

                moves = _moves_from_row(row)
                if not moves:
                    dropped += 1
                    dropped_rows_total += 1
                    continue
                splices = _runtime_splice_indices_for_moves(
                    moves=moves,
                    min_context=runtime_min_context,
                    min_target=runtime_min_target,
                    max_samples_per_game=runtime_max_samples_per_game,
                    seed=seed,
                    game_id=str(row.get("game_id", "")),
                )
                if not splices:
                    dropped += 1
                    dropped_rows_total += 1
                    continue
                if build_vocab:
                    for tok in moves:
                        if tok not in vocab:
                            vocab[tok] = len(vocab)
                for splice_i in splices:
                    path_ids.append(int(path_id))
                    offsets.append(int(offset))
                    splice_indices.append(int(splice_i))
                    kept += 1
                    total_rows += 1
        rows_by_file[path] = int(kept)
        dropped_by_file[path] = int(dropped)
    return (
        vocab,
        path_strs,
        path_ids,
        offsets,
        splice_indices,
        rows_by_file,
        int(total_rows),
        dropped_by_file,
        int(dropped_rows_total),
    )


class IndexedDualSequenceDataset(Dataset):
    def __init__(
        self,
        *,
        paths: List[str],
        path_ids: array,
        offsets: array,
        splice_indices: array,
        schema: str,
        vocab: Dict[str, int],
        horizon: int,
        mate_bias_enabled: bool,
        mate_in_x: int,
        mate_weight: float,
    ) -> None:
        self.paths = paths
        self.path_ids = path_ids
        self.offsets = offsets
        self.splice_indices = splice_indices
        self.schema = str(schema)
        self.vocab = vocab
        self.horizon = int(max(1, horizon))
        self.mate_bias_enabled = bool(mate_bias_enabled)
        self.mate_in_x = int(max(0, mate_in_x))
        self.mate_weight = float(max(1.0, mate_weight))
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
        path_id = int(self.path_ids[idx])
        offset = int(self.offsets[idx])
        splice_i = int(self.splice_indices[idx])
        h = self._handle_for(path_id)
        h.seek(offset)
        row = json.loads(h.readline().decode("utf-8"))

        phase_name = str(row.get("phase", "")).strip().lower()
        if self.schema == "spliced":
            context = list(row.get("context", []) or [])
            target = list(row.get("target", []) or [])
        else:
            moves = _moves_from_row(row)
            context = moves[: splice_i + 1]
            target = moves[splice_i + 1 :]
            if not phase_name:
                phase_name = str(classify_context_phase(context).get("phase", "")).strip().lower()

        if not context or not target:
            raise IndexError("Dual sequence dataset row resolved to empty context/target")

        context_ids = encode_tokens(context, self.vocab)
        side_to_move = side_to_move_id_from_context_len(len(context))
        seq_targets, seq_mask = _encode_fixed_horizon_targets(target, self.vocab, self.horizon)
        mate_bias_weights = _compute_mate_bias_weights(
            context=context,
            target=target,
            horizon=self.horizon,
            mask=seq_mask,
            phase_name_hint=phase_name,
            enabled=self.mate_bias_enabled,
            mate_in_x=self.mate_in_x,
            mate_weight=self.mate_weight,
        )
        return context_ids, side_to_move, seq_targets, seq_mask, mate_bias_weights


def collate_dual_sequence_batch(
    batch: List[Tuple[List[int], int, List[int], List[int], List[float]]]
):
    lengths = torch.tensor([len(x[0]) for x in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    horizon = len(batch[0][2]) if batch else 1
    tokens = torch.zeros((len(batch), max_len), dtype=torch.long)
    sides = torch.tensor([x[1] for x in batch], dtype=torch.long)
    targets = torch.zeros((len(batch), horizon), dtype=torch.long)
    mask = torch.zeros((len(batch), horizon), dtype=torch.bool)
    mate_weights = torch.ones((len(batch), horizon), dtype=torch.float32)
    for i, (ctx, side, seq_targets, seq_mask, seq_mate_weights) in enumerate(batch):
        tokens[i, : len(ctx)] = torch.tensor(ctx, dtype=torch.long)
        sides[i] = int(side)
        targets[i] = torch.tensor(seq_targets, dtype=torch.long)
        mask[i] = torch.tensor(seq_mask, dtype=torch.bool)
        mate_weights[i] = torch.tensor(seq_mate_weights, dtype=torch.float32)
    return tokens, lengths, sides, targets, mask, mate_weights


def _build_step_weights(horizon: int, step_loss_decay: float) -> torch.Tensor:
    h = max(1, int(horizon))
    d = float(step_loss_decay)
    if d <= 0.0:
        d = 1.0
    out = [1.0]
    for _ in range(1, h):
        out.append(out[-1] * d)
    return torch.tensor(out, dtype=torch.float32)


def _masked_sequence_loss(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    step_weights: torch.Tensor,
    mate_weights: Optional[torch.Tensor] = None,
    criterion: Optional[nn.Module] = None,
) -> torch.Tensor:
    if criterion is None:
        criterion = nn.CrossEntropyLoss(reduction="none")
    bsz, horizon, vocab_size = logits.shape
    losses = criterion(logits.reshape(bsz * horizon, vocab_size), targets.reshape(bsz * horizon)).reshape(bsz, horizon)
    weights = mask.float() * step_weights.view(1, horizon).to(logits.device)
    if mate_weights is not None:
        weights = weights * mate_weights.float()
    return (losses * weights).sum() / weights.sum().clamp_min(1e-12)


def compute_sequence_match_metrics(
    *, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> Dict[str, float]:
    preds = logits.argmax(dim=-1)
    matches = (preds == targets) & mask
    valid = mask.sum().item()
    match_count = matches.sum().item()

    row_has_valid = mask.any(dim=1)
    row_exact = ((matches | (~mask)).all(dim=1) & row_has_valid)
    rows_valid = row_has_valid.sum().item()
    full_exact_hits = row_exact.sum().item()
    out = {
        "ply_match_count": float(match_count),
        "valid_ply_count": float(valid),
        "ply_match_rate": float(match_count / valid) if valid > 0 else 0.0,
        "rows_with_valid_targets": float(rows_valid),
        "full_seq_exact_hit_count": float(full_exact_hits),
        "full_seq_exact_hit_rate": float(full_exact_hits / rows_valid) if rows_valid > 0 else 0.0,
    }
    return out


def evaluate_dual_sequence_loader(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    step_weights: torch.Tensor,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    model.eval()
    total_weighted_loss = 0.0
    total_rows = 0
    ply_match_count = 0.0
    valid_ply_count = 0.0
    rows_with_valid_targets = 0.0
    full_seq_exact_hit_count = 0.0
    with torch.no_grad():
        for batch in loader:
            tokens, lengths, sides, targets, mask, mate_weights = batch
            tokens = tokens.to(device)
            lengths = lengths.to(device)
            sides = sides.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            mate_weights = mate_weights.to(device)
            logits = model(tokens, lengths, sides)
            loss = _masked_sequence_loss(
                logits=logits,
                targets=targets,
                mask=mask,
                step_weights=step_weights.to(device),
                mate_weights=mate_weights,
                criterion=criterion,
            )
            batch_rows = int(tokens.shape[0])
            total_rows += batch_rows
            total_weighted_loss += float(loss.item()) * float(batch_rows)

            m = compute_sequence_match_metrics(logits=logits, targets=targets, mask=mask)
            ply_match_count += float(m["ply_match_count"])
            valid_ply_count += float(m["valid_ply_count"])
            rows_with_valid_targets += float(m["rows_with_valid_targets"])
            full_seq_exact_hit_count += float(m["full_seq_exact_hit_count"])
    return {
        "val_loss": float(total_weighted_loss / max(total_rows, 1)),
        "ply_match_count": float(ply_match_count),
        "valid_ply_count": float(valid_ply_count),
        "ply_match_rate": float(ply_match_count / valid_ply_count) if valid_ply_count > 0 else 0.0,
        "rows_with_valid_targets": float(rows_with_valid_targets),
        "full_seq_exact_hit_count": float(full_seq_exact_hit_count),
        "full_seq_exact_hit_rate": (
            float(full_seq_exact_hit_count / rows_with_valid_targets) if rows_with_valid_targets > 0 else 0.0
        ),
    }


def train_dual_sequence_model_from_jsonl_paths(
    *,
    train_paths: List[str],
    val_paths: List[str],
    model_side: str,
    out_model_path: str,
    out_metrics_path: Optional[str] = None,
    seed: int = 7,
    horizon: int = 8,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 2e-4,
    embed_dim: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 2,
    dropout: float = 0.15,
    use_side_to_move_feature: bool = True,
    side_to_move_embed_dim: int = 4,
    step_loss_decay: float = 1.0,
    runtime_min_context: int = 8,
    runtime_min_target: int = 1,
    runtime_max_samples_per_game: int = 0,
    num_workers: int = 0,
    device_str: str = "cpu",
    mate_bias_enabled: bool = True,
    mate_in_x: int = 3,
    mate_weight: float = 1.25,
    verbose: bool = False,
) -> Dict[str, Any]:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    schema = _sniff_paths_schema(list(train_paths) + list(val_paths))
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    (
        vocab,
        train_path_strs,
        train_path_ids,
        train_offsets,
        train_splice_indices,
        train_rows_by_file,
        train_rows_total,
        train_dropped_by_file,
        train_dropped_total,
    ) = _index_paths_for_model_side(
        paths=train_paths,
        model_side=model_side,
        schema=schema,
        build_vocab=True,
        runtime_min_context=runtime_min_context,
        runtime_min_target=runtime_min_target,
        runtime_max_samples_per_game=runtime_max_samples_per_game,
        seed=seed,
    )
    (
        _unused_vocab,
        val_path_strs,
        val_path_ids,
        val_offsets,
        val_splice_indices,
        val_rows_by_file,
        val_rows_total,
        val_dropped_by_file,
        val_dropped_total,
    ) = _index_paths_for_model_side(
        paths=val_paths,
        model_side=model_side,
        schema=schema,
        build_vocab=False,
        runtime_min_context=runtime_min_context,
        runtime_min_target=runtime_min_target,
        runtime_max_samples_per_game=runtime_max_samples_per_game,
        seed=seed,
    )
    if int(train_rows_total) <= 0:
        raise RuntimeError(f"No training rows after winner-side filtering for model_side={model_side}")
    if int(val_rows_total) <= 0:
        raise RuntimeError(f"No validation rows after winner-side filtering for model_side={model_side}")

    train_ds = IndexedDualSequenceDataset(
        paths=train_path_strs,
        path_ids=train_path_ids,
        offsets=train_offsets,
        splice_indices=train_splice_indices,
        schema=schema,
        vocab=vocab,
        horizon=horizon,
        mate_bias_enabled=mate_bias_enabled,
        mate_in_x=mate_in_x,
        mate_weight=mate_weight,
    )
    val_ds = IndexedDualSequenceDataset(
        paths=val_path_strs,
        path_ids=val_path_ids,
        offsets=val_offsets,
        splice_indices=val_splice_indices,
        schema=schema,
        vocab=vocab,
        horizon=horizon,
        mate_bias_enabled=mate_bias_enabled,
        mate_in_x=mate_in_x,
        mate_weight=mate_weight,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(num_workers),
        collate_fn=collate_dual_sequence_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_dual_sequence_batch,
    )

    model = NextMoveSeqLSTM(
        vocab_size=len(vocab),
        horizon=int(horizon),
        embed_dim=int(embed_dim),
        hidden_dim=int(hidden_dim),
        num_layers=int(num_layers),
        dropout=float(dropout),
        side_to_move_embed_dim=int(side_to_move_embed_dim),
        use_side_to_move=bool(use_side_to_move_feature),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    criterion = nn.CrossEntropyLoss(reduction="none")
    step_weights = _build_step_weights(int(horizon), float(step_loss_decay)).to(device)

    history: List[Dict[str, Any]] = []
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_loss = float("inf")

    for epoch in range(1, int(epochs) + 1):
        model.train()
        running = 0.0
        seen_rows = 0
        for batch in train_loader:
            tokens, lengths, sides, targets, mask, mate_weights = batch
            tokens = tokens.to(device)
            lengths = lengths.to(device)
            sides = sides.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            mate_weights = mate_weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens, lengths, sides)
            loss = _masked_sequence_loss(
                logits=logits,
                targets=targets,
                mask=mask,
                step_weights=step_weights,
                mate_weights=mate_weights,
                criterion=criterion,
            )
            loss.backward()
            optimizer.step()
            batch_rows = int(tokens.shape[0])
            running += float(loss.item()) * float(batch_rows)
            seen_rows += batch_rows
        train_loss = float(running / max(seen_rows, 1))
        val_metrics = evaluate_dual_sequence_loader(
            model=model,
            loader=val_loader,
            device=device,
            step_weights=step_weights,
            criterion=criterion,
        )
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            **val_metrics,
        }
        history.append(row)
        if float(val_metrics["val_loss"]) < float(best_val_loss):
            best_val_loss = float(val_metrics["val_loss"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(
                {
                    "event": "epoch_end",
                    "epoch": int(epoch),
                    "train_loss": round(train_loss, 6),
                    "val_loss": round(float(val_metrics["val_loss"]), 6),
                    "ply_match_rate": round(float(val_metrics["ply_match_rate"]), 6),
                    "full_seq_exact_hit_rate": round(float(val_metrics["full_seq_exact_hit_rate"]), 6),
                }
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    artifact = {
        "artifact_format_version": 1,
        "model_family": "dual_side_sequence_lstm",
        "training_objective": "one_shot_sequence_match",
        "model_side": str(model_side).strip().lower(),
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "config": {
            "horizon": int(horizon),
            "embed_dim": int(embed_dim),
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
            "use_side_to_move": bool(use_side_to_move_feature),
            "side_to_move_embed_dim": int(side_to_move_embed_dim),
        },
        "runtime": {
            "device": str(device),
            "schema": str(schema),
            "step_loss_decay": float(step_loss_decay),
            "mate_bias_enabled": bool(mate_bias_enabled),
            "mate_in_x": int(mate_in_x),
            "mate_weight": float(mate_weight),
            "train_paths": [str(p) for p in train_paths],
            "val_paths": [str(p) for p in val_paths],
        },
    }
    torch.save(artifact, out_model_path)

    final_metrics: Dict[str, Any] = {
        "model_side": str(model_side).strip().lower(),
        "model_family": "dual_side_sequence_lstm",
        "training_objective": "one_shot_sequence_match",
        "schema": str(schema),
        "seed": int(seed),
        "horizon": int(horizon),
        "train_rows_total": int(train_rows_total),
        "val_rows_total": int(val_rows_total),
        "train_rows_by_file": train_rows_by_file,
        "val_rows_by_file": val_rows_by_file,
        "train_dropped_rows_total": int(train_dropped_total),
        "val_dropped_rows_total": int(val_dropped_total),
        "train_dropped_rows_by_file": train_dropped_by_file,
        "val_dropped_rows_by_file": val_dropped_by_file,
        "history": history,
        "best_val_loss": float(best_val_loss),
        "model_path": str(out_model_path),
        "runtime_splice": {
            "min_context": int(runtime_min_context),
            "min_target": int(runtime_min_target),
            "max_samples_per_game": int(runtime_max_samples_per_game),
        },
        "step_loss_decay": float(step_loss_decay),
        "mate_bias_enabled": bool(mate_bias_enabled),
        "mate_in_x": int(mate_in_x),
        "mate_weight": float(mate_weight),
    }
    if out_metrics_path:
        with open(out_metrics_path, "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=2)
    return {
        "artifact": artifact,
        "metrics": final_metrics,
    }
