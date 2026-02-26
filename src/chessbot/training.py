import json
import os
import random
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.chessbot.model import (
    NextMoveLSTM,
    build_vocab,
    compute_topk,
    encode_tokens,
    side_to_move_id_from_context_len,
    winner_to_id,
)
from src.chessbot.phase import PHASE_MIDDLEGAME, PHASE_OPENING, PHASE_ENDGAME, PHASE_UNKNOWN, phase_to_id


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
        # Keep prefetch small to limit host RAM growth from queued batches.
        train_loader_kwargs["prefetch_factor"] = 1
        # Avoid persistent workers; they retain Python/Torch allocator caches across epochs.
        train_loader_kwargs["persistent_workers"] = False
    train_loader = DataLoader(train_ds, shuffle=True, **train_loader_kwargs)

    # Validation is infrequent and not throughput-critical; keeping it single-process
    # avoids a second worker pool and substantially reduces host RAM pressure.
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

            with torch.amp.autocast("cuda", enabled=use_amp):
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
        },
    }

    return artifact, history


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
):
    random.seed(seed)
    torch.manual_seed(seed)

    train_paths = [os.fspath(p) for p in train_paths]
    val_paths = [os.fspath(p) for p in val_paths]

    # Stream train files once to build vocabulary and exact row counts.
    vocab, train_rows_by_file, train_rows_total = _build_vocab_and_count_rows_from_train_paths(train_paths)
    # Count validation rows without loading them.
    val_rows_by_file, val_rows_total = _count_rows_in_jsonl_paths(val_paths)

    if train_rows_total <= 0:
        raise RuntimeError("No training rows found")

    # Build line-offset indexes for on-demand row loading.
    train_ds = IndexedJsonlDataset(*_index_jsonl_paths(train_paths), vocab=vocab)
    val_ds = IndexedJsonlDataset(*_index_jsonl_paths(val_paths), vocab=vocab)

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
                    "data_loading": "indexed_jsonl_on_demand",
                }
            }
        )
    if progress_callback is not None:
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
                "data_loading": "indexed_jsonl_on_demand",
            }
        )

    history: List[Dict] = []
    for epoch in range(1, epochs + 1):
        if verbose:
            print(f"[train] epoch {epoch}/{epochs} start")
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "epoch_start",
                    "epoch": int(epoch),
                    "epochs": int(epochs),
                    "train_batches_total": int(len(train_loader)),
                }
            )
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

            with torch.amp.autocast("cuda", enabled=use_amp):
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
        if progress_callback is not None:
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
                    },
                }
            )
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
                    if progress_callback is not None:
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
        "data_loading": "indexed_jsonl_on_demand",
    }
    if progress_callback is not None:
        progress_callback(
            {
                "event": "train_complete",
                "epochs_completed": int(len(history)),
                "epochs_requested": int(epochs),
                "best_checkpoint": best_checkpoint_info,
                "early_stopping": early_stop_info,
            }
        )
    return artifact, history, dataset_info
