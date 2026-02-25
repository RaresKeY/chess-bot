import random
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.chessbot.model import NextMoveLSTM, build_vocab, compute_topk, encode_tokens, winner_to_id


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
        return context_ids, label, winner


def collate_train(batch: List[Tuple[List[int], int, int]]):
    lengths = torch.tensor([len(x[0]) for x in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    tokens = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.tensor([x[1] for x in batch], dtype=torch.long)
    winners = torch.tensor([x[2] for x in batch], dtype=torch.long)

    for i, (ctx, _, _) in enumerate(batch):
        tokens[i, : len(ctx)] = torch.tensor(ctx, dtype=torch.long)
    return tokens, lengths, labels, winners


def evaluate_loader(model, loader, device, topks=(1, 5), criterion=None, winner_weight: float = 1.0):
    model.eval()
    totals = {k: 0.0 for k in topks}
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for tokens, lengths, labels, winners in loader:
            tokens = tokens.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            winners = winners.to(device, non_blocking=True)
            logits = model(tokens, lengths, winners)
            if criterion is not None:
                losses = criterion(logits, labels)
                winner_mask = ((winners == 0) | (winners == 1)).float()
                weights = 1.0 + winner_mask * (winner_weight - 1.0)
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
    device_str: str = "auto",
    num_workers: int = 0,
    pin_memory: bool = True,
    amp: bool = False,
    restore_best: bool = True,
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
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction="none")
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    has_val_rows = len(val_rows) > 0
    best_state_dict = None
    best_epoch = None
    best_val_loss = None
    best_top1 = None
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
                    "use_winner": use_winner,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "restore_best": bool(restore_best),
                    "restore_best_has_val": bool(has_val_rows),
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
        for batch_idx, (tokens, lengths, labels, winners) in enumerate(train_loader, start=1):
            tokens = tokens.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            winners = winners.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(tokens, lengths, winners)
                losses = criterion(logits, labels)
                winner_mask = ((winners == 0) | (winners == 1)).float()
                weights = 1.0 + winner_mask * (winner_weight - 1.0)
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
        )
        row = {"epoch": epoch, "train_loss": train_loss, "device": str(device), "amp": use_amp, **val_metrics}
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
        },
        "runtime": {
            "device": str(device),
            "amp": use_amp,
            "best_checkpoint": best_checkpoint_info,
        },
    }

    return artifact, history
