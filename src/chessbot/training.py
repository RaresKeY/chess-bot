import random
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


def evaluate_loader(model, loader, device, topks=(1, 5)):
    model.eval()
    totals = {k: 0.0 for k in topks}
    n = 0
    with torch.no_grad():
        for tokens, lengths, labels, winners in loader:
            tokens = tokens.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            winners = winners.to(device, non_blocking=True)
            logits = model(tokens, lengths, winners)
            batch_metrics = compute_topk(logits, labels, topks)
            bs = labels.size(0)
            n += bs
            for k in topks:
                totals[k] += batch_metrics[k] * bs
    return {f"top{k}": (totals[k] / n if n else 0.0) for k in topks}


def train_next_move_model(
    train_rows: List[Dict],
    val_rows: List[Dict],
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    embed_dim: int,
    hidden_dim: int,
    winner_weight: float,
    use_winner: bool,
    device_str: str = "auto",
    num_workers: int = 0,
    pin_memory: bool = True,
    amp: bool = False,
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
        use_winner=use_winner,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(reduction="none")
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: List[Dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for tokens, lengths, labels, winners in train_loader:
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

        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate_loader(model, val_loader, device, topks=(1, 5))
        history.append({"epoch": epoch, "train_loss": train_loss, "device": str(device), "amp": use_amp, **val_metrics})

    artifact = {
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "config": {
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "use_winner": use_winner,
        },
        "runtime": {
            "device": str(device),
            "amp": use_amp,
        },
    }

    return artifact, history
