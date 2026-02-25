from typing import Dict, List

import chess
import torch
from torch.utils.data import DataLoader

from src.chessbot.model import NextMoveLSTM, compute_topk, encode_tokens, winner_to_id


def collate_eval(batch):
    lengths = torch.tensor([len(x["context_ids"]) for x in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    tokens = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    winners = torch.tensor([x["winner"] for x in batch], dtype=torch.long)
    for i, row in enumerate(batch):
        ctx = row["context_ids"]
        tokens[i, : len(ctx)] = torch.tensor(ctx, dtype=torch.long)
    return tokens, lengths, labels, winners, batch


def legal_rate_from_predictions(pred_ids: List[int], inv_vocab: Dict[int, str], batch_rows: List[Dict]) -> float:
    legal = 0
    total = 0
    for pred_id, row in zip(pred_ids, batch_rows):
        board = chess.Board()
        ok = True
        for uci in row["context"]:
            try:
                mv = chess.Move.from_uci(uci)
            except Exception:
                ok = False
                break
            if mv not in board.legal_moves:
                ok = False
                break
            board.push(mv)
        if not ok:
            continue

        token = inv_vocab.get(pred_id, "")
        try:
            pred_mv = chess.Move.from_uci(token)
        except Exception:
            total += 1
            continue

        total += 1
        if pred_mv in board.legal_moves:
            legal += 1
    return legal / total if total else 0.0


def evaluate_artifact(artifact: Dict, rows: List[Dict], batch_size: int = 128, device_str: str = "cpu") -> Dict:
    vocab = artifact["vocab"]
    inv_vocab = {idx: tok for tok, idx in vocab.items()}
    cfg = artifact["config"]

    prepared = []
    unk = vocab["<UNK>"]
    for row in rows:
        prepared.append(
            {
                **row,
                "context_ids": encode_tokens(row["context"], vocab),
                "label": vocab.get(row["next_move"], unk),
                "winner": winner_to_id(row.get("winner_side", "?")),
            }
        )

    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    loader = DataLoader(
        prepared,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_eval,
        pin_memory=(device.type == "cuda"),
    )
    model = NextMoveLSTM(vocab_size=len(vocab), **cfg)
    model.load_state_dict(artifact["state_dict"])
    model.to(device)
    model.eval()

    totals = {1: 0.0, 5: 0.0}
    n = 0
    legal_total = 0.0
    batches = 0

    with torch.no_grad():
        for tokens, lengths, labels, winners, batch_rows in loader:
            tokens = tokens.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            winners = winners.to(device, non_blocking=True)
            logits = model(tokens, lengths, winners)
            metrics = compute_topk(logits, labels, (1, 5))
            bs = labels.size(0)
            n += bs
            totals[1] += metrics[1] * bs
            totals[5] += metrics[5] * bs

            pred_ids = logits.argmax(dim=1).tolist()
            legal_total += legal_rate_from_predictions(pred_ids, inv_vocab, batch_rows)
            batches += 1

    return {
        "rows": len(rows),
        "top1": totals[1] / n if n else 0.0,
        "top5": totals[5] / n if n else 0.0,
        "legal_rate_top1": legal_total / batches if batches else 0.0,
        "device": str(device),
    }
