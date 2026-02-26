from typing import Dict, List

import chess
import torch
from torch.utils.data import DataLoader

from src.chessbot.model import (
    NextMoveLSTM,
    compute_topk,
    encode_tokens,
    side_to_move_id_from_context_len,
    winner_to_id,
)
from src.chessbot.phase import (
    PHASE_UNKNOWN,
    PHASE_RULE_VERSION,
    board_from_context,
    classify_board_phase,
    phase_to_id,
    remaining_plies_bucket,
)


def collate_eval(batch):
    lengths = torch.tensor([len(x["context_ids"]) for x in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    tokens = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    winners = torch.tensor([x["winner"] for x in batch], dtype=torch.long)
    phases = torch.tensor([x["phase_id"] for x in batch], dtype=torch.long)
    side_to_moves = torch.tensor([x["side_to_move_id"] for x in batch], dtype=torch.long)
    for i, row in enumerate(batch):
        ctx = row["context_ids"]
        tokens[i, : len(ctx)] = torch.tensor(ctx, dtype=torch.long)
    return tokens, lengths, labels, winners, phases, side_to_moves, batch


def _empty_bucket() -> Dict[str, float]:
    return {
        "rows": 0,
        "top1_hits": 0,
        "top5_hits": 0,
        "legal_top1_numerator": 0,
        "legal_top1_denominator": 0,
    }


def _update_bucket(
    buckets: Dict[str, Dict[str, float]],
    key: str,
    *,
    top1_hit: bool,
    top5_hit: bool,
    legal_num: int,
    legal_den: int,
) -> None:
    rec = buckets.setdefault(key, _empty_bucket())
    rec["rows"] += 1
    rec["top1_hits"] += int(top1_hit)
    rec["top5_hits"] += int(top5_hit)
    rec["legal_top1_numerator"] += int(legal_num)
    rec["legal_top1_denominator"] += int(legal_den)


def _finalize_buckets(buckets: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for key, rec in buckets.items():
        rows = int(rec["rows"])
        legal_den = int(rec["legal_top1_denominator"])
        out[key] = {
            "rows": rows,
            "top1": (float(rec["top1_hits"]) / rows) if rows else 0.0,
            "top5": (float(rec["top5_hits"]) / rows) if rows else 0.0,
            "legal_rate_top1": (float(rec["legal_top1_numerator"]) / legal_den) if legal_den else 0.0,
            "legal_rate_top1_denominator": legal_den,
        }
    return out


def _phase_from_row_or_board(row: Dict, board: chess.Board, board_ok: bool) -> str:
    phase = str(row.get("phase", "")).strip().lower()
    if phase:
        return phase
    if not board_ok:
        return "unknown"
    return str(classify_board_phase(board, ply=len(row.get("context", []))).get("phase", "unknown"))


def _remaining_bucket_from_row(row: Dict) -> str:
    bucket = str(row.get("plies_remaining_bucket", "")).strip()
    if bucket:
        return bucket
    plies_remaining = row.get("plies_remaining")
    if plies_remaining is None and row.get("plies_total") is not None:
        if row.get("ply") is not None:
            plies_remaining = int(row["plies_total"]) - int(row["ply"])
        elif row.get("splice_index") is not None:
            plies_remaining = int(row["plies_total"]) - (int(row["splice_index"]) + 1)
    return remaining_plies_bucket(plies_remaining if plies_remaining is not None else -1)


def _legal_top1_outcome(pred_id: int, inv_vocab: Dict[int, str], board: chess.Board, board_ok: bool) -> tuple[int, int]:
    if not board_ok:
        return 0, 0
    token = inv_vocab.get(pred_id, "")
    try:
        pred_mv = chess.Move.from_uci(token)
    except Exception:
        return 0, 1
    return (1, 1) if pred_mv in board.legal_moves else (0, 1)


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
                "phase_id": phase_to_id(str(row.get("phase", PHASE_UNKNOWN))),
                "side_to_move_id": side_to_move_id_from_context_len(len(row.get("context", []))),
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
    legal_num_total = 0
    legal_den_total = 0
    by_phase: Dict[str, Dict[str, float]] = {}
    by_remaining: Dict[str, Dict[str, float]] = {}

    with torch.no_grad():
        for tokens, lengths, labels, winners, phases, side_to_moves, batch_rows in loader:
            tokens = tokens.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            winners = winners.to(device, non_blocking=True)
            phases = phases.to(device, non_blocking=True)
            side_to_moves = side_to_moves.to(device, non_blocking=True)
            logits = model(tokens, lengths, winners, phases, side_to_moves)
            metrics = compute_topk(logits, labels, (1, 5))
            bs = labels.size(0)
            n += bs
            totals[1] += metrics[1] * bs
            totals[5] += metrics[5] * bs

            top5_ids = logits.topk(5, dim=1).indices.detach().cpu().tolist()
            label_ids = labels.detach().cpu().tolist()
            for row, row_top5, label_id in zip(batch_rows, top5_ids, label_ids):
                top1_pred_id = int(row_top5[0]) if row_top5 else -1
                top1_hit = bool(row_top5 and top1_pred_id == int(label_id))
                top5_hit = bool(int(label_id) in row_top5)
                board, board_ok = board_from_context(row.get("context", []))
                legal_num, legal_den = _legal_top1_outcome(top1_pred_id, inv_vocab, board, board_ok)
                legal_num_total += legal_num
                legal_den_total += legal_den
                phase_key = _phase_from_row_or_board(row, board, board_ok)
                remaining_key = _remaining_bucket_from_row(row)
                _update_bucket(
                    by_phase,
                    phase_key,
                    top1_hit=top1_hit,
                    top5_hit=top5_hit,
                    legal_num=legal_num,
                    legal_den=legal_den,
                )
                _update_bucket(
                    by_remaining,
                    remaining_key,
                    top1_hit=top1_hit,
                    top5_hit=top5_hit,
                    legal_num=legal_num,
                    legal_den=legal_den,
                )

    return {
        "rows": len(rows),
        "top1": totals[1] / n if n else 0.0,
        "top5": totals[5] / n if n else 0.0,
        "legal_rate_top1": (legal_num_total / legal_den_total) if legal_den_total else 0.0,
        "legal_rate_top1_denominator": legal_den_total,
        "phase_rule_version": PHASE_RULE_VERSION,
        "by_phase": _finalize_buckets(by_phase),
        "by_remaining_plies": _finalize_buckets(by_remaining),
        "device": str(device),
    }
