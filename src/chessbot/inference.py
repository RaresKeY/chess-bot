from typing import Dict, List

import chess
import torch

from src.chessbot.model import NextMoveLSTM, encode_tokens, winner_to_id


def parse_context(text: str) -> List[str]:
    return [x.strip() for x in text.split() if x.strip()]


def best_legal_from_topk(topk_tokens: List[str], context: List[str]) -> str:
    board = chess.Board()
    for uci in context:
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            raise ValueError(f"Illegal context move: {uci}")
        board.push(mv)

    for tok in topk_tokens:
        try:
            mv = chess.Move.from_uci(tok)
        except Exception:
            continue
        if mv in board.legal_moves:
            return tok
    return ""


def infer_from_artifact(artifact: Dict, context: List[str], winner_side: str, topk: int) -> Dict:
    vocab = artifact["vocab"]
    inv_vocab = {idx: tok for tok, idx in vocab.items()}
    cfg = artifact["config"]

    model = NextMoveLSTM(vocab_size=len(vocab), **cfg)
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    context_ids = encode_tokens(context, vocab)
    tokens = torch.tensor([context_ids], dtype=torch.long)
    lengths = torch.tensor([len(context_ids)], dtype=torch.long)
    winners = torch.tensor([winner_to_id(winner_side)], dtype=torch.long)

    with torch.no_grad():
        logits = model(tokens, lengths, winners)
        pred_ids = logits.topk(topk, dim=1).indices[0].tolist()

    topk_tokens = [inv_vocab.get(i, "") for i in pred_ids]
    legal = best_legal_from_topk(topk_tokens, context)
    return {
        "topk": topk_tokens,
        "best_legal": legal,
    }


def infer_from_artifact_on_device(
    artifact: Dict, context: List[str], winner_side: str, topk: int, device_str: str = "cpu"
) -> Dict:
    vocab = artifact["vocab"]
    inv_vocab = {idx: tok for tok, idx in vocab.items()}
    cfg = artifact["config"]
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    model = NextMoveLSTM(vocab_size=len(vocab), **cfg).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    context_ids = encode_tokens(context, vocab)
    tokens = torch.tensor([context_ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(context_ids)], dtype=torch.long, device=device)
    winners = torch.tensor([winner_to_id(winner_side)], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(tokens, lengths, winners)
        pred_ids = logits.topk(topk, dim=1).indices[0].detach().cpu().tolist()

    topk_tokens = [inv_vocab.get(i, "") for i in pred_ids]
    legal = best_legal_from_topk(topk_tokens, context)
    return {
        "topk": topk_tokens,
        "best_legal": legal,
        "device": str(device),
    }
