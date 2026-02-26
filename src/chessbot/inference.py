from typing import Dict, List

import chess
import torch

from src.chessbot.model import NextMoveLSTM, encode_tokens, side_to_move_id_from_context_len, winner_to_id
from src.chessbot.phase import PHASE_UNKNOWN, classify_context_phase, phase_to_id


def parse_context(text: str) -> List[str]:
    return [x.strip() for x in text.split() if x.strip()]


def artifact_model_family(artifact: Dict) -> str:
    return str(artifact.get("model_family") or "next_move_lstm")


def artifact_training_objective(artifact: Dict) -> str:
    runtime = artifact.get("runtime") or {}
    return str(runtime.get("training_objective") or artifact.get("training_objective") or "single_step_next_move")


def artifact_rollout_horizon(artifact: Dict) -> int:
    runtime = artifact.get("runtime") or {}
    try:
        return max(0, int(runtime.get("rollout_horizon") or 0))
    except Exception:
        return 0


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

    original_context_len = len(context)
    context_ids = encode_tokens(context, vocab)
    if not context_ids:
        context_ids = [vocab.get("<UNK>", 1)]
    tokens = torch.tensor([context_ids], dtype=torch.long)
    lengths = torch.tensor([len(context_ids)], dtype=torch.long)
    winners = torch.tensor([winner_to_id(winner_side)], dtype=torch.long)
    phase_name = str(classify_context_phase(context).get("phase", PHASE_UNKNOWN))
    phases = torch.tensor([phase_to_id(phase_name)], dtype=torch.long)
    side_to_moves = torch.tensor([side_to_move_id_from_context_len(original_context_len)], dtype=torch.long)

    with torch.no_grad():
        logits = model(tokens, lengths, winners, phases, side_to_moves)
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

    original_context_len = len(context)
    context_ids = encode_tokens(context, vocab)
    if not context_ids:
        context_ids = [vocab.get("<UNK>", 1)]
    tokens = torch.tensor([context_ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(context_ids)], dtype=torch.long, device=device)
    winners = torch.tensor([winner_to_id(winner_side)], dtype=torch.long, device=device)
    phase_name = str(classify_context_phase(context).get("phase", PHASE_UNKNOWN))
    phases = torch.tensor([phase_to_id(phase_name)], dtype=torch.long, device=device)
    side_to_moves = torch.tensor([side_to_move_id_from_context_len(original_context_len)], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(tokens, lengths, winners, phases, side_to_moves)
        pred_ids = logits.topk(topk, dim=1).indices[0].detach().cpu().tolist()

    topk_tokens = [inv_vocab.get(i, "") for i in pred_ids]
    legal = best_legal_from_topk(topk_tokens, context)
    return {
        "topk": topk_tokens,
        "best_legal": legal,
        "device": str(device),
    }


def infer_rollout_from_artifact_on_device(
    artifact: Dict,
    context: List[str],
    winner_side: str,
    topk: int,
    rollout_plies: int,
    device_str: str = "cpu",
    fallback_legal: bool = False,
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

    board = chess.Board()
    for uci in context:
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            raise ValueError(f"Illegal context move: {uci}")
        board.push(mv)

    rollout: List[str] = []
    step_debug: List[Dict] = []
    local_context = list(context)
    max_steps = max(0, int(rollout_plies))
    for _step in range(max_steps):
        original_context_len = len(local_context)
        context_ids = encode_tokens(local_context, vocab)
        if not context_ids:
            context_ids = [vocab.get("<UNK>", 1)]
        tokens = torch.tensor([context_ids], dtype=torch.long, device=device)
        lengths = torch.tensor([len(context_ids)], dtype=torch.long, device=device)
        winners = torch.tensor([winner_to_id(winner_side)], dtype=torch.long, device=device)
        phase_name = str(classify_context_phase(local_context).get("phase", PHASE_UNKNOWN))
        phases = torch.tensor([phase_to_id(phase_name)], dtype=torch.long, device=device)
        side_to_moves = torch.tensor([side_to_move_id_from_context_len(original_context_len)], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(tokens, lengths, winners, phases, side_to_moves)
            k = max(1, min(int(topk), int(logits.shape[-1])))
            pred_ids = logits.topk(k, dim=1).indices[0].detach().cpu().tolist()
        topk_tokens = [inv_vocab.get(i, "") for i in pred_ids]
        legal = best_legal_from_topk(topk_tokens, local_context)
        chosen = legal
        fallback_used = False
        if not chosen and fallback_legal and not board.is_game_over(claim_draw=True):
            fallback_mv = next(iter(board.legal_moves), None)
            if fallback_mv is not None:
                chosen = fallback_mv.uci()
                fallback_used = True
        step_debug.append(
            {
                "topk": topk_tokens,
                "best_legal": legal,
                "chosen": chosen,
                "fallback": fallback_used,
            }
        )
        if not chosen:
            break
        mv = chess.Move.from_uci(chosen)
        if mv not in board.legal_moves:
            break
        board.push(mv)
        local_context.append(chosen)
        rollout.append(chosen)
        if board.is_game_over(claim_draw=True):
            break

    return {
        "rollout": rollout,
        "first_move": rollout[0] if rollout else "",
        "steps_generated": len(rollout),
        "fallback_moves": sum(1 for x in step_debug if x.get("fallback")),
        "step_debug": step_debug,
        "device": str(device),
    }


def infer_first_move_auto_from_artifact_on_device(
    artifact: Dict,
    context: List[str],
    winner_side: str,
    topk: int,
    device_str: str = "cpu",
    policy_mode: str = "auto",
    rollout_plies: int = 0,
    rollout_fallback_legal: bool = False,
) -> Dict:
    mode = str(policy_mode or "auto").strip().lower()
    if mode not in {"auto", "next", "rollout"}:
        raise ValueError(f"Unsupported policy_mode: {policy_mode}")

    objective = artifact_training_objective(artifact)
    fam = artifact_model_family(artifact)
    if fam != "next_move_lstm":
        raise RuntimeError(f"Unsupported artifact model family for current inference path: {fam}")

    use_rollout = False
    rollout_len = max(0, int(rollout_plies))
    if mode == "rollout":
        use_rollout = True
        if rollout_len <= 0:
            rollout_len = max(artifact_rollout_horizon(artifact), 1)
    elif mode == "auto":
        if objective.startswith("multistep_"):
            use_rollout = True
            if rollout_len <= 0:
                rollout_len = max(artifact_rollout_horizon(artifact), 1)
    # mode == next keeps legacy path regardless of artifact metadata.

    if use_rollout:
        out = infer_rollout_from_artifact_on_device(
            artifact=artifact,
            context=context,
            winner_side=winner_side,
            topk=topk,
            rollout_plies=rollout_len,
            device_str=device_str,
            fallback_legal=bool(rollout_fallback_legal),
        )
        return {
            **out,
            "policy_mode_requested": mode,
            "policy_mode_used": "rollout",
            "artifact_training_objective": objective,
            "artifact_model_family": fam,
            "move_uci": out.get("first_move", ""),
            "fallback": bool(out.get("fallback_moves", 0) > 0 and (out.get("first_move") or "")),
        }

    out = infer_from_artifact_on_device(
        artifact=artifact,
        context=context,
        winner_side=winner_side,
        topk=topk,
        device_str=device_str,
    )
    return {
        **out,
        "policy_mode_requested": mode,
        "policy_mode_used": "next",
        "artifact_training_objective": objective,
        "artifact_model_family": fam,
        "move_uci": out.get("best_legal", ""),
        "fallback": False,
    }
