from typing import Dict, List, Optional, Sequence

import chess
import torch

from src.chessbot.board_features import board_state_planes_from_context
from src.chessbot.model import (
    NextMoveLSTM,
    NextMoveSeqBoardLSTM,
    NextMoveSeqLSTM,
    encode_tokens,
    side_to_move_id_from_context_len,
    winner_to_id,
)
from src.chessbot.phase import PHASE_UNKNOWN, classify_context_phase, phase_to_id

MODEL_FAMILY_DUAL_SEQUENCE = "dual_side_sequence_lstm"
MODEL_FAMILY_DUAL_SEQUENCE_BOARD = "dual_side_sequence_board_lstm"


def parse_context(text: str) -> List[str]:
    return [x.strip() for x in text.split() if x.strip()]


def _coerce_topk_multipliers(values: Optional[Sequence[int]]) -> List[int]:
    if values is None:
        return [1, 2, 5]
    out: List[int] = []
    for raw in values:
        try:
            v = int(raw)
        except Exception:
            continue
        if v <= 0:
            continue
        if v not in out:
            out.append(v)
    if not out:
        return [1, 2, 5]
    return out


def resolve_topk_attempts(base_topk: int, vocab_size: int, topk_multipliers: Optional[Sequence[int]] = None) -> List[int]:
    base = max(1, int(base_topk))
    vmax = max(1, int(vocab_size))
    multipliers = _coerce_topk_multipliers(topk_multipliers)
    out: List[int] = []
    for m in multipliers:
        k = min(vmax, max(1, base * int(m)))
        if k not in out:
            out.append(k)
    if not out:
        out = [min(vmax, base)]
    return out


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


def artifact_model_side(artifact: Dict) -> str:
    return str(artifact.get("model_side") or "").strip().lower()


def artifact_sequence_horizon(artifact: Dict) -> int:
    cfg = artifact.get("config") or {}
    try:
        return max(1, int(cfg.get("horizon") or 1))
    except Exception:
        return 1


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


def _legal_token_ids_for_board(board: chess.Board, vocab: Dict[str, int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for mv in board.legal_moves:
        tok = mv.uci()
        tok_id = vocab.get(tok)
        if tok_id is None:
            continue
        tok_id = int(tok_id)
        if tok_id in seen:
            continue
        seen.add(tok_id)
        out.append(tok_id)
    return out


def _topk_legal_tokens_for_board_from_logits(
    *,
    logits_1d: torch.Tensor,
    board: chess.Board,
    vocab: Dict[str, int],
    inv_vocab: Dict[int, str],
    k: int,
) -> List[str]:
    legal_ids = _legal_token_ids_for_board(board, vocab)
    if not legal_ids:
        return []
    legal_idx = torch.tensor(legal_ids, dtype=torch.long, device=logits_1d.device)
    legal_logits = logits_1d.index_select(0, legal_idx)
    k_eff = min(max(1, int(k)), int(legal_logits.shape[0]))
    best_local = legal_logits.topk(k_eff, dim=-1).indices
    best_global = legal_idx.index_select(0, best_local).detach().cpu().tolist()
    out: List[str] = []
    for tok_id in best_global:
        tok = inv_vocab.get(int(tok_id), "")
        if tok:
            out.append(tok)
    return out


def _decode_best_legal_from_next_logits(
    *,
    logits_1d: torch.Tensor,
    vocab: Dict[str, int],
    inv_vocab: Dict[int, str],
    context: List[str],
    topk: int,
    fallback_topk_multipliers: Optional[Sequence[int]] = None,
) -> Dict:
    board = chess.Board()
    for uci in context:
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            raise ValueError(f"Illegal context move: {uci}")
        board.push(mv)

    vocab_size = int(logits_1d.shape[-1])
    attempts = resolve_topk_attempts(topk, vocab_size, fallback_topk_multipliers)
    topk_tokens_last: List[str] = []
    used_k = attempts[-1] if attempts else max(1, min(int(topk), vocab_size))
    for k in attempts:
        topk_tokens = _topk_legal_tokens_for_board_from_logits(
            logits_1d=logits_1d,
            board=board,
            vocab=vocab,
            inv_vocab=inv_vocab,
            k=int(k),
        )
        legal = str(topk_tokens[0] if topk_tokens else "")
        topk_tokens_last = topk_tokens
        used_k = int(k)
        if legal:
            return {
                "best_legal": legal,
                "topk": topk_tokens,
                "decode_topk_attempts": attempts,
                "decode_topk_used": int(k),
            }
    return {
        "best_legal": "",
        "topk": topk_tokens_last,
        "decode_topk_attempts": attempts,
        "decode_topk_used": int(used_k),
    }


def infer_from_artifact(
    artifact: Dict,
    context: List[str],
    winner_side: str,
    topk: int,
    fallback_topk_multipliers: Optional[Sequence[int]] = None,
) -> Dict:
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
        decoded = _decode_best_legal_from_next_logits(
            logits_1d=logits[0],
            vocab=vocab,
            inv_vocab=inv_vocab,
            context=context,
            topk=topk,
            fallback_topk_multipliers=fallback_topk_multipliers,
        )
    return {
        "topk": list(decoded.get("topk") or []),
        "best_legal": str(decoded.get("best_legal") or ""),
        "decode_topk_attempts": list(decoded.get("decode_topk_attempts") or []),
        "decode_topk_used": int(decoded.get("decode_topk_used") or 0),
    }


def infer_from_artifact_on_device(
    artifact: Dict,
    context: List[str],
    winner_side: str,
    topk: int,
    device_str: str = "cpu",
    fallback_topk_multipliers: Optional[Sequence[int]] = None,
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
        decoded = _decode_best_legal_from_next_logits(
            logits_1d=logits[0],
            vocab=vocab,
            inv_vocab=inv_vocab,
            context=context,
            topk=topk,
            fallback_topk_multipliers=fallback_topk_multipliers,
        )
    return {
        "topk": list(decoded.get("topk") or []),
        "best_legal": str(decoded.get("best_legal") or ""),
        "decode_topk_attempts": list(decoded.get("decode_topk_attempts") or []),
        "decode_topk_used": int(decoded.get("decode_topk_used") or 0),
        "device": str(device),
    }


def _rollout_sequence_from_first_token(
    *,
    first_token: str,
    board_start: chess.Board,
    step_logits: torch.Tensor,
    vocab: Dict[str, int],
    inv_vocab: Dict[int, str],
    k: int,
) -> List[str]:
    board = board_start.copy(stack=False)
    try:
        first_mv = chess.Move.from_uci(first_token)
    except Exception:
        return []
    if first_mv not in board.legal_moves:
        return []
    board.push(first_mv)
    out = [first_token]
    horizon = int(step_logits.shape[0])
    for step in range(1, horizon):
        topk_tokens = _topk_legal_tokens_for_board_from_logits(
            logits_1d=step_logits[step],
            board=board,
            vocab=vocab,
            inv_vocab=inv_vocab,
            k=int(k),
        )
        chosen = str(topk_tokens[0] if topk_tokens else "")
        if not chosen:
            break
        mv = chess.Move.from_uci(chosen)
        if mv not in board.legal_moves:
            break
        board.push(mv)
        out.append(chosen)
        if board.is_game_over(claim_draw=True):
            break
    return out


def _sequence_with_probs(
    *,
    sequence_tokens: List[str],
    vocab: Dict[str, int],
    step_probs: torch.Tensor,
    step_log_probs: torch.Tensor,
) -> List[Dict]:
    out: List[Dict] = []
    steps = min(len(sequence_tokens), int(step_probs.shape[0]))
    for step_i in range(steps):
        tok = str(sequence_tokens[step_i] or "")
        tok_id = vocab.get(tok)
        if tok_id is None:
            prob = 0.0
            log_prob = float("-inf")
        else:
            prob = float(step_probs[step_i, int(tok_id)].item())
            log_prob = float(step_log_probs[step_i, int(tok_id)].item())
        out.append(
            {
                "ply": int(step_i + 1),
                "move_uci": tok,
                "probability": prob,
                "log_probability": log_prob,
            }
        )
    return out


def _decode_sequence_first_move(
    *,
    logits: torch.Tensor,
    vocab: Dict[str, int],
    inv_vocab: Dict[int, str],
    context: List[str],
    topk: int,
    sequence_decode_policy: str,
    fallback_topk_multipliers: Optional[Sequence[int]],
) -> Dict:
    policy = str(sequence_decode_policy or "sequence_path").strip().lower()
    if policy not in {"sequence_path", "step1_legal"}:
        raise ValueError(f"Unsupported sequence_decode_policy: {sequence_decode_policy}")

    board_start = chess.Board()
    for uci in context:
        mv = chess.Move.from_uci(uci)
        if mv not in board_start.legal_moves:
            raise ValueError(f"Illegal context move: {uci}")
        board_start.push(mv)

    vocab_size = int(logits.shape[-1])
    attempts = resolve_topk_attempts(topk, vocab_size, fallback_topk_multipliers)
    log_probs = torch.log_softmax(logits[0], dim=-1)

    topk_step1_last: List[str] = []
    topk_ids_all_last: List[List[int]] = []
    used_k = attempts[-1] if attempts else max(1, min(int(topk), vocab_size))
    step_logits = logits[0]
    for k in attempts:
        topk_ids_all = logits.topk(int(k), dim=-1).indices[0].detach().cpu().tolist()
        topk_step1 = _topk_legal_tokens_for_board_from_logits(
            logits_1d=step_logits[0],
            board=board_start,
            vocab=vocab,
            inv_vocab=inv_vocab,
            k=int(k),
        )
        legal_first = list(topk_step1)
        topk_step1_last = topk_step1
        topk_ids_all_last = topk_ids_all
        used_k = int(k)
        if not legal_first:
            continue

        if policy == "step1_legal":
            chosen_first = legal_first[0]
            chosen_sequence = _rollout_sequence_from_first_token(
                first_token=chosen_first,
                board_start=board_start,
                step_logits=step_logits,
                vocab=vocab,
                inv_vocab=inv_vocab,
                k=int(k),
            )
            return {
                "best_legal": chosen_first,
                "legal_sequence": chosen_sequence,
                "topk_step1": topk_step1,
                "decode_topk_attempts": attempts,
                "decode_topk_used": int(k),
                "sequence_decode_policy_used": policy,
            }

        best_first = ""
        best_sequence: List[str] = []
        best_score = float("-inf")
        for tok in legal_first:
            sequence = _rollout_sequence_from_first_token(
                first_token=tok,
                board_start=board_start,
                step_logits=step_logits,
                vocab=vocab,
                inv_vocab=inv_vocab,
                k=int(k),
            )
            if not sequence:
                continue
            score = 0.0
            for step_i, move_uci in enumerate(sequence):
                tok_id = vocab.get(move_uci)
                if tok_id is None:
                    score += -50.0
                    continue
                score += float(log_probs[step_i, int(tok_id)].item())
            if (score > best_score) or (
                score == best_score and len(sequence) > len(best_sequence)
            ):
                best_score = score
                best_first = tok
                best_sequence = sequence
        if best_first:
            return {
                "best_legal": best_first,
                "legal_sequence": best_sequence,
                "topk_step1": topk_step1,
                "decode_topk_attempts": attempts,
                "decode_topk_used": int(k),
                "sequence_decode_policy_used": policy,
            }

    topk_tokens_all_last = [[inv_vocab.get(i, "") for i in step_ids] for step_ids in topk_ids_all_last]
    return {
        "best_legal": "",
        "legal_sequence": [],
        "topk_step1": topk_step1_last,
        "topk_tokens_all_last": topk_tokens_all_last,
        "decode_topk_attempts": attempts,
        "decode_topk_used": int(used_k),
        "sequence_decode_policy_used": policy,
    }


def infer_sequence_from_artifact_on_device(
    artifact: Dict,
    context: List[str],
    topk: int,
    device_str: str = "cpu",
    sequence_decode_policy: str = "sequence_path",
    fallback_topk_multipliers: Optional[Sequence[int]] = None,
) -> Dict:
    vocab = artifact["vocab"]
    inv_vocab = {idx: tok for tok, idx in vocab.items()}
    cfg = artifact["config"]
    fam = artifact_model_family(artifact)
    if fam not in {MODEL_FAMILY_DUAL_SEQUENCE, MODEL_FAMILY_DUAL_SEQUENCE_BOARD}:
        raise RuntimeError(f"Unsupported model family for sequence inference: {fam}")

    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    if fam == MODEL_FAMILY_DUAL_SEQUENCE_BOARD:
        model = NextMoveSeqBoardLSTM(vocab_size=len(vocab), **cfg).to(device)
    else:
        model = NextMoveSeqLSTM(vocab_size=len(vocab), **cfg).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()

    original_context_len = len(context)
    context_ids = encode_tokens(context, vocab)
    if not context_ids:
        context_ids = [vocab.get("<UNK>", 1)]
    tokens = torch.tensor([context_ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(context_ids)], dtype=torch.long, device=device)
    side_to_moves = torch.tensor([side_to_move_id_from_context_len(original_context_len)], dtype=torch.long, device=device)
    board_state = None
    if fam == MODEL_FAMILY_DUAL_SEQUENCE_BOARD:
        board_state = board_state_planes_from_context(context).unsqueeze(0).to(device)
    with torch.no_grad():
        if fam == MODEL_FAMILY_DUAL_SEQUENCE_BOARD:
            logits = model(tokens, lengths, side_to_moves, board_state=board_state)
        else:
            logits = model(tokens, lengths, side_to_moves)
        pred_ids = logits.argmax(dim=-1)[0].detach().cpu().tolist()
        step_probs = torch.softmax(logits[0], dim=-1).detach().cpu()
        step_log_probs = torch.log_softmax(logits[0], dim=-1).detach().cpu()
        decoded = _decode_sequence_first_move(
            logits=logits,
            vocab=vocab,
            inv_vocab=inv_vocab,
            context=context,
            topk=topk,
            sequence_decode_policy=sequence_decode_policy,
            fallback_topk_multipliers=fallback_topk_multipliers,
        )

    topk_step1 = list(decoded.get("topk_step1") or [])
    best_legal = str(decoded.get("best_legal") or "")
    predicted_sequence = [inv_vocab.get(i, "") for i in pred_ids]
    legal_sequence = list(decoded.get("legal_sequence") or [])
    predicted_sequence_with_probs = _sequence_with_probs(
        sequence_tokens=predicted_sequence,
        vocab=vocab,
        step_probs=step_probs,
        step_log_probs=step_log_probs,
    )
    legal_sequence_with_probs = _sequence_with_probs(
        sequence_tokens=legal_sequence,
        vocab=vocab,
        step_probs=step_probs,
        step_log_probs=step_log_probs,
    )

    return {
        "topk_step1": topk_step1,
        "best_legal": best_legal,
        "predicted_sequence": predicted_sequence,
        "predicted_sequence_with_probs": predicted_sequence_with_probs,
        "legal_sequence": legal_sequence,
        "legal_sequence_with_probs": legal_sequence_with_probs,
        "first_move": best_legal,
        "horizon": int(artifact_sequence_horizon(artifact)),
        "model_side": artifact_model_side(artifact),
        "sequence_decode_policy_used": str(decoded.get("sequence_decode_policy_used") or sequence_decode_policy),
        "decode_topk_attempts": list(decoded.get("decode_topk_attempts") or []),
        "decode_topk_used": int(decoded.get("decode_topk_used") or 0),
        "device": str(device),
    }


def infer_first_move_dual_artifacts_on_device(
    white_artifact: Dict,
    black_artifact: Dict,
    context: List[str],
    topk: int,
    device_str: str = "cpu",
    sequence_decode_policy: str = "sequence_path",
    fallback_topk_multipliers: Optional[Sequence[int]] = None,
) -> Dict:
    stm = side_to_move_id_from_context_len(len(context))
    side_name = "white" if int(stm) == 0 else "black"
    artifact = white_artifact if side_name == "white" else black_artifact
    out = infer_sequence_from_artifact_on_device(
        artifact=artifact,
        context=context,
        topk=topk,
        device_str=device_str,
        sequence_decode_policy=sequence_decode_policy,
        fallback_topk_multipliers=fallback_topk_multipliers,
    )
    return {
        **out,
        "selected_model_side": side_name,
        "move_uci": out.get("best_legal", ""),
        "fallback": not bool(out.get("best_legal", "")),
    }


def infer_rollout_from_artifact_on_device(
    artifact: Dict,
    context: List[str],
    winner_side: str,
    topk: int,
    rollout_plies: int,
    device_str: str = "cpu",
    fallback_legal: bool = False,
    fallback_topk_multipliers: Optional[Sequence[int]] = None,
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
            decoded = _decode_best_legal_from_next_logits(
                logits_1d=logits[0],
                vocab=vocab,
                inv_vocab=inv_vocab,
                context=local_context,
                topk=topk,
                fallback_topk_multipliers=fallback_topk_multipliers,
            )
        topk_tokens = list(decoded.get("topk") or [])
        legal = str(decoded.get("best_legal") or "")
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
                "decode_topk_attempts": list(decoded.get("decode_topk_attempts") or []),
                "decode_topk_used": int(decoded.get("decode_topk_used") or 0),
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
    sequence_decode_policy: str = "sequence_path",
    fallback_topk_multipliers: Optional[Sequence[int]] = None,
) -> Dict:
    mode = str(policy_mode or "auto").strip().lower()
    if mode not in {"auto", "next", "rollout"}:
        raise ValueError(f"Unsupported policy_mode: {policy_mode}")

    objective = artifact_training_objective(artifact)
    fam = artifact_model_family(artifact)
    if fam in {MODEL_FAMILY_DUAL_SEQUENCE, MODEL_FAMILY_DUAL_SEQUENCE_BOARD}:
        out = infer_sequence_from_artifact_on_device(
            artifact=artifact,
            context=context,
            topk=topk,
            device_str=device_str,
            sequence_decode_policy=sequence_decode_policy,
            fallback_topk_multipliers=fallback_topk_multipliers,
        )
        move_uci = str(out.get("best_legal") or "")
        return {
            **out,
            "policy_mode_requested": mode,
            "policy_mode_used": "sequence",
            "artifact_training_objective": objective,
            "artifact_model_family": fam,
            "move_uci": move_uci,
            "fallback": not bool(move_uci),
        }
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
            fallback_topk_multipliers=fallback_topk_multipliers,
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
        fallback_topk_multipliers=fallback_topk_multipliers,
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
