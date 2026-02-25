from typing import Dict, List, Tuple

import chess

PHASE_UNKNOWN = "unknown"
PHASE_OPENING = "opening"
PHASE_MIDDLEGAME = "middlegame"
PHASE_ENDGAME = "endgame"

PHASE_TO_ID = {
    PHASE_UNKNOWN: 0,
    PHASE_OPENING: 1,
    PHASE_MIDDLEGAME: 2,
    PHASE_ENDGAME: 3,
}

PHASE_RULE_VERSION = "material_castling_v1"

_NONPAWN_VALUES = {
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}


def phase_to_id(name: str) -> int:
    return PHASE_TO_ID.get((name or "").strip().lower(), PHASE_TO_ID[PHASE_UNKNOWN])


def phase_name_from_id(idx: int) -> str:
    for name, phase_id in PHASE_TO_ID.items():
        if phase_id == idx:
            return name
    return PHASE_UNKNOWN


def remaining_plies_bucket(plies_remaining: int) -> str:
    try:
        value = int(plies_remaining)
    except Exception:
        return "unknown"
    if value < 0:
        return "unknown"
    if value <= 10:
        return "<=10"
    if value <= 20:
        return "11-20"
    return ">20"


def relative_progress_bucket(ply: int, plies_total: int) -> str:
    try:
        ply_i = int(ply)
        total_i = int(plies_total)
    except Exception:
        return "unknown"
    if total_i <= 0:
        return "unknown"
    frac = float(ply_i) / float(total_i)
    if frac < 0.33:
        return "early"
    if frac < 0.66:
        return "mid"
    return "late"


def _total_nonpawn_material(board: chess.Board) -> int:
    total = 0
    for piece_type, value in _NONPAWN_VALUES.items():
        total += value * len(board.pieces(piece_type, chess.WHITE))
        total += value * len(board.pieces(piece_type, chess.BLACK))
    return total


def classify_board_phase(board: chess.Board, ply: int | None = None) -> Dict[str, object]:
    total_nonpawn = _total_nonpawn_material(board)
    queens_off = (
        len(board.pieces(chess.QUEEN, chess.WHITE)) == 0
        and len(board.pieces(chess.QUEEN, chess.BLACK)) == 0
    )
    castling_rights_present = bool(
        board.has_kingside_castling_rights(chess.WHITE)
        or board.has_queenside_castling_rights(chess.WHITE)
        or board.has_kingside_castling_rights(chess.BLACK)
        or board.has_queenside_castling_rights(chess.BLACK)
    )

    if total_nonpawn <= 14:
        phase = PHASE_ENDGAME
        reason = "low_nonpawn_material"
    elif queens_off and total_nonpawn <= 20:
        phase = PHASE_ENDGAME
        reason = "queens_off_low_material"
    elif ply is not None and int(ply) <= 20 and castling_rights_present:
        phase = PHASE_OPENING
        reason = "early_with_castling_rights"
    else:
        phase = PHASE_MIDDLEGAME
        reason = "default_transition"

    return {
        "phase": phase,
        "phase_reason": reason,
        "phase_rule_version": PHASE_RULE_VERSION,
        "total_nonpawn_material": total_nonpawn,
        "queens_off": queens_off,
        "castling_rights_present": castling_rights_present,
        "ply": (int(ply) if ply is not None else None),
    }


def board_from_context(context: List[str]) -> Tuple[chess.Board, bool]:
    board = chess.Board()
    for uci in context:
        try:
            mv = chess.Move.from_uci(uci)
        except Exception:
            return board, False
        if mv not in board.legal_moves:
            return board, False
        board.push(mv)
    return board, True


def classify_context_phase(context: List[str]) -> Dict[str, object]:
    board, ok = board_from_context(context)
    if not ok:
        return {
            "phase": PHASE_UNKNOWN,
            "phase_reason": "invalid_context",
            "phase_rule_version": PHASE_RULE_VERSION,
            "ply": len(context),
        }
    return classify_board_phase(board, ply=len(context))
