import glob
import hashlib
import os
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import chess.pgn


class _ValidationVisitor(chess.pgn.BaseVisitor[Dict]):
    """PGN visitor that captures only the data needed for dataset validation."""

    def begin_game(self) -> None:
        self.headers = chess.pgn.Headers({})
        self.moves_uci: List[str] = []
        self.errors: List[str] = []
        self.board = None

    def begin_headers(self) -> chess.pgn.Headers:
        return self.headers

    def visit_header(self, tagname: str, tagvalue: str) -> None:
        self.headers[tagname] = tagvalue

    def begin_variation(self):
        # Dataset validation only uses the mainline; skipping variations avoids
        # unnecessary parsing work and object churn.
        return chess.pgn.SKIP

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        self.moves_uci.append(move.uci())

    def visit_board(self, board: chess.Board) -> None:
        self.board = board

    def visit_result(self, result: str) -> None:
        if self.headers.get("Result", "*") == "*":
            self.headers["Result"] = result

    def handle_error(self, error: Exception) -> None:
        # Collect parse/parser errors without logging to keep validation fast.
        self.errors.append(str(error))

    def result(self) -> Dict:
        board_result = "*"
        if self.board is not None and self.board.is_game_over(claim_draw=True):
            board_result = self.board.result(claim_draw=True)
        return {
            "headers": dict(self.headers),
            "moves_uci": self.moves_uci,
            "errors": self.errors,
            "board_result": board_result,
        }


def winner_from_result(result: str) -> str:
    if result == "1-0":
        return "W"
    if result == "0-1":
        return "B"
    if result == "1/2-1/2":
        return "D"
    return "?"


def game_id_for(headers: Dict[str, str], source_file: str, first_moves: List[str]) -> str:
    basis = "|".join(
        [
            source_file,
            headers.get("Event", ""),
            headers.get("Site", ""),
            headers.get("Date", ""),
            headers.get("Round", ""),
            headers.get("White", ""),
            headers.get("Black", ""),
            " ".join(first_moves[:16]),
        ]
    )
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


def validate_game(game: chess.pgn.Game, source_file: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    headers = {k: v for k, v in game.headers.items()}
    result = headers.get("Result", "*")
    board = game.board()
    moves_uci: List[str] = []

    try:
        for ply, move in enumerate(game.mainline_moves(), start=1):
            if move not in board.legal_moves:
                return None, {
                    "source_file": source_file,
                    "reason": "illegal_move",
                    "offending_ply": ply,
                    "result": result,
                }
            moves_uci.append(move.uci())
            board.push(move)
    except Exception as exc:
        return None, {
            "source_file": source_file,
            "reason": "parse_error",
            "offending_ply": len(moves_uci) + 1,
            "result": result,
            "error": str(exc),
        }

    if getattr(game, "errors", None):
        return None, {
            "source_file": source_file,
            "reason": "pgn_errors",
            "offending_ply": len(moves_uci),
            "result": result,
            "error": "; ".join(str(e) for e in game.errors[:3]),
        }

    if not moves_uci:
        return None, {
            "source_file": source_file,
            "reason": "empty_game",
            "offending_ply": 0,
            "result": result,
        }

    board_result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
    if board_result != "*" and result != "*" and board_result != result:
        return None, {
            "source_file": source_file,
            "reason": "result_mismatch",
            "offending_ply": len(moves_uci),
            "result": result,
            "board_result": board_result,
        }

    gid = game_id_for(headers, source_file, moves_uci)
    valid = {
        "game_id": gid,
        "source_file": source_file,
        "headers": headers,
        "result": result,
        "winner_side": winner_from_result(result),
        "plies": len(moves_uci),
        "moves_uci": moves_uci,
    }
    return valid, None


def iter_pgn_games(paths: Iterable[str]):
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                yield path, game


def _iter_parsed_games_fast(paths: Iterable[str]) -> Iterator[Tuple[str, Dict]]:
    for path in paths:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                parsed = chess.pgn.read_game(f, Visitor=_ValidationVisitor)
                if parsed is None:
                    break
                yield path, parsed


def resolve_inputs(input_arg: str) -> List[str]:
    if os.path.isdir(input_arg):
        return sorted(glob.glob(os.path.join(input_arg, "*.pgn")))
    if any(c in input_arg for c in "*?[]"):
        return sorted(glob.glob(input_arg))
    return [input_arg]


def _invalid_row_from_valid_too_short(valid: Dict, source_file: str) -> Dict:
    return {
        "game_id": valid["game_id"],
        "source_file": source_file,
        "reason": "too_short",
        "offending_ply": valid["plies"],
        "result": valid["result"],
    }


def _invalid_row_from_invalid(invalid: Dict, source_file: str) -> Dict:
    return {
        "game_id": "",
        "source_file": invalid.get("source_file", source_file),
        "reason": invalid.get("reason", "unknown"),
        "offending_ply": invalid.get("offending_ply", 0),
        "result": invalid.get("result", "*"),
    }


def _validate_parsed_game(parsed: Dict, source_file: str) -> Tuple[Optional[Dict], Optional[Dict]]:
    headers = parsed.get("headers", {})
    moves_uci = parsed.get("moves_uci", [])
    errors = parsed.get("errors", [])
    result = headers.get("Result", "*")

    if errors:
        return None, {
            "source_file": source_file,
            "reason": "pgn_errors",
            "offending_ply": len(moves_uci),
            "result": result,
            "error": "; ".join(errors[:3]),
        }

    if not moves_uci:
        return None, {
            "source_file": source_file,
            "reason": "empty_game",
            "offending_ply": 0,
            "result": result,
        }

    board_result = parsed.get("board_result", "*")
    if board_result != "*" and result != "*" and board_result != result:
        return None, {
            "source_file": source_file,
            "reason": "result_mismatch",
            "offending_ply": len(moves_uci),
            "result": result,
            "board_result": board_result,
        }

    gid = game_id_for(headers, source_file, moves_uci)
    valid = {
        "game_id": gid,
        "source_file": source_file,
        "headers": headers,
        "result": result,
        "winner_side": winner_from_result(result),
        "plies": len(moves_uci),
        "moves_uci": moves_uci,
    }
    return valid, None


def iter_validation_events(paths: List[str], min_plies: int) -> Iterator[Tuple[str, Dict]]:
    """Yield ('valid'|'invalid', row) events for streaming validation writes."""
    for source_file, parsed in _iter_parsed_games_fast(paths):
        valid, invalid = _validate_parsed_game(parsed, source_file)
        if valid is not None:
            if valid["plies"] < min_plies:
                yield "invalid", _invalid_row_from_valid_too_short(valid, source_file)
            else:
                yield "valid", valid
            continue

        assert invalid is not None
        yield "invalid", _invalid_row_from_invalid(invalid, source_file)


def validate_games(paths: List[str], min_plies: int) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
    valid_rows: List[Dict] = []
    invalid_rows: List[Dict] = []
    reason_counts: Dict[str, int] = {}

    for kind, row in iter_validation_events(paths=paths, min_plies=min_plies):
        if kind == "valid":
            valid_rows.append(row)
        else:
            invalid_rows.append(row)
            reason = row.get("reason", "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return valid_rows, invalid_rows, reason_counts
