from typing import List

import chess
import torch

from src.chessbot.phase import board_from_context

BOARD_FEATURE_PLANES = 18

_PIECE_PLANE = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,
    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}


def board_state_planes_from_board(board: chess.Board) -> torch.Tensor:
    planes = torch.zeros((BOARD_FEATURE_PLANES, 8, 8), dtype=torch.float32)
    for square, piece in board.piece_map().items():
        plane = _PIECE_PLANE.get((piece.piece_type, piece.color))
        if plane is None:
            continue
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        planes[plane, rank, file] = 1.0

    planes[12].fill_(1.0 if board.turn == chess.WHITE else 0.0)
    planes[13].fill_(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    planes[14].fill_(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    planes[15].fill_(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    planes[16].fill_(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
    if board.ep_square is not None:
        ep_rank = chess.square_rank(board.ep_square)
        ep_file = chess.square_file(board.ep_square)
        planes[17, ep_rank, ep_file] = 1.0
    return planes


def board_state_planes_from_context(context: List[str]) -> torch.Tensor:
    board, ok = board_from_context(context)
    if not ok:
        raise ValueError("Illegal context; cannot derive board-state planes")
    return board_state_planes_from_board(board)
