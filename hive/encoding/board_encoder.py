"""Encode Hive board state as a fixed-size tensor for neural network input."""

from __future__ import annotations
import numpy as np
from ..core.hex import Hex
from ..core.pieces import Piece, PieceColor, PieceType, PIECE_COUNTS
from ..core.game import Game

# Grid dimensions: 23x23 centered hex grid
GRID_SIZE = 23
GRID_CENTER = GRID_SIZE // 2  # 11

# Channel layout:
# 0-4:   White pieces on top (Q, S, B, G, A) - binary
# 5-9:   Black pieces on top (Q, S, B, G, A) - binary
# 10-14: White pieces anywhere in stack (Q, S, B, G, A)
# 15-19: Black pieces anywhere in stack (Q, S, B, G, A)
# 20:    Stack height (normalized by /7)
# 21:    Current player's pieces (1 = current player, 0 = opponent)
# 22:    Is current player white (constant plane)
# Total: 23 channels

NUM_CHANNELS = 23

PIECE_TYPE_INDEX = {
    PieceType.QUEEN: 0,
    PieceType.SPIDER: 1,
    PieceType.BEETLE: 2,
    PieceType.GRASSHOPPER: 3,
    PieceType.ANT: 4,
}

# Reserve vector: 5 piece types x 2 colors = 10 values
RESERVE_SIZE = 10


def hex_to_grid(h: Hex) -> tuple[int, int]:
    """Convert axial hex coordinate to grid position."""
    col = h.q + GRID_CENTER
    row = h.r + GRID_CENTER
    return (row, col)


def grid_to_hex(row: int, col: int) -> Hex:
    """Convert grid position back to axial hex coordinate."""
    return Hex(col - GRID_CENTER, row - GRID_CENTER)


def in_grid(row: int, col: int) -> bool:
    return 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE


def encode_board(game: Game) -> tuple[np.ndarray, np.ndarray]:
    """Encode game state as (board_tensor, reserve_vector).

    Returns:
        board_tensor: shape (NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
        reserve_vector: shape (RESERVE_SIZE,) normalized counts
    """
    board = game.board
    tensor = np.zeros((NUM_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    is_white_turn = game.turn_color == PieceColor.WHITE

    for pos, stack in board._stacks.items():
        row, col = hex_to_grid(pos)
        if not in_grid(row, col):
            continue

        # Top piece
        top = stack[-1]
        offset = 0 if top.color == PieceColor.WHITE else 5
        ch = PIECE_TYPE_INDEX[top.piece_type]
        tensor[offset + ch, row, col] = 1.0

        # All pieces in stack
        for piece in stack:
            offset2 = 10 if piece.color == PieceColor.WHITE else 15
            ch2 = PIECE_TYPE_INDEX[piece.piece_type]
            tensor[offset2 + ch2, row, col] += 1.0

        # Stack height
        tensor[20, row, col] = len(stack) / 7.0

        # Current player's piece
        if (top.color == PieceColor.WHITE) == is_white_turn:
            tensor[21, row, col] = 1.0

    # Current player plane
    if is_white_turn:
        tensor[22, :, :] = 1.0

    # Reserve vector
    reserve = np.zeros(RESERVE_SIZE, dtype=np.float32)
    for i, pt in enumerate(PIECE_TYPE_INDEX):
        max_count = PIECE_COUNTS[pt]
        # White reserve
        w_count = sum(1 for p in game.reserve(PieceColor.WHITE) if p.piece_type == pt)
        reserve[i] = w_count / max_count if max_count > 0 else 0.0
        # Black reserve
        b_count = sum(1 for p in game.reserve(PieceColor.BLACK) if p.piece_type == pt)
        reserve[5 + i] = b_count / max_count if max_count > 0 else 0.0

    return tensor, reserve
