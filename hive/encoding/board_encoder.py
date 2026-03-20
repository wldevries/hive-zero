"""Encode Hive board state as a fixed-size tensor for neural network input."""

from __future__ import annotations
import numpy as np
from ..core.hex import Hex
from ..core.pieces import Piece, PieceColor, PieceType, PIECE_COUNTS
from ..core.game import Game

# Grid dimensions: 23x23 centered hex grid
GRID_SIZE = 23
GRID_CENTER = GRID_SIZE // 2  # 11

# Channel layout (all channels are current-player-relative):
# 0-4:   Current player's pieces on top (Q, S, B, G, A) - binary
# 5-9:   Opponent's pieces on top (Q, S, B, G, A) - binary
# 10-14: Current player's pieces in stack (count)
# 15-19: Opponent's pieces in stack (count)
# 20:    Stack height (normalized by /7)
# 21:    Current player's pieces marker (1 = current player's top piece)
# Total: 22 channels
#
# Reserve vector (current-player-relative):
# 0-4:   Current player's reserve counts (normalized)
# 5-9:   Opponent's reserve counts (normalized)

NUM_CHANNELS = 22

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

    All channels are current-player-relative: channel 0-4 are always the
    current player's pieces, channels 5-9 are always the opponent's.

    Returns:
        board_tensor: shape (NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
        reserve_vector: shape (RESERVE_SIZE,) normalized counts
    """
    board = game.board
    tensor = np.zeros((NUM_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    is_white_turn = game.turn_color == PieceColor.WHITE
    cur_color = PieceColor.WHITE if is_white_turn else PieceColor.BLACK
    opp_color = PieceColor.BLACK if is_white_turn else PieceColor.WHITE

    def is_mine(color: PieceColor) -> bool:
        return color == cur_color

    for pos, stack in board._stacks.items():
        row, col = hex_to_grid(pos)
        if not in_grid(row, col):
            continue

        # Top piece — current player's in 0-4, opponent's in 5-9
        top = stack[-1]
        offset = 0 if is_mine(top.color) else 5
        ch = PIECE_TYPE_INDEX[top.piece_type]
        tensor[offset + ch, row, col] = 1.0

        # All pieces in stack — current player's in 10-14, opponent's in 15-19
        for piece in stack:
            offset2 = 10 if is_mine(piece.color) else 15
            ch2 = PIECE_TYPE_INDEX[piece.piece_type]
            tensor[offset2 + ch2, row, col] += 1.0

        # Stack height
        tensor[20, row, col] = len(stack) / 7.0

        # Current player's piece marker
        if is_mine(top.color):
            tensor[21, row, col] = 1.0

    # Reserve vector — current player first (0-4), opponent second (5-9)
    reserve = np.zeros(RESERVE_SIZE, dtype=np.float32)
    for i, pt in enumerate(PIECE_TYPE_INDEX):
        max_count = PIECE_COUNTS[pt]
        cur_count = sum(1 for p in game.reserve(cur_color) if p.piece_type == pt)
        reserve[i] = cur_count / max_count if max_count > 0 else 0.0
        opp_count = sum(1 for p in game.reserve(opp_color) if p.piece_type == pt)
        reserve[5 + i] = opp_count / max_count if max_count > 0 else 0.0

    return tensor, reserve
