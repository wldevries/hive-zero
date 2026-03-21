"""Encode Hive board state as a fixed-size tensor for neural network input."""

from __future__ import annotations
import numpy as np
from ..core.hex import Hex
from ..core.pieces import Piece, PieceColor, PieceType, PIECE_COUNTS
from ..core.game import Game

# Grid dimensions: 23x23 centered hex grid
GRID_SIZE = 23
GRID_CENTER = GRID_SIZE // 2  # 11

# Channel layout (all channels current-player-relative):
#
# Base layer — piece at depth 0 of each hex (binary, one per piece):
#   0-10:  Current player's pieces  (Q, S1, S2, B1, B2, G1, G2, G3, A1, A2, A3)
#   11-21: Opponent's pieces
#
# Stacked beetles — beetle at depth D above the base (binary):
#   22-25: Current player's Beetle1 at depths 1-4
#   26-29: Current player's Beetle2 at depths 1-4
#   30-33: Opponent's Beetle1 at depths 1-4
#   34-37: Opponent's Beetle2 at depths 1-4
#
#   Channel = 22 + player_offset(0 or 8) + (beetle_number-1)*4 + (depth-1)
#
# 38: Stack height (normalized /7)
#
# Reserve vector (current-player-relative):
#   0-4:  Current player's reserve counts (normalized by max)
#   5-9:  Opponent's reserve counts

NUM_CHANNELS = 39

PIECE_TYPE_INDEX = {
    PieceType.QUEEN: 0,
    PieceType.SPIDER: 1,
    PieceType.BEETLE: 2,
    PieceType.GRASSHOPPER: 3,
    PieceType.ANT: 4,
}

# Reserve vector: 5 piece types x 2 colors = 10 values
RESERVE_SIZE = 10

# Cumulative piece-index offsets within a player (matches Piece::linear_index() % 11)
# Q=1, S=2, B=2, G=3, A=3 → offsets [0, 1, 3, 5, 8]
_BASE_TYPES = [PieceType.QUEEN, PieceType.SPIDER, PieceType.BEETLE,
               PieceType.GRASSHOPPER, PieceType.ANT]
_TYPE_OFFSET: dict[PieceType, int] = {}
_off = 0
for _pt, _cnt in zip(_BASE_TYPES, [PIECE_COUNTS[pt] for pt in _BASE_TYPES]):
    _TYPE_OFFSET[_pt] = _off
    _off += _cnt

_STACKED_BEETLE_BASE = 22
_STACK_HEIGHT_CH = 38


def _piece_idx(piece: Piece) -> int:
    """Piece index within its player (0-10). Mirrors Rust linear_index() % 11."""
    return _TYPE_OFFSET[piece.piece_type] + (piece.number - 1)


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
    cur_color = game.turn_color
    opp_color = PieceColor.BLACK if cur_color == PieceColor.WHITE else PieceColor.WHITE

    def is_mine(color: PieceColor) -> bool:
        return color == cur_color

    for pos, stack in board._stacks.items():
        row, col = hex_to_grid(pos)
        if not in_grid(row, col):
            continue

        # Stack height
        tensor[_STACK_HEIGHT_CH, row, col] = len(stack) / 7.0

        # stack[0] = depth 0 (base piece), stack[-1] = top
        for depth, piece in enumerate(stack):
            mine = is_mine(piece.color)
            idx = _piece_idx(piece)

            if depth == 0:
                # Base layer
                ch = idx if mine else 11 + idx
                tensor[ch, row, col] = 1.0
            else:
                # Stacked beetle
                player_offset = 0 if mine else 8
                beetle_offset = (piece.number - 1) * 4
                depth_offset = min(depth - 1, 3)  # depths 1-4 → offsets 0-3
                ch = _STACKED_BEETLE_BASE + player_offset + beetle_offset + depth_offset
                tensor[ch, row, col] = 1.0

    # Reserve vector — current player first (0-4), opponent second (5-9)
    reserve = np.zeros(RESERVE_SIZE, dtype=np.float32)
    for i, pt in enumerate(_BASE_TYPES):
        max_count = PIECE_COUNTS[pt]
        cur_count = sum(1 for p in game.reserve(cur_color) if p.piece_type == pt)
        reserve[i] = cur_count / max_count if max_count > 0 else 0.0
        opp_count = sum(1 for p in game.reserve(opp_color) if p.piece_type == pt)
        reserve[5 + i] = opp_count / max_count if max_count > 0 else 0.0

    return tensor, reserve
