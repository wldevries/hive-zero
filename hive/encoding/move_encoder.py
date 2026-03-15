"""Encode/decode Hive moves for neural network policy output."""

from __future__ import annotations
import numpy as np
from typing import Optional

from ..core.hex import Hex
from ..core.pieces import Piece, PieceColor, PieceType, PIECE_COUNTS
from ..core.game import Game
from .board_encoder import GRID_SIZE, GRID_CENTER, hex_to_grid, grid_to_hex, in_grid

# Move encoding:
# Movement moves: source_grid_pos * GRID_SIZE^2 + dest_grid_pos
# Placement moves: (GRID_SIZE^2 + piece_type_slot) * GRID_SIZE^2 + dest_grid_pos
#
# Piece type slots for placement (5 types per color, but since it's always
# the current player placing, we use 5 slots):
# 0=Queen, 1=Spider, 2=Beetle, 3=Grasshopper, 4=Ant
NUM_PIECE_SLOTS = 5  # one per piece type for placement
NUM_SOURCES = GRID_SIZE * GRID_SIZE + NUM_PIECE_SLOTS  # grid positions + placement slots
NUM_DESTS = GRID_SIZE * GRID_SIZE
POLICY_SIZE = NUM_SOURCES * NUM_DESTS  # total flat policy space

PIECE_TYPE_SLOT = {
    PieceType.QUEEN: 0,
    PieceType.SPIDER: 1,
    PieceType.BEETLE: 2,
    PieceType.GRASSHOPPER: 3,
    PieceType.ANT: 4,
}


def _grid_index(row: int, col: int) -> int:
    return row * GRID_SIZE + col


def encode_move(piece: Piece, from_pos: Optional[Hex], to_pos: Hex) -> int:
    """Encode a move as a flat policy index."""
    dest_row, dest_col = hex_to_grid(to_pos)
    dest_idx = _grid_index(dest_row, dest_col)

    if from_pos is None:
        # Placement move
        slot = PIECE_TYPE_SLOT[piece.piece_type]
        src_idx = GRID_SIZE * GRID_SIZE + slot
    else:
        # Movement move
        src_row, src_col = hex_to_grid(from_pos)
        src_idx = _grid_index(src_row, src_col)

    return src_idx * NUM_DESTS + dest_idx


def decode_move(index: int) -> tuple[int, int]:
    """Decode a flat policy index into (source_index, dest_index).

    Returns raw indices - caller must interpret whether source is a grid
    position or a placement slot.
    """
    src_idx = index // NUM_DESTS
    dst_idx = index % NUM_DESTS
    return src_idx, dst_idx


def get_legal_move_mask(game: Game) -> tuple[np.ndarray, list]:
    """Create a binary mask over the policy space for legal moves.

    Returns:
        mask: shape (POLICY_SIZE,) with 1.0 for legal moves
        moves: list of (piece, from_pos, to_pos) in same order as mask indices
    """
    mask = np.zeros(POLICY_SIZE, dtype=np.float32)
    valid_moves = game.valid_moves()
    indexed_moves = []

    for piece, from_pos, to_pos in valid_moves:
        idx = encode_move(piece, from_pos, to_pos)
        if 0 <= idx < POLICY_SIZE:
            mask[idx] = 1.0
            indexed_moves.append((idx, piece, from_pos, to_pos))

    return mask, indexed_moves


def policy_to_moves(policy: np.ndarray, game: Game) -> list[tuple[float, Piece, Optional[Hex], Hex]]:
    """Convert masked policy distribution to sorted (prob, piece, from, to) list."""
    mask, indexed_moves = get_legal_move_mask(game)

    # Apply mask and renormalize
    masked_policy = policy * mask
    total = masked_policy.sum()
    if total > 0:
        masked_policy /= total

    result = []
    for idx, piece, from_pos, to_pos in indexed_moves:
        prob = masked_policy[idx]
        result.append((prob, piece, from_pos, to_pos))

    result.sort(key=lambda x: -x[0])
    return result
