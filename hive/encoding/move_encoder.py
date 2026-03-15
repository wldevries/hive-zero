"""Encode/decode Hive moves for neural network policy output.

Policy layout: 12 channels x 23 x 23 grid = 6,348 total policy logits.

Channels 0-5:  Movement by sliding/jumping. The destination cell stores the logit.
               Channel index = direction FROM which the piece came:
               0=from E, 1=from NE, 2=from NW, 3=from W, 4=from SW, 5=from SE
               (i.e., the piece at dest+DIRECTIONS[ch] moves to dest)
Channel 6:     Beetle stacking / staying on same hex (piece moves onto occupied dest)
Channels 7-11: Placement (piece from reserve). 7=Queen, 8=Spider, 9=Beetle,
               10=Grasshopper, 11=Ant. Destination cell stores the logit.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

from ..core.hex import Hex, DIRECTIONS
from ..core.pieces import Piece, PieceColor, PieceType, PIECE_COUNTS
from ..core.game import Game
from .board_encoder import GRID_SIZE, GRID_CENTER, hex_to_grid, grid_to_hex, in_grid

NUM_POLICY_CHANNELS = 12
POLICY_SIZE = NUM_POLICY_CHANNELS * GRID_SIZE * GRID_SIZE  # 6,348

PIECE_TYPE_CHANNEL = {
    PieceType.QUEEN: 7,
    PieceType.SPIDER: 8,
    PieceType.BEETLE: 9,
    PieceType.GRASSHOPPER: 10,
    PieceType.ANT: 11,
}


def _policy_index(channel: int, row: int, col: int) -> int:
    """Flat index into policy vector: channel * GRID_SIZE^2 + row * GRID_SIZE + col."""
    return channel * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col


def encode_move(piece: Piece, from_pos: Optional[Hex], to_pos: Hex) -> int:
    """Encode a move as a flat policy index."""
    dest_row, dest_col = hex_to_grid(to_pos)

    if from_pos is None:
        # Placement move
        channel = PIECE_TYPE_CHANNEL[piece.piece_type]
    else:
        # Movement move - find which direction the piece came from
        diff = from_pos - to_pos  # vector from dest to source
        channel = 6  # default: stacking (beetle on top, or grasshopper landing)
        for i, d in enumerate(DIRECTIONS):
            if diff == d:
                channel = i
                break

    return _policy_index(channel, dest_row, dest_col)


def decode_move(index: int) -> tuple[int, int, int]:
    """Decode a flat policy index into (channel, row, col)."""
    channel = index // (GRID_SIZE * GRID_SIZE)
    remainder = index % (GRID_SIZE * GRID_SIZE)
    row = remainder // GRID_SIZE
    col = remainder % GRID_SIZE
    return channel, row, col


def get_legal_move_mask(game: Game) -> tuple[np.ndarray, list]:
    """Create a binary mask over the policy space for legal moves.

    Returns:
        mask: shape (POLICY_SIZE,) with 1.0 for legal moves
        moves: list of (idx, piece, from_pos, to_pos) for legal moves
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
