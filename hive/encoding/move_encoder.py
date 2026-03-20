"""Encode/decode Hive moves for neural network policy output.

Policy layout: 11 channels x 23 x 23 grid = 5,819 total policy logits.

Channel = piece index within current player (0-10), for both placement and movement.
Destination cell stores the logit. Piece identity is the only discriminator — no
direction encoding, no separate placement channels.

Channel mapping (mirrors Rust Piece::linear_index() % PIECES_PER_PLAYER):
  0: Queen
  1: Spider1,  2: Spider2
  3: Beetle1,  4: Beetle2
  5: Grasshopper1, 6: Grasshopper2, 7: Grasshopper3
  8: Ant1, 9: Ant2, 10: Ant3

Canonical placement ordering is enforced by valid_moves(): only the lowest-numbered
piece of each type in reserve is offered, so piece numbers are unambiguous.
"""

from __future__ import annotations
import numpy as np

from ..core.pieces import Piece, PieceType, PIECE_COUNTS
from ..core.game import Game
from .board_encoder import GRID_SIZE, hex_to_grid

NUM_POLICY_CHANNELS = 11
PIECES_PER_PLAYER = 11
POLICY_SIZE = NUM_POLICY_CHANNELS * GRID_SIZE * GRID_SIZE  # 5,819

# Cumulative offsets for piece type within a player, matching PIECE_COUNTS order.
# Q=1, S=2, B=2, G=3, A=3 → offsets [0, 1, 3, 5, 8]
_BASE_TYPES = [PieceType.QUEEN, PieceType.SPIDER, PieceType.BEETLE, PieceType.GRASSHOPPER, PieceType.ANT]
_TYPE_OFFSET: dict[PieceType, int] = {}
_offset = 0
for _pt, _count in zip(_BASE_TYPES, [PIECE_COUNTS[pt] for pt in _BASE_TYPES]):
    _TYPE_OFFSET[_pt] = _offset
    _offset += _count


def _piece_channel(piece: Piece) -> int:
    """Channel for a piece: its index within the current player (0-10).

    Mirrors Rust: piece.linear_index() % PIECES_PER_PLAYER.
    """
    return _TYPE_OFFSET[piece.piece_type] + (piece.number - 1)


def _policy_index(channel: int, row: int, col: int) -> int:
    """Flat index into policy vector: channel * GRID_SIZE^2 + row * GRID_SIZE + col."""
    return channel * GRID_SIZE * GRID_SIZE + row * GRID_SIZE + col


def encode_move(piece: Piece, from_pos, to_pos) -> int:  # noqa: ARG001 (from_pos unused — source is implicit in piece identity)
    """Encode a move as a flat policy index."""
    dest_row, dest_col = hex_to_grid(to_pos)
    channel = _piece_channel(piece)
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


def policy_to_moves(policy: np.ndarray, game: Game) -> list[tuple]:
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
