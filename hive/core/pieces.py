"""Piece definitions for Hive."""

from __future__ import annotations
from enum import Enum
from typing import NamedTuple


class PieceType(Enum):
    QUEEN = "Q"
    SPIDER = "S"
    BEETLE = "B"
    GRASSHOPPER = "G"
    ANT = "A"


class PieceColor(Enum):
    WHITE = "w"
    BLACK = "b"


class Piece(NamedTuple):
    color: PieceColor
    piece_type: PieceType
    number: int  # 1-indexed (Queen is always 1)

    def __str__(self) -> str:
        if self.piece_type == PieceType.QUEEN:
            return f"{self.color.value}{self.piece_type.value}"
        return f"{self.color.value}{self.piece_type.value}{self.number}"

    @staticmethod
    def from_str(s: str) -> Piece:
        """Parse piece string like 'wQ1', 'bS2', 'wA3'."""
        color = PieceColor(s[0])
        ptype = PieceType(s[1])
        num = int(s[2]) if len(s) > 2 else 1
        return Piece(color, ptype, num)


# How many of each piece type per player (base game)
PIECE_COUNTS = {
    PieceType.QUEEN: 1,
    PieceType.SPIDER: 2,
    PieceType.BEETLE: 2,
    PieceType.GRASSHOPPER: 3,
    PieceType.ANT: 3,
}

# All pieces for one player
def player_pieces(color: PieceColor) -> list[Piece]:
    pieces = []
    for pt, count in PIECE_COUNTS.items():
        for i in range(1, count + 1):
            pieces.append(Piece(color, pt, i))
    return pieces
