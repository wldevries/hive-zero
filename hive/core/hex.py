"""Axial hex coordinate system for Hive board."""

from __future__ import annotations
from typing import NamedTuple


class Hex(NamedTuple):
    """Axial hex coordinate (q, r). Flat-top orientation."""
    q: int
    r: int

    def __add__(self, other: Hex) -> Hex:
        return Hex(self.q + other.q, self.r + other.r)

    def __sub__(self, other: Hex) -> Hex:
        return Hex(self.q - other.q, self.r - other.r)

    def neighbors(self) -> list[Hex]:
        """Return all 6 adjacent hex positions."""
        return [self + d for d in DIRECTIONS]

    def distance(self, other: Hex) -> int:
        """Hex Manhattan distance."""
        dq = self.q - other.q
        dr = self.r - other.r
        return (abs(dq) + abs(dr) + abs(dq + dr)) // 2


# Six directions in axial coordinates (flat-top hexagons).
# Order: E, NE, NW, W, SW, SE (clockwise from right)
DIRECTIONS = [
    Hex(1, 0),    # E  (right)          UHP: -
    Hex(0, -1),   # NE (top-right)      UHP: /
    Hex(-1, -1),  # NW (top-left)       UHP: \  (from target's perspective)
    Hex(-1, 0),   # W  (left)
    Hex(0, 1),    # SW (bottom-left)
    Hex(1, 1),    # SE (bottom-right)
]

# Maps direction index to its opposite
OPPOSITE_DIR = {0: 3, 1: 4, 2: 5, 3: 0, 4: 1, 5: 2}
