"""Board representation for Hive with beetle stacking."""

from __future__ import annotations
from collections import defaultdict
from typing import Optional

from .hex import Hex, DIRECTIONS
from .pieces import Piece, PieceColor, PieceType


class Board:
    """Hive board storing piece positions with stacking support.

    Each hex position has a stack (list) of pieces. Index 0 is the bottom.
    Max stack height is 7 (beetle stacking).
    """

    def __init__(self):
        # hex -> list of pieces (bottom to top)
        self._stacks: dict[Hex, list[Piece]] = defaultdict(list)
        # piece -> hex position (for quick lookup)
        self._piece_positions: dict[Piece, Hex] = {}

    def copy(self) -> Board:
        b = Board()
        for pos, stack in self._stacks.items():
            b._stacks[pos] = list(stack)
        b._piece_positions = dict(self._piece_positions)
        return b

    @property
    def occupied(self) -> set[Hex]:
        """All hex positions that have at least one piece."""
        return set(self._stacks.keys())

    def top_piece(self, pos: Hex) -> Optional[Piece]:
        """Return the top piece at a position, or None."""
        stack = self._stacks.get(pos)
        if stack:
            return stack[-1]
        return None

    def stack_at(self, pos: Hex) -> list[Piece]:
        """Return the full stack at a position (bottom to top)."""
        return list(self._stacks.get(pos, []))

    def stack_height(self, pos: Hex) -> int:
        return len(self._stacks.get(pos, []))

    def piece_position(self, piece: Piece) -> Optional[Hex]:
        """Return the position of a piece, or None if not on board."""
        return self._piece_positions.get(piece)

    def pieces_on_board(self, color: Optional[PieceColor] = None) -> list[Piece]:
        """All pieces currently on the board, optionally filtered by color."""
        if color is None:
            return list(self._piece_positions.keys())
        return [p for p in self._piece_positions if p.color == color]

    def place_piece(self, piece: Piece, pos: Hex):
        """Place a piece at a position (on top of any existing stack)."""
        if piece in self._piece_positions:
            raise ValueError(f"{piece} is already on the board")
        self._stacks[pos].append(piece)
        self._piece_positions[piece] = pos

    def remove_piece(self, piece: Piece) -> Hex:
        """Remove a piece from the board. Returns its position.

        The piece must be the top of its stack.
        """
        pos = self._piece_positions.get(piece)
        if pos is None:
            raise ValueError(f"{piece} is not on the board")
        stack = self._stacks[pos]
        if stack[-1] != piece:
            raise ValueError(f"{piece} is not on top of its stack at {pos}")
        stack.pop()
        if not stack:
            del self._stacks[pos]
        del self._piece_positions[piece]
        return pos

    def move_piece(self, piece: Piece, dest: Hex):
        """Move a piece from its current position to dest (handles stacking)."""
        self.remove_piece(piece)
        self.place_piece(piece, dest)

    def neighbors_of(self, pos: Hex) -> list[Hex]:
        """Return occupied neighbor positions."""
        return [n for n in pos.neighbors() if n in self._stacks]

    def is_connected(self, exclude: Optional[Piece] = None) -> bool:
        """Check if all pieces form a single connected group (One Hive Rule).

        If exclude is given, check connectivity as if that piece were removed.
        """
        positions = set(self._stacks.keys())
        if exclude is not None:
            epos = self._piece_positions.get(exclude)
            if epos is not None and self._stacks.get(epos) and self._stacks[epos][-1] == exclude:
                # If stack has only this piece, remove position entirely
                if len(self._stacks[epos]) == 1:
                    positions.discard(epos)
                # If stack has more pieces below, position stays

        if len(positions) <= 1:
            return True

        # BFS from any position
        start = next(iter(positions))
        visited = {start}
        queue = [start]
        while queue:
            current = queue.pop()
            for n in current.neighbors():
                if n in positions and n not in visited:
                    visited.add(n)
                    queue.append(n)

        return len(visited) == len(positions)

    def can_slide(self, from_pos: Hex, to_pos: Hex) -> bool:
        """Check if a ground-level piece can slide from from_pos to to_pos.

        The two positions must be adjacent. A slide is blocked (gate) if both
        common neighbors of from_pos and to_pos are occupied.
        """
        # Find the two common neighbors
        from_neighbors = set(from_pos.neighbors())
        to_neighbors = set(to_pos.neighbors())
        common = from_neighbors & to_neighbors

        # Both common neighbors must NOT be occupied for the gate to block
        occupied_common = [c for c in common if c in self._stacks and c != from_pos]
        return len(occupied_common) < 2

    def empty_neighbors(self, pos: Hex) -> list[Hex]:
        """Return unoccupied neighbor positions."""
        return [n for n in pos.neighbors() if n not in self._stacks]

    def all_top_pieces(self) -> list[tuple[Hex, Piece]]:
        """Return (position, top_piece) for all occupied positions."""
        return [(pos, stack[-1]) for pos, stack in self._stacks.items()]
