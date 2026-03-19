"""Game state management for Hive."""

from __future__ import annotations
from enum import Enum
from typing import Optional

from .hex import Hex
from .board import Board
from .pieces import Piece, PieceColor, PieceType, PIECE_COUNTS, player_pieces
from .rules import get_moves, get_placements


class GameState(Enum):
    NOT_STARTED = "NotStarted"
    IN_PROGRESS = "InProgress"
    DRAW = "Draw"
    WHITE_WINS = "WhiteWins"
    BLACK_WINS = "BlackWins"


class Game:
    """Full Hive game state with move history and undo support."""

    def __init__(self):
        self.board = Board()
        self.state = GameState.NOT_STARTED
        self.turn_color = PieceColor.WHITE
        self.turn_number = 1  # increments after both players move
        self.move_count = 0  # total half-moves played
        self.move_history: list[tuple[Optional[Piece], Optional[Hex], Optional[Hex]]] = []
        # (piece, from_pos_or_None_for_placement, to_pos)

        # Track which pieces each player has in reserve
        self._white_reserve = set(player_pieces(PieceColor.WHITE))
        self._black_reserve = set(player_pieces(PieceColor.BLACK))

        # History snapshots for undo (store reserve sets)
        self._history_reserves: list[tuple[set[Piece], set[Piece]]] = []

    def copy(self) -> Game:
        g = Game()
        g.board = self.board.copy()
        g.state = self.state
        g.turn_color = self.turn_color
        g.turn_number = self.turn_number
        g.move_count = self.move_count
        g.move_history = list(self.move_history)
        g._white_reserve = set(self._white_reserve)
        g._black_reserve = set(self._black_reserve)
        g._history_reserves = [(set(w), set(b)) for w, b in self._history_reserves]
        return g

    @property
    def current_reserve(self) -> set[Piece]:
        if self.turn_color == PieceColor.WHITE:
            return self._white_reserve
        return self._black_reserve

    def reserve(self, color: PieceColor) -> set[Piece]:
        if color == PieceColor.WHITE:
            return self._white_reserve
        return self._black_reserve

    @property
    def is_game_over(self) -> bool:
        return self.state in (GameState.DRAW, GameState.WHITE_WINS, GameState.BLACK_WINS)

    def _must_place_queen(self) -> bool:
        """Player must place queen by their 4th turn (move 7 or 8)."""
        color = self.turn_color
        reserve = self.reserve(color)
        queen = Piece(color, PieceType.QUEEN, 1)
        if queen not in reserve:
            return False  # already placed

        # Count how many moves this player has made
        player_moves = sum(1 for p, _, _ in self.move_history if p is not None and p.color == color)
        # Also count passes
        player_moves += sum(1 for p, _, _ in self.move_history if p is None)
        # Actually, let's count by half-moves for this color
        if color == PieceColor.WHITE:
            player_moves = (self.move_count + 1) // 2
        else:
            player_moves = self.move_count // 2

        # On their 4th turn (0-indexed: move index 3), queen MUST be placed
        return player_moves >= 3

    def _queen_placed(self, color: PieceColor) -> bool:
        queen = Piece(color, PieceType.QUEEN, 1)
        return queen not in self.reserve(color)

    def valid_moves(self) -> list[tuple[Optional[Piece], Optional[Hex], Hex]]:
        """Return all valid moves as (piece, from_hex_or_None, to_hex).

        from_hex is None for placement moves.
        If no moves are available, returns empty list (player must pass).
        """
        if self.is_game_over:
            return []

        color = self.turn_color
        reserve = self.reserve(color)
        moves = []

        must_queen = self._must_place_queen()

        # Placement moves
        placement_hexes = get_placements(color, self.board)
        if placement_hexes:
            # Tournament rule: cannot place queen on first move
            is_first_move = (color == PieceColor.WHITE and self.move_count == 0) or \
                            (color == PieceColor.BLACK and self.move_count == 1)

            # Determine which piece types can be placed.
            # Only include lowest-numbered piece per type (UHP ordering rule).
            seen_types = set()
            placeable = []
            for piece in sorted(reserve, key=lambda p: (str(p.piece_type.value), p.number)):
                if must_queen and piece.piece_type != PieceType.QUEEN:
                    continue
                if is_first_move and piece.piece_type == PieceType.QUEEN:
                    continue
                if piece.piece_type in seen_types:
                    continue
                seen_types.add(piece.piece_type)
                placeable.append(piece)

            for piece in sorted(placeable, key=str):
                for pos in sorted(placement_hexes):
                    moves.append((piece, None, pos))

        # Movement moves (only if queen is placed)
        if not must_queen and self._queen_placed(color):
            # Compute articulation points once for all pieces (O(V+E))
            articulation_pts = self.board.articulation_points()
            on_board = self.board.pieces_on_board(color)
            for piece in sorted(on_board, key=str):
                destinations = get_moves(piece, self.board, articulation_pts)
                for dest in sorted(destinations):
                    pos = self.board.piece_position(piece)
                    moves.append((piece, pos, dest))

        return moves

    def play_move(self, piece: Optional[Piece], from_pos: Optional[Hex], to_pos: Hex):
        """Execute a move. piece=None for pass (to_pos ignored)."""
        # Save state for undo
        self._history_reserves.append(
            (set(self._white_reserve), set(self._black_reserve))
        )

        if piece is not None:
            if from_pos is None:
                # Placement
                self.reserve(piece.color).discard(piece)
                self.board.place_piece(piece, to_pos)
            else:
                # Movement
                self.board.move_piece(piece, to_pos)

        self.move_history.append((piece, from_pos, to_pos))

        if self.state == GameState.NOT_STARTED:
            self.state = GameState.IN_PROGRESS

        # Advance turn
        self.move_count += 1
        if self.turn_color == PieceColor.BLACK:
            self.turn_number += 1
        self.turn_color = (
            PieceColor.BLACK if self.turn_color == PieceColor.WHITE else PieceColor.WHITE
        )

        # Check for game end
        self._check_game_end()

    def play_pass(self):
        """Execute a pass move."""
        self._history_reserves.append(
            (set(self._white_reserve), set(self._black_reserve))
        )
        self.move_history.append((None, None, None))

        self.move_count += 1
        if self.turn_color == PieceColor.BLACK:
            self.turn_number += 1
        self.turn_color = (
            PieceColor.BLACK if self.turn_color == PieceColor.WHITE else PieceColor.WHITE
        )

        self._check_game_end()

    def undo(self, count: int = 1):
        """Undo the last count moves."""
        for _ in range(count):
            if not self.move_history:
                raise ValueError("No moves to undo")

            piece, from_pos, to_pos = self.move_history.pop()
            wr, br = self._history_reserves.pop()
            self._white_reserve = wr
            self._black_reserve = br

            if piece is not None:
                if from_pos is None:
                    # Undo placement
                    self.board.remove_piece(piece)
                else:
                    # Undo movement
                    self.board.move_piece(piece, from_pos)

            self.move_count -= 1
            if self.turn_color == PieceColor.WHITE:
                self.turn_number -= 1
            self.turn_color = (
                PieceColor.BLACK if self.turn_color == PieceColor.WHITE else PieceColor.WHITE
            )

        # Re-evaluate game state
        if self.move_count == 0:
            self.state = GameState.NOT_STARTED
        else:
            self.state = GameState.IN_PROGRESS
            self._check_game_end()

    def _check_game_end(self):
        """Check if any queen is surrounded (game over)."""
        white_queen = Piece(PieceColor.WHITE, PieceType.QUEEN, 1)
        black_queen = Piece(PieceColor.BLACK, PieceType.QUEEN, 1)

        w_pos = self.board.piece_position(white_queen)
        b_pos = self.board.piece_position(black_queen)

        w_surrounded = False
        b_surrounded = False

        if w_pos is not None:
            w_surrounded = all(n in self.board.occupied for n in w_pos.neighbors())
        if b_pos is not None:
            b_surrounded = all(n in self.board.occupied for n in b_pos.neighbors())

        if w_surrounded and b_surrounded:
            self.state = GameState.DRAW
        elif w_surrounded:
            self.state = GameState.BLACK_WINS
        elif b_surrounded:
            self.state = GameState.WHITE_WINS

    # ---- UHP string support ----

    @property
    def turn_string(self) -> str:
        """e.g. 'White[1]', 'Black[3]'"""
        color_name = "White" if self.turn_color == PieceColor.WHITE else "Black"
        return f"{color_name}[{self.turn_number}]"

    @property
    def game_type_string(self) -> str:
        return "Base"

    @property
    def game_string(self) -> str:
        """Full UHP GameString."""
        parts = [self.game_type_string, self.state.value, self.turn_string]
        replay = Game()
        for piece, from_pos, to_pos in self.move_history:
            if piece is None:
                replay.play_pass()
                parts.append("pass")
            else:
                replay.play_move(piece, from_pos, to_pos)
                parts.append(replay._encode_move_string(piece, from_pos, to_pos))
        return ";".join(parts)

    def _move_to_uhp(self, piece: Piece, from_pos: Optional[Hex], to_pos: Hex) -> str:
        """Convert a move to UHP MoveString notation."""
        piece_str = str(piece)

        # First move of the game: just the piece name
        if self.board.stack_height(to_pos) == 1 and len(self.move_history) == 0:
            # This is called after the move is already done for history
            pass

        # If beetle stacking (destination has another piece), reference that piece directly
        # For the first overall move, just return piece string
        # We need to reconstruct what the board looked like before this move

        # This is complex - delegate to the UHP module for full conversion
        # For now, store in a simple format
        return self._encode_move_string(piece, from_pos, to_pos)

    def _encode_move_string(self, piece: Piece, from_pos: Optional[Hex], to_pos: Hex) -> str:
        """Encode move to UHP string. Called AFTER the move is applied."""
        piece_str = str(piece)

        # First move of the game
        if len(self.move_history) <= 1 and from_pos is None:
            return piece_str

        # Beetle/piece stacking on top of another piece
        stack = self.board.stack_at(to_pos)
        if len(stack) > 1 and from_pos is not None:
            # Piece under us
            under = stack[-2]
            return f"{piece_str} {under}"

        # Find a reference piece adjacent to to_pos
        return self._position_to_uhp(piece_str, to_pos)

    def _position_to_uhp(self, piece_str: str, pos: Hex) -> str:
        """Encode a destination position as UHP relative notation."""
        from .hex import DIRECTIONS

        # Direction symbols: index -> (prefix, suffix) for "target is in direction D from reference"
        # UHP uses: -, /, \ for right, top-right, top-left from reference piece
        # And the reverse for the other three directions
        # Mapping from direction index (direction from pos to neighbor):
        # If reference is to the E (dir 0) of pos, pos is to the W (dir 3) of ref
        # UHP: piece is placed at ref's left -> ref-piece notation would be: piece_str ref\\ or similar
        #
        # UHP position notation: [piece] [refPiece][direction]
        # direction is where the NEW piece is relative to refPiece:
        # - = right (E), / = top-right (NE), \\ = top-left (NW)
        # for the other three: -refPiece, /refPiece, \\refPiece (direction on left side)

        dir_to_uhp = {
            0: ("", "-"),    # E:  refPiece-
            1: ("", "/"),    # NE: refPiece/
            2: ("\\", ""),   # NW: \refPiece  (prefix \)
            3: ("-", ""),    # W:  -refPiece
            4: ("/", ""),    # SW: /refPiece
            5: ("", "\\"),   # SE: refPiece\  (suffix \)
        }

        for i, d in enumerate(DIRECTIONS):
            neighbor = pos + d
            top = self.board.top_piece(neighbor)
            if top is not None and str(top) != piece_str:
                prefix, suffix = dir_to_uhp[i]
                # The direction from pos to this neighbor is i
                # But UHP wants: where is pos relative to the reference piece?
                # pos is in the OPPOSITE direction from the reference
                # Actually: if reference is at pos+d (direction i from pos),
                # then pos is in direction opposite(i) from reference.
                # UHP notation: pos is at direction opposite(i) from reference.
                opp = (i + 3) % 6
                prefix, suffix = dir_to_uhp[opp]
                ref_str = str(top)
                return f"{piece_str} {prefix}{ref_str}{suffix}"

        # Fallback: shouldn't happen in a valid game
        return piece_str
