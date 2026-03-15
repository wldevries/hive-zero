"""UHP (Universal Hive Protocol) stdin/stdout engine."""

from __future__ import annotations
import sys
from typing import Optional

from ..core.hex import Hex, DIRECTIONS
from ..core.pieces import Piece, PieceColor, PieceType
from ..core.board import Board
from ..core.game import Game, GameState


ENGINE_NAME = "HiveEngine"
ENGINE_VERSION = "0.1.0"


class UHPEngine:
    """UHP-compliant engine communicating over stdin/stdout."""

    def __init__(self):
        self.game: Optional[Game] = None
        self._running = True

    def run(self):
        """Main loop: read commands from stdin, write responses to stdout."""
        # Output info on startup
        self._cmd_info()
        sys.stdout.flush()

        while self._running:
            try:
                line = input().strip()
            except EOFError:
                break
            if not line:
                continue

            self._dispatch(line)
            sys.stdout.flush()

    def _dispatch(self, line: str):
        parts = line.split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        handlers = {
            "info": lambda: self._cmd_info(),
            "newgame": lambda: self._cmd_newgame(args),
            "play": lambda: self._cmd_play(args),
            "pass": lambda: self._cmd_pass(),
            "validmoves": lambda: self._cmd_validmoves(),
            "bestmove": lambda: self._cmd_bestmove(args),
            "undo": lambda: self._cmd_undo(args),
            "options": lambda: self._cmd_options(args),
        }

        handler = handlers.get(cmd)
        if handler:
            try:
                handler()
            except Exception as e:
                self._respond(f"err {e}")
        else:
            self._respond(f"err Unknown command: {cmd}")

    def _respond(self, *lines: str):
        for line in lines:
            print(line)
        print("ok")

    def _cmd_info(self):
        self._respond(f"id {ENGINE_NAME} v{ENGINE_VERSION}")

    def _cmd_newgame(self, args: str):
        args = args.strip()

        if not args or args == "Base":
            self.game = Game()
            self._respond(self.game.game_string)
            return

        # Check if it's a game type string
        if args.startswith("Base"):
            if "+" in args:
                self._respond("err Expansions not supported")
                return
            self.game = Game()
            self._respond(self.game.game_string)
            return

        # Try to load as a GameString
        try:
            self.game = self._parse_game_string(args)
            self._respond(self.game.game_string)
        except Exception as e:
            self._respond(f"err Invalid game string: {e}")

    def _cmd_play(self, args: str):
        if self.game is None:
            self._respond("err No game in progress")
            return

        if self.game.is_game_over:
            self._respond("invalidmove Game is over")
            return

        move_str = args.strip()
        if move_str.lower() == "pass":
            self._cmd_pass()
            return

        try:
            piece, from_pos, to_pos = self._parse_move(move_str)
        except ValueError as e:
            self._respond(f"invalidmove {e}")
            return

        # Validate the move
        valid = self.game.valid_moves()
        if not self._move_in_list(piece, from_pos, to_pos, valid):
            self._respond(f"invalidmove {move_str} is not valid")
            return

        self.game.play_move(piece, from_pos, to_pos)
        self._respond(self.game.game_string)

    def _cmd_pass(self):
        if self.game is None:
            self._respond("err No game in progress")
            return

        if self.game.is_game_over:
            self._respond("invalidmove Game is over")
            return

        # Pass is only valid when no moves are available
        valid = self.game.valid_moves()
        if valid:
            self._respond("invalidmove Pass is not valid when moves are available")
            return

        self.game.play_pass()
        self._respond(self.game.game_string)

    def _cmd_validmoves(self):
        if self.game is None:
            self._respond("err No game in progress")
            return

        if self.game.is_game_over:
            self._respond("")
            return

        valid = self.game.valid_moves()
        if not valid:
            self._respond("pass")
            return

        move_strings = []
        for piece, from_pos, to_pos in valid:
            ms = self._format_move(piece, from_pos, to_pos)
            move_strings.append(ms)

        self._respond(";".join(move_strings))

    def _cmd_bestmove(self, args: str):
        if self.game is None:
            self._respond("err No game in progress")
            return

        if self.game.is_game_over:
            self._respond("err Game is over")
            return

        # Parse time or depth constraint
        parts = args.strip().split()
        if len(parts) >= 2:
            constraint_type = parts[0].lower()
            constraint_value = parts[1]
        else:
            constraint_type = "depth"
            constraint_value = "1"

        # For now, just pick the first valid move (placeholder for MCTS)
        valid = self.game.valid_moves()
        if not valid:
            self._respond("pass")
            return

        # Determine simulations count
        if constraint_type == "depth":
            max_sims = int(constraint_value) * 100
        else:
            max_sims = 800

        # Try Rust MCTS first, then Python MCTS, then fallback
        try:
            from hive_engine import RustGame, RustMCTS
            # Convert current game state to RustGame by replaying moves
            rust_game = RustGame()
            for hist_piece, hist_from, hist_to in self.game.move_history:
                if hist_piece is None:
                    rust_game.play_pass()
                else:
                    p_str = str(hist_piece)
                    f = (hist_from.q, hist_from.r) if hist_from is not None else None
                    t = (hist_to.q, hist_to.r)
                    rust_game.play_move(p_str, f, t)

            mcts = RustMCTS()
            # Uniform policy eval (no model loaded in UHP engine)
            import numpy as np
            from ..encoding.move_encoder import POLICY_SIZE
            def uniform_eval(board_batch, reserve_batch):
                n = board_batch.shape[0]
                policy = np.ones((n, POLICY_SIZE), dtype=np.float32) / POLICY_SIZE
                value = np.zeros(n, dtype=np.float32)
                return policy, value

            result = mcts.search(rust_game, uniform_eval, max_sims)
            if result is None:
                self._respond("pass")
            else:
                piece_str, from_pos, to_pos = result
                piece = Piece.from_str(piece_str)
                fp = Hex(*from_pos) if from_pos is not None else None
                tp = Hex(*to_pos)
                self._respond(self._format_move(piece, fp, tp))
            return
        except ImportError:
            pass

        # Fallback: first valid move
        piece, from_pos, to_pos = valid[0]
        self._respond(self._format_move(piece, from_pos, to_pos))

    def _cmd_undo(self, args: str):
        if self.game is None:
            self._respond("err No game in progress")
            return

        count = 1
        if args.strip():
            try:
                count = int(args.strip())
            except ValueError:
                self._respond("err Invalid undo count")
                return

        try:
            self.game.undo(count)
            self._respond(self.game.game_string)
        except ValueError as e:
            self._respond(f"err {e}")

    def _cmd_options(self, args: str):
        # Minimal options support
        parts = args.strip().split(None, 2)
        if not parts:
            # List all options
            self._respond()
            return
        sub = parts[0].lower()
        if sub == "get":
            self._respond("err Unknown option")
        elif sub == "set":
            self._respond("err Unknown option")
        else:
            self._respond("err Invalid options command")

    # ---- Move parsing ----

    def _parse_move(self, move_str: str) -> tuple[Piece, Optional[Hex], Hex]:
        """Parse UHP MoveString into (piece, from_pos_or_None, to_pos)."""
        parts = move_str.split()

        if len(parts) == 1:
            # First move of the game: just piece name, placed at origin
            piece = Piece.from_str(parts[0])
            from_pos = self.game.board.piece_position(piece)
            if from_pos is not None:
                raise ValueError(f"{piece} is already on the board")
            return (piece, None, Hex(0, 0))

        if len(parts) == 2:
            piece = Piece.from_str(parts[0])
            pos_str = parts[1]

            # Check if it's a stacking move (destination is another piece name)
            # e.g., "wB1 wS1" means beetle climbs on top of wS1
            # Must be purely a piece name with no direction prefix/suffix
            is_pure_piece = (pos_str[0] in ('w', 'b') and len(pos_str) >= 2
                             and pos_str[1].isupper()
                             and pos_str[-1] not in ('-', '/', '\\')
                             and pos_str[0] not in ('-', '/', '\\'))
            if is_pure_piece:
                target_piece = Piece.from_str(pos_str)
                target_pos = self.game.board.piece_position(target_piece)
                if target_pos is None:
                    raise ValueError(f"Reference piece {target_piece} not found")
                from_pos = self.game.board.piece_position(piece)
                return (piece, from_pos, target_pos)

            # Relative position notation
            to_pos = self._parse_position(pos_str)
            from_pos = self.game.board.piece_position(piece)
            return (piece, from_pos, to_pos)

        raise ValueError(f"Cannot parse move: {move_str}")

    def _parse_position(self, pos_str: str) -> Hex:
        """Parse UHP position string like 'wS1/', '-wS1', 'wA1-', etc."""
        # Determine direction prefix/suffix and reference piece
        prefix = ""
        suffix = ""
        ref_str = pos_str

        # Check for direction prefix: -, /, or backslash
        if pos_str[0] in ('-', '/', '\\'):
            prefix = pos_str[0]
            ref_str = pos_str[1:]
        # Check for direction suffix
        elif pos_str[-1] in ('-', '/', '\\'):
            suffix = pos_str[-1]
            ref_str = pos_str[:-1]

        # Parse reference piece
        ref_piece = Piece.from_str(ref_str)
        ref_pos = self.game.board.piece_position(ref_piece)
        if ref_pos is None:
            raise ValueError(f"Reference piece {ref_piece} not found on board")

        # UHP direction notation (flat-top hexagons):
        #   suffix '-' = E,  suffix '/' = NE,  suffix '\' = SE
        #   prefix '-' = W,  prefix '/' = SW,  prefix '\' = NW
        dir_map = {
            ("", "-"): 0,   # E
            ("", "/"): 1,   # NE
            ("\\", ""): 2,  # NW (prefix \)
            ("-", ""): 3,   # W
            ("/", ""): 4,   # SW
            ("", "\\"): 5,  # SE (suffix \)
        }

        key = (prefix, suffix)
        if key not in dir_map:
            raise ValueError(f"Invalid position notation: {pos_str}")

        dir_idx = dir_map[key]
        return ref_pos + DIRECTIONS[dir_idx]

    def _format_move(self, piece: Piece, from_pos: Optional[Hex], to_pos: Hex) -> str:
        """Format a move as UHP MoveString."""
        piece_str = str(piece)

        # First overall move
        if self.game.move_count == 0 and from_pos is None:
            return piece_str

        # Second overall move
        if self.game.move_count == 1 and from_pos is None:
            return self._format_position(piece_str, to_pos)

        # Stacking (beetle on top of another piece)
        if from_pos is not None:
            top_at_dest = self.game.board.top_piece(to_pos)
            if top_at_dest is not None:
                return f"{piece_str} {top_at_dest}"

        # Normal move or placement
        return self._format_position(piece_str, to_pos)

    def _format_position(self, piece_str: str, pos: Hex) -> str:
        """Format destination as UHP relative position."""
        dir_to_uhp = {
            0: ("", "-"),    # E  -> suffix '-'
            1: ("", "/"),    # NE -> suffix '/'
            2: ("\\", ""),   # NW -> prefix '\'
            3: ("-", ""),    # W  -> prefix '-'
            4: ("/", ""),    # SW -> prefix '/'
            5: ("", "\\"),   # SE -> suffix '\'
        }

        # Find an adjacent occupied hex to use as reference
        for i, d in enumerate(DIRECTIONS):
            neighbor = pos + d
            top = self.game.board.top_piece(neighbor)
            if top is not None and str(top) != piece_str:
                # pos is in direction opposite(i) from neighbor
                opp = (i + 3) % 6
                prefix, suffix = dir_to_uhp[opp]
                ref_str = str(top)
                return f"{piece_str} {prefix}{ref_str}{suffix}"

        # If we can only find ourselves as reference (shouldn't happen normally)
        for i, d in enumerate(DIRECTIONS):
            neighbor = pos + d
            top = self.game.board.top_piece(neighbor)
            if top is not None:
                opp = (i + 3) % 6
                prefix, suffix = dir_to_uhp[opp]
                ref_str = str(top)
                return f"{piece_str} {prefix}{ref_str}{suffix}"

        return piece_str

    def _move_in_list(self, piece, from_pos, to_pos, valid_moves) -> bool:
        """Check if a move matches any in the valid moves list."""
        for vp, vf, vt in valid_moves:
            if vp == piece and vt == to_pos:
                # from_pos match: both None (placement) or both same position
                if vf == from_pos:
                    return True
                # Also accept if from_pos doesn't match but piece and dest do
                # (from_pos may differ in how we parsed vs generated)
                if from_pos is not None and vf is not None:
                    return True
        return False

    def _parse_game_string(self, game_string: str) -> Game:
        """Parse a UHP GameString and replay to reconstruct the game."""
        parts = game_string.split(";")
        if len(parts) < 3:
            raise ValueError("GameString must have at least 3 parts")

        game_type = parts[0]
        if game_type != "Base":
            raise ValueError(f"Unsupported game type: {game_type}")

        # parts[1] is GameStateString (we'll derive it from replay)
        # parts[2] is TurnString (we'll derive it from replay)
        # parts[3:] are moves

        game = Game()
        for move_str in parts[3:]:
            move_str = move_str.strip()
            if not move_str:
                continue
            if move_str.lower() == "pass":
                game.play_pass()
            else:
                piece, from_pos, to_pos = self._parse_move_in_context(game, move_str)
                game.play_move(piece, from_pos, to_pos)

        return game

    def _parse_move_in_context(self, game: Game, move_str: str) -> tuple[Piece, Optional[Hex], Hex]:
        """Parse a move string in the context of a specific game state."""
        saved_game = self.game
        self.game = game
        try:
            return self._parse_move(move_str)
        finally:
            self.game = saved_game
