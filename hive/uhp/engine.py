"""UHP (Universal Hive Protocol) stdin/stdout engine."""

from __future__ import annotations
import sys
from typing import Optional

from engine_zero import HiveGame


ENGINE_NAME = "HiveEngine"
ENGINE_VERSION = "0.1.0"


class UHPEngine:
    """UHP-compliant engine communicating over stdin/stdout."""

    def __init__(self, model=None, device: str = "cpu", simulations: int = 800,
                 onnx_path: str | None = None, grid_size: int | None = None):
        self.game: Optional[HiveGame] = None
        self._running = True
        self.model = model
        self.device = device
        self.simulations = simulations
        self.grid_size = grid_size if grid_size is not None else getattr(model, "grid_size", None)
        self.ort_session = None
        if onnx_path is not None:
            from engine_zero import HiveOrtSession
            self.ort_session = HiveOrtSession(onnx_path)

    def run(self):
        """Main loop: read commands from stdin, write responses to stdout."""
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
            "info":      lambda: self._cmd_info(),
            "newgame":   lambda: self._cmd_newgame(args),
            "play":      lambda: self._cmd_play(args),
            "pass":      lambda: self._cmd_pass(),
            "validmoves":lambda: self._cmd_validmoves(),
            "bestmove":  lambda: self._cmd_bestmove(args),
            "undo":      lambda: self._cmd_undo(args),
            "options":   lambda: self._cmd_options(args),
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

        if not args or args.startswith("Base"):
            if "+" in args:
                self._respond("err Expansions not supported")
                return
            self.game = self._create_game()
            self._respond(self.game.game_string)
            return

        # Load from a GameString
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

        if self.game.play_move_uhp(move_str):
            self._respond(self.game.game_string)
        else:
            self._respond(f"invalidmove {move_str} is not valid")

    def _cmd_pass(self):
        if self.game is None:
            self._respond("err No game in progress")
            return
        if self.game.is_game_over:
            self._respond("invalidmove Game is over")
            return
        if self.game.valid_moves():
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

        move_strings = [
            self.game.format_move_uhp(piece_str, from_pos, to_pos)
            for piece_str, from_pos, to_pos in valid
        ]
        self._respond(";".join(move_strings))

    def _cmd_bestmove(self, args: str):
        if self.game is None:
            self._respond("err No game in progress")
            return
        if self.game.is_game_over:
            self._respond("err Game is over")
            return
        if not self.game.valid_moves():
            self._respond("pass")
            return
        if self.ort_session is None and self.model is None:
            self._respond("err No model loaded")
            return

        if self.ort_session is not None:
            move_str = self.game.best_move_ort(self.ort_session, simulations=self.simulations)
        else:
            import torch
            import numpy as np

            model = self.model
            device = self.device

            def eval_fn(board_batch, reserve_batch):
                b = torch.tensor(np.asarray(board_batch)).to(device)
                r = torch.tensor(np.asarray(reserve_batch)).to(device)
                with torch.no_grad():
                    policy_logits, wdl, _ = model(b, r)
                policy = torch.softmax(policy_logits, dim=1).cpu().numpy()
                vals = (wdl[:, 0] - wdl[:, 2]).cpu().numpy().flatten()
                return policy.astype(np.float32), vals.astype(np.float32)

            move_str = self.game.best_move(eval_fn, simulations=self.simulations)

        self._respond(move_str)

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

        for _ in range(count):
            self.game.undo()
        self._respond(self.game.game_string)

    def _cmd_options(self, args: str):
        parts = args.strip().split(None, 2)
        if not parts:
            self._respond()
            return
        sub = parts[0].lower()
        if sub in ("get", "set"):
            self._respond("err Unknown option")
        else:
            self._respond("err Invalid options command")

    def _parse_game_string(self, game_string: str) -> HiveGame:
        """Parse a UHP GameString and replay to reconstruct the game."""
        parts = game_string.split(";")
        if len(parts) < 3:
            raise ValueError("GameString must have at least 3 parts")
        if parts[0] != "Base":
            raise ValueError(f"Unsupported game type: {parts[0]}")

        game = self._create_game()
        for move_str in parts[3:]:
            move_str = move_str.strip()
            if not move_str:
                continue
            if not game.play_move_uhp(move_str):
                raise ValueError(f"Invalid move in game string: {move_str}")
        return game

    def _create_game(self) -> HiveGame:
        if self.grid_size is None:
            return HiveGame()
        return HiveGame(grid_size=self.grid_size)
