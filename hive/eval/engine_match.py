"""Engine-vs-engine match runner over UHP protocol.

Plays games between two UHP-compatible engines (e.g., our model vs Mzinga)
to evaluate strength. Supports both subprocess engines (any UHP engine) and
an in-process engine backed by our model + Rust MCTS.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

import numpy as np


import re

# Pattern: piece name like wQ or bQ without a trailing number
_QUEEN_NO_NUM = re.compile(r'([wb]Q)(?![0-9])')
# Pattern: piece name like wQ1 or bQ1 with the number 1
_QUEEN_WITH_NUM = re.compile(r'([wb]Q)1')


def _normalize_uhp(move_str: str) -> str:
    """Normalize UHP move string: add '1' to queen names missing a number.

    Our engine uses wQ1/bQ1, Mzinga uses wQ/bQ.
    """
    return _QUEEN_NO_NUM.sub(r'\g<1>1', move_str)


def _strip_queen_num(move_str: str) -> str:
    """Strip the '1' from queen names for Mzinga compatibility."""
    return _QUEEN_WITH_NUM.sub(r'\g<1>', move_str)


@dataclass
class EngineConfig:
    """Configuration for a UHP subprocess engine."""
    path: str
    args: list[str] = field(default_factory=list)
    bestmove_args: str = "time 00:00:05"
    name: str = ""
    options: dict[str, str] = field(default_factory=dict)


@dataclass
class GameResult:
    """Result of a single game."""
    white_engine: str
    black_engine: str
    result: str  # "white", "black", "draw"
    moves: int
    move_list: list[str] = field(default_factory=list)
    game_string: str = ""


class Engine(Protocol):
    """Interface for a game-playing engine."""
    name: str
    def start(self) -> None: ...
    def send(self, command: str) -> list[str]: ...
    def stop(self) -> None: ...


class UHPProcess:
    """Manages a UHP engine subprocess."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.name = config.name

    def start(self):
        cmd = [self.config.path] + self.config.args
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        info = self._read_response()
        if not self.name:
            for line in info:
                if line.startswith("id "):
                    self.name = line[3:]
                    break
        # Set any engine options
        for key, value in self.config.options.items():
            self.send(f"options set {key} {value}")

    def send(self, command: str) -> list[str]:
        assert self.process is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        return self._read_response()

    def _read_response(self) -> list[str]:
        lines = []
        while True:
            line = self.process.stdout.readline().strip()
            if line == "ok":
                break
            if line:
                lines.append(line)
        return lines

    def stop(self):
        if self.process:
            try:
                self.process.stdin.write("exit\n")
                self.process.stdin.flush()
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()
            self.process = None


class ModelEngine:
    """In-process engine using our model + Rust MCTS.

    Avoids subprocess overhead and reuses the already-loaded model.
    """

    def __init__(self, model, device: str = "cpu", simulations: int = 800,
                 name: str = "HiveZero"):
        self.model = model
        self.device = device
        self.simulations = simulations
        self.name = name
        self._game = None

    def start(self):
        pass

    def stop(self):
        self._game = None

    def send(self, command: str) -> list[str]:
        parts = command.strip().split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "newgame":
            return self._newgame()
        elif cmd == "play":
            return self._play(args.strip())
        elif cmd == "pass":
            return self._do_pass()
        elif cmd == "bestmove":
            return self._bestmove()
        return []

    def _newgame(self) -> list[str]:
        from hive_engine import RustGame
        self._game = RustGame()
        return [self._game.game_string]

    def _play(self, move_str: str) -> list[str]:
        if move_str.lower() == "pass":
            return self._do_pass()

        game = self._game

        # Parse the UHP move to get piece name and target hex
        piece_name, target_hex = self._parse_uhp_to_hex(move_str)

        # Find matching valid move
        for piece_str, from_pos, to_pos in game.valid_moves():
            # Match piece (handle queen: wQ == wQ1)
            if self._piece_matches(piece_str, piece_name) and to_pos == target_hex:
                game.play_move(piece_str, from_pos, to_pos)
                return [game.game_string]

        raise ValueError(f"Could not match move: {move_str}")

    @staticmethod
    def _piece_matches(internal: str, uhp: str) -> bool:
        """Check if internal piece name matches UHP name (handles wQ vs wQ1)."""
        if internal == uhp:
            return True
        # wQ1 matches wQ, bQ1 matches bQ
        if len(uhp) == 2 and uhp[1] == 'Q' and internal == uhp + '1':
            return True
        return False

    def _parse_uhp_to_hex(self, move_str: str) -> tuple[str, tuple[int, int]]:
        """Parse a UHP move string into (piece_name, target_hex).

        Returns the piece name and the absolute hex coordinate of the target.
        """
        parts = move_str.split()
        piece_name = parts[0]

        if len(parts) == 1:
            # First move: place at origin
            return (piece_name, (0, 0))

        pos_str = parts[1]
        game = self._game

        # Check for stacking: destination is just a piece name
        # e.g., "wB1 wS1" means beetle climbs on wS1
        is_pure_piece = (len(pos_str) >= 2
                         and pos_str[0] in ('w', 'b')
                         and pos_str[1].isupper()
                         and pos_str[-1] not in ('-', '/', '\\')
                         and pos_str[0] not in ('-', '/', '\\'))
        if is_pure_piece and len(parts) == 2:
            # Find position of reference piece
            ref_name = pos_str
            for p_str, _, p_to in game.valid_moves():
                # The reference piece should be on the board; find its position
                # by checking all top pieces
                pass
            # Search all top pieces for the reference
            for (q, r), top_str in game.all_top_pieces():
                if self._piece_matches(top_str, ref_name):
                    return (piece_name, (q, r))

        # Direction-based reference
        # UHP: suffix '-' = E, '/' = NE, '\' = SE
        #       prefix '-' = W, '/' = SW, '\' = NW
        DIRECTIONS = [
            (1, 0),   # 0: E
            (1, -1),  # 1: NE
            (0, -1),  # 2: NW
            (-1, 0),  # 3: W
            (-1, 1),  # 4: SW
            (0, 1),   # 5: SE
        ]

        prefix = ""
        suffix = ""
        ref_str = pos_str

        if pos_str[0] in ('-', '/', '\\'):
            prefix = pos_str[0]
            ref_str = pos_str[1:]
        elif pos_str[-1] in ('-', '/', '\\'):
            suffix = pos_str[-1]
            ref_str = pos_str[:-1]

        dir_map = {
            ("", "-"): 0,   # E
            ("", "/"): 1,   # NE
            ("\\", ""): 2,  # NW
            ("-", ""): 3,   # W
            ("/", ""): 4,   # SW
            ("", "\\"): 5,  # SE
        }

        key = (prefix, suffix)
        if key not in dir_map:
            raise ValueError(f"Invalid UHP position notation: {pos_str}")

        dir_idx = dir_map[key]
        dq, dr = DIRECTIONS[dir_idx]

        # Find reference piece position
        for (q, r), top_str in game.all_top_pieces():
            if self._piece_matches(top_str, ref_str):
                return (piece_name, (q + dq, r + dr))

        raise ValueError(f"Reference piece {ref_str} not found on board")

    def _do_pass(self) -> list[str]:
        self._game.play_pass()
        return [self._game.game_string]

    def _bestmove(self) -> list[str]:
        import torch
        from hive_engine import RustMCTS
        from ..encoding.move_encoder import POLICY_SIZE

        game = self._game
        valid = game.valid_moves()
        if not valid:
            return ["pass"]

        mcts = RustMCTS()
        model = self.model
        device = self.device

        def eval_fn(board_batch, reserve_batch):
            board_4d = np.asarray(board_batch)
            reserves = np.asarray(reserve_batch)
            bt = torch.tensor(board_4d).to(device)
            rv = torch.tensor(reserves).to(device)
            with torch.no_grad():
                policy_logits, values = model(bt, rv)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()
            vals = values.cpu().numpy().flatten()
            return policy.astype(np.float32), vals.astype(np.float32)

        result = mcts.search(game, eval_fn, self.simulations)
        if result is None:
            return ["pass"]

        piece_str, from_pos, to_pos = result
        uhp_str = game.format_move_uhp(piece_str, from_pos, to_pos)
        return [uhp_str]


def play_game(
    white: Engine,
    black: Engine,
    max_moves: int = 200,
    verbose: bool = False,
) -> GameResult:
    """Play a single game between two engines."""
    white.send("newgame Base")
    black.send("newgame Base")

    engines = [white, black]
    move_list: list[str] = []
    move_count = 0
    game_string = ""

    while move_count < max_moves:
        active = engines[move_count % 2]
        passive = engines[1 - move_count % 2]

        # Get best move from active engine
        bestmove_cmd = "bestmove"
        if isinstance(active, UHPProcess):
            bestmove_cmd = f"bestmove {active.config.bestmove_args}"

        response = active.send(bestmove_cmd)
        if not response:
            break

        move = response[0]

        # Handle error responses
        if move.startswith("err"):
            if verbose:
                print(f"  Engine error: {move}")
            break

        if verbose:
            color = "W" if move_count % 2 == 0 else "B"
            print(f"  {color} {move_count + 1}: {move}")

        move_list.append(move)

        # Play the move on both engines
        if move.lower() == "pass":
            active_gs = active.send("pass")
            passive_gs = passive.send("pass")
        else:
            active_gs = active.send(f"play {move}")
            passive_gs = passive.send(f"play {move}")

        # Check for errors
        for gs in [active_gs, passive_gs]:
            if gs and gs[0].startswith(("err", "invalidmove")):
                if verbose:
                    print(f"  ERROR: {gs[0]}")
                return GameResult(white.name, black.name, "draw",
                                  move_count + 1, move_list, "")

        if active_gs:
            game_string = active_gs[0]
            parts = game_string.split(";")
            if len(parts) >= 2:
                state = parts[1]
                if state == "WhiteWins":
                    return GameResult(white.name, black.name, "white",
                                      move_count + 1, move_list, game_string)
                elif state == "BlackWins":
                    return GameResult(white.name, black.name, "black",
                                      move_count + 1, move_list, game_string)
                elif state == "Draw":
                    return GameResult(white.name, black.name, "draw",
                                      move_count + 1, move_list, game_string)

        move_count += 1

    return GameResult(white.name, black.name, "draw",
                      move_count, move_list, game_string)


def run_match(
    engine1: Engine,
    engine2: Engine,
    num_games: int = 10,
    max_moves: int = 200,
    verbose: bool = True,
) -> dict:
    """Run a match of multiple games, alternating colors."""
    engine1.start()
    engine2.start()

    if verbose:
        print(f"Match: {engine1.name} vs {engine2.name}")
        print(f"Games: {num_games}, Max moves: {max_moves}")
        print()

    results = []
    e1_wins = 0
    e2_wins = 0
    draws = 0

    try:
        for i in range(num_games):
            if i % 2 == 0:
                white, black = engine1, engine2
            else:
                white, black = engine2, engine1

            if verbose:
                print(f"Game {i + 1}/{num_games}: "
                      f"W={white.name} vs B={black.name}")

            t0 = time.time()
            result = play_game(white, black, max_moves, verbose)
            elapsed = time.time() - t0
            results.append(result)

            if result.result == "white":
                winner = white.name
                if white is engine1:
                    e1_wins += 1
                else:
                    e2_wins += 1
            elif result.result == "black":
                winner = black.name
                if black is engine1:
                    e1_wins += 1
                else:
                    e2_wins += 1
            else:
                winner = "draw"
                draws += 1

            if verbose:
                print(f"  Result: {winner} in {result.moves} moves "
                      f"({elapsed:.1f}s)\n")

    finally:
        engine1.stop()
        engine2.stop()

    total = e1_wins + e2_wins + draws
    summary = {
        "engine1": engine1.name,
        "engine2": engine2.name,
        "engine1_wins": e1_wins,
        "engine2_wins": e2_wins,
        "draws": draws,
        "total": total,
        "engine1_score": (e1_wins + 0.5 * draws) / total if total else 0,
        "results": results,
    }

    if verbose:
        print("=" * 40)
        print(f"Results: {engine1.name} {e1_wins}W / {draws}D / {e2_wins}L")
        print(f"Score: {summary['engine1_score']:.1%}")

    return summary
