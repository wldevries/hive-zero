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
    Records board states and MCTS policies for training feedback.
    """

    def __init__(self, model, device: str = "cpu", simulations: int = 800,
                 name: str = "HiveZero", opening_temp_moves: int = 8):
        self.model = model
        self.device = device
        self.simulations = simulations
        self.name = name
        self.opening_temp_moves = opening_temp_moves
        self._game = None
        self._move_count = 0
        # Training sample collection: (board_tensor, reserve_vec, policy_vec, turn_color)
        self._history: list[tuple] = []

    def start(self):
        pass

    def stop(self):
        self._game = None

    def new_game_history(self):
        """Clear history for a new game."""
        self._history = []
        self._move_count = 0

    def get_samples(self, outcome: dict[str, float], weight: float = 1.0) -> list[tuple]:
        """Build training samples from recorded history with game outcome.

        Args:
            outcome: {"w": float, "b": float} value targets.
            weight: sample weight for the replay buffer.

        Returns:
            List of (board_tensor, reserve_vec, policy_vec, value, weight).
        """
        samples = []
        for bt, rv, pv, color in self._history:
            samples.append((bt, rv, pv, outcome[color], weight))
        return samples

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
        from hive_engine import RustBatchMCTS
        from ..encoding.move_encoder import POLICY_SIZE

        game = self._game
        valid = game.valid_moves()
        if not valid:
            return ["pass"]

        # Record board state before searching
        bt, rv = game.encode_board()
        bt = np.asarray(bt)
        rv = np.asarray(rv)
        turn_color = game.turn_color

        model = self.model
        device = self.device

        # Batch eval for initial policy + run_simulations callback
        def eval_fn(board_batch, reserve_batch):
            board_4d = np.asarray(board_batch)
            reserves = np.asarray(reserve_batch)
            b = torch.tensor(board_4d).to(device)
            r = torch.tensor(reserves).to(device)
            with torch.no_grad():
                policy_logits, values, _ = model(b, r)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()
            vals = values.cpu().numpy().flatten()
            return policy.astype(np.float32), vals.astype(np.float32)

        # Initial policy for root
        init_policy, _ = eval_fn(bt.reshape(1, *bt.shape), rv.reshape(1, -1))

        batch_mcts = RustBatchMCTS(num_games=1, c_puct=1.5, leaf_batch_size=16)
        batch_mcts.init_searches([game], init_policy)
        batch_mcts.run_simulations([0], [self.simulations], eval_fn)

        moves, probs = batch_mcts.visit_distributions([0])[0]

        if not moves:
            return ["pass"]

        # Build policy vector from MCTS visit distribution
        policy_vector = np.zeros(POLICY_SIZE, dtype=np.float32)
        for (piece_str, from_pos, to_pos), prob in zip(moves, probs):
            if piece_str != "pass":
                idx = game.encode_move(piece_str, from_pos, to_pos)
                if 0 <= idx < POLICY_SIZE:
                    policy_vector[idx] = prob

        # Record for training
        self._history.append((bt, rv, policy_vector, turn_color))

        # Opening moves: sample with temperature 1 for diversity; then argmax
        if self._move_count < self.opening_temp_moves:
            p = np.array(probs, dtype=np.float64)
            p /= p.sum()
            best_idx = int(np.random.choice(len(probs), p=p))
        else:
            best_idx = int(np.argmax(probs))
        self._move_count += 1
        piece_str, from_pos, to_pos = moves[best_idx]
        if piece_str == "pass":
            return ["pass"]

        uhp_str = game.format_move_uhp(piece_str, from_pos, to_pos)
        return [uhp_str]


def play_game(
    white: Engine,
    black: Engine,
    max_moves: int = 200,
    verbose: bool = False,
) -> GameResult:
    """Play a single game between two engines."""
    # Clear training history for ModelEngine(s)
    if isinstance(white, ModelEngine):
        white.new_game_history()
    if isinstance(black, ModelEngine):
        black.new_game_history()

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


def run_parallel_match(
    engine1: ModelEngine,
    engine2: ModelEngine,
    num_games: int = 10,
    max_moves: int = 200,
    verbose: bool = True,
    show_progress: bool = False,
) -> dict:
    """Run a match with all games played in parallel using batched MCTS.

    Both engines must be ModelEngine instances. Games are played concurrently,
    with MCTS evaluations batched per-model for GPU efficiency.
    Even-indexed games: engine1=white, engine2=black.
    Odd-indexed games: engine2=white, engine1=black.
    """
    import torch
    from hive_engine import RustGame, RustBatchMCTS
    from ..encoding.move_encoder import POLICY_SIZE

    NUM_CH = 23
    GS = 23

    if verbose:
        print(f"Match: {engine1.name} vs {engine2.name} ({num_games} games parallel)")

    # Initialize all games
    games = [RustGame() for _ in range(num_games)]
    move_counts = [0] * num_games
    active = set(range(num_games))
    results: list[Optional[GameResult]] = [None] * num_games

    # Track which engine plays which color per game
    # Even games: engine1=white(0), engine2=black(1)
    # Odd games: engine2=white(0), engine1=black(1)
    def get_engine(gi: int, turn: int) -> ModelEngine:
        """Return the engine that moves on this turn for game gi."""
        is_white_turn = (turn % 2 == 0)
        if gi % 2 == 0:
            return engine1 if is_white_turn else engine2
        else:
            return engine2 if is_white_turn else engine1

    def make_eval_fn(model, device):
        def eval_fn(board_batch, reserve_batch):
            board_4d = np.asarray(board_batch)
            reserves = np.asarray(reserve_batch)
            bt = torch.tensor(board_4d).to(device)
            rv = torch.tensor(reserves).to(device)
            with torch.no_grad():
                policy_logits, values, _ = model(bt, rv)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()
            vals = values.cpu().numpy().flatten()
            return policy.astype(np.float32), vals.astype(np.float32)
        return eval_fn

    eval_fn1 = make_eval_fn(engine1.model, engine1.device)
    eval_fn2 = make_eval_fn(engine2.model, engine2.device)

    while active:
        # Handle passes for games with no valid moves
        for gi in list(active):
            if not games[gi].valid_moves():
                games[gi].play_pass()
                move_counts[gi] += 1
                if games[gi].is_game_over or move_counts[gi] >= max_moves:
                    active.discard(gi)

        if not active:
            break

        # Group active games by which engine needs to move
        e1_games = []  # game indices where engine1 moves
        e2_games = []  # game indices where engine2 moves
        for gi in active:
            eng = get_engine(gi, move_counts[gi])
            if eng is engine1:
                e1_games.append(gi)
            else:
                e2_games.append(gi)

        # Process each engine's games as a batch
        for engine, game_indices, eval_fn in [
            (engine1, e1_games, eval_fn1),
            (engine2, e2_games, eval_fn2),
        ]:
            if not game_indices:
                continue

            # Encode boards and get initial policies
            board_list = []
            reserve_list = []
            for gi in game_indices:
                bt, rv = games[gi].encode_board()
                board_list.append(np.asarray(bt).reshape(1, NUM_CH, GS, GS))
                reserve_list.append(np.asarray(rv).reshape(1, -1))

            board_batch = np.concatenate(board_list)
            reserve_batch = np.concatenate(reserve_list)
            init_policies, _ = eval_fn(board_batch, reserve_batch)

            # Set up batch MCTS
            batch_mcts = RustBatchMCTS(
                num_games=len(game_indices),
                c_puct=1.5,
                leaf_batch_size=512,
            )
            game_refs = [games[gi] for gi in game_indices]
            batch_mcts.init_searches(game_refs, init_policies)

            # Check which games have moves
            child_counts = batch_mcts.root_child_counts()
            searching = []
            for bi, gi in enumerate(game_indices):
                if child_counts[bi] > 0:
                    searching.append(bi)
                else:
                    games[gi].play_pass()
                    move_counts[gi] += 1
                    if games[gi].is_game_over or move_counts[gi] >= max_moves:
                        active.discard(gi)

            if searching:
                batch_mcts.run_simulations(searching, [engine.simulations] * len(searching), eval_fn)

            # Collect results and play moves
            all_dists = batch_mcts.visit_distributions(list(range(len(game_indices))))
            for bi, gi in enumerate(game_indices):
                if child_counts[bi] == 0:
                    continue

                moves, probs = all_dists[bi]
                if not moves:
                    games[gi].play_pass()
                    move_counts[gi] += 1
                    if games[gi].is_game_over or move_counts[gi] >= max_moves:
                        active.discard(gi)
                    continue

                # Always argmax in eval
                best_idx = int(np.argmax(probs))
                piece_str, from_pos, to_pos = moves[best_idx]
                if piece_str == "pass":
                    games[gi].play_pass()
                else:
                    games[gi].play_move(piece_str, from_pos, to_pos)

                move_counts[gi] += 1
                if games[gi].is_game_over or move_counts[gi] >= max_moves:
                    active.discard(gi)

        if show_progress:
            done = num_games - len(active)
            print(f"\r  Eval: {done}/{num_games} games done", end="", flush=True)

    if show_progress:
        print()

    # Tally results
    e1_wins = 0
    e2_wins = 0
    draws = 0
    game_results = []

    for gi in range(num_games):
        state = games[gi].state
        if gi % 2 == 0:
            w_name, b_name = engine1.name, engine2.name
        else:
            w_name, b_name = engine2.name, engine1.name

        if state == "WhiteWins":
            result_str = "white"
            if gi % 2 == 0:
                e1_wins += 1
            else:
                e2_wins += 1
        elif state == "BlackWins":
            result_str = "black"
            if gi % 2 == 0:
                e2_wins += 1
            else:
                e1_wins += 1
        else:
            result_str = "draw"
            draws += 1

        game_results.append(GameResult(w_name, b_name, result_str, move_counts[gi]))

    total = e1_wins + e2_wins + draws
    summary = {
        "engine1": engine1.name,
        "engine2": engine2.name,
        "engine1_wins": e1_wins,
        "engine2_wins": e2_wins,
        "draws": draws,
        "total": total,
        "engine1_score": (e1_wins + 0.5 * draws) / total if total else 0,
        "results": game_results,
        "training_samples": [],
    }

    if verbose:
        print(f"  Results: {engine1.name} {e1_wins}W / {draws}D / {e2_wins}L")
        print(f"  Score: {summary['engine1_score']:.0%}")

    return summary


def run_match(
    engine1: Engine,
    engine2: Engine,
    num_games: int = 10,
    max_moves: int = 200,
    verbose: bool = True,
    show_progress: bool = False,
) -> dict:
    """Run a match of multiple games, alternating colors."""
    engine1.start()
    engine2.start()

    if verbose:
        print(f"Match: {engine1.name} vs {engine2.name}")
        print(f"Games: {num_games}, Max moves: {max_moves}")
        print()

    results = []
    all_samples = []
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
            if show_progress:
                done = i + 1
                score = (e1_wins + 0.5 * draws) / done
                print(f"  Games: {done}/{num_games} done  "
                      f"{engine1.name}={e1_wins} {engine2.name}={e2_wins} D={draws}  score={score:.0%}",
                      end="\r" if done < num_games else "\n", flush=True)

            # Collect training samples from ModelEngine(s)
            if result.result in ("white", "black"):
                outcome = {"w": 1.0, "b": -1.0} if result.result == "white" \
                    else {"w": -1.0, "b": 1.0}
            else:
                outcome = {"w": 0.0, "b": 0.0}

            for eng in (engine1, engine2):
                if isinstance(eng, ModelEngine) and eng._history:
                    all_samples.extend(eng.get_samples(outcome))

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
        "training_samples": all_samples,
    }

    if verbose:
        print("=" * 40)
        print(f"Results: {engine1.name} {e1_wins}W / {draws}D / {e2_wins}L")
        print(f"Score: {summary['engine1_score']:.1%}")
        if all_samples:
            print(f"Training samples collected: {len(all_samples)}")

    return summary
