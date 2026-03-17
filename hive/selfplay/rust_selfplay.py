"""Rust-accelerated self-play for training.

Uses RustGame for game logic and encoding, keeping NN evaluation in Python/PyTorch.
Falls back to Python self-play if Rust extension is not available.
"""

from __future__ import annotations
import numpy as np
from typing import Optional

try:
    from hive_engine import RustGame, RustMCTS
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

from ..encoding.move_encoder import POLICY_SIZE


DRAW_PENALTY = -0.5  # both sides penalized for not finishing
DECISIVE_WEIGHT = 10.0  # upweight positions from games with a winner


def _rust_heuristic_value(game: RustGame) -> dict[str, float]:
    """Heuristic value for unfinished games using RustGame."""
    state = game.state
    if state == "WhiteWins":
        return {"w": 1.0, "b": -1.0}
    elif state == "BlackWins":
        return {"w": -1.0, "b": 1.0}
    elif state == "Draw":
        return {"w": DRAW_PENALTY, "b": DRAW_PENALTY}

    # Heuristic based on queen neighbor pressure + beetle-on-queen
    # Plus draw penalty to incentivize decisive play
    w_score, b_score = game.heuristic_value()
    w_score = max(w_score + DRAW_PENALTY, -1.0)
    b_score = max(b_score + DRAW_PENALTY, -1.0)
    return {"w": w_score, "b": b_score}


class RustParallelSelfPlay:
    """Batched MCTS self-play using Rust game engine with rayon parallelism.

    Cross-game batching: MCTS tree ops (select, encode, expand, backprop)
    are parallelized across games with rayon, while NN inference is batched
    into single GPU calls.

    Supports playout cap randomization (KataGo): each turn is randomly
    assigned as "full" (probability playout_cap_p) or "fast" (1-p).
    Full turns use `simulations` playouts with Dirichlet noise and record
    policy targets. Fast turns use `fast_cap` playouts without noise and
    only contribute to value training.
    """

    LEAF_BATCH_SIZE = 512

    def __init__(self, model, device: str = "cpu",
                 simulations: int = 100, max_moves: int = 200,
                 temperature: float = 1.0, temp_threshold: int = 30,
                 resign_threshold: float = -0.97, resign_moves: int = 5,
                 calibration_frac: float = 0.1,
                 playout_cap_p: float = 0.0,
                 fast_cap: int = 20,
                 **kwargs):
        self.model = model
        self.device = device
        self.simulations = simulations
        self.max_moves = max_moves
        self.temperature = temperature
        self.temp_threshold = temp_threshold
        self.resign_threshold = resign_threshold
        self.resign_moves = resign_moves
        self.calibration_frac = calibration_frac
        # Playout cap randomization: if playout_cap_p > 0, each turn has
        # probability p of being a full search (simulations) and (1-p) of
        # being a fast search (fast_cap). Only full-search turns record
        # policy targets. playout_cap_p=0 disables (all turns are full).
        self.playout_cap_p = playout_cap_p
        self.fast_cap = fast_cap

    def _eval_batch(self, board_batch_4d, reserve_batch):
        """Evaluate a batch of positions. Returns (policies, values)."""
        import torch
        if self.model is None:
            n = board_batch_4d.shape[0]
            return (np.ones((n, POLICY_SIZE), dtype=np.float32) / POLICY_SIZE,
                    np.zeros(n, dtype=np.float32))

        bt = torch.tensor(board_batch_4d).to(self.device)
        rv = torch.tensor(reserve_batch).to(self.device)
        with torch.no_grad():
            policy_logits, values = self.model(bt, rv)
        policy = torch.softmax(policy_logits, dim=1).cpu().numpy()
        vals = values.cpu().numpy().flatten()
        return policy.astype(np.float32), vals.astype(np.float32)

    def _eval_fn(self):
        """Return a callable for Rust's run_simulations callback."""
        import torch
        model = self.model
        device = self.device

        def eval_fn(board_batch, reserve_batch):
            board_4d = np.asarray(board_batch)
            reserves = np.asarray(reserve_batch)
            if model is None:
                n = board_4d.shape[0]
                return (np.ones((n, POLICY_SIZE), dtype=np.float32) / POLICY_SIZE,
                        np.zeros(n, dtype=np.float32))
            bt = torch.tensor(board_4d).to(device)
            rv = torch.tensor(reserves).to(device)
            with torch.no_grad():
                policy_logits, values = model(bt, rv)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()
            vals = values.cpu().numpy().flatten()
            return policy.astype(np.float32), vals.astype(np.float32)

        return eval_fn

    def play_games(self, num_games: int) -> tuple[list[list[tuple]], list]:
        """Play num_games with cross-game batched MCTS + rayon parallelism."""
        from hive_engine import RustBatchMCTS

        NUM_CH = 23
        GS = 23

        games = [RustGame() for _ in range(num_games)]
        histories = [[] for _ in range(num_games)]
        move_counts = [0] * num_games
        active = set(range(num_games))
        finished_count = 0

        # Resignation state
        resign_counters = [0] * num_games
        resigned_as: dict[int, str] = {}
        if self.resign_threshold is not None:
            num_cal = max(1, int(num_games * self.calibration_frac))
            calibration_games: set[int] = set(
                np.random.choice(num_games, num_cal, replace=False).tolist()
            )
        else:
            calibration_games = set()
        calibration_would_resign: dict[int, str] = {}

        # Playout cap randomization state
        use_playout_cap = self.playout_cap_p > 0

        while active:
            # --- Handle passes ---
            mcts_games = []
            for gi in list(active):
                if not games[gi].valid_moves():
                    games[gi].play_pass()
                    move_counts[gi] += 1
                    if games[gi].is_game_over or move_counts[gi] >= self.max_moves:
                        active.discard(gi)
                        finished_count += 1
                else:
                    mcts_games.append(gi)

            if not mcts_games:
                continue

            # --- Playout cap: decide fast vs full for each game this turn ---
            if use_playout_cap:
                is_full_search = {
                    gi: np.random.random() < self.playout_cap_p
                    for gi in mcts_games
                }
            else:
                is_full_search = {gi: True for gi in mcts_games}

            # --- Record positions & batch initial policy eval ---
            positions = {}
            board_list = []
            reserve_list = []
            for gi in mcts_games:
                bt, rv = games[gi].encode_board()
                positions[gi] = (bt, rv)
                board_list.append(np.asarray(bt).reshape(1, NUM_CH, GS, GS))
                reserve_list.append(np.asarray(rv).reshape(1, -1))

            board_batch = np.concatenate(board_list)
            reserve_batch = np.concatenate(reserve_list)
            init_policies, init_values = self._eval_batch(board_batch, reserve_batch)

            # --- Init batch MCTS (rayon-parallel) ---
            # Forced playouts + policy target pruning enabled for all games;
            # only meaningful for full-search games that have Dirichlet noise
            batch_mcts = RustBatchMCTS(
                num_games=len(mcts_games),
                c_puct=1.5,
                leaf_batch_size=self.LEAF_BATCH_SIZE,
                use_forced_playouts=True,
            )
            game_refs = [games[gi] for gi in mcts_games]
            batch_mcts.init_searches(game_refs, init_policies)

            # Only apply Dirichlet noise to full-search games
            full_bis = [bi for bi, gi in enumerate(mcts_games) if is_full_search[gi]]
            if full_bis:
                batch_mcts.apply_root_dirichlet(full_bis)

            # Map from batch index to game index
            batch_to_gi = list(mcts_games)
            child_counts = batch_mcts.root_child_counts()

            # Split into fast and full search groups
            fast_searching = []
            full_searching = []
            for bi, gi in enumerate(batch_to_gi):
                if child_counts[bi] > 0:
                    if is_full_search[gi]:
                        full_searching.append(bi)
                    else:
                        fast_searching.append(bi)
                else:
                    games[gi].play_pass()
                    move_counts[gi] += 1
                    if games[gi].is_game_over or move_counts[gi] >= self.max_moves:
                        active.discard(gi)
                        finished_count += 1

            # --- Run simulations: fast cap for fast games, full cap for full ---
            if fast_searching:
                batch_mcts.run_simulations(
                    fast_searching, self.fast_cap, self._eval_fn(),
                )
            if full_searching:
                batch_mcts.run_simulations(
                    full_searching, self.simulations, self._eval_fn(),
                )

            # --- Collect results: visit distributions, play moves ---
            all_bi = list(range(len(batch_to_gi)))
            all_dists = batch_mcts.visit_distributions(all_bi)

            for bi, gi in enumerate(batch_to_gi):
                if child_counts[bi] == 0:
                    continue

                game = games[gi]

                # Resignation check (value is from current player's perspective)
                if self.resign_threshold is not None:
                    val = float(init_values[bi])
                    if val < self.resign_threshold:
                        resign_counters[gi] += 1
                    else:
                        resign_counters[gi] = 0
                    if resign_counters[gi] >= self.resign_moves:
                        if gi in calibration_games:
                            if gi not in calibration_would_resign:
                                calibration_would_resign[gi] = game.turn_color
                        else:
                            resigned_as[gi] = game.turn_color
                            active.discard(gi)
                            finished_count += 1
                            continue

                move_num = move_counts[gi]
                temp = self.temperature if move_num < self.temp_threshold else 0.0
                bt, rv = positions[gi]

                moves, visit_probs = all_dists[bi]
                if not moves:
                    game.play_pass()
                    move_counts[gi] += 1
                    if game.is_game_over or move_counts[gi] >= self.max_moves:
                        active.discard(gi)
                        finished_count += 1
                    continue

                # Apply temperature
                probs = np.array(visit_probs, dtype=np.float32)
                if temp == 0:
                    best = np.argmax(probs)
                    probs = np.zeros_like(probs)
                    probs[best] = 1.0
                else:
                    probs = probs ** (1.0 / temp)
                    total = probs.sum()
                    if total > 0:
                        probs /= total
                    else:
                        probs = np.ones_like(probs) / len(probs)

                # Build policy vector for training
                policy_vector = np.zeros(POLICY_SIZE, dtype=np.float32)
                for (piece_str, from_pos, to_pos), prob in zip(moves, probs):
                    if piece_str != "pass":
                        idx = game.encode_move(piece_str, from_pos, to_pos)
                        if 0 <= idx < POLICY_SIZE:
                            policy_vector[idx] = prob

                # Full search: record for both policy and value training
                # Fast search: record for value training only (policy_vector=None)
                if is_full_search[gi]:
                    histories[gi].append((bt, rv, policy_vector, game.turn_color))
                else:
                    histories[gi].append((bt, rv, None, game.turn_color))

                # Sample move (fast turns play strongest move for game quality)
                if not is_full_search[gi]:
                    move_idx = np.argmax(probs)
                else:
                    move_idx = np.random.choice(len(moves), p=probs)
                piece_str, from_pos, to_pos = moves[move_idx]
                if piece_str == "pass":
                    game.play_pass()
                else:
                    game.play_move(piece_str, from_pos, to_pos)

                move_counts[gi] += 1
                if game.is_game_over or move_counts[gi] >= self.max_moves:
                    active.discard(gi)
                    finished_count += 1

            total_moves = sum(move_counts)
            resign_str = f", {len(resigned_as)} resigned" if resigned_as else ""
            print(f"\r  Games: {finished_count}/{num_games} done, "
                  f"{len(active)} active, "
                  f"{total_moves} total moves{resign_str}", end="", flush=True)

        print()

        # Build samples with outcomes
        all_game_samples = []
        wins_w, wins_b, draws, resignations = 0, 0, 0, 0
        for gi in range(num_games):
            game = games[gi]
            if gi in resigned_as:
                resigning_color = resigned_as[gi]
                if resigning_color == "w":
                    outcome = {"w": -1.0, "b": 1.0}
                    wins_b += 1
                else:
                    outcome = {"w": 1.0, "b": -1.0}
                    wins_w += 1
                decisive = True
                resignations += 1
            else:
                state = game.state
                if state == "WhiteWins":
                    outcome = {"w": 1.0, "b": -1.0}
                    decisive = True
                    wins_w += 1
                elif state == "BlackWins":
                    outcome = {"w": -1.0, "b": 1.0}
                    decisive = True
                    wins_b += 1
                else:
                    outcome = _rust_heuristic_value(game)
                    decisive = False
                    draws += 1

            weight = DECISIVE_WEIGHT if decisive else 1.0
            samples = []
            for bt, rv, pv, color in histories[gi]:
                if pv is None:
                    # Fast-search turn: value-only, zero policy
                    pv = np.zeros(POLICY_SIZE, dtype=np.float32)
                    samples.append((bt, rv, pv, outcome[color], weight, True))
                else:
                    samples.append((bt, rv, pv, outcome[color], weight, False))
            all_game_samples.append(samples)

        resign_suffix = f" (resigned={resignations})" if resignations else ""
        if use_playout_cap:
            total_turns = sum(len(h) for h in histories)
            full_turns = sum(
                1 for h in histories for entry in h if entry[2] is not None
            )
            print(f"  Results: W={wins_w} B={wins_b} D/unfinished={draws}{resign_suffix}")
            print(f"  Playout cap: {full_turns}/{total_turns} full-search turns "
                  f"({100*full_turns/max(total_turns,1):.0f}%)")
        else:
            print(f"  Results: W={wins_w} B={wins_b} D/unfinished={draws}{resign_suffix}")

        # Calibration report
        if calibration_games and calibration_would_resign:
            false_pos = 0
            for gi, would_resign_color in calibration_would_resign.items():
                state = games[gi].state
                if (would_resign_color == "w" and state == "WhiteWins") or \
                   (would_resign_color == "b" and state == "BlackWins"):
                    false_pos += 1
            print(f"  Calibration: {len(calibration_would_resign)}/{len(calibration_games)} "
                  f"would resign, {false_pos} false positives")

        return all_game_samples, games
