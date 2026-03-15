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


def _rust_heuristic_value(game: RustGame) -> dict[str, float]:
    """Heuristic value for unfinished games using RustGame."""
    state = game.state
    if state == "WhiteWins":
        return {"w": 1.0, "b": -1.0}
    elif state == "BlackWins":
        return {"w": -1.0, "b": 1.0}
    elif state == "Draw":
        return {"w": 0.0, "b": 0.0}

    # Heuristic based on queen neighbor pressure + beetle-on-queen
    w_score, b_score = game.heuristic_value()
    return {"w": w_score, "b": b_score}


class RustFastSelfPlay:
    """Fast self-play using Rust game engine + raw network policy (no MCTS).

    Replaces FastSelfPlay with Rust-accelerated game logic and encoding.
    """

    def __init__(self, model, device: str = "cpu", max_moves: int = 80,
                 temperature: float = 1.0, temp_threshold: int = 30):
        self.model = model
        self.device = device
        self.max_moves = max_moves
        self.temperature = temperature
        self.temp_threshold = temp_threshold

    def play_games(self, num_games: int) -> tuple[list[list[tuple]], list]:
        """Play num_games using raw policy with Rust game engine."""
        import torch
        import sys

        games = [RustGame() for _ in range(num_games)]
        histories = [[] for _ in range(num_games)]
        active = list(range(num_games))
        move_counts = [0] * num_games
        recent_positions = [[] for _ in range(num_games)]

        while active:
            # Batch encode all active positions using Rust
            boards = []
            reserves = []
            for gi in active:
                bt, rv = games[gi].encode_board()
                boards.append(bt)
                reserves.append(rv)

            # Single GPU call for all active games
            if self.model is not None:
                bt_batch = torch.tensor(np.stack(boards)).to(self.device)
                rv_batch = torch.tensor(np.stack(reserves)).to(self.device)
                with torch.no_grad():
                    policy_logits, _ = self.model(bt_batch, rv_batch)
                policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
            else:
                policies = np.ones((len(active), POLICY_SIZE), dtype=np.float32) / POLICY_SIZE

            newly_finished = []
            for i, gi in enumerate(active):
                game = games[gi]
                move_num = move_counts[gi]
                temp = self.temperature if move_num < self.temp_threshold else 0.0

                # Get legal moves and mask using Rust
                mask, indexed_moves = game.get_legal_move_mask()

                if not indexed_moves:
                    game.play_pass()
                    move_counts[gi] += 1
                    if game.is_game_over or move_counts[gi] >= self.max_moves:
                        newly_finished.append(gi)
                    continue

                # Mask and normalize
                policy = policies[i] * mask
                total = policy.sum()
                if total > 0:
                    policy /= total
                else:
                    policy = mask / mask.sum()

                # Dirichlet noise
                num_legal = len(indexed_moves)
                noise = np.random.dirichlet([0.3] * num_legal)
                noise_full = np.zeros(POLICY_SIZE, dtype=np.float32)
                for j, (idx, _, _, _) in enumerate(indexed_moves):
                    noise_full[idx] = noise[j]
                policy = 0.75 * policy + 0.25 * noise_full

                # Policy target for training: uniform over legal moves
                # (using the model's own output would be circular)
                policy_vector = np.zeros(POLICY_SIZE, dtype=np.float32)
                legal_probs = []
                legal_moves = []
                uniform_prob = 1.0 / len(indexed_moves)
                for idx, piece_str, from_pos, to_pos in indexed_moves:
                    policy_vector[idx] = uniform_prob
                    legal_probs.append(policy[idx])
                    legal_moves.append((piece_str, from_pos, to_pos))

                # Repetition detection
                # Use valid_moves as a position hash proxy
                pos_key = frozenset(
                    (piece_str, from_pos, to_pos)
                    for piece_str, from_pos, to_pos in game.valid_moves()
                )
                recent = recent_positions[gi]
                if recent.count(pos_key) >= 3:
                    newly_finished.append(gi)
                    continue
                recent.append(pos_key)

                # Record training data
                histories[gi].append((boards[i], reserves[i], policy_vector, game.turn_color))
                if len(recent) > 20:
                    recent.pop(0)

                # Sample move
                legal_probs = np.array(legal_probs, dtype=np.float32)
                if temp == 0:
                    move_idx = np.argmax(legal_probs)
                else:
                    legal_probs = legal_probs ** (1.0 / temp)
                    total = legal_probs.sum()
                    if total > 0:
                        legal_probs /= total
                    else:
                        legal_probs = np.ones(len(legal_probs)) / len(legal_probs)
                    move_idx = np.random.choice(len(legal_moves), p=legal_probs)

                piece_str, from_pos, to_pos = legal_moves[move_idx]
                game.play_move(piece_str, from_pos, to_pos)
                move_counts[gi] += 1

                if game.is_game_over or move_counts[gi] >= self.max_moves:
                    newly_finished.append(gi)

            for gi in newly_finished:
                active.remove(gi)

            total_moves = sum(move_counts)
            finished = num_games - len(active)
            print(f"\r  Games: {finished}/{num_games} done, "
                  f"{len(active)} active, "
                  f"{total_moves} total moves", end="", flush=True)

        print()

        # Build samples with outcomes
        from ..core.pieces import PieceColor
        all_game_samples = []
        wins_w, wins_b, draws = 0, 0, 0
        for gi in range(num_games):
            game = games[gi]
            state = game.state
            if state == "WhiteWins":
                outcome = {"w": 1.0, "b": -1.0}
                wins_w += 1
            elif state == "BlackWins":
                outcome = {"w": -1.0, "b": 1.0}
                wins_b += 1
            else:
                outcome = _rust_heuristic_value(game)
                draws += 1

            samples = []
            for bt, rv, pv, color in histories[gi]:
                samples.append((bt, rv, pv, outcome[color]))
            all_game_samples.append(samples)

        print(f"  Results: W={wins_w} B={wins_b} D/unfinished={draws}")
        return all_game_samples, games


class RustParallelSelfPlay:
    """Batched MCTS self-play using Rust game engine with rayon parallelism.

    Cross-game batching: MCTS tree ops (select, encode, expand, backprop)
    are parallelized across games with rayon, while NN inference is batched
    into single GPU calls.
    """

    LEAF_BATCH_SIZE = 512

    def __init__(self, model, device: str = "cpu",
                 simulations: int = 100, max_moves: int = 200,
                 temperature: float = 1.0, temp_threshold: int = 30,
                 **kwargs):
        self.model = model
        self.device = device
        self.simulations = simulations
        self.max_moves = max_moves
        self.temperature = temperature
        self.temp_threshold = temp_threshold

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
            init_policies, _ = self._eval_batch(board_batch, reserve_batch)

            # --- Init batch MCTS (rayon-parallel) ---
            batch_mcts = RustBatchMCTS(
                num_games=len(mcts_games),
                c_puct=1.5,
                leaf_batch_size=self.LEAF_BATCH_SIZE,
            )
            game_refs = [games[gi] for gi in mcts_games]
            batch_mcts.init_searches(game_refs, init_policies)

            # Map from batch index to game index
            batch_to_gi = list(mcts_games)
            child_counts = batch_mcts.root_child_counts()

            searching = []  # batch indices with valid trees
            for bi, gi in enumerate(batch_to_gi):
                if child_counts[bi] > 0:
                    searching.append(bi)
                else:
                    games[gi].play_pass()
                    move_counts[gi] += 1
                    if games[gi].is_game_over or move_counts[gi] >= self.max_moves:
                        active.discard(gi)
                        finished_count += 1

            # --- Run full simulation loop (rayon + single GPU callback) ---
            if searching:
                batch_mcts.run_simulations(
                    searching, self.simulations, self._eval_fn(),
                )

            # --- Collect results: visit distributions, play moves ---
            all_bi = list(range(len(batch_to_gi)))
            all_dists = batch_mcts.visit_distributions(all_bi)

            for bi, gi in enumerate(batch_to_gi):
                if child_counts[bi] == 0:
                    continue

                game = games[gi]
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

                histories[gi].append((bt, rv, policy_vector, game.turn_color))

                # Sample move
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
            print(f"\r  Games: {finished_count}/{num_games} done, "
                  f"{len(active)} active, "
                  f"{total_moves} total moves", end="", flush=True)

        print()

        # Build samples with outcomes
        all_game_samples = []
        wins_w, wins_b, draws = 0, 0, 0
        for gi in range(num_games):
            game = games[gi]
            state = game.state
            if state == "WhiteWins":
                outcome = {"w": 1.0, "b": -1.0}
                wins_w += 1
            elif state == "BlackWins":
                outcome = {"w": -1.0, "b": 1.0}
                wins_b += 1
            else:
                outcome = _rust_heuristic_value(game)
                draws += 1

            samples = []
            for bt, rv, pv, color in histories[gi]:
                samples.append((bt, rv, pv, outcome[color]))
            all_game_samples.append(samples)

        print(f"  Results: W={wins_w} B={wins_b} D/unfinished={draws}")
        return all_game_samples, games
