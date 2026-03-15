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
    # Count queen neighbors via the board
    # RustGame doesn't expose queen neighbor counting directly,
    # so we use the valid_moves approach: check game state
    state = game.state
    if state == "WhiteWins":
        return {"w": 1.0, "b": -1.0}
    elif state == "BlackWins":
        return {"w": -1.0, "b": 1.0}
    elif state == "Draw":
        return {"w": 0.0, "b": 0.0}

    # For unfinished games, return neutral (the NN value will be used instead)
    # A proper heuristic would need queen position info exposed from Rust
    return {"w": 0.0, "b": 0.0}


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

                # Clean policy for training
                policy_clean = policies[i] * mask
                total_clean = policy_clean.sum()
                if total_clean > 0:
                    policy_clean /= total_clean
                policy_vector = np.zeros(POLICY_SIZE, dtype=np.float32)
                legal_probs = []
                legal_moves = []
                for idx, piece_str, from_pos, to_pos in indexed_moves:
                    policy_vector[idx] = policy_clean[idx]
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
    """Parallel MCTS self-play using Rust game engine and RustMCTS.

    All tree search happens in Rust; only NN evaluation crosses to Python.
    """

    def __init__(self, model, device: str = "cpu", num_parallel: int = 8,
                 simulations: int = 100, max_moves: int = 200,
                 temperature: float = 1.0, temp_threshold: int = 30):
        self.model = model
        self.device = device
        self.num_parallel = num_parallel
        self.simulations = simulations
        self.max_moves = max_moves
        self.temperature = temperature
        self.temp_threshold = temp_threshold

    def _make_eval_fn(self):
        """Create a batch NN evaluation callback for RustMCTS."""
        import torch
        model = self.model
        device = self.device

        def eval_fn(board_batch, reserve_batch):
            if model is None:
                n = board_batch.shape[0]
                policy = np.ones((n, POLICY_SIZE), dtype=np.float32) / POLICY_SIZE
                value = np.zeros(n, dtype=np.float32)
                return policy, value

            bt = torch.tensor(board_batch).to(device)
            rv = torch.tensor(reserve_batch).to(device)
            with torch.no_grad():
                policy_logits, values = model(bt, rv)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()
            vals = values.cpu().numpy().flatten()
            return policy.astype(np.float32), vals.astype(np.float32)

        return eval_fn

    def play_games(self, num_games: int) -> tuple[list[list[tuple]], list]:
        """Play num_games via parallel MCTS self-play."""
        all_samples = []
        all_games = []
        remaining = num_games
        while remaining > 0:
            wave_size = min(remaining, self.num_parallel)
            wave_samples, wave_games = self._play_wave(wave_size)
            all_samples.extend(wave_samples)
            all_games.extend(wave_games)
            remaining -= wave_size
        return all_samples, all_games

    def _play_wave(self, wave_size: int) -> tuple[list[list[tuple]], list]:
        """Play a wave of games using RustMCTS."""
        import sys

        games = [RustGame() for _ in range(wave_size)]
        histories = [[] for _ in range(wave_size)]
        move_counts = [0] * wave_size
        active = list(range(wave_size))
        finished_count = 0
        eval_fn = self._make_eval_fn()

        while active:
            # For each active game, run RustMCTS search
            for gi in list(active):
                game = games[gi]
                move_num = move_counts[gi]
                temp = self.temperature if move_num < self.temp_threshold else 0.0

                # Record position before MCTS
                bt, rv = game.encode_board()

                # Check for valid moves
                valid = game.valid_moves()
                if not valid:
                    # Must pass
                    game.play_pass()
                    move_counts[gi] += 1
                    if game.is_game_over or move_counts[gi] >= self.max_moves:
                        active.remove(gi)
                        finished_count += 1
                    continue

                # Run RustMCTS
                mcts = RustMCTS(c_puct=1.5, batch_size=16)
                moves, probs = mcts.get_policy(game, eval_fn, self.simulations, temp if temp > 0 else 1.0)

                if not moves:
                    game.play_pass()
                    move_counts[gi] += 1
                    if game.is_game_over or move_counts[gi] >= self.max_moves:
                        active.remove(gi)
                        finished_count += 1
                    continue

                # Apply temperature to visit distribution
                probs = np.array(probs, dtype=np.float32)
                if temp == 0:
                    probs = np.zeros_like(probs)
                    probs[np.argmax(probs)] = 1.0
                    # Fix: use original probs for argmax
                    _, orig_probs = mcts.get_policy(game, eval_fn, 0, 1.0)
                    probs = np.zeros(len(moves), dtype=np.float32)
                    probs[np.argmax(orig_probs)] = 1.0

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
                    active.remove(gi)
                    finished_count += 1

            total_moves = sum(move_counts)
            print(f"\r  Games: {finished_count}/{wave_size} done, "
                  f"{len(active)} active, "
                  f"{total_moves} total moves", end="", flush=True)

        print()

        # Build samples with outcomes
        all_game_samples = []
        wins_w, wins_b, draws = 0, 0, 0
        for gi in range(wave_size):
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
