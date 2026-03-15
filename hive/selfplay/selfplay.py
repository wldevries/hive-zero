"""Self-play training loop for Hive AI."""

from __future__ import annotations
import numpy as np
import os
from typing import Optional

from ..core.game import Game, GameState
from ..core.pieces import Piece, PieceColor, PieceType
from ..encoding.board_encoder import encode_board
from ..encoding.move_encoder import encode_move, get_legal_move_mask, POLICY_SIZE


def heuristic_value(game: Game) -> dict[PieceColor, float]:
    """Estimate game value for unfinished games based on queen safety.

    Returns value in [-1, 1] for each color. Positive = winning.
    Based on how surrounded each queen is: more surrounded opponent queen = better.
    """
    board = game.board
    wq = Piece(PieceColor.WHITE, PieceType.QUEEN, 1)
    bq = Piece(PieceColor.BLACK, PieceType.QUEEN, 1)

    wq_pos = board.piece_position(wq)
    bq_pos = board.piece_position(bq)

    # Count how many of the 6 hexes around each queen are occupied
    w_surrounded = 0
    b_surrounded = 0

    if wq_pos is not None:
        w_surrounded = sum(1 for n in wq_pos.neighbors() if n in board.occupied)
    if bq_pos is not None:
        b_surrounded = sum(1 for n in bq_pos.neighbors() if n in board.occupied)

    # Value from white's perspective: opponent queen more surrounded = good
    # Scale: 6 surrounded = 1.0 (win), 0 surrounded = 0.0
    # Net value = (opponent_surrounded - my_surrounded) / 6
    w_value = (b_surrounded - w_surrounded) / 6.0

    # Clamp to [-0.8, 0.8] so heuristic values don't overpower real wins
    w_value = max(-0.8, min(0.8, w_value))

    return {PieceColor.WHITE: w_value, PieceColor.BLACK: -w_value}
from ..nn.model import HiveNet, create_model, save_checkpoint, load_checkpoint
from ..nn.training import HiveDataset, Trainer
from ..mcts.mcts import MCTS


class FastSelfPlay:
    """Fast self-play using raw network policy (no MCTS).

    One forward pass per move per game. All games in a batch share a single
    GPU call, making this ~30-50x faster than MCTS-based self-play.
    Used for early training iterations when the network is still random.
    """

    def __init__(self, model, device: str = "cpu", max_moves: int = 80,
                 temperature: float = 1.0, temp_threshold: int = 30):
        self.model = model
        self.device = device
        self.max_moves = max_moves
        self.temperature = temperature
        self.temp_threshold = temp_threshold

    def play_games(self, num_games: int) -> list[list[tuple]]:
        """Play num_games using raw policy. Returns list of game samples."""
        import torch
        import sys

        games = [Game() for _ in range(num_games)]
        histories = [[] for _ in range(num_games)]
        active = list(range(num_games))
        move_counts = [0] * num_games
        recent_positions = [[] for _ in range(num_games)]  # track for repetition detection

        while active:
            # Batch encode all active positions
            boards = []
            reserves = []
            for gi in active:
                bt, rv = encode_board(games[gi])
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

            # Apply policy to each game
            newly_finished = []
            for i, gi in enumerate(active):
                game = games[gi]
                move_num = move_counts[gi]
                temp = self.temperature if move_num < self.temp_threshold else 0.0

                # Get legal moves and mask policy
                mask, indexed_moves = get_legal_move_mask(game)

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

                # Add Dirichlet noise for exploration (like AlphaZero)
                num_legal = len(indexed_moves)
                noise = np.random.dirichlet([0.3] * num_legal)
                noise_full = np.zeros(POLICY_SIZE, dtype=np.float32)
                for j, (idx, _, _, _) in enumerate(indexed_moves):
                    noise_full[idx] = noise[j]
                policy = 0.75 * policy + 0.25 * noise_full

                # Build training policy vector (store the clean policy, not noised)
                policy_clean = policies[i] * mask
                total_clean = policy_clean.sum()
                if total_clean > 0:
                    policy_clean /= total_clean
                policy_vector = np.zeros(POLICY_SIZE, dtype=np.float32)
                legal_probs = []
                legal_moves = []
                for idx, piece, from_pos, to_pos in indexed_moves:
                    policy_vector[idx] = policy_clean[idx]
                    legal_probs.append(policy[idx])  # use noised policy for sampling
                    legal_moves.append((piece, from_pos, to_pos))

                # Repetition detection: hash position by piece positions
                pos_key = frozenset(
                    (str(p), h) for h, p in game.board.all_top_pieces()
                )
                recent = recent_positions[gi]
                if recent.count(pos_key) >= 3:
                    # Threefold repetition → draw
                    newly_finished.append(gi)
                    continue
                recent.append(pos_key)

                # Record position (after repetition check so we don't store dead-end states)
                histories[gi].append((boards[i], reserves[i], policy_vector, game.turn_color))
                if len(recent) > 20:
                    recent.pop(0)

                # Sample move
                legal_probs = np.array(legal_probs, dtype=np.float32)
                if temp == 0:
                    move_idx = np.argmax(legal_probs)
                else:
                    # Apply temperature
                    legal_probs = legal_probs ** (1.0 / temp)
                    total = legal_probs.sum()
                    if total > 0:
                        legal_probs /= total
                    else:
                        legal_probs = np.ones(len(legal_probs)) / len(legal_probs)
                    move_idx = np.random.choice(len(legal_moves), p=legal_probs)

                piece, from_pos, to_pos = legal_moves[move_idx]
                game.play_move(piece, from_pos, to_pos)
                move_counts[gi] += 1

                if game.is_game_over or move_counts[gi] >= self.max_moves:
                    newly_finished.append(gi)

            for gi in newly_finished:
                active.remove(gi)

            # Progress
            total_moves = sum(move_counts)
            finished = num_games - len(active)
            print(f"\r  Games: {finished}/{num_games} done, "
                  f"{len(active)} active, "
                  f"{total_moves} total moves", end="", flush=True)

        print()  # newline

        # Build samples with outcomes and report stats
        all_game_samples = []
        wins_w, wins_b, draws = 0, 0, 0
        for gi in range(num_games):
            game = games[gi]
            if game.state == GameState.WHITE_WINS:
                outcome = {PieceColor.WHITE: 1.0, PieceColor.BLACK: -1.0}
                wins_w += 1
            elif game.state == GameState.BLACK_WINS:
                outcome = {PieceColor.WHITE: -1.0, PieceColor.BLACK: 1.0}
                wins_b += 1
            else:
                outcome = heuristic_value(game)
                draws += 1

            samples = []
            for bt, rv, pv, color in histories[gi]:
                samples.append((bt, rv, pv, outcome[color]))
            all_game_samples.append(samples)

        print(f"  Results: W={wins_w} B={wins_b} D/unfinished={draws}")
        return all_game_samples


class ParallelSelfPlay:
    """Run multiple self-play games in parallel, batching GPU evaluations
    across all active games for maximum throughput."""

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

    def play_games(self, num_games: int) -> list[list[tuple]]:
        """Play num_games via parallel self-play. Returns list of game samples."""
        all_samples = []
        # Run games in waves of num_parallel
        remaining = num_games
        while remaining > 0:
            wave_size = min(remaining, self.num_parallel)
            wave_samples = self._play_wave(wave_size)
            all_samples.extend(wave_samples)
            remaining -= wave_size
        return all_samples

    def _play_wave(self, wave_size: int) -> list[list[tuple]]:
        """Play a wave of games simultaneously."""
        import sys
        import torch

        # Initialize games
        games = [Game() for _ in range(wave_size)]
        histories = [[] for _ in range(wave_size)]  # per-game: (bt, rv, pv, color)
        move_counts = [0] * wave_size
        active = list(range(wave_size))  # indices of games still in progress
        finished_count = 0

        # Per-game MCTS roots (reset each move)
        mcts_states = [None] * wave_size  # will hold (root, sims_done)

        loop_count = 0
        while active:
            loop_count += 1
            # Print progress periodically
            if loop_count % 50 == 0:
                total_moves = sum(move_counts)
                print(f"\r  Games: {finished_count}/{wave_size} done, "
                      f"{len(active)} active, "
                      f"{total_moves} total moves", end="", flush=True)

            # Phase 1: For each active game that needs a new MCTS search,
            # initialize or continue the search
            needs_init = []
            needs_sim = []

            for gi in active:
                if mcts_states[gi] is None:
                    # Start new MCTS for this game's current position
                    needs_init.append(gi)
                else:
                    root, sims_done = mcts_states[gi]
                    if sims_done >= self.simulations:
                        # MCTS done - pick move and advance game
                        self._finalize_move(gi, games, histories, move_counts, mcts_states)
                    else:
                        needs_sim.append(gi)

            # Re-check which games are still active after finalizing
            newly_finished = []
            for gi in active:
                if games[gi].is_game_over or move_counts[gi] >= self.max_moves:
                    newly_finished.append(gi)
            for gi in newly_finished:
                active.remove(gi)
                mcts_states[gi] = None
                finished_count += 1
                total_moves = sum(move_counts)
                print(f"\r  Games: {finished_count}/{wave_size} done, "
                      f"{len(active)} active, "
                      f"{total_moves} total moves", end="", flush=True)

            if not active:
                print()  # newline after progress
                break

            # Initialize MCTS roots for games that need it
            if needs_init:
                # Batch evaluate root positions
                root_games = [games[gi] for gi in needs_init]
                if self.model is not None:
                    policies, values = self._batch_evaluate(root_games)
                else:
                    policies = [np.ones(POLICY_SIZE, dtype=np.float32) / POLICY_SIZE] * len(needs_init)

                for i, gi in enumerate(needs_init):
                    from ..mcts.mcts import MCTSNode
                    root = MCTSNode(games[gi].copy())
                    self._expand_node(root, policies[i])
                    if root.children:
                        mcts_states[gi] = (root, 0)
                    else:
                        # No valid moves - pass
                        games[gi].play_pass()
                        move_counts[gi] += 1
                        mcts_states[gi] = None

            # Phase 2: Run one batch of MCTS simulations across all active games
            # Collect leaves from all active games
            all_leaves = []  # (game_index, leaf_node)
            seen_leaves = set()  # track by id for O(1) duplicate check
            active_with_mcts = [gi for gi in active if mcts_states[gi] is not None]

            for gi in active_with_mcts:
                root, sims_done = mcts_states[gi]
                remaining_sims = self.simulations - sims_done
                batch = min(16, remaining_sims)

                for _ in range(batch):
                    leaf = root.select_leaf()
                    if leaf.game.is_game_over:
                        value = self._terminal_value(leaf.game, root.game.turn_color)
                        self._backpropagate(leaf, value)
                    elif id(leaf) in seen_leaves:
                        self._backpropagate(leaf, 0.0)
                    else:
                        seen_leaves.add(id(leaf))
                        all_leaves.append((gi, leaf))

                mcts_states[gi] = (root, sims_done + batch)

            # Batch evaluate all collected leaves
            if all_leaves and self.model is not None:
                leaf_games = [leaf.game for _, leaf in all_leaves]
                policies, values = self._batch_evaluate(leaf_games)

                for (gi, leaf), policy, value in zip(all_leaves, policies, values):
                    self._expand_node(leaf, policy)
                    root = mcts_states[gi][0]
                    if leaf.game.turn_color != root.game.turn_color:
                        value = -value
                    self._backpropagate(leaf, value)
            elif all_leaves:
                for gi, leaf in all_leaves:
                    policy = np.ones(POLICY_SIZE, dtype=np.float32) / POLICY_SIZE
                    self._expand_node(leaf, policy)
                    self._backpropagate(leaf, 0.0)

        # Convert histories to samples with outcomes
        all_game_samples = []
        wins_w, wins_b, draws = 0, 0, 0
        for gi in range(wave_size):
            game = games[gi]
            if game.state == GameState.WHITE_WINS:
                outcome = {PieceColor.WHITE: 1.0, PieceColor.BLACK: -1.0}
                wins_w += 1
            elif game.state == GameState.BLACK_WINS:
                outcome = {PieceColor.WHITE: -1.0, PieceColor.BLACK: 1.0}
                wins_b += 1
            else:
                outcome = heuristic_value(game)
                draws += 1

            samples = []
            for bt, rv, pv, color in histories[gi]:
                samples.append((bt, rv, pv, outcome[color]))
            all_game_samples.append(samples)

        print(f"  Results: W={wins_w} B={wins_b} D/unfinished={draws}")
        return all_game_samples

    def _finalize_move(self, gi, games, histories, move_counts, mcts_states):
        """Pick a move from completed MCTS, record training data, advance game."""
        root, _ = mcts_states[gi]
        game = games[gi]
        move_num = move_counts[gi]
        temp = self.temperature if move_num < self.temp_threshold else 0.0

        # Record position before making move
        bt, rv = encode_board(game)

        # Get visit count distribution
        moves = [c.move for c in root.children]
        visit_counts = np.array([c.visit_count for c in root.children], dtype=np.float32)

        if temp == 0:
            probs = np.zeros_like(visit_counts)
            probs[np.argmax(visit_counts)] = 1.0
        else:
            visit_counts_t = visit_counts ** (1.0 / temp)
            total = visit_counts_t.sum()
            probs = visit_counts_t / total if total > 0 else np.ones_like(visit_counts_t) / len(visit_counts_t)

        # Build policy vector
        policy_vector = np.zeros(POLICY_SIZE, dtype=np.float32)
        for move, prob in zip(moves, probs):
            if move is not None:
                piece, from_pos, to_pos = move
                idx = encode_move(piece, from_pos, to_pos)
                if 0 <= idx < POLICY_SIZE:
                    policy_vector[idx] = prob

        histories[gi].append((bt, rv, policy_vector, game.turn_color))

        # Sample and play move
        move_idx = np.random.choice(len(moves), p=probs)
        move = moves[move_idx]
        if move is None:
            game.play_pass()
        else:
            piece, from_pos, to_pos = move
            game.play_move(piece, from_pos, to_pos)

        move_counts[gi] += 1
        mcts_states[gi] = None  # will reinitialize next iteration

    def _batch_evaluate(self, games: list[Game]) -> tuple[list[np.ndarray], list[float]]:
        """Batch neural network evaluation."""
        import torch

        boards = []
        reserves = []
        for game in games:
            bt, rv = encode_board(game)
            boards.append(bt)
            reserves.append(rv)

        bt_batch = torch.tensor(np.stack(boards)).to(self.device)
        rv_batch = torch.tensor(np.stack(reserves)).to(self.device)

        with torch.no_grad():
            policy_logits, values = self.model(bt_batch, rv_batch)

        policies = torch.softmax(policy_logits, dim=1).cpu().numpy()
        values_np = values.cpu().numpy().flatten()

        return [policies[i] for i in range(len(games))], values_np.tolist()

    def _expand_node(self, node, policy):
        """Expand a node with precomputed policy."""
        from ..mcts.mcts import MCTSNode
        node.is_expanded = True
        game = node.game

        valid_moves = game.valid_moves()
        if not valid_moves:
            child_game = game.copy()
            child_game.play_pass()
            child = MCTSNode(child_game, parent=node, move=None, prior=1.0)
            node.children.append(child)
            return

        mask, indexed_moves = get_legal_move_mask(game)
        total_prior = 0.0

        for idx, piece, from_pos, to_pos in indexed_moves:
            prior = policy[idx]
            total_prior += prior
            child_game = game.copy()
            child_game.play_move(piece, from_pos, to_pos)
            child = MCTSNode(child_game, parent=node,
                             move=(piece, from_pos, to_pos), prior=prior)
            node.children.append(child)

        if total_prior > 0:
            for child in node.children:
                child.prior /= total_prior

    def _terminal_value(self, game, perspective_color):
        if game.state == GameState.DRAW:
            return 0.0
        if game.state == GameState.WHITE_WINS:
            return 1.0 if perspective_color == PieceColor.WHITE else -1.0
        if game.state == GameState.BLACK_WINS:
            return 1.0 if perspective_color == PieceColor.BLACK else -1.0
        return 0.0

    def _backpropagate(self, node, value):
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            value = -value
            current = current.parent


class SelfPlayTrainer:
    """Full self-play training pipeline."""

    def __init__(self, model_path: str = "model.pt", device: str = "cpu",
                 num_blocks: int = 6, channels: int = 64,
                 checkpoint_dir: str = "checkpoints"):
        self.model_path = model_path
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.num_blocks = num_blocks
        self.channels = channels
        self.start_iteration = 0

        if os.path.exists(model_path):
            self.model, ckpt = load_checkpoint(model_path)
            self.start_iteration = ckpt.get("iteration", 0)
            print(f"Resumed from {model_path} (iteration {self.start_iteration})")
        else:
            self.model = create_model(num_blocks, channels)
            print(f"Created new model ({num_blocks} blocks, {channels} channels)")

        self.model.to(device)
        self.trainer = Trainer(self.model, device=device)

        # Save initial model immediately so it exists on disk
        if not os.path.exists(model_path):
            save_checkpoint(self.model, model_path, self.start_iteration)
            print(f"  Saved initial model to {model_path}")

    def run(self, num_iterations: int = 100, games_per_iter: int = 10,
            simulations: int = 100, epochs_per_iter: int = 5,
            batch_size: int = 64, max_moves: int = 200,
            time_limit_minutes: float | None = None,
            num_parallel: int = 8, mcts_after: int = 20):
        """Run the full training loop.

        Args:
            mcts_after: Use MCTS self-play after this many iterations.
                Before that, use fast raw-policy self-play.
        """
        import time
        start_time = time.time()

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Replay buffer: keep last ~50k positions across iterations
        replay_buffer = HiveDataset(max_size=50_000)

        for i in range(num_iterations):
            iteration = self.start_iteration + i + 1

            if time_limit_minutes is not None:
                elapsed = (time.time() - start_time) / 60.0
                if elapsed >= time_limit_minutes:
                    print(f"\nTime limit reached ({elapsed:.1f}m / {time_limit_minutes}m)")
                    break

            elapsed_str = f" [{(time.time() - start_time) / 60:.1f}m]" if time_limit_minutes else ""
            use_mcts = iteration > mcts_after
            if use_mcts and iteration == mcts_after + 1 and len(replay_buffer) > 0:
                replay_buffer = HiveDataset(max_size=50_000)
                print("  Cleared replay buffer for MCTS transition")
            mode = "MCTS" if use_mcts else "fast"
            print(f"\n=== Iteration {iteration} ({mode}){elapsed_str} ===")

            # Generate self-play games
            iter_start = time.time()

            if use_mcts:
                sp = ParallelSelfPlay(
                    model=self.model, device=self.device,
                    num_parallel=num_parallel, simulations=simulations,
                    max_moves=max_moves,
                )
            else:
                sp = FastSelfPlay(
                    model=self.model, device=self.device,
                    max_moves=max_moves,
                )

            all_game_samples = sp.play_games(games_per_iter)

            total_positions = 0
            for gi, samples in enumerate(all_game_samples):
                for bt, rv, pv, vt in samples:
                    replay_buffer.add_sample(bt, rv, pv, vt)
                total_positions += len(samples)

            game_time = time.time() - iter_start
            print(f"  {games_per_iter} games: {total_positions} new positions "
                  f"({game_time:.1f}s, {total_positions / max(game_time, 0.1):.0f} pos/s), "
                  f"buffer: {len(replay_buffer)}")

            # Train on replay buffer
            for epoch in range(epochs_per_iter):
                losses = self.trainer.train_epoch(replay_buffer, batch_size=batch_size)
                print(f"  Epoch {epoch + 1}: loss={losses['total_loss']:.4f} "
                      f"(policy={losses['policy_loss']:.4f}, value={losses['value_loss']:.4f})")

            # Save latest model + periodic checkpoint
            metadata = {
                "total_loss": losses["total_loss"],
                "policy_loss": losses["policy_loss"],
                "value_loss": losses["value_loss"],
                "samples": len(replay_buffer),
            }
            save_checkpoint(self.model, self.model_path, iteration, metadata)

            if iteration % 10 == 0:
                ckpt_path = os.path.join(self.checkpoint_dir, f"model_iter{iteration:04d}.pt")
                save_checkpoint(self.model, ckpt_path, iteration, metadata)
                print(f"  Checkpoint saved to {ckpt_path}")

            print(f"  Model saved to {self.model_path} (iteration {iteration})")


def main():
    """Entry point for self-play training."""
    import argparse
    parser = argparse.ArgumentParser(description="Hive self-play training")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", type=str, default="model.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--parallel", type=int, default=8)
    args = parser.parse_args()

    trainer = SelfPlayTrainer(
        model_path=args.model, device=args.device,
        num_blocks=args.blocks, channels=args.channels
    )
    trainer.run(
        num_iterations=args.iterations, games_per_iter=args.games,
        simulations=args.simulations, epochs_per_iter=args.epochs,
        batch_size=args.batch_size, num_parallel=args.parallel
    )


if __name__ == "__main__":
    main()
