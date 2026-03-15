"""Self-play training loop for Hive AI."""

from __future__ import annotations
import numpy as np
import os
from typing import Optional

from ..core.game import Game, GameState
from ..core.pieces import PieceColor
from ..encoding.board_encoder import encode_board
from ..encoding.move_encoder import encode_move, POLICY_SIZE
from ..nn.model import HiveNet, create_model, save_model, load_model
from ..nn.training import HiveDataset, Trainer
from ..mcts.mcts import MCTS


class SelfPlayGame:
    """Generate a single self-play game."""

    def __init__(self, mcts: MCTS, temperature: float = 1.0,
                 temp_threshold: int = 30):
        self.mcts = mcts
        self.temperature = temperature
        self.temp_threshold = temp_threshold

    def play(self, max_moves: int = 200, simulations: int = 100) -> list[tuple]:
        """Play a game, return list of (board_tensor, reserve, policy, value_target).

        value_target is filled in after game ends based on outcome.
        """
        game = Game()
        history = []  # (board_tensor, reserve_vector, policy_vector, turn_color)

        for move_num in range(max_moves):
            if game.is_game_over:
                break

            temp = self.temperature if move_num < self.temp_threshold else 0.0

            # Encode current state
            board_tensor, reserve_vector = encode_board(game)

            # Get MCTS policy
            moves, probs = self.mcts.get_policy(game, max_simulations=simulations,
                                                 temperature=temp)

            if not moves:
                game.play_pass()
                continue

            # Build full policy vector
            policy_vector = np.zeros(POLICY_SIZE, dtype=np.float32)
            for (piece, from_pos, to_pos), prob in zip(moves, probs):
                idx = encode_move(piece, from_pos, to_pos)
                if 0 <= idx < POLICY_SIZE:
                    policy_vector[idx] = prob

            history.append((board_tensor, reserve_vector, policy_vector, game.turn_color))

            # Sample a move
            move_idx = np.random.choice(len(moves), p=probs)
            piece, from_pos, to_pos = moves[move_idx]
            game.play_move(piece, from_pos, to_pos)

        # Determine game outcome
        if game.state == GameState.WHITE_WINS:
            outcome = {PieceColor.WHITE: 1.0, PieceColor.BLACK: -1.0}
        elif game.state == GameState.BLACK_WINS:
            outcome = {PieceColor.WHITE: -1.0, PieceColor.BLACK: 1.0}
        else:
            outcome = {PieceColor.WHITE: 0.0, PieceColor.BLACK: 0.0}

        # Fill in value targets
        samples = []
        for board_tensor, reserve_vector, policy_vector, color in history:
            value_target = outcome[color]
            samples.append((board_tensor, reserve_vector, policy_vector, value_target))

        return samples


class SelfPlayTrainer:
    """Full self-play training pipeline."""

    def __init__(self, model_path: str = "model.pt", device: str = "cpu",
                 num_blocks: int = 6, channels: int = 64):
        self.model_path = model_path
        self.device = device
        self.num_blocks = num_blocks
        self.channels = channels

        if os.path.exists(model_path):
            self.model = load_model(model_path, num_blocks, channels)
        else:
            self.model = create_model(num_blocks, channels)

        self.model.to(device)
        self.trainer = Trainer(self.model, device=device)

    def run(self, num_iterations: int = 100, games_per_iter: int = 10,
            simulations: int = 100, epochs_per_iter: int = 5,
            batch_size: int = 64):
        """Run the full training loop."""
        for iteration in range(num_iterations):
            print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

            # Generate self-play games
            dataset = HiveDataset()
            mcts = MCTS(model=self.model, device=self.device)
            sp = SelfPlayGame(mcts)

            for g in range(games_per_iter):
                print(f"  Game {g + 1}/{games_per_iter}...", end=" ", flush=True)
                samples = sp.play(simulations=simulations)
                for bt, rv, pv, vt in samples:
                    dataset.add_sample(bt, rv, pv, vt)
                print(f"{len(samples)} positions")

            print(f"  Total training samples: {len(dataset)}")

            # Train on collected data
            for epoch in range(epochs_per_iter):
                losses = self.trainer.train_epoch(dataset, batch_size=batch_size)
                print(f"  Epoch {epoch + 1}: loss={losses['total_loss']:.4f} "
                      f"(policy={losses['policy_loss']:.4f}, value={losses['value_loss']:.4f})")

            # Save model
            save_model(self.model, self.model_path)
            print(f"  Model saved to {self.model_path}")


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
    args = parser.parse_args()

    trainer = SelfPlayTrainer(
        model_path=args.model, device=args.device,
        num_blocks=args.blocks, channels=args.channels
    )
    trainer.run(
        num_iterations=args.iterations, games_per_iter=args.games,
        simulations=args.simulations, epochs_per_iter=args.epochs,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
