"""Entry point for the Hive AI engine."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Hive AI Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # UHP engine (default)
    subparsers.add_parser("uhp", help="Run UHP engine (stdin/stdout)")

    # Training
    train_parser = subparsers.add_parser("train", help="Run self-play training")
    train_parser.add_argument("--iterations", type=int, default=100,
                              help="Number of training iterations")
    train_parser.add_argument("--games", type=int, default=20,
                              help="Self-play games per iteration")
    train_parser.add_argument("--simulations", type=int, default=100,
                              help="MCTS simulations per move")
    train_parser.add_argument("--epochs", type=int, default=1,
                              help="Training epochs per iteration")
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--model", type=str, default="model.pt",
                              help="Model file path")
    train_parser.add_argument("--device", type=str, default="cuda",
                              help="Device: cuda or cpu")
    train_parser.add_argument("--blocks", type=int, default=6,
                              help="Residual blocks in network")
    train_parser.add_argument("--channels", type=int, default=64,
                              help="Channels in network")
    train_parser.add_argument("--max-moves", type=int, default=200,
                              help="Max moves per self-play game")
    train_parser.add_argument("--time-limit", type=float, default=None,
                              help="Training time limit in minutes (stops after current iteration)")
    train_parser.add_argument("--mcts-after", type=int, default=0,
                              help="Skip fast/full cycling and use full MCTS after this iteration (0=disabled)")
    train_parser.add_argument("--fast-iters", type=int, default=10,
                              help="Number of fast iterations per cycle (default: 10)")
    train_parser.add_argument("--full-iters", type=int, default=2,
                              help="Number of full MCTS iterations per cycle (default: 2)")
    train_parser.add_argument("--warmup-positions", type=int, default=10_000,
                              help="Fill buffer to this many positions before training (0=skip)")

    args = parser.parse_args()

    if args.command == "train":
        from hive.selfplay.selfplay import SelfPlayTrainer
        trainer = SelfPlayTrainer(
            model_path=args.model, device=args.device,
            num_blocks=args.blocks, channels=args.channels
        )
        trainer.run(
            num_iterations=args.iterations, games_per_iter=args.games,
            simulations=args.simulations, epochs_per_iter=args.epochs,
            batch_size=args.batch_size, max_moves=args.max_moves,
            time_limit_minutes=args.time_limit,
            mcts_after=args.mcts_after,
            fast_iters=args.fast_iters,
            full_iters=args.full_iters,
            warmup_positions=args.warmup_positions,
        )
    else:
        # Default: UHP engine
        from hive.uhp.engine import UHPEngine
        engine = UHPEngine()
        engine.run()


if __name__ == "__main__":
    main()
