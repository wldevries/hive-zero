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
    train_parser.add_argument("--iterations", type=int, default=None,
                              help="Number of training iterations (default: infinite)")
    train_parser.add_argument("--time-limit", type=float, default=None,
                              help="Training time limit in minutes (stops after current iteration)")
    train_parser.add_argument("--games", type=int, default=20,
                              help="Self-play games per iteration")
    train_parser.add_argument("--simulations", type=int, default=100,
                              help="MCTS simulations per move")
    train_parser.add_argument("--epochs", type=int, default=1,
                              help="Training epochs per iteration")
    train_parser.add_argument("--training-batch-size", type=int, default=512)
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
    train_parser.add_argument("--replay-window", type=int, default=8,
                              help="Replay buffer size in iterations (buffer = replay-window * games * max-moves)")
    train_parser.add_argument("--playout-cap-p", type=float, default=0.0,
                              help="Playout cap randomization: probability of full search per turn (0=disabled, 0.25=recommended)")
    train_parser.add_argument("--fast-cap", type=int, default=20,
                              help="Simulations for fast-search turns when playout cap is enabled (default: 20)")
    train_parser.add_argument("--checkpoint-every", type=int, default=10,
                              help="Save checkpoint every N iterations (default: 10)")
    train_parser.add_argument("--checkpoint-eval", action="store_true",
                              help="Run model-vs-best eval at each checkpoint")
    train_parser.add_argument("--eval-every", type=int, default=0,
                              help="Run evaluation vs Mzinga every N iterations (0=disabled)")
    train_parser.add_argument("--eval-games", type=int, default=6,
                              help="Number of evaluation games per eval round")
    train_parser.add_argument("--eval-simulations", type=int, default=200,
                              help="MCTS simulations per move during eval")
    train_parser.add_argument("--mzinga-path", type=str, default="mzinga/MzingaEngine.exe",
                              help="Path to MzingaEngine for evaluation")
    train_parser.add_argument("--play-batch-size", type=int, default=512,
                              help="Leaf batch size for self-play GPU inference (default: 512)")
    train_parser.add_argument("--lr", type=float, default=0.02,
                              help="Learning rate for SGD optimizer (default: 0.02)")
    train_parser.add_argument("--mzinga-time", type=int, default=2,
                              help="Mzinga search time in seconds per move during eval")

    # Evaluation
    eval_parser = subparsers.add_parser("eval", help="Evaluate model against Mzinga")
    eval_parser.add_argument("--model", type=str, default="model.pt",
                             help="Model file path")
    eval_parser.add_argument("--device", type=str, default="cuda",
                             help="Device: cuda or cpu")
    eval_parser.add_argument("--games", type=int, default=10,
                             help="Number of games to play")
    eval_parser.add_argument("--simulations", type=int, default=800,
                             help="MCTS simulations per move for our engine")
    eval_parser.add_argument("--mzinga-path", type=str, default="mzinga/MzingaEngine.exe",
                             help="Path to MzingaEngine executable")
    eval_parser.add_argument("--mzinga-time", type=int, default=5,
                             help="Mzinga search time in seconds per move")
    eval_parser.add_argument("--mzinga-depth", type=int, default=None,
                             help="Mzinga search depth (overrides --mzinga-time)")
    eval_parser.add_argument("--max-moves", type=int, default=200,
                             help="Max moves per game")
    eval_parser.add_argument("--verbose", action="store_true",
                             help="Print each move")

    args = parser.parse_args()

    if args.command == "eval":
        from hive.eval.engine_match import EngineConfig, UHPProcess, ModelEngine, run_match
        from hive.nn.model import load_checkpoint

        model, _ = load_checkpoint(args.model)
        model.to(args.device)
        model.eval()

        our_engine = ModelEngine(
            model=model, device=args.device,
            simulations=args.simulations, name="HiveZero",
        )

        if args.mzinga_depth is not None:
            bestmove_args = f"depth {args.mzinga_depth}"
        else:
            h = args.mzinga_time // 3600
            m = (args.mzinga_time % 3600) // 60
            s = args.mzinga_time % 60
            bestmove_args = f"time {h:02d}:{m:02d}:{s:02d}"

        import os
        mzinga_path = args.mzinga_path
        if not os.path.isabs(mzinga_path):
            mzinga_path = os.path.join(os.path.dirname(__file__) or '.', mzinga_path)
        mzinga_config = EngineConfig(
            path=mzinga_path,
            bestmove_args=bestmove_args,
        )
        mzinga = UHPProcess(mzinga_config)

        run_match(our_engine, mzinga, num_games=args.games,
                  max_moves=args.max_moves, verbose=args.verbose)

    elif args.command == "train":
        from hive.selfplay.selfplay import SelfPlayTrainer
        trainer = SelfPlayTrainer(
            model_path=args.model, device=args.device,
            num_blocks=args.blocks, channels=args.channels, lr=args.lr
        )
        eval_config = None
        if args.eval_every > 0:
            eval_config = {
                "every": args.eval_every,
                "games": args.eval_games,
                "simulations": args.eval_simulations,
                "mzinga_path": args.mzinga_path,
                "mzinga_time": args.mzinga_time,
            }
        trainer.run(
            num_iterations=args.iterations, games_per_iter=args.games,
            simulations=args.simulations, epochs_per_iter=args.epochs,
            batch_size=args.training_batch_size, max_moves=args.max_moves,
            time_limit_minutes=args.time_limit,
            eval_config=eval_config,
            checkpoint_every=args.checkpoint_every,
            checkpoint_eval=args.checkpoint_eval,
            playout_cap_p=args.playout_cap_p,
            fast_cap=args.fast_cap,
            replay_window=args.replay_window,
            leaf_batch_size=args.play_batch_size,
        )
    else:
        # Default: UHP engine
        from hive.uhp.engine import UHPEngine
        engine = UHPEngine()
        engine.run()


if __name__ == "__main__":
    main()
