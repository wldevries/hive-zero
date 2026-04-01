"""CLI entry point for the Hive AI engine."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Hive AI Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # UHP engine (default)
    uhp_parser = subparsers.add_parser("uhp", help="Run UHP engine (stdin/stdout)")
    uhp_parser.add_argument("--model", type=str, default=None,
                            help="Model file path (optional; uses random MCTS if omitted)")
    uhp_parser.add_argument("--device", type=str, default="cuda",
                            help="Device: cuda or cpu (default: cuda)")
    uhp_parser.add_argument("--simulations", type=int, default=800,
                            help="MCTS simulations per move (default: 800)")

    # Training
    train_parser = subparsers.add_parser("train", help="Run self-play training")
    train_parser.add_argument("--generations", type=int, default=None,
                              help="Number of training generations (default: infinite)")
    train_parser.add_argument("--time-limit", type=float, default=None,
                              help="Training time limit in minutes (stops after current generation)")
    train_parser.add_argument("--games", type=int, default=20,
                              help="Self-play games per generation")
    train_parser.add_argument("--simulations", type=int, default=100,
                              help="MCTS simulations per move")
    train_parser.add_argument("--epochs", type=int, default=1,
                              help="Training epochs per generation")
    train_parser.add_argument("--training-batch-size", type=int, default=512)
    train_parser.add_argument("--model", type=str, default="model.pt",
                              help="Model file path")
    train_parser.add_argument("--device", type=str, default="cuda",
                              help="Device: cuda or cpu")
    train_parser.add_argument("--blocks", type=int, default=6,
                              help="Residual blocks in network")
    train_parser.add_argument("--channels", type=int, default=64,
                              help="Channels in network")
    train_parser.add_argument("--grid-size", type=int, default=23,
                              help="NN encoding grid size (must be odd, default 23)")
    train_parser.add_argument("--max-moves", type=int, default=200,
                              help="Max moves per self-play game")
    train_parser.add_argument("--replay-window", type=int, default=8,
                              help="Replay buffer size in generations (buffer = replay-window * games * max-moves)")
    train_parser.add_argument("--playout-cap-p", type=float, default=0.0,
                              help="Playout cap randomization: probability of full search per turn (0=disabled, 0.25=recommended)")
    train_parser.add_argument("--fast-cap", type=int, default=20,
                              help="Simulations for fast-search turns when playout cap is enabled (default: 20)")
    train_parser.add_argument("--checkpoint-every", type=int, default=10,
                              help="Save checkpoint every N generations (default: 10)")
    train_parser.add_argument("--checkpoint-eval", action="store_true",
                              help="Run model-vs-best eval at each checkpoint")
    train_parser.add_argument("--eval-every", type=int, default=0,
                              help="Run evaluation vs Mzinga every N generations (0=disabled)")
    train_parser.add_argument("--eval-games", type=int, default=6,
                              help="Number of evaluation games per eval round")
    train_parser.add_argument("--eval-simulations", type=int, default=200,
                              help="MCTS simulations per move during eval")
    train_parser.add_argument("--mzinga-path", type=str, default="mzinga/MzingaEngine.exe",
                              help="Path to MzingaEngine for evaluation")
    train_parser.add_argument("--play-batch-size", type=int, default=1,
                              help="Rounds of leaf selection to accumulate before a GPU inference "
                                   "call during self-play MCTS. 1 (default) = flush every round "
                                   "(batch ≈ active game count). N > 1 = N × active games per call.")
    train_parser.add_argument("--temperature", type=float, default=1.0,
                              help="MCTS temperature for move selection (default: 1.0)")
    train_parser.add_argument("--temp-threshold", type=int, default=30,
                              help="Move number after which temperature drops to 0 (default: 30)")
    train_parser.add_argument("--c-puct", type=float, default=1.5,
                              help="PUCT exploration constant (default: 1.5)")
    train_parser.add_argument("--dir-alpha", type=float, default=0.3,
                              help="Dirichlet noise alpha (default: 0.3)")
    train_parser.add_argument("--dir-epsilon", type=float, default=0.25,
                              help="Dirichlet noise weight (default: 0.25)")
    train_parser.add_argument("--lr", type=float, default=0.02,
                              help="Learning rate for SGD optimizer (default: 0.02)")
    train_parser.add_argument("--resign-threshold", type=float, default=-0.97,
                              help="Resign when value < threshold for N consecutive moves (default: -0.97)")
    train_parser.add_argument("--resign-min-moves", type=int, default=20,
                              help="Minimum move count before resign can trigger (default: 20)")
    train_parser.add_argument("--comment", type=str, default="",
                              help="Comment to append to every row in the training log")
    train_parser.add_argument("--mzinga-time", type=int, default=2,
                              help="Mzinga search time in seconds per move during eval")
    def _opening_moves(s):
        if '-' in s:
            lo, hi = s.split('-', 1)
            return (int(lo), int(hi))
        return int(s)
    train_parser.add_argument("--random-opening-moves", type=_opening_moves, default=0,
                              help="Play N (or N-M for a random range) random moves at the start of each game before MCTS")
    train_parser.add_argument("--opening-book", type=str, default=None,
                              help="Path to game_outcomes.csv to enable boardspace opening positions")
    train_parser.add_argument("--opening-boardspace-dir", type=str, default="games/hive/boardspace",
                              help="Directory containing boardspace SGF zip archives (default: games/hive/boardspace)")
    train_parser.add_argument("--boardspace-frac", type=float, default=1.0,
                              help="Fraction of games using book openings; remainder use --random-opening-moves (default: 1.0)")
    train_parser.add_argument("--skip-timeout-games", action="store_true",
                              help="Exclude positions from games that hit the move limit (timeouts) from training")
    train_parser.add_argument("--augment-symmetry", action="store_true",
                              help="Apply random D6 hex symmetry augmentation during training (12x effective data)")
    train_parser.add_argument("--use-ort", action="store_true",
                              help="Use Rust-native ORT inference instead of Python eval (requires .onnx model)")

    # Supervised pre-training
    pretrain_parser = subparsers.add_parser(
        "pretrain", help="Supervised pre-training from human game archives"
    )
    pretrain_parser.add_argument("--games-csv", default="games/game_outcomes.csv")
    pretrain_parser.add_argument("--elo-csv", default="games/player_elo.csv")
    pretrain_parser.add_argument("--boardspace-dir", default="games/hive/boardspace")
    pretrain_parser.add_argument("--min-elo", type=float, default=1600.0,
                                 help="Minimum ELO for both players (default: 1600)")
    pretrain_parser.add_argument("--min-games", type=int, default=20,
                                 help="Minimum games played for ELO to count (default: 20)")
    pretrain_parser.add_argument("--model", default="model.pt")
    pretrain_parser.add_argument("--device", default="cuda")
    pretrain_parser.add_argument("--blocks", type=int, default=6)
    pretrain_parser.add_argument("--channels", type=int, default=64)
    pretrain_parser.add_argument("--grid-size", type=int, default=23,
                                 help="NN encoding grid size (must be odd, default 23)")
    pretrain_parser.add_argument("--lr", type=float, default=0.005,
                                 help="Learning rate (default: 0.005)")
    pretrain_parser.add_argument("--epochs", type=int, default=3,
                                 help="Full passes over the game list (default: 3)")
    pretrain_parser.add_argument("--batch-size", type=int, default=512)
    pretrain_parser.add_argument("--buffer-size", type=int, default=100_000,
                                 help="Positions per training chunk (default: 100k)")
    pretrain_parser.add_argument("--epochs-per-chunk", type=int, default=3,
                                 help="SGD epochs per buffer fill (default: 3)")
    pretrain_parser.add_argument("--checkpoint-every", type=int, default=10,
                                 help="Checkpoint every N chunks (default: 10)")
    pretrain_parser.add_argument("--checkpoint-dir", default="checkpoints")
    pretrain_parser.add_argument("--verbose-samples", action="store_true",
                                 help="Print skipped moves (useful for diagnosing bad SGFs)")
    pretrain_parser.add_argument("--exclude-players", nargs="*", default=["Dumbot"],
                                 help="Players to exclude from training data (default: Dumbot)")

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

    if args.command == "pretrain":
        from hive.supervised.pretrain import load_filtered_games, build_zip_index, Pretrainer
        print("Loading filtered game list...")
        games = load_filtered_games(
            args.games_csv, args.elo_csv,
            min_elo=args.min_elo, min_games=args.min_games,
            exclude_players=set(args.exclude_players),
        )
        print(f"  {len(games)} qualifying games (ELO≥{args.min_elo}, games≥{args.min_games})")
        print("Indexing zip archives...")
        zip_index = build_zip_index(args.boardspace_dir)
        print(f"  {len(zip_index)} zip files found")
        pretrainer = Pretrainer(
            model_path=args.model, device=args.device,
            num_blocks=args.blocks, channels=args.channels, lr=args.lr,
            grid_size=args.grid_size,
        )
        pretrainer.run(
            games=games, zip_index=zip_index,
            num_epochs=args.epochs, batch_size=args.batch_size,
            buffer_size=args.buffer_size, epochs_per_chunk=args.epochs_per_chunk,
            checkpoint_every_chunks=args.checkpoint_every,
            checkpoint_dir=args.checkpoint_dir,
            verbose_samples=args.verbose_samples,
        )

    elif args.command == "eval":
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
        if args.resign_threshold > 0:
            parser.error(f"--resign-threshold must be negative (e.g. -0.95), got {args.resign_threshold}")
        from hive.selfplay.selfplay import SelfPlayTrainer
        trainer = SelfPlayTrainer(
            model_path=args.model, device=args.device,
            num_blocks=args.blocks, channels=args.channels, lr=args.lr,
            grid_size=args.grid_size,
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
            num_generations=args.generations, games_per_gen=args.games,
            simulations=args.simulations, epochs_per_gen=args.epochs,
            batch_size=args.training_batch_size, max_moves=args.max_moves,
            time_limit_minutes=args.time_limit,
            eval_config=eval_config,
            checkpoint_every=args.checkpoint_every,
            checkpoint_eval=args.checkpoint_eval,
            playout_cap_p=args.playout_cap_p,
            fast_cap=args.fast_cap,
            replay_window=args.replay_window,
            leaf_batch_size=args.play_batch_size,
            temperature=args.temperature,
            temp_threshold=args.temp_threshold,
            c_puct=args.c_puct,
            dir_alpha=args.dir_alpha,
            dir_epsilon=args.dir_epsilon,
            resign_threshold=args.resign_threshold,
            resign_min_moves=args.resign_min_moves,
            random_opening_moves=args.random_opening_moves,
            opening_games_csv=args.opening_book,
            opening_boardspace_dir=args.opening_boardspace_dir,
            boardspace_frac=args.boardspace_frac,
            skip_timeout_games=args.skip_timeout_games,
            augment_symmetry=args.augment_symmetry,
            comment=args.comment,
            use_ort=args.use_ort,
        )
    else:
        # Default: UHP engine
        from hive.uhp.engine import UHPEngine
        model = None
        device = "cuda"
        simulations = 800
        if hasattr(args, "model") and args.model is not None:
            from hive.nn.model import load_checkpoint
            import torch
            device = args.device if hasattr(args, "device") else "cuda"
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            model, _ = load_checkpoint(args.model)
            model.to(device)
            model.eval()
            simulations = args.simulations if hasattr(args, "simulations") else 800
        engine = UHPEngine(model=model, device=device, simulations=simulations)
        engine.run()


if __name__ == "__main__":
    main()
