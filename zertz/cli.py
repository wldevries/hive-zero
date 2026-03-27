"""CLI entry point for the Zertz AI engine."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Zertz AI Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Self-play training
    train_parser = subparsers.add_parser("train", help="Run self-play training")
    train_parser.add_argument("--model", type=str, default="zertz.pt")
    train_parser.add_argument("--device", type=str, default="cuda")
    train_parser.add_argument("--iterations", type=int, default=None)
    train_parser.add_argument(
        "--time-limit", type=float, default=None, help="Training time limit in minutes"
    )
    train_parser.add_argument("--games", type=int, default=20)
    train_parser.add_argument("--simulations", type=int, default=100)
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--training-batch-size", type=int, default=256)
    train_parser.add_argument("--blocks", type=int, default=6)
    train_parser.add_argument("--channels", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=0.02)
    train_parser.add_argument("--max-moves", type=int, default=40)
    train_parser.add_argument("--replay-window", type=int, default=8)
    train_parser.add_argument("--checkpoint-every", type=int, default=10)
    train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/zertz")
    train_parser.add_argument("--playout-cap-p", type=float, default=0.0)
    train_parser.add_argument("--fast-cap", type=int, default=20)
    train_parser.add_argument("--temp-threshold", type=int, default=30,
                              help="Move number after which temperature drops to 0 (default: 30)")
    train_parser.add_argument(
        "--play-batch-size",
        type=int,
        default=2,
        help="MCTS simulation rounds per GPU inference call. "
        "Actual batch = play_batch_size × active_games.",
    )
    train_parser.add_argument("--comment", type=str, default="")
    train_parser.add_argument("--augment-symmetry", action="store_true",
                              help="Apply random D6 hex symmetry augmentation during training (12x effective data)")

    # Play mode
    play_parser = subparsers.add_parser("play", help="Play against the AI")
    play_parser.add_argument("--model", type=str, default=None, help="Model checkpoint (omit for random AI)")
    play_parser.add_argument("--device", type=str, default="cuda")
    play_parser.add_argument("--simulations", type=int, default=200)
    play_parser.add_argument("--color", type=str, default=None, help="p1 or p2 (default: random)")

    args = parser.parse_args()

    if args.command == "play":
        from zertz.play import run
        run(
            model_path=args.model,
            device=args.device,
            simulations=args.simulations,
            human_color=args.color,
        )
    elif args.command == "train":
        from zertz.selfplay.selfplay import SelfPlayTrainer

        trainer = SelfPlayTrainer(
            model_path=args.model,
            device=args.device,
            num_blocks=args.blocks,
            channels=args.channels,
            lr=args.lr,
            checkpoint_dir=args.checkpoint_dir,
        )
        trainer.run(
            num_iterations=args.iterations,
            games_per_iter=args.games,
            simulations=args.simulations,
            epochs_per_iter=args.epochs,
            batch_size=args.training_batch_size,
            max_moves=args.max_moves,
            replay_window=args.replay_window,
            checkpoint_every=args.checkpoint_every,
            playout_cap_p=args.playout_cap_p,
            fast_cap=args.fast_cap,
            temp_threshold=args.temp_threshold,
            play_batch_size=args.play_batch_size,
            time_limit_minutes=args.time_limit,
            comment=args.comment,
            augment_symmetry=args.augment_symmetry,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
