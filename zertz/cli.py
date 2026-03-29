"""CLI entry point for the Zertz AI engine."""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Zertz AI Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Self-play training
    train_parser = subparsers.add_parser("train", help="Run self-play training")
    train_parser.add_argument("--model", type=str, default="zertz.pt")
    train_parser.add_argument("--device", type=str, default="cuda")
    train_parser.add_argument("--generations", type=int, default=None)
    train_parser.add_argument(
        "--time-limit", type=float, default=None, help="Training time limit in minutes"
    )
    train_parser.add_argument("--games", type=int, default=20)
    train_parser.add_argument("--simulations", type=int, default=100)
    train_parser.add_argument("--epochs-per-gen", type=int, default=1)
    train_parser.add_argument("--training-batch-size", type=int, default=256)
    train_parser.add_argument("--blocks", type=int, default=6)
    train_parser.add_argument("--channels", type=int, default=64)
    train_parser.add_argument("--lr", type=float, default=0.02)
    train_parser.add_argument(
        "--lr-schedule", type=str, default=None,
        help="Stepped LR schedule as iter:lr pairs, e.g. '0:0.1,20:0.02,40:0.01'. Overrides --lr."
    )
    train_parser.add_argument("--max-moves", type=int, default=40)
    train_parser.add_argument("--replay-window", type=int, default=8)
    train_parser.add_argument("--checkpoint-every", type=int, default=10)
    train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/zertz")
    train_parser.add_argument("--playout-cap-p", type=float, default=0.0)
    train_parser.add_argument("--fast-cap", type=int, default=20)
    train_parser.add_argument("--temperature", type=float, default=1.0,
                              help="MCTS temperature for move selection (default: 1.0)")
    train_parser.add_argument("--temp-threshold", type=int, default=10,
                              help="Move number after which temperature drops to 0 (default: 10)")
    train_parser.add_argument("--c-puct", type=float, default=1.5,
                              help="PUCT exploration constant (default: 1.5)")
    train_parser.add_argument("--dir-alpha", type=float, default=0.3,
                              help="Dirichlet noise alpha (default: 0.3)")
    train_parser.add_argument("--dir-epsilon", type=float, default=0.25,
                              help="Dirichlet noise weight (default: 0.25)")
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
    play_parser.add_argument("--simulations", type=int, default=800)
    play_parser.add_argument("--color", type=str, default=None, help="p1 or p2 (default: random)")

    # Battle mode
    battle_parser = subparsers.add_parser("battle", help="Pit two models against each other")
    battle_parser.add_argument("model1", type=str, help="Path to first model checkpoint")
    battle_parser.add_argument("model2", type=str, help="Path to second model checkpoint")
    battle_parser.add_argument("--games", type=int, default=100, help="Number of games to play (default: 100)")
    battle_parser.add_argument("--simulations", type=int, default=None, help="Simulations per move (default: from checkpoint metadata or 800)")
    battle_parser.add_argument("--device", type=str, default="cuda")
    battle_parser.add_argument("--max-moves", type=int, default=40)
    battle_parser.add_argument("--play-batch-size", type=int, default=2)

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

        lr_schedule = None
        if args.lr_schedule:
            lr_schedule = []
            for pair in args.lr_schedule.split(","):
                it, lr_val = pair.split(":")
                lr_schedule.append((int(it), float(lr_val)))
            lr_schedule.sort()

        trainer = SelfPlayTrainer(
            model_path=args.model,
            device=args.device,
            num_blocks=args.blocks,
            channels=args.channels,
            lr=args.lr,
            lr_schedule=lr_schedule,
            checkpoint_dir=args.checkpoint_dir,
        )
        trainer.run(
            num_generations=args.generations,
            games_per_gen=args.games,
            simulations=args.simulations,
            epochs_per_gen=args.epochs_per_gen,
            batch_size=args.training_batch_size,
            max_moves=args.max_moves,
            replay_window=args.replay_window,
            checkpoint_every=args.checkpoint_every,
            playout_cap_p=args.playout_cap_p,
            fast_cap=args.fast_cap,
            temperature=args.temperature,
            temp_threshold=args.temp_threshold,
            c_puct=args.c_puct,
            dir_alpha=args.dir_alpha,
            dir_epsilon=args.dir_epsilon,
            play_batch_size=args.play_batch_size,
            time_limit_minutes=args.time_limit,
            comment=args.comment,
            augment_symmetry=args.augment_symmetry,
        )
    elif args.command == "battle":
        from zertz.selfplay.battle import run_battle
        run_battle(
            model1_path=args.model1,
            model2_path=args.model2,
            num_games=args.games,
            simulations=args.simulations,
            device=args.device,
            max_moves=args.max_moves,
            play_batch_size=args.play_batch_size,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
