"""CLI entry point for the Tic-Tac-Toe AI engine."""

import argparse

from shared.lr_scheduler import lr_scheduler_from_string


def main():
    parser = argparse.ArgumentParser(description="Tic-Tac-Toe AI Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Self-play training
    train_parser = subparsers.add_parser("train", help="Run self-play training")
    train_parser.add_argument("--model", type=str, default="tictactoe.pt")
    train_parser.add_argument("--device", type=str, default="cpu")
    train_parser.add_argument("--generations", type=int, default=None)
    train_parser.add_argument(
        "--time-limit", type=float, default=None, help="Training time limit in minutes"
    )
    train_parser.add_argument("--games", type=int, default=200)
    train_parser.add_argument("--simulations", type=int, default=100)
    train_parser.add_argument("--epochs-per-gen", type=int, default=1)
    train_parser.add_argument("--training-batch-size", type=int, default=64)
    train_parser.add_argument("--blocks", type=int, default=2)
    train_parser.add_argument("--channels", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=0.005)
    train_parser.add_argument(
        "--lr-schedule", type=str, default=None,
        help="Stepped LR schedule as iter:lr pairs, e.g. '0:0.1,20:0.02,40:0.01'. Overrides --lr."
    )
    train_parser.add_argument("--max-moves", type=int, default=9)
    train_parser.add_argument("--replay-window", type=int, default=3)
    train_parser.add_argument("--checkpoint-every", type=int, default=10)
    train_parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/tictactoe")
    train_parser.add_argument("--playout-cap-p", type=float, default=0.0)
    train_parser.add_argument("--fast-cap", type=int, default=10)
    train_parser.add_argument("--temperature", type=float, default=1.0)
    train_parser.add_argument("--temp-threshold", type=int, default=5,
                              help="Move number after which temperature drops to 0 (default: 5)")
    train_parser.add_argument("--c-puct", type=float, default=1.5)
    train_parser.add_argument("--dir-alpha", type=float, default=0.5,
                              help="Dirichlet noise alpha (default: 0.5)")
    train_parser.add_argument("--dir-epsilon", type=float, default=0.25)
    train_parser.add_argument("--comment", type=str, default="")
    train_parser.add_argument("--value-loss-scale", type=float, default=1.0)

    # Play mode
    play_parser = subparsers.add_parser("play", help="Play against the AI")
    play_parser.add_argument("--model", type=str, default=None, help="Model checkpoint")
    play_parser.add_argument("--device", type=str, default="cuda")
    play_parser.add_argument("--simulations", type=int, default=800)
    play_parser.add_argument("--color", type=str, default=None,
                             help="Your color: x/1 for X (first), o/2 for O (second). Default: random")

    args = parser.parse_args()

    if args.command == "play":
        from tictactoe.play import run
        run(
            model_path=args.model,
            device=args.device,
            simulations=args.simulations,
            human_color=args.color,
        )
    elif args.command == "train":
        from tictactoe.selfplay.selfplay import SelfPlayTrainer

        lr_scheduler = None
        if args.lr_schedule:
            lr_scheduler = lr_scheduler_from_string(args.lr_schedule)

        trainer = SelfPlayTrainer(
            model_path=args.model,
            device=args.device,
            num_blocks=args.blocks,
            channels=args.channels,
            lr=args.lr,
            lr_scheduler=lr_scheduler,
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
            time_limit_minutes=args.time_limit,
            comment=args.comment,
            value_loss_scale=args.value_loss_scale,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
