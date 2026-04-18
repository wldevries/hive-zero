"""CLI entry point for the Yinsh AI engine."""

import argparse

from shared.lr_scheduler import lr_scheduler_from_string


def main():
    parser = argparse.ArgumentParser(description="Yinsh AI Engine")
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # train
    t = sub.add_parser("train", help="Run self-play training")
    t.add_argument("--name", type=str, default="yinsh",
                   help="Model name; all paths derived as models/{name}/")
    t.add_argument("--device", type=str, default="cuda")
    t.add_argument("--generations", type=int, default=None)
    t.add_argument("--time-limit", type=float, default=None,
                   help="Training time limit in minutes")
    t.add_argument("--games", type=int, default=16)
    t.add_argument("--simulations", type=int, default=200)
    t.add_argument("--epochs-per-gen", type=int, default=1)
    t.add_argument("--training-batch-size", type=int, default=256)
    t.add_argument("--blocks", type=int, default=8)
    t.add_argument("--channels", type=int, default=96)
    t.add_argument("--lr", type=float, default=0.02)
    t.add_argument("--lr-schedule", type=str, default=None,
                   help="Stepped LR schedule, e.g. '0:0.1,20:0.02'. Overrides --lr.")
    t.add_argument("--max-moves", type=int, default=400)
    t.add_argument("--replay-window", type=int, default=8)
    t.add_argument("--checkpoint-every", type=int, default=10)
    t.add_argument("--playout-cap-p", type=float, default=0.0)
    t.add_argument("--fast-cap", type=int, default=30)
    t.add_argument("--temperature", type=float, default=1.0)
    t.add_argument("--temp-threshold", type=int, default=20)
    t.add_argument("--c-puct", type=float, default=1.5)
    t.add_argument("--dir-alpha", type=float, default=0.3)
    t.add_argument("--dir-epsilon", type=float, default=0.25)
    t.add_argument("--play-batch-size", type=int, default=8,
                   help="MCTS rounds per inference batch (× active games)")
    t.add_argument("--comment", type=str, default="")
    t.add_argument("--augment-symmetry", action="store_true",
                   help="Apply Yinsh-valid D6 hex symmetry augmentation in the dataset")
    t.add_argument("--use-ort", action="store_true",
                   help="Use Rust-native ORT inference (requires .onnx export)")
    t.add_argument("--value-loss-scale", type=float, default=1.0)
    t.add_argument("--buf-dir", type=str, default=None,
                   help="Override replay buffer directory (default: models/{name}/)")

    # battle
    b = sub.add_parser("battle", help="Pit two models against each other")
    b.add_argument("model1", type=str)
    b.add_argument("model2", type=str)
    b.add_argument("--games", type=int, default=20)
    b.add_argument("--simulations", type=int, default=None)
    b.add_argument("--device", type=str, default="cuda")
    b.add_argument("--max-moves", type=int, default=400)
    b.add_argument("--play-batch-size", type=int, default=8)

    # play
    p = sub.add_parser("play", help="REPL: list valid moves and play interactively")
    p.add_argument("--model", type=str, default=None,
                   help="Optional checkpoint; if omitted no AI is used")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--simulations", type=int, default=400)

    args = parser.parse_args()

    if args.command == "train":
        from yinsh.selfplay.selfplay import SelfPlayTrainer

        scheduler = None
        if args.lr_schedule:
            scheduler = lr_scheduler_from_string(args.lr_schedule)

        trainer = SelfPlayTrainer(
            name=args.name,
            device=args.device,
            num_blocks=args.blocks,
            channels=args.channels,
            lr=args.lr,
            lr_scheduler=scheduler,
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
            use_ort=args.use_ort,
            value_loss_scale=args.value_loss_scale,
            buf_dir=args.buf_dir,
        )
    elif args.command == "battle":
        from yinsh.selfplay.battle import run_battle
        run_battle(
            model1_path=args.model1,
            model2_path=args.model2,
            num_games=args.games,
            simulations=args.simulations,
            device=args.device,
            max_moves=args.max_moves,
            play_batch_size=args.play_batch_size,
        )
    elif args.command == "play":
        _run_play(args)
    else:
        parser.print_help()


def _run_play(args):
    """Minimal interactive REPL: list legal moves, accept user input, optionally
    let the AI suggest a move via MCTS."""
    from engine_zero import YinshGame

    model = None
    if args.model:
        import numpy as np
        import torch

        from yinsh.nn.model import load_checkpoint

        model, _ = load_checkpoint(args.model)
        model.to(args.device).eval()

        def eval_fn(board_np, reserve_np):
            board = torch.from_numpy(np.array(board_np)).to(args.device, dtype=torch.float32)
            reserve = torch.from_numpy(np.array(reserve_np)).to(args.device, dtype=torch.float32)
            with torch.no_grad():
                policy, value = model(board, reserve)
            return policy.float().cpu().numpy(), value.float().cpu().numpy().squeeze(1)
    else:
        eval_fn = None

    game = YinshGame()
    while game.outcome() == "ongoing":
        print(
            f"\n[{game.phase()}] {game.current_player()} to move "
            f"(score W{game.white_score()} - B{game.black_score()}, "
            f"pool {game.markers_in_pool()})"
        )
        moves = game.valid_moves()
        for i, m in enumerate(moves[:30]):
            print(f"  {i:2d}: {m}")
        if len(moves) > 30:
            print(f"  ... ({len(moves) - 30} more)")
        suffix = " (or 'ai' for MCTS suggestion)" if eval_fn else ""
        choice = input(f"Move index, notation, 'q' to quit{suffix}: ").strip()
        if choice in ("q", "quit", "exit"):
            return
        if choice == "ai" and eval_fn is not None:
            best = game.best_move(eval_fn, args.simulations, 1.5)
            print(f"  AI suggests: {best}")
            game.play(best)
            continue
        if choice.isdigit() and 0 <= int(choice) < len(moves):
            game.play(moves[int(choice)])
        else:
            try:
                game.play(choice)
            except Exception as e:
                print(f"  invalid: {e}")
    print(f"\nGame over: {game.outcome()} (W{game.white_score()} - B{game.black_score()})")


if __name__ == "__main__":
    main()
