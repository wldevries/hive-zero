"""CLI entry point for the Yinsh AI engine."""

import argparse
import json
import os

from shared.lr_scheduler import lr_scheduler_from_string

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_DEFAULT_MODEL_CONFIG_PATH = os.path.join(_REPO_ROOT, "configs", "yinsh", "medium.json")

_FALLBACK_MODEL_CONFIG = {
    "channels": 96,
    "num_blocks": 8,
}


def _load_model_config(path: str | None) -> dict:
    """Load model config from a JSON file, falling back to the repo default."""
    if path is None:
        path = _DEFAULT_MODEL_CONFIG_PATH
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        if path != _DEFAULT_MODEL_CONFIG_PATH:
            raise
        return _FALLBACK_MODEL_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Yinsh AI Engine")
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # train
    t = sub.add_parser("train", help="Run self-play training")
    t.add_argument("--name", type=str, default="yinsh",
                   help="Model name; all paths derived as models/{name}/")
    t.add_argument("--device", type=str, default="cuda")
    t.add_argument("--model-config", type=str, default=None,
                   help="Path to JSON model config (default: configs/yinsh/medium.json)."
                        " Ignored when resuming from an existing checkpoint.")
    t.add_argument("--generations", type=int, default=None)
    t.add_argument("--time-limit", type=float, default=None,
                   help="Training time limit in minutes")
    t.add_argument("--games", type=int, default=16)
    t.add_argument("--simulations", type=int, default=200)
    t.add_argument("--epochs-per-gen", type=int, default=1)
    t.add_argument("--training-batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=0.02)
    t.add_argument("--lr-schedule", type=str, default=None,
                   help="Stepped LR schedule, e.g. '0:0.1,20:0.02'. Overrides --lr.")
    t.add_argument("--replay-window", type=int, default=8)
    t.add_argument("--checkpoint-every", type=int, default=10)
    t.add_argument("--playout-cap-p", type=float, default=0.0)
    t.add_argument("--fast-cap", type=int, default=30)
    t.add_argument("--temperature", type=float, default=1.0)
    t.add_argument("--temp-threshold", type=int, default=20)
    t.add_argument("--c-puct", type=float, default=1.5)
    t.add_argument("--dir-alpha", type=float, default=0.3)
    t.add_argument("--dir-epsilon", type=float, default=0.25)
    t.add_argument("--play-batch-sims", type=int, default=8,
                   help="MCTS rounds per inference batch (× active games)")
    t.add_argument("--comment", type=str, default="")
    t.add_argument("--augment-symmetry", action=argparse.BooleanOptionalAction, default=True,
                   help="Apply Yinsh-valid D6 hex symmetry augmentation in the dataset (default: on)")
    t.add_argument("--use-ort", action="store_true",
                   help="Use Rust-native ORT inference (requires .onnx export)")
    t.add_argument("--value-loss-scale", type=float, default=1.0)
    t.add_argument("--buf-dir", type=str, default=None,
                   help="Override replay buffer directory (default: models/{name}/)")
    t.add_argument("--draw-contempt", type=float, default=0.0,
                   help="Draw contempt: MCTS value = W - L - contempt * D. "
                        "Positive = avoid draws (default: 0.0)")
    t.add_argument("--show-timing", action="store_true",
                   help="Print per-phase wall-clock breakdown (select / encode / "
                        "eval / expand) and ORT eval subphase times each gen.")
    t.add_argument("--forced-playouts", action="store_true",
                   help="Use KataGo-style forced playouts at the root: every legal "
                        "move gets a visit floor proportional to sqrt(P*N). "
                        "Pairs well with low c_puct (default: off)")

    def _opening_moves(s):
        if "-" in s:
            lo, hi = s.split("-", 1)
            return (int(lo), int(hi))
        return int(s)
    t.add_argument("--random-opening-moves", type=_opening_moves, default=0,
                   help="Play N (or N-M for a random range) random moves at the start of "
                        "each game before MCTS takes over")

    # battle
    b = sub.add_parser("battle", help="Pit two models against each other")
    b.add_argument("model1", type=str)
    b.add_argument(
        "model2",
        type=str,
        help="Path to second model checkpoint, or the literal 'alphabeta' to play "
             "against the heuristic alpha-beta bot",
    )
    b.add_argument("--games", type=int, default=20)
    b.add_argument("--simulations", type=int, default=None)
    b.add_argument("--device", type=str, default="cuda")
    b.add_argument("--play-batch-sims", type=int, default=8)
    b.add_argument(
        "--bot-depth",
        type=int,
        default=3,
        help="Alpha-beta search depth when model2 is 'alphabeta' (default: 3)",
    )

    # play
    p = sub.add_parser("play", help="REPL: list valid moves and play interactively")
    p.add_argument("--model", type=str, default=None,
                   help="Optional checkpoint; if omitted no AI is used (unless --bot alphabeta)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--simulations", type=int, default=400)
    p.add_argument("--bot", type=str, default=None, choices=["nn", "alphabeta"],
                   help="Bot to use for 'ai' command. Defaults to 'nn' if --model is given, "
                        "else 'alphabeta'.")
    p.add_argument("--depth", type=int, default=3,
                   help="Search depth for --bot alphabeta (default: 3)")

    args = parser.parse_args()

    if args.command == "train":
        from shared.selfplay_config import MctsConfig, OpeningRandomConfig, PlayoutCapConfig
        from yinsh.selfplay.selfplay import SelfPlayTrainer

        scheduler = None
        if args.lr_schedule:
            scheduler = lr_scheduler_from_string(args.lr_schedule)

        trainer = SelfPlayTrainer(
            name=args.name,
            device=args.device,
            model_config=_load_model_config(args.model_config),
            lr=args.lr,
            lr_scheduler=scheduler,
        )
        mcts = MctsConfig(
            simulations=args.simulations,
            c_puct=args.c_puct,
            dir_alpha=args.dir_alpha,
            dir_epsilon=args.dir_epsilon,
            forced_playouts=args.forced_playouts,
            draw_contempt=args.draw_contempt,
            play_batch_size=args.play_batch_sims,
            temperature=args.temperature,
            temp_threshold=args.temp_threshold,
        )
        playout_cap = PlayoutCapConfig(p=args.playout_cap_p, fast_cap=args.fast_cap)
        opening = OpeningRandomConfig.from_cli_arg(args.random_opening_moves)
        trainer.run(
            mcts=mcts,
            playout_cap=playout_cap,
            opening=opening,
            num_generations=args.generations,
            games_per_gen=args.games,
            epochs_per_gen=args.epochs_per_gen,
            batch_size=args.training_batch_size,
            replay_window=args.replay_window,
            checkpoint_every=args.checkpoint_every,
            time_limit_minutes=args.time_limit,
            comment=args.comment,
            augment_symmetry=args.augment_symmetry,
            use_ort=args.use_ort,
            value_loss_scale=args.value_loss_scale,
            buf_dir=args.buf_dir,
            show_timing=args.show_timing,
        )
    elif args.command == "battle":
        from yinsh.selfplay.battle import run_battle
        run_battle(
            model1_path=args.model1,
            model2_path=args.model2,
            num_games=args.games,
            simulations=args.simulations,
            device=args.device,
            play_batch_size=args.play_batch_sims,
            bot_depth=args.bot_depth,
        )
    elif args.command == "play":
        _run_play(args)
    else:
        parser.print_help()


def _run_play(args):
    """Minimal interactive REPL: list legal moves, accept user input, optionally
    let an AI suggest a move via MCTS or alpha-beta."""
    from engine_zero import YinshGame

    bot = args.bot or ("nn" if args.model else "alphabeta")

    eval_fn = None
    if bot == "nn":
        if not args.model:
            raise SystemExit("--bot nn requires --model")
        import numpy as np
        import torch
        import torch.nn.functional as F

        from yinsh.nn.model import load_checkpoint

        model, _ = load_checkpoint(args.model)
        model.to(args.device).eval()

        def eval_fn(board_np, reserve_np):
            board = torch.from_numpy(np.array(board_np)).to(args.device, dtype=torch.float32)
            reserve = torch.from_numpy(np.array(reserve_np)).to(args.device, dtype=torch.float32)
            with torch.no_grad():
                policy, wdl_logits = model(board, reserve)
            wdl = F.softmax(wdl_logits, dim=1)
            return policy.float().cpu().numpy(), wdl.float().cpu().numpy()

    bot_label = f"alphabeta depth={args.depth}" if bot == "alphabeta" else f"NN MCTS sims={args.simulations}"
    print(f"AI: {bot_label}")

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
        choice = input("Move index, notation, 'ai' for AI move, 'q' to quit: ").strip()
        if choice in ("q", "quit", "exit"):
            return
        if choice == "ai":
            if bot == "alphabeta":
                best = game.best_move_alphabeta(args.depth)
            else:
                best = game.best_move(eval_fn, args.simulations, 1.5)
            print(f"  AI plays: {best}")
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
