"""CLI entry point for the Yinsh AI engine."""

import argparse
import json
import os

from shared.lr_scheduler import lr_scheduler_from_string
from shared.optimizer_defaults import resolve_optimizer_defaults as _resolve_optimizer_defaults

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
    t.add_argument("--asymmetric-contempt", action="store_true", default=False,
                   help="Apply --draw-contempt asymmetrically: each self-play game "
                        "randomly designates one side as the contempt side; the other "
                        "side searches with contempt 0. Per-node UCB scoring keys on "
                        "whose turn it is so the contempt side correctly models opponent "
                        "as playing without draw aversion. Intended to break draw "
                        "lock-in by producing decisive outcomes (default: off).")
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
    t.add_argument("--opening-book", type=str, default=None,
                   help="Path to game_outcomes.csv to enable boardspace opening positions. "
                        "The first --random-opening-moves (max) moves of a sampled human "
                        "game are replayed before MCTS takes over.")
    t.add_argument("--opening-boardspace-dir", type=str, default="games/yinsh/boardspace",
                   help="Directory containing boardspace SGF zip archives "
                        "(default: games/yinsh/boardspace)")
    t.add_argument("--boardspace-frac", type=float, default=1.0,
                   help="Fraction of games using book openings; remainder use "
                        "--random-opening-moves (default: 1.0)")
    t.add_argument("--opening-min-elo", type=float, default=1600.0,
                   help="Minimum ELO for both players in opening book games (default: 1600)")

    # pretrain
    pt = sub.add_parser("pretrain", help="Supervised pre-training from human SGF archives")
    pt.add_argument("--games-csv", default="games/yinsh/boardspace/game_outcomes.csv")
    pt.add_argument("--elo-csv", default="games/yinsh/boardspace/player_elo.csv")
    pt.add_argument("--boardspace-dir", default="games/yinsh/boardspace")
    pt.add_argument("--min-elo", type=float, default=1600.0,
                    help="Minimum ELO for both players (default: 1600)")
    pt.add_argument("--min-games", type=int, default=20,
                    help="Minimum games played for ELO to count (default: 20)")
    pt.add_argument("--model", default="yinsh.pt")
    pt.add_argument("--device", default="cuda")
    pt.add_argument("--model-config", type=str, default=None,
                    help="Path to JSON model config (default: configs/yinsh/medium.json). "
                         "Ignored when resuming from an existing checkpoint.")
    pt.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw"],
                    help="Optimizer kind (default: adamw — typically converges faster on a fixed dataset)")
    pt.add_argument("--lr", type=float, default=None,
                    help="Learning rate (default: 0.005 for sgd, 1e-3 for adamw)")
    pt.add_argument("--weight-decay", type=float, default=None,
                    help="L2 weight decay (default: 1e-4 for sgd, 1e-2 for adamw — adamw decouples decay from lr)")
    pt.add_argument("--lr-warmup-steps", type=int, default=None,
                    help="Linearly ramp lr from 0 to target over the first N optimizer steps "
                         "(default: 0 for sgd, 500 for adamw). Set 0 to disable.")
    pt.add_argument("--epochs", type=int, default=3,
                    help="Full passes over the game list (default: 3)")
    pt.add_argument("--batch-size", type=int, default=256)
    pt.add_argument("--buffer-size", type=int, default=100_000,
                    help="Positions per training chunk (default: 100k)")
    pt.add_argument("--epochs-per-chunk", type=int, default=3,
                    help="SGD epochs per buffer fill (default: 3)")
    pt.add_argument("--checkpoint-dir", default="checkpoints")
    pt.add_argument("--verbose-samples", action="store_true",
                    help="Print skipped moves (useful for diagnosing bad SGFs)")
    pt.add_argument("--augment-symmetry", action=argparse.BooleanOptionalAction, default=True,
                    help="Apply Yinsh-valid D6 symmetry augmentation (default: on)")
    pt.add_argument("--exclude-players", nargs="*", default=["Dumbot"],
                    help="Players to exclude from training data (default: Dumbot)")
    pt.add_argument("--value-loss-scale", type=float, default=1.0)
    pt.add_argument("--val-frac", type=float, default=0.1,
                    help="Fraction of games held out as validation set (deterministic by sgf_name hash). "
                         "After each chunk, val loss is logged and the best-by-val checkpoint is saved "
                         "as <model>.best.pt. Set 0 to disable.")

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
            asymmetric_contempt=args.asymmetric_contempt,
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
            opening_games_csv=args.opening_book,
            opening_boardspace_dir=args.opening_boardspace_dir,
            boardspace_frac=args.boardspace_frac,
            opening_min_elo=args.opening_min_elo,
        )
    elif args.command == "pretrain":
        from yinsh.supervised.pretrain import (
            Pretrainer,
            build_zip_index,
            load_filtered_games,
        )

        print("Loading filtered game list...")
        games = load_filtered_games(
            args.games_csv,
            args.elo_csv,
            min_elo=args.min_elo,
            min_games=args.min_games,
            exclude_players=set(args.exclude_players),
        )
        print(
            f"  {len(games)} qualifying games (ELO≥{args.min_elo}, games≥{args.min_games})"
        )
        print("Indexing zip archives...")
        zip_index = build_zip_index(args.boardspace_dir)
        print(f"  {len(zip_index)} zip files found")
        lr, weight_decay, warmup_steps = _resolve_optimizer_defaults(
            args.optimizer, args.lr, args.weight_decay, args.lr_warmup_steps,
        )
        pretrainer = Pretrainer(
            model_path=args.model,
            device=args.device,
            model_config=_load_model_config(args.model_config),
            lr=lr,
            optimizer=args.optimizer,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
        )
        pretrainer.run(
            games=games,
            zip_index=zip_index,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            epochs_per_chunk=args.epochs_per_chunk,
            checkpoint_dir=args.checkpoint_dir,
            verbose_samples=args.verbose_samples,
            augment_symmetry=args.augment_symmetry,
            value_loss_scale=args.value_loss_scale,
            val_frac=args.val_frac,
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
