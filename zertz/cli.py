"""CLI entry point for the Zertz AI engine."""

import argparse
import json
import os

from shared.lr_scheduler import lr_scheduler_from_string

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_DEFAULT_MODEL_CONFIG_PATH = os.path.join(_REPO_ROOT, "configs", "zertz", "medium.json")

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
    parser = argparse.ArgumentParser(description="Zertz AI Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Self-play training
    train_parser = subparsers.add_parser("train", help="Run self-play training")
    train_parser.add_argument("--name", type=str, default="zertz",
                              help="Model name; all paths derived as models/{name}/")
    train_parser.add_argument("--device", type=str, default="cuda")
    train_parser.add_argument("--model-config", type=str, default=None,
                              help="Path to JSON model config (default: configs/zertz/medium.json)."
                                   " Ignored when resuming from an existing checkpoint.")
    train_parser.add_argument("--generations", type=int, default=None)
    train_parser.add_argument(
        "--time-limit", type=float, default=None, help="Training time limit in minutes"
    )
    train_parser.add_argument("--games", type=int, default=20)
    train_parser.add_argument("--simulations", type=int, default=100)
    train_parser.add_argument("--epochs-per-gen", type=int, default=1)
    train_parser.add_argument("--training-batch-size", type=int, default=256)
    train_parser.add_argument(
        "--training-num-workers", type=int, default=0,
        help="DataLoader worker processes for training. 0 = main thread "
             "(default). Workers open the replay h5 'r' in parallel so fetch "
             "overlaps GPU compute; the main process detaches 'r+' for the "
             "training phase and reattaches between generations. Try 2-4.",
    )
    train_parser.add_argument("--lr", type=float, default=None,
                              help="Learning rate (default: restore from checkpoint, or 0.02 if none).")
    train_parser.add_argument(
        "--lr-schedule", type=str, default=None,
        help="Stepped LR schedule as iter:lr pairs, e.g. '0:0.1,20:0.02,40:0.01'. Overrides --lr."
    )
    train_parser.add_argument("--replay-window", type=int, default=8)
    train_parser.add_argument("--checkpoint-every", type=int, default=10)
    train_parser.add_argument(
        "--playout-cap-p", type=float, default=0.0,
        help="KataGo playout cap randomization: probability of a FULL search per turn. "
             "The remaining (1-p) turns use --fast-cap sims and train value only. "
             "KataGo paper §3.1 uses p=0.25 (0=disabled).",
    )
    train_parser.add_argument(
        "--fast-cap", type=int, default=20,
        help="Simulations per fast-search turn under playout cap randomization "
             "(KataGo's `n`; default: 20). Ignored when --playout-cap-p=0.",
    )
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
        "--play-batch-sims",
        type=int,
        default=2,
        help="MCTS simulation rounds per GPU inference call. "
        "Actual batch = play_batch_sims × active_games.",
    )
    train_parser.add_argument("--comment", type=str, default="")
    train_parser.add_argument("--augment-symmetry", action=argparse.BooleanOptionalAction, default=True,
                              help="Apply random D6 hex symmetry augmentation during training, 12x effective data (default: on)")
    train_parser.add_argument("--use-ort", action="store_true",
                              help="Use Rust-native ORT inference instead of Python eval (requires .onnx model)")
    train_parser.add_argument("--value-loss-scale", type=float, default=1.0,
                              help="Scale factor for value loss in combined loss (default: 1.0)")
    train_parser.add_argument("--buf-dir", type=str, default=None,
                              help="Override replay buffer directory (default: models/{name}/)")
    train_parser.add_argument("--forced-playouts", action="store_true",
                              help="KataGo-style forced playouts at the root: every legal move "
                                   "gets a visit floor proportional to sqrt(P*N). Pairs well with "
                                   "low c_puct (default: off)")
    train_parser.add_argument("--draw-contempt", type=float, default=0.0,
                              help="Draw contempt: MCTS value = W - L - contempt * D. "
                                   "Positive = avoid draws (default: 0.0)")

    # Play mode
    play_parser = subparsers.add_parser("play", help="Play against the AI")
    play_parser.add_argument("--model", type=str, default=None, help="Model checkpoint (omit for random AI)")
    play_parser.add_argument("--device", type=str, default="cuda")
    play_parser.add_argument("--simulations", type=int, default=800)
    play_parser.add_argument("--color", type=str, default=None, help="p1 or p2 (default: random)")

    # Battle mode
    battle_parser = subparsers.add_parser("battle", help="Pit two models against each other")
    battle_parser.add_argument("model1", type=str, help="Path to first model checkpoint")
    battle_parser.add_argument(
        "model2",
        type=str,
        help="Path to second model checkpoint, or the literal 'alphabeta' to play "
             "against the heuristic alpha-beta bot",
    )
    battle_parser.add_argument("--games", type=int, default=100, help="Number of games to play (default: 100)")
    battle_parser.add_argument("--simulations", type=int, default=None, help="Simulations per move (default: from checkpoint metadata or 800)")
    battle_parser.add_argument("--device", type=str, default="cuda")
    battle_parser.add_argument("--play-batch-sims", type=int, default=2)
    battle_parser.add_argument(
        "--bot-depth",
        type=int,
        default=3,
        help="Alpha-beta search depth when model2 is 'alphabeta' (default: 3)",
    )
    battle_parser.add_argument(
        "--bot-time-ms",
        type=int,
        default=None,
        help="Per-move wall-clock budget for the alphabeta bot in milliseconds. "
             "Iterative deepening polls the deadline and falls back to the "
             "deepest fully-completed iteration. --bot-depth still caps the "
             "search; whichever expires first wins. Default: no time cap.",
    )
    battle_parser.add_argument(
        "--use-ort",
        action="store_true",
        help="Run the NN through a Rust-native pipelined ORT worker (instead of "
             "PyTorch via Python). Requires an .onnx alongside each .pt "
             "checkpoint. Currently only wired up for model-vs-alphabeta.",
    )

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
        from shared.selfplay_config import MctsConfig, PlayoutCapConfig
        from zertz.selfplay.selfplay import SelfPlayTrainer

        lr_scheduler = None
        if args.lr_schedule:
            lr_scheduler = lr_scheduler_from_string(args.lr_schedule)

        trainer = SelfPlayTrainer(
            name=args.name,
            device=args.device,
            model_config=_load_model_config(args.model_config),
            lr=args.lr,
            lr_scheduler=lr_scheduler,
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
        trainer.run(
            mcts=mcts,
            playout_cap=playout_cap,
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
            training_num_workers=args.training_num_workers,
        )
    elif args.command == "battle":
        from zertz.selfplay.battle import run_battle
        run_battle(
            model1_path=args.model1,
            model2_path=args.model2,
            num_games=args.games,
            simulations=args.simulations,
            device=args.device,
            play_batch_size=args.play_batch_sims,
            bot_depth=args.bot_depth,
            bot_time_ms=args.bot_time_ms,
            use_ort=args.use_ort,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
