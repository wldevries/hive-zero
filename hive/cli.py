"""CLI entry point for the Hive AI engine."""

import argparse
import json
import os

from shared.lr_scheduler import lr_scheduler_from_string
from shared.optimizer_defaults import resolve_optimizer_defaults as _resolve_optimizer_defaults

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
_DEFAULT_MODEL_CONFIG_PATH = os.path.join(_REPO_ROOT, "configs", "hive", "medium.json")

_FALLBACK_MODEL_CONFIG = {
    "channels": 64,
    "grid_size": 17,
    "trunk": [{"type": "res", "count": 6}],
    "bilinear_dim": 32,
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


def _resolve_device(device: str) -> str:
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def main():
    parser = argparse.ArgumentParser(description="Hive AI Engine")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # UHP engine (default)
    uhp_parser = subparsers.add_parser("uhp", help="Run UHP engine (stdin/stdout)")
    uhp_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model file path (optional; uses random MCTS if omitted)",
    )
    uhp_parser.add_argument(
        "--device", type=str, default="cuda", help="Device: cuda or cpu (default: cuda)"
    )
    uhp_parser.add_argument(
        "--simulations",
        type=int,
        default=800,
        help="MCTS simulations per move (default: 800)",
    )
    uhp_parser.add_argument(
        "--onnx-model",
        type=str,
        default=None,
        help="ONNX model path for Rust-native ORT/QNN inference (export with --batch-size 1 for QNN)",
    )
    uhp_parser.add_argument(
        "--grid-size",
        type=int,
        default=None,
        help="NN encoding grid size (must be odd; overrides value stored in checkpoint)",
    )

    # Training
    train_parser = subparsers.add_parser("train", help="Run self-play training")
    train_parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Number of training generations (default: infinite)",
    )
    train_parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        help="Training time limit in minutes (stops after current generation)",
    )
    train_parser.add_argument(
        "--games", type=int, default=20, help="Self-play games per generation"
    )
    train_parser.add_argument(
        "--simulations", type=int, default=100, help="MCTS simulations per move"
    )
    train_parser.add_argument(
        "--epochs", type=int, default=1, help="Training epochs per generation"
    )
    train_parser.add_argument("--training-batch-size", type=int, default=512)
    train_parser.add_argument(
        "--name", type=str, default="hive",
        help="Model name; all paths derived as models/{name}/",
    )
    train_parser.add_argument(
        "--device", type=str, default="cuda", help="Device: cuda or cpu"
    )
    train_parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to model config JSON (default: configs/hive/default.json). "
             "Ignored when resuming from an existing checkpoint.",
    )
    train_parser.add_argument(
        "--max-moves", type=int, default=200, help="Max moves per self-play game"
    )
    train_parser.add_argument(
        "--replay-window",
        type=int,
        default=8,
        help="Replay buffer size in generations (buffer = replay-window * games * max-moves)",
    )
    train_parser.add_argument(
        "--playout-cap-p",
        type=float,
        default=0.0,
        help="Playout cap randomization: probability of full search per turn (0=disabled, 0.25=recommended)",
    )
    train_parser.add_argument(
        "--fast-cap",
        type=int,
        default=20,
        help="Simulations for fast-search turns when playout cap is enabled (default: 20)",
    )
    train_parser.add_argument(
        "--forced-playouts",
        action="store_true",
        help="KataGo-style forced playouts at the root: every legal move gets a "
             "visit floor proportional to sqrt(P*N), and forced visits are pruned "
             "from the policy training target. Pairs well with low c_puct (default: off)",
    )
    train_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save checkpoint every N generations (default: 10)",
    )
    train_parser.add_argument(
        "--checkpoint-eval",
        action="store_true",
        help="Run model-vs-best eval at each checkpoint",
    )
    train_parser.add_argument(
        "--play-batch-sims",
        type=int,
        default=1,
        help="Rounds of sim steps to accumulate before an inference call "
        "(batch ≈ N × active games). Default 1 = flush every round.",
    )
    train_parser.add_argument(
        "--fixed-batch-size",
        type=int,
        default=None,
        help="Fixed inference batch size in positions (for QNN/NPU). "
        "Every inference call uses exactly this many positions, "
        "chunking larger batches and padding the last chunk with zeros.",
    )
    train_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="MCTS temperature for move selection (default: 1.0)",
    )
    train_parser.add_argument(
        "--temp-threshold",
        type=int,
        default=30,
        help="Move number after which temperature drops to 0 (default: 30)",
    )
    train_parser.add_argument(
        "--c-puct",
        type=float,
        default=1.5,
        help="PUCT exploration constant (default: 1.5)",
    )
    train_parser.add_argument(
        "--dir-alpha",
        type=float,
        default=0.3,
        help="Dirichlet noise alpha (default: 0.3)",
    )
    train_parser.add_argument(
        "--dir-epsilon",
        type=float,
        default=0.25,
        help="Dirichlet noise weight (default: 0.25)",
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate for SGD optimizer (default: restore from checkpoint, or 0.02 if none).",
    )
    train_parser.add_argument(
        "--lr-schedule",
        type=str,
        default=None,
        help="LR schedule as gen:lr pairs. Linear interpolation by default, prefix 's:' for steps, 'l:' for explicit linear. E.g. '0:0.02,100:0.005' or 's:0:0.02,30:0.01'. Overrides --lr.",
    )
    train_parser.add_argument(
        "--value-loss-scale",
        type=float,
        default=1.0,
        help="Scale factor for value loss in combined loss (default: 1.0)",
    )
    train_parser.add_argument(
        "--aux-loss-scale",
        type=float,
        default=1.0,
        help="Scale factor for auxiliary loss (default: 1.0, set to 0 to disable)",
    )
    train_parser.add_argument(
        "--resign-threshold",
        type=float,
        default=-0.97,
        help="Resign when value < threshold for N consecutive moves (default: -0.97)",
    )
    train_parser.add_argument(
        "--resign-min-moves",
        type=int,
        default=20,
        help="Minimum move count before resign can trigger (default: 20)",
    )
    train_parser.add_argument(
        "--comment",
        type=str,
        default="",
        help="Comment to append to every row in the training log",
    )

    def _opening_moves(s):
        if "-" in s:
            lo, hi = s.split("-", 1)
            return (int(lo), int(hi))
        return int(s)

    train_parser.add_argument(
        "--random-opening-moves",
        type=_opening_moves,
        default=0,
        help="Play N (or N-M for a random range) random moves at the start of each game before MCTS",
    )
    train_parser.add_argument(
        "--opening-book",
        type=str,
        default=None,
        help="Path to game_outcomes.csv to enable boardspace opening positions",
    )
    train_parser.add_argument(
        "--opening-boardspace-dir",
        type=str,
        default="games/hive/boardspace",
        help="Directory containing boardspace SGF zip archives (default: games/hive/boardspace)",
    )
    train_parser.add_argument(
        "--boardspace-frac",
        type=float,
        default=1.0,
        help="Fraction of games using book openings; remainder use --random-opening-moves (default: 1.0)",
    )
    train_parser.add_argument(
        "--skip-timeout-games",
        action="store_true",
        help="Exclude positions from games that hit the move limit (timeouts) from training",
    )
    train_parser.add_argument(
        "--use-heuristic",
        action="store_true",
        default=False,
        help="Use heuristic value for draw/timeout training targets",
    )
    train_parser.add_argument(
        "--draw-contempt",
        type=float,
        default=0.0,
        help="Draw contempt: MCTS value = W - L - contempt * D; terminal draws = -contempt. "
             "Positive = avoid draws, negative = seek draws (default: 0.0)",
    )
    train_parser.add_argument(
        "--asymmetric-contempt",
        action="store_true",
        default=False,
        help="Apply --draw-contempt asymmetrically: each self-play game randomly "
             "designates one side as the contempt side; the other side searches with "
             "contempt 0. Within MCTS the bias is applied per-node based on whose turn "
             "it is, so the contempt side correctly models opponent as playing "
             "without draw aversion. Intended to break draw lock-in by producing "
             "decisive outcomes from genuinely asymmetric play (default: off).",
    )
    train_parser.add_argument(
        "--augment-symmetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply random D6 hex symmetry augmentation during training, 12x effective data (default: on)",
    )
    train_parser.add_argument(
        "--use-ort",
        action="store_true",
        help="Use Rust-native ORT inference instead of Python eval (requires .onnx model)",
    )
    train_parser.add_argument(
        "--buf-dir",
        type=str,
        default=None,
        help="Directory for HDF5 replay buffer (default: same directory as --model)",
    )
    train_parser.add_argument(
        "--no-export-sgf",
        dest="export_sgf",
        action="store_false",
        help="Disable per-generation SGF zip export (default: enabled)",
    )
    train_parser.set_defaults(export_sgf=True)
    train_parser.add_argument(
        "--opponent-bot",
        choices=["none", "alphabeta"],
        default="none",
        help="Opponent type for self-play. 'alphabeta' makes one randomly-chosen "
             "side play heuristic alphabeta moves (no NN), so the value head sees "
             "decisive games. (default: none = pure MCTS-vs-MCTS)",
    )
    train_parser.add_argument(
        "--bot-depth",
        type=int,
        default=2,
        help="Alphabeta search depth for the bot opponent. 2 is ~15ms/move, 3 is ~460ms/move "
             "on heavy positions; depth 4+ is too slow for self-play (default: 2)",
    )
    train_parser.add_argument(
        "--bot-frac",
        type=float,
        default=0.5,
        help="Fraction of self-play games where one side is the bot. The other (1-frac) "
             "are pure MCTS-vs-MCTS so policy training still sees the self-play distribution "
             "(default: 0.5)",
    )

    # Supervised pre-training
    pretrain_parser = subparsers.add_parser(
        "pretrain", help="Supervised pre-training from human game archives"
    )
    pretrain_parser.add_argument("--games-csv", default="games/game_outcomes.csv")
    pretrain_parser.add_argument("--elo-csv", default="games/player_elo.csv")
    pretrain_parser.add_argument("--boardspace-dir", default="games/hive/boardspace")
    pretrain_parser.add_argument(
        "--min-elo",
        type=float,
        default=1600.0,
        help="Minimum ELO for both players (default: 1600)",
    )
    pretrain_parser.add_argument(
        "--min-games",
        type=int,
        default=20,
        help="Minimum games played for ELO to count (default: 20)",
    )
    pretrain_parser.add_argument("--model", default="model.pt")
    pretrain_parser.add_argument("--device", default="cuda")
    pretrain_parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Path to model config JSON (default: configs/hive/default.json). "
             "Ignored when resuming from an existing checkpoint.",
    )
    pretrain_parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["sgd", "adamw"],
        help="Optimizer kind (default: adamw — typically converges faster on a fixed dataset)",
    )
    pretrain_parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (default: 0.005 for sgd, 1e-3 for adamw)",
    )
    pretrain_parser.add_argument(
        "--weight-decay", type=float, default=None,
        help="L2 weight decay (default: 1e-4 for sgd, 1e-2 for adamw — adamw decouples decay from lr)",
    )
    pretrain_parser.add_argument(
        "--lr-warmup-steps", type=int, default=None,
        help="Linearly ramp lr from 0 to target over the first N optimizer steps "
             "(default: 0 for sgd, 500 for adamw — adamw's running variance estimates "
             "are noisy until they've seen enough batches). Set 0 to disable.",
    )
    pretrain_parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Full passes over the game list (default: 3)",
    )
    pretrain_parser.add_argument("--batch-size", type=int, default=512)
    pretrain_parser.add_argument(
        "--buffer-size",
        type=int,
        default=100_000,
        help="Positions per training chunk (default: 100k)",
    )
    pretrain_parser.add_argument(
        "--epochs-per-chunk",
        type=int,
        default=3,
        help="SGD epochs per buffer fill (default: 3)",
    )
    pretrain_parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Checkpoint every N chunks (default: 10)",
    )
    pretrain_parser.add_argument("--checkpoint-dir", default="checkpoints")
    pretrain_parser.add_argument(
        "--verbose-samples",
        action="store_true",
        help="Print skipped moves (useful for diagnosing bad SGFs)",
    )
    pretrain_parser.add_argument(
        "--augment-symmetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply D6 hex symmetry augmentation during training (default: on)",
    )
    pretrain_parser.add_argument(
        "--exclude-players",
        nargs="*",
        default=["Dumbot"],
        help="Players to exclude from training data (default: Dumbot)",
    )
    pretrain_parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of games held out as validation set (deterministic by sgf_name hash). "
             "After each chunk, val loss is logged and the best-by-val checkpoint is saved as "
             "<model>.best.pt. Set 0 to disable.",
    )

    # Battle: model vs model (parallel) or model vs Mzinga (sequential)
    battle_parser = subparsers.add_parser(
        "battle", help="Pit a model against another model or Mzinga"
    )
    battle_parser.add_argument("model1", type=str, help="Path to model checkpoint")
    battle_parser.add_argument(
        "model2",
        type=str,
        nargs="?",
        default=None,
        help="Path to second model checkpoint (omit when using --mzinga)",
    )
    battle_parser.add_argument(
        "--mzinga",
        action="store_true",
        help="Battle against MzingaEngine instead of a second model",
    )
    battle_parser.add_argument(
        "--mzinga-path",
        type=str,
        default="mzinga/MzingaEngine.exe",
        help="Path to MzingaEngine executable (default: mzinga/MzingaEngine.exe)",
    )
    battle_parser.add_argument(
        "--mzinga-time",
        type=int,
        default=2,
        help="Mzinga search time in seconds per move (default: 2)",
    )
    battle_parser.add_argument(
        "--mzinga-depth",
        type=int,
        default=None,
        help="Mzinga search depth (overrides --mzinga-time)",
    )
    battle_parser.add_argument(
        "--games", type=int, default=100, help="Number of games to play (default: 100)"
    )
    battle_parser.add_argument(
        "--simulations",
        type=int,
        default=None,
        help="MCTS simulations per move (default: read from checkpoint metadata)",
    )
    battle_parser.add_argument(
        "--device", type=str, default="cuda", help="Device: cuda or cpu (default: cuda)"
    )
    battle_parser.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="Max moves per game (default: 200)",
    )
    battle_parser.add_argument(
        "--c-puct",
        type=float,
        default=1.5,
        help="PUCT exploration constant (default: 1.5)",
    )
    battle_parser.add_argument(
        "--play-batch-sims",
        type=int,
        default=1,
        help="MCTS sim rounds per inference call (default: 1)",
    )

    args = parser.parse_args()

    if args.command == "pretrain":
        from hive.supervised.pretrain import (
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
            device=_resolve_device(args.device),
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
            val_frac=args.val_frac,
            augment_symmetry=args.augment_symmetry,
        )

    elif args.command == "battle":
        if args.mzinga:
            if args.model2 is not None:
                parser.error("--mzinga cannot be combined with a model2 argument")
            from hive.selfplay.battle import run_battle_mzinga

            run_battle_mzinga(
                model_path=args.model1,
                mzinga_path=args.mzinga_path,
                mzinga_time=args.mzinga_time,
                mzinga_depth=args.mzinga_depth,
                num_games=args.games,
                simulations=args.simulations,
                device=_resolve_device(args.device),
                max_moves=args.max_moves,
                c_puct=args.c_puct,
            )
        else:
            if args.model2 is None:
                parser.error("battle requires either model2 or --mzinga")
            from hive.selfplay.battle import run_battle

            run_battle(
                model1_path=args.model1,
                model2_path=args.model2,
                num_games=args.games,
                simulations=args.simulations,
                device=_resolve_device(args.device),
                max_moves=args.max_moves,
                c_puct=args.c_puct,
                play_batch_size=args.play_batch_sims,
            )

    elif args.command == "train":
        from shared.selfplay_config import MctsConfig, OpeningRandomConfig, PlayoutCapConfig
        from hive.selfplay.selfplay import SelfPlayTrainer

        if args.resign_threshold > 0:
            parser.error(
                f"--resign-threshold must be negative (e.g. -0.95), got {args.resign_threshold}"
            )

        lr_scheduler = None
        if args.lr_schedule:
            lr_scheduler = lr_scheduler_from_string(args.lr_schedule)

        trainer = SelfPlayTrainer(
            name=args.name,
            device=_resolve_device(args.device),
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
            epochs_per_gen=args.epochs,
            batch_size=args.training_batch_size,
            max_moves=args.max_moves,
            time_limit_minutes=args.time_limit,
            checkpoint_every=args.checkpoint_every,
            checkpoint_eval=args.checkpoint_eval,
            replay_window=args.replay_window,
            fixed_batch_size=args.fixed_batch_size,
            resign_threshold=args.resign_threshold,
            resign_min_moves=args.resign_min_moves,
            opening_games_csv=args.opening_book,
            opening_boardspace_dir=args.opening_boardspace_dir,
            boardspace_frac=args.boardspace_frac,
            skip_timeout_games=args.skip_timeout_games,
            augment_symmetry=args.augment_symmetry,
            comment=args.comment,
            use_ort=args.use_ort,
            value_loss_scale=args.value_loss_scale,
            aux_loss_scale=args.aux_loss_scale,
            use_heuristic=args.use_heuristic,
            buf_dir=args.buf_dir,
            export_sgf=args.export_sgf,
            bot_frac=args.bot_frac if args.opponent_bot == "alphabeta" else 0.0,
            bot_depth=args.bot_depth,
        )
    else:
        # Default: UHP engine
        from hive.uhp.engine import UHPEngine

        model = None
        device = "cuda"
        simulations = getattr(args, "simulations", 800)
        onnx_path = getattr(args, "onnx_model", None)

        if onnx_path is None and hasattr(args, "model") and args.model is not None:
            from hive.nn.model import load_checkpoint

            device = _resolve_device(getattr(args, "device", "cuda"))
            model, _ = load_checkpoint(args.model)
            model.to(device)
            model.eval()

        grid_size = getattr(args, "grid_size", None)
        engine = UHPEngine(model=model, device=device, simulations=simulations,
                           onnx_path=onnx_path, grid_size=grid_size)
        engine.run()


if __name__ == "__main__":
    main()
