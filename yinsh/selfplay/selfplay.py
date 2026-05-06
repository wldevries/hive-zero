"""Self-play training pipeline for Yinsh."""

from __future__ import annotations

import os
import statistics
import time
from typing import Optional

import colorama
import numpy as np
import torch
from tqdm import tqdm

colorama.init()
_RESET = colorama.Style.RESET_ALL
_BRIGHT = colorama.Style.BRIGHT
_cg = lambda v: f"{colorama.Fore.GREEN}{_BRIGHT}{v}{_RESET}"
_cy = lambda v: f"{colorama.Fore.YELLOW}{_BRIGHT}{v}{_RESET}"
_cr = lambda v: f"{colorama.Fore.RED}{_BRIGHT}{v}{_RESET}"
_cc = lambda v: f"{colorama.Fore.CYAN}{_BRIGHT}{v}{_RESET}"

from shared.lr_scheduler import LRScheduler
from shared.optimizer_defaults import resolve_resumed_lr
from shared.selfplay_config import MctsConfig, OpeningRandomConfig, PlayoutCapConfig
from shared.training_log import csv_comment

from ..nn.model import (
    _describe_trunk,
    create_model,
    export_onnx,
    load_checkpoint,
    save_checkpoint,
)
from ..nn.training import Trainer, YinshDataset

LOG_HEADER = (
    "gen,epoch,"
    "simulations,games,positions,buffer,"
    "wins_p1,wins_p2,draws,timeouts,"
    "avg_game_len,med_game_len,min_game_len,max_game_len,"
    "loss,policy_loss,value_loss,"
    "lr,duration_s,"
    "mcts_top1_mean,mcts_top1_std,mcts_depth_mean,mcts_depth_std,mcts_moves_mean,mcts_moves_std,"
    "comment\n"
)


def _median(lst):
    return int(statistics.median(lst)) if lst else 0


class SelfPlayTrainer:
    """Self-play trainer orchestrating Rust self-play + Python NN training."""

    def __init__(
        self,
        name: str = "yinsh",
        device: str = "cuda",
        model_config: Optional[dict] = None,
        lr: Optional[float] = None,
        lr_scheduler: Optional[LRScheduler] = None,
    ):
        self.name = name
        self.model_dir = os.path.join("models", name)
        self.model_path = os.path.join(self.model_dir, f"{name}.pt")
        self.onnx_path = os.path.join(self.model_dir, f"{name}.onnx")
        self.log_path = os.path.join(self.model_dir, f"{name}_log.csv")
        self.checkpoint_dir = os.path.join(self.model_dir, "checkpoints")
        self.device = device
        self.start_generation = 0

        ckpt: dict = {}
        if os.path.exists(self.model_path):
            self.model, ckpt = load_checkpoint(self.model_path)
            self.start_generation = ckpt.get("generation", 0)
            ch = self.model.input_conv.out_channels
            trunk_desc = _describe_trunk(self.model.trunk_spec)
            params = sum(p.numel() for p in self.model.parameters())
            print(
                f"Resumed from {self.model_path} (gen {self.start_generation}, "
                f"{ch}ch, {trunk_desc}, {params / 1e6:.2f}M params)"
            )
        else:
            os.makedirs(self.model_dir, exist_ok=True)
            self.model = create_model(model_config)
            ch = self.model.input_conv.out_channels
            trunk_desc = _describe_trunk(self.model.trunk_spec)
            params = sum(p.numel() for p in self.model.parameters())
            print(
                f"New model '{name}': {ch}ch, {trunk_desc}, "
                f"{params / 1e6:.2f}M params"
            )

        self.model.to(device)
        self.lr_scheduler = lr_scheduler
        opt_state = ckpt.get("optimizer_state_dict")
        effective_lr, lr_source = resolve_resumed_lr(lr, opt_state)
        self.trainer = Trainer(model=self.model, device=device, lr=effective_lr)
        if opt_state is not None:
            try:
                self.trainer.optimizer.load_state_dict(opt_state)
                for pg in self.trainer.optimizer.param_groups:
                    pg["lr"] = effective_lr
                print(f"  Restored optimizer state (momentum buffers); lr={effective_lr:g} from {lr_source}")
            except (ValueError, KeyError) as e:
                print(f"  Skipped optimizer state (incompatible: {e})")

    def _load_opening_book(
        self,
        games_csv: str,
        boardspace_dir: str,
        min_elo: float = 1600.0,
    ) -> list[list[str]]:
        """Load boardspace YINSH game move sequences for use as opening positions.

        Returns a list of move-string lists (one per game) in canonical Yinsh
        notation (`E5`, `E5 G5`, `Claim E5 E6 E7 E8 E9 D3`). Short games (< 12
        moves) are skipped.
        """
        from engine_zero import parse_yinsh_sgf_moves
        from tqdm import tqdm

        from ..supervised.pretrain import build_zip_index, load_filtered_games

        elo_csv = os.path.join(os.path.dirname(games_csv) or ".", "player_elo.csv")
        games = load_filtered_games(games_csv, elo_csv, min_elo=min_elo, min_games=20)
        zip_index = build_zip_index(boardspace_dir)

        import zipfile
        sequences: list[list[str]] = []
        errors = 0
        for zip_file, sgf_name, _result, _p0, _p1 in tqdm(
            games, desc="Loading opening book", unit="game"
        ):
            zip_path = zip_index.get(zip_file)
            if not zip_path:
                continue
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    content = zf.read(sgf_name).decode("iso-8859-1")
                moves = parse_yinsh_sgf_moves(content)
                if len(moves) >= 12:
                    sequences.append(moves)
            except Exception:
                errors += 1

        print(f"  Opening book: {len(sequences)} games loaded ({errors} errors)")
        return sequences

    def _eval_fn(self, board_tensor_np, reserve_np):
        """Python NN eval callback for the Rust self-play loop.

        Rust passes `(board[N,9,11,11], reserve[N,6])`; we return
        `(policy[N,7139], wdl[N,3])` as float32 numpy arrays.
        """
        board = torch.from_numpy(np.array(board_tensor_np)).to(
            self.device, dtype=torch.float32
        )
        reserve = torch.from_numpy(np.array(reserve_np)).to(
            self.device, dtype=torch.float32
        )
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda" if self.device != "cpu" else "cpu",
                dtype=torch.bfloat16,
            ):
                policy, wdl_logits = self.model(board, reserve)
            wdl = torch.softmax(wdl_logits, dim=1)
        return (
            policy.float().cpu().numpy(),
            wdl.float().cpu().numpy(),
        )

    def run(
        self,
        mcts: MctsConfig,
        playout_cap: PlayoutCapConfig = PlayoutCapConfig(),
        opening: OpeningRandomConfig = OpeningRandomConfig(),
        num_generations: Optional[int] = None,
        games_per_gen: int = 16,
        epochs_per_gen: int = 1,
        batch_size: int = 256,
        replay_window: int = 8,
        checkpoint_every: int = 10,
        time_limit_minutes: Optional[float] = None,
        comment: str = "",
        augment_symmetry: bool = True,
        use_ort: bool = False,
        value_loss_scale: float = 1.0,
        buf_dir: Optional[str] = None,
        show_timing: bool = False,
        opening_games_csv: Optional[str] = None,
        opening_boardspace_dir: Optional[str] = None,
        boardspace_frac: float = 1.0,
        opening_min_elo: float = 1600.0,
    ):
        from engine_zero import YinshSelfPlaySession

        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write(LOG_HEADER)

        resolved_buf_dir = buf_dir if buf_dir is not None else self.model_dir
        max_buffer = games_per_gen * 91 * replay_window
        dataset = YinshDataset(max_size=max_buffer, buf_dir=resolved_buf_dir)
        dataset.augment_symmetry = augment_symmetry
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Flat metadata stored in the checkpoint — keep keys stable for any
        # downstream tooling that reads them.
        train_params = {
            "simulations": mcts.simulations,
            "games_per_gen": games_per_gen,
            "epochs_per_gen": epochs_per_gen,
            "batch_size": batch_size,
            "replay_window": replay_window,
            "playout_cap_p": playout_cap.p,
            "fast_cap": playout_cap.fast_cap,
            "temperature": mcts.temperature,
            "temp_threshold": mcts.temp_threshold,
            "c_puct": mcts.c_puct,
            "dir_alpha": mcts.dir_alpha,
            "dir_epsilon": mcts.dir_epsilon,
            "play_batch_size": mcts.play_batch_size,
            "augment_symmetry": augment_symmetry,
            "draw_contempt": mcts.draw_contempt,
            "asymmetric_contempt": mcts.asymmetric_contempt,
            "random_opening_moves": (opening.min, opening.max) if opening.max != opening.min else opening.min,
            "forced_playouts": mcts.forced_playouts,
        }

        # Opening book: load boardspace game sequences if configured.
        opening_book: list[list[str]] = []
        if opening_games_csv and opening_boardspace_dir:
            opening_book = self._load_opening_book(
                opening_games_csv,
                opening_boardspace_dir,
                opening_min_elo,
            )

        start_time = time.time()
        generation = self.start_generation

        while True:
            if (
                num_generations is not None
                and (generation - self.start_generation) >= num_generations
            ):
                break
            if time_limit_minutes is not None:
                if (time.time() - start_time) / 60 >= time_limit_minutes:
                    break

            generation += 1
            gen_start = time.time()

            if self.lr_scheduler is not None:
                scheduled_lr = self.lr_scheduler.get_scheduled_lr(generation)
                if scheduled_lr is not None:
                    for pg in self.trainer.optimizer.param_groups:
                        pg["lr"] = scheduled_lr

            cap_str = (
                f", fast={playout_cap.fast_cap}, cap={int(playout_cap.p * 100)}%"
                if playout_cap.p > 0
                else ""
            )
            rand_str = (
                f"{opening.min}-{opening.max}"
                if opening.min != opening.max
                else str(opening.min)
            )
            opening_enabled = opening.max > 0
            opening_label = ""
            if opening_book:
                opening_label = (
                    f", book={boardspace_frac:.0%} rand={rand_str}"
                    if opening_enabled
                    else f", book={boardspace_frac:.0%}"
                )
            elif opening_enabled:
                opening_label = f", rand={rand_str}"
            print(
                f"\n=== {_cc(self.name)}  Gen {generation}  "
                f"[sims={mcts.simulations}{cap_str}{opening_label}] ==="
            )

            onnx_path = None
            if use_ort:
                onnx_path = self.onnx_path
                if not os.path.exists(onnx_path):
                    print(f"  ORT requested but {onnx_path} not found, exporting...")
                    export_onnx(self.model, onnx_path)

            if not use_ort:
                self.model.eval()

            session = YinshSelfPlaySession(
                num_games=games_per_gen,
                **mcts.session_kwargs(),
                **playout_cap.session_kwargs(),
                **opening.session_kwargs(),
            )

            # Empirical max with the joint ClaimRow encoding: previous max was 91
            # (with RemoveRow + RemoveRing as separate turns). At most 5 rings can
            # come off in a game (winner 3, loser ≤2), and each claim now saves
            # one MCTS decision, so max drops by 5 to 86.
            pbar = tqdm(total=86, unit="turn", desc="  Self-play", leave=False)
            turn = [0]

            def progress_fn(finished, total, active, total_moves):
                turn[0] += 1
                advance = turn[0] - pbar.n
                if advance > 0:
                    pbar.update(advance)
                pbar.set_postfix(active=f"{active}/{total}")

            opening_sequences = (
                _make_opening_sequences(
                    games_per_gen,
                    opening_book,
                    boardspace_frac,
                    opening.min,
                    opening.max,
                )
                if opening_book
                else None
            )

            play_start = time.time()
            if onnx_path:
                result = session.play_games(
                    progress_fn=progress_fn,
                    onnx_path=onnx_path,
                    opening_sequences=opening_sequences,
                )
            else:
                result = session.play_games(
                    self._eval_fn,
                    progress_fn=progress_fn,
                    opening_sequences=opening_sequences,
                )
            play_time = time.time() - play_start
            pbar.update(pbar.total - pbar.n)
            pbar.close()

            p1 = result.wins_p1
            p2 = result.wins_p2
            d = result.draws
            lengths = list(result.game_lengths)

            print(
                f"  Results: white={_cg(f'{p1}')}  black={_cr(f'{p2}')}  "
                f"draw={_cy(f'{d}')}"
            )

            if lengths:
                avg = sum(lengths) / len(lengths)
                med = _median(lengths)
                mn, mx = min(lengths), max(lengths)
                print(
                    f"  Game length:  min=\033[1;37m{mn}\033[0m  "
                    f"avg=\033[1;37m{avg:.1f}\033[0m  "
                    f"med=\033[1;37m{med}\033[0m  max=\033[1;37m{mx}\033[0m"
                )

            ss = result.search_stats
            print(
                f"  MCTS: top-1 {ss.top1_mean:.2f}±{ss.top1_std:.2f}  "
                f"depth {ss.depth_mean:.1f}±{ss.depth_std:.1f}  "
                f"moves {ss.valid_moves_mean:.1f}±{ss.valid_moves_std:.1f}"
            )

            if playout_cap.p > 0:
                fs = result.full_search_turns
                tt = result.total_turns
                pct = 100 * fs / tt if tt > 0 else 0
                print(f"  Playout cap: {fs}/{tt} full-search turns ({pct:.0f}%)")

            buf_start = time.time()
            boards, reserves, policies, values, value_only, phase_flags = (
                result.training_data()
            )
            dataset.add_batch(
                board_tensors=np.array(boards),
                reserve_vectors=np.array(reserves),
                policy_targets=np.array(policies),
                value_targets=np.array(values),
                value_only=list(value_only),
                phase_flags=list(phase_flags),
                generation=generation,
            )
            buf_time = time.time() - buf_start

            n_vo = sum(1 for v in value_only if v)
            pos_per_s = result.num_samples / play_time if play_time > 0 else 0
            print(
                f"  {games_per_gen} games: {result.num_samples} new positions "
                f"({n_vo} value-only) "
                f"(play={play_time:.1f}s, buf={buf_time:.2f}s, {pos_per_s:.0f} pos/s), "
                f"buffer: {len(dataset)}"
            )

            # Per-phase wall-clock breakdown (gated by --show-timing). `other`
            # covers root NN evals, move sampling, training-record bookkeeping,
            # etc. — anything outside the four instrumented phases.
            if show_timing:
                t = result.timing
                t_sum = t.select_s + t.encode_s + t.eval_s + t.expand_s
                t_other = max(0.0, play_time - t_sum)
                denom = play_time if play_time > 0 else 1.0
                print(
                    f"  Phase: select={t.select_s:.1f}s ({100*t.select_s/denom:.0f}%)  "
                    f"encode={t.encode_s:.1f}s ({100*t.encode_s/denom:.0f}%)  "
                    f"eval={t.eval_s:.1f}s ({100*t.eval_s/denom:.0f}%)  "
                    f"expand={t.expand_s:.1f}s ({100*t.expand_s/denom:.0f}%)  "
                    f"other={t_other:.1f}s ({100*t_other/denom:.0f}%)"
                )
                gpu_total = t.eval_input_s + t.eval_run_s + t.eval_extract_s
                if gpu_total > 0.0:
                    evl_denom = t.eval_s if t.eval_s > 0 else 1.0
                    print(
                        f"    eval split: input={t.eval_input_s:.1f}s ({100*t.eval_input_s/evl_denom:.0f}%)  "
                        f"run={t.eval_run_s:.1f}s ({100*t.eval_run_s/evl_denom:.0f}%)  "
                        f"extract={t.eval_extract_s:.1f}s ({100*t.eval_extract_s/evl_denom:.0f}%)"
                    )

            samples = result.sample_boards()
            if samples:
                board_strs = [b for _, b in samples]
                labels = [lbl for lbl, _ in samples]
                rendered = _render_boards_horizontally(board_strs, labels=labels)
                print("\n".join("    " + line for line in rendered.split("\n")))

            avg_gl = f"{sum(lengths) / len(lengths):.1f}" if lengths else ""
            med_gl = str(_median(lengths)) if lengths else ""
            min_gl = str(min(lengths)) if lengths else ""
            max_gl = str(max(lengths)) if lengths else ""

            losses = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}
            lr = self.trainer._current_lr
            for epoch in range(epochs_per_gen):
                losses = self.trainer.train_epoch(
                    dataset, batch_size=batch_size, value_loss_scale=value_loss_scale
                )
                lr = self.trainer._current_lr
                total_s = f"{losses['total_loss']:.4f}"
                policy_s = f"{losses['policy_loss']:.4f}"
                value_s = f"{losses['value_loss']:.4f}"
                print(
                    f"  Epoch {epoch + 1}: loss={_cr(total_s)} "
                    f"(policy={_cy(policy_s)}, value={_cy(value_s)}, lr={lr:.4f})"
                )

                duration = time.time() - gen_start

                with open(self.log_path, "a") as f:
                    f.write(
                        f"{generation},{epoch + 1},"
                        f"{mcts.simulations},{games_per_gen},{result.num_samples},{len(dataset)},"
                        f"{result.wins_p1},{result.wins_p2},{result.draws},{result.timeouts},"
                        f"{avg_gl},{med_gl},{min_gl},{max_gl},"
                        f"{losses['total_loss']:.6f},"
                        f"{losses['policy_loss']:.6f},"
                        f"{losses['value_loss']:.6f},"
                        f"{lr:.6f},{duration:.1f},"
                        f"{ss.top1_mean:.4f},{ss.top1_std:.4f},"
                        f"{ss.depth_mean:.4f},{ss.depth_std:.4f},"
                        f"{ss.valid_moves_mean:.4f},{ss.valid_moves_std:.4f},"
                        f"{csv_comment(comment)}\n"
                    )
                comment = ""

            metadata = {**train_params, "lr": lr}
            save_checkpoint(self.model, self.model_path, generation=generation,
                            metadata=metadata, optimizer=self.trainer.optimizer)
            try:
                export_onnx(self.model, self.onnx_path)
            except Exception as e:
                print(f"  ONNX export failed (non-fatal): {e}")
            print(f"  Model saved: {self.model_path} (gen {generation})")

            if generation % checkpoint_every == 0:
                ckpt_path = os.path.join(
                    self.checkpoint_dir, f"{self.name}_gen{generation:05d}.pt"
                )
                save_checkpoint(self.model, ckpt_path, generation=generation,
                                metadata=metadata, optimizer=self.trainer.optimizer)
                print(f"  Checkpoint saved to {ckpt_path}")

        print(f"\nTraining complete after gen {generation}. Final model: {self.model_path}")


def _make_opening_sequences(
    num_games: int,
    book: list[list[str]],
    boardspace_frac: float,
    opening_moves_min: int,
    opening_moves_max: int,
) -> list[list[str]]:
    """Generate per-game opening sequences from the book.

    `boardspace_frac` of games get a prefix sampled from a random human game;
    the rest get an empty list (Rust falls back to `random_opening_moves` for
    those games). The prefix length is sampled uniformly per game from
    `[opening_moves_min, opening_moves_max]`. Games shorter than
    `opening_moves_min` are excluded from the eligible pool; if the sampled
    length exceeds a chosen game's length, the full game is used.
    """
    import random as _random

    eligible = [g for g in book if len(g) >= opening_moves_min]
    sequences: list[list[str]] = []
    for _ in range(num_games):
        if eligible and _random.random() < boardspace_frac:
            game_moves = _random.choice(eligible)
            n = _random.randint(opening_moves_min, opening_moves_max)
            sequences.append(game_moves[:n])
        else:
            sequences.append([])
    return sequences


def _render_boards_horizontally(
    board_strings: list, labels: list | None = None, sep: str = "   "
) -> str:
    """Render multiple board strings side-by-side, handling ANSI escape codes."""
    import re

    _ANSI = re.compile(r"\033\[[0-9;]*m")

    def visual_len(s: str) -> int:
        return len(_ANSI.sub("", s))

    boards_lines = [b.split("\n") for b in board_strings]
    board_widths = [
        max((visual_len(line) for line in lines), default=0) for lines in boards_lines
    ]

    if labels:
        board_widths = [max(w, visual_len(labels[i])) for i, w in enumerate(board_widths)]

    max_height = max(len(lines) for lines in boards_lines)
    all_lines = []

    if labels:
        all_lines.append(
            sep.join(
                lbl + " " * (board_widths[i] - visual_len(lbl)) for i, lbl in enumerate(labels)
            ).rstrip()
        )

    for row in range(max_height):
        parts = []
        for bi, lines in enumerate(boards_lines):
            if row < len(lines):
                line = lines[row]
                padding = board_widths[bi] - visual_len(line)
                parts.append(line + " " * padding)
            else:
                parts.append(" " * board_widths[bi])
        all_lines.append(sep.join(parts).rstrip())

    return "\n".join(all_lines)
