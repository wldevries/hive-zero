"""Multi-iteration token self-play training loop.

Drives the standard AlphaZero-style cycle:

    1. Export current model → temporary ONNX file
    2. Load it via HiveOrtSession (Rust-side ORT engine — keeps the
       Python GIL out of the per-batch inference path, which is what
       lets long runs not blow up the host process)
    3. Play N self-play games through the ORT session; per-ply samples
       land in a HiveDataset (HDF5-backed if a buf_dir is configured)
    4. Train K epochs on the buffer
    5. Save checkpoint, log stats, repeat

This is the minimal viable trainer for evaluating the token transformer
direction. It deliberately omits the bells and whistles of the legacy
CNN-side `SelfPlayTrainer` (LR scheduling, multi-process pipelining,
battle vs prior gen, SGF export, opening book, q-mix lambda, resignation,
calibration, alphabeta-bot opponent). Once the architecture proves
itself, those can be ported one at a time.
"""
from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from shared.console import cg as _cg, cy as _cy, cr as _cr, cc as _cc, bold_white

from ..nn.training import HiveDataset, Trainer
from ..nn.transformer import (
    HiveTransformer, create_model, export_onnx, save_checkpoint, load_checkpoint,
)
from .token_selfplay import GameResult, play_one_game
from .token_selfplay_concurrent import play_games_concurrent, SearchStats

# Same column layout as hive/selfplay/selfplay.py's CNN trainer so
# scripts/plot_log.py works on either log without changes.
_LOG_HEADER = (
    "iter,mode,simulations,wins_w,wins_b,"
    "draws,draws_repetition,draws_timeout,draws_real,"
    "resignations,positions,buffer,"
    "loss,policy_loss,value_loss,qd_loss,lr,duration_s,comment,qe_loss,mob_loss,"
    "avg_game_len,med_game_len,avg_decisive_len,med_decisive_len,"
    "mcts_top1_mean,mcts_top1_std,mcts_depth_mean,mcts_depth_std,mcts_moves_mean,mcts_moves_std\n"
)


@dataclass
class TrainConfig:
    name: str = "hive-token"                         # checkpoint dir = models/{name}/
    generations: Optional[int] = None                # None = infinite
    games_per_gen: int = 8
    simulations: int = 100
    play_batch: int = 8
    epochs_per_gen: int = 1
    training_batch_size: int = 64
    num_workers: int = 0
    max_moves: int = 200
    temperature_moves: int = 12
    temperature_start: float = 1.0
    dir_alpha: float = 0.3
    dir_epsilon: float = 0.25
    c_puct: float = 1.5
    lr: float = 0.02
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    optimizer: str = "sgd"
    device: str = "cuda"
    grid_size: int = 23
    tournament_mode: bool = False
    skip_timeout_data: bool = True
    augment_symmetry: bool = True
    buffer_size: int = 50_000
    onnx_precision: str = "fp16"
    model_config: dict = field(default_factory=dict)
    time_limit_minutes: Optional[float] = None
    checkpoint_every: int = 1
    playout_cap_p: float = 0.0
    fast_cap: int = 20
    random_opening_moves_min: int = 0
    random_opening_moves_max: int = 0


def _ckpt_paths(name: str) -> tuple[str, str, str]:
    """Return (models_dir, live_ckpt_path, checkpoint_dir).
    `replay.h5` lives directly in `models_dir`, matching Zertz/Yinsh."""
    models_dir = os.path.join("models", name)
    os.makedirs(models_dir, exist_ok=True)
    ckpt = os.path.join(models_dir, f"{name}.pt")
    checkpoint_dir = os.path.join(models_dir, "checkpoints")
    return models_dir, ckpt, checkpoint_dir


def _load_or_init_model(cfg: TrainConfig, ckpt_path: str) -> tuple[HiveTransformer, int]:
    """Resume from `ckpt_path` if it exists, else build a fresh model."""
    if os.path.exists(ckpt_path):
        model, meta = load_checkpoint(ckpt_path)
        gen = int(meta.get("generation", 0))
        print(f"  Resumed gen={gen} from {ckpt_path}")
        return model, gen
    model = create_model(cfg.model_config)
    return model, 0


def _render_boards_horizontally(
    board_strings: list[str], labels: list[str] | None = None, sep: str = "   "
) -> str:
    """Render multiple board strings side-by-side, handling ANSI escape codes.
    Copied from hive/selfplay/selfplay.py so the token trainer matches the
    legacy CNN per-gen sample-board layout."""
    import re
    _ANSI = re.compile(r"\033\[[0-9;]*m")

    def visual_len(s: str) -> int:
        return len(_ANSI.sub("", s))

    boards_lines = [b.split("\n") for b in board_strings]
    board_widths = [
        max((visual_len(line) for line in lines), default=0) for lines in boards_lines
    ]
    if labels:
        board_widths = [max(w, len(labels[i])) for i, w in enumerate(board_widths)]
    max_height = max(len(lines) for lines in boards_lines)
    out_lines: list[str] = []
    if labels:
        out_lines.append(
            sep.join(lbl.ljust(board_widths[i]) for i, lbl in enumerate(labels))
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
        out_lines.append(sep.join(parts))
    return "\n".join(out_lines)


def _print_results_line(results: list[GameResult]) -> None:
    """One-line outcome summary in CNN-trainer style:
        Results: W=3 B=2 D=3 (rep=1, timeout=2)
    Draws are split into repetition / timeout / real-draw where we can
    distinguish them (`Draw` vs `DrawByRepetition` vs `InProgress`)."""
    def count(*keys):
        return sum(1 for r in results if r.outcome in keys)

    wins_w = count("WhiteWins")
    wins_b = count("BlackWins")
    draws_real = count("Draw")
    draws_rep  = count("DrawByRepetition")
    timeouts   = count("InProgress")
    draws_total = draws_real + draws_rep + timeouts

    parts = [f"W={_cg(wins_w)} B={_cg(wins_b)}"]
    sub: list[str] = []
    if draws_rep:    sub.append(f"rep={_cy(draws_rep)}")
    if timeouts:     sub.append(f"timeout={_cy(timeouts)}")
    if draws_real:   sub.append(f"real={_cy(draws_real)}")
    if sub:
        parts.append(f"D={_cy(draws_total)} ({', '.join(sub)})")
    elif draws_total:
        parts.append(f"D={_cy(draws_total)}")
    print(f"  Results: {' '.join(parts)}")


def _print_game_length_stats(results: list[GameResult]) -> None:
    """Game length stats over all games + decisive subset, in the same
    "Game length: min=… avg=… med=… max=…   decisive: avg=… med=…" form
    used by hive/selfplay/selfplay.py and Zertz."""
    if not results:
        return
    lengths = sorted(r.move_count for r in results)
    decisive = sorted(
        r.move_count for r in results
        if r.outcome in ("WhiteWins", "BlackWins")
    )
    mn = lengths[0]; mx = lengths[-1]
    avg = sum(lengths) / len(lengths)
    med = lengths[len(lengths) // 2]
    print(
        f"  Game length:  min={bold_white(mn)}  avg={bold_white(f'{avg:.0f}')}  "
        f"med={bold_white(med)}  max={bold_white(mx)}",
        end="",
    )
    if decisive:
        d_avg = sum(decisive) / len(decisive)
        d_med = decisive[len(decisive) // 2]
        print(f"   decisive: avg={bold_white(f'{d_avg:.0f}')}  med={bold_white(d_med)}")
    else:
        print()


def _print_mcts_stats(stats: SearchStats) -> None:
    """MCTS top-1 / depth / moves line, matching the Zertz/Yinsh format."""
    print(
        f"  MCTS: top-1 {bold_white(f'{stats.top1_mean:.2f}')}±{bold_white(f'{stats.top1_std:.2f}')}  "
        f"depth {bold_white(f'{stats.depth_mean:.1f}')}±{bold_white(f'{stats.depth_std:.1f}')}  "
        f"moves {bold_white(f'{stats.valid_moves_mean:.1f}')}±{bold_white(f'{stats.valid_moves_std:.1f}')}"
    )


def _print_sample_boards(results: list[GameResult]) -> None:
    """Render up to three representative final boards side-by-side, in the
    same horizontal layout as the CNN trainer (hive/selfplay/selfplay.py:
    `Decisive games (3 most recent)` block)."""
    if not results:
        return
    seen_uhps: set[str] = set()
    picks: list[GameResult] = []
    for outcome_class in (
        ("WhiteWins",),
        ("BlackWins",),
        ("Draw", "DrawByRepetition"),
        ("InProgress",),
    ):
        for r in results:
            if r.outcome in outcome_class and r.final_uhp not in seen_uhps:
                picks.append(r)
                seen_uhps.add(r.final_uhp)
                break
        if len(picks) >= 3:
            break
    if len(picks) < 3:
        for r in results:
            if r.final_uhp in seen_uhps:
                continue
            picks.append(r)
            seen_uhps.add(r.final_uhp)
            if len(picks) >= 3:
                break
    if not picks:
        picks = [results[0]]

    label_for = {
        "WhiteWins": "White wins",
        "BlackWins": "Black wins",
        "Draw": "Draw",
        "DrawByRepetition": "Draw (rep)",
        "InProgress": "Timeout",
    }
    labels = [f"{label_for.get(r.outcome, r.outcome)} ({r.move_count} moves)" for r in picks]
    board_strs = [r.final_board_render for r in picks]
    rendered = _render_boards_horizontally(board_strs, labels=labels)
    print("\n".join("    " + line for line in rendered.split("\n")))


def run_training(cfg: TrainConfig) -> None:
    """Drive the play → train → export → repeat loop until `generations`
    are done or `time_limit_minutes` elapses."""
    models_dir, ckpt_path, checkpoint_dir = _ckpt_paths(cfg.name)
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    model, gen = _load_or_init_model(cfg, ckpt_path)
    model.to(device)
    trainer = Trainer(
        model=model, device=device,
        lr=cfg.lr, weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip, optimizer=cfg.optimizer,
    )
    dataset = HiveDataset(max_size=cfg.buffer_size, buf_dir=models_dir)
    dataset.augment_symmetry = cfg.augment_symmetry

    log_path = os.path.join(models_dir, f"{cfg.name}_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write(_LOG_HEADER)

    t_started = time.time()

    cap_str = (
        f"  playout_cap_p={cfg.playout_cap_p}  fast_cap={cfg.fast_cap}\n"
        if cfg.playout_cap_p > 0 else ""
    )
    opening_str = (
        f"  random_opening_moves={cfg.random_opening_moves_min}-{cfg.random_opening_moves_max}\n"
        if cfg.random_opening_moves_max > 0 else ""
    )
    print(
        f"Token self-play training started:\n"
        f"  name={_cc(cfg.name)}  device={device}  grid_size={cfg.grid_size}\n"
        f"  games/gen={cfg.games_per_gen}  sims={cfg.simulations}  "
        f"batch={cfg.training_batch_size}  epochs={cfg.epochs_per_gen}\n"
        f"  augment_symmetry={cfg.augment_symmetry}  buffer={cfg.buffer_size}\n"
        f"{cap_str}{opening_str}",
        end="",
    )
    print()

    while True:
        gen += 1
        if cfg.generations is not None and gen > cfg.generations:
            break
        if cfg.time_limit_minutes is not None:
            if (time.time() - t_started) / 60.0 >= cfg.time_limit_minutes:
                print(f"  Time limit ({cfg.time_limit_minutes:.1f} min) reached")
                break

        print(
            f"\n=== {_cc(cfg.name)}  Gen {gen}  [sims={cfg.simulations}] ==="
        )

        # ---- 1. Export current model to a temp ONNX for ORT inference ----
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx_path = f.name
        try:
            export_onnx(model, onnx_path, batch_size=None,
                        precision=cfg.onnx_precision)

            # ---- 2. Load ORT session (Rust-side, GIL-free hot loop) ------
            from engine_zero import HiveOrtSession
            session = HiveOrtSession(onnx_path)

            # ---- 3. Self-play N games concurrently with cross-game ------
            # batching. The Rust runner drives all K games through one
            # shared ORT session, batching MCTS leaves from every active
            # game into single inference calls (effective batch size up
            # to K * play_batch). One ply-level tqdm bar updated from a
            # progress callback in the Rust round-loop.
            t0 = time.time()
            prev_count = dataset._count
            pbar = tqdm(
                total=cfg.max_moves,
                desc=f"  Self-play",
                unit="turn",
                dynamic_ncols=True,
                leave=False,
            )

            def _progress(round_idx: int, active: int, batch: int,
                          per_game: list[tuple[int, int]]):
                # `per_game` lists ACTIVE games only. Track the furthest-
                # ahead active game's move count vs max_moves, matching the
                # old CNN/Yinsh "Self-play  N/max_moves [s/turn]" style.
                max_turn = max((m for (m, _) in per_game), default=cfg.max_moves)
                advance = max_turn - pbar.n
                if advance > 0:
                    pbar.update(advance)
                pbar.set_postfix_str(
                    f"active={active}/{cfg.games_per_gen}",
                    refresh=True,
                )

            results, search_stats = play_games_concurrent(
                session,
                dataset,
                num_games=cfg.games_per_gen,
                simulations=cfg.simulations,
                play_batch=cfg.play_batch,
                c_puct=cfg.c_puct,
                draw_contempt=0.0,
                dir_alpha=cfg.dir_alpha,
                dir_epsilon=cfg.dir_epsilon,
                max_moves=cfg.max_moves,
                temperature_moves=cfg.temperature_moves,
                temperature_start=cfg.temperature_start,
                grid_size=cfg.grid_size,
                tournament_mode=cfg.tournament_mode,
                skip_timeout_data=cfg.skip_timeout_data,
                rng_seed=gen * 10_000,
                progress_cb=_progress,
                playout_cap_p=cfg.playout_cap_p,
                fast_cap=cfg.fast_cap,
                random_opening_moves_min=cfg.random_opening_moves_min,
                random_opening_moves_max=cfg.random_opening_moves_max,
            )
            pbar.update(pbar.total - pbar.n)
            pbar.close()
            t_play = time.time() - t0
            total_samples = sum(r.samples_written for r in results)

            _print_results_line(results)
            _print_game_length_stats(results)
            _print_mcts_stats(search_stats)
            pos_per_s = total_samples / t_play if t_play > 0 else 0
            fast_str = (
                f" [cap_p={cfg.playout_cap_p}, fast={cfg.fast_cap}]"
                if cfg.playout_cap_p > 0 else ""
            )
            # Buffer writes happen inline in play_games_concurrent, so we
            # don't break them out here — `play` already includes them.
            print(
                f"  {cfg.games_per_gen} games: {total_samples} new positions{fast_str} "
                f"(play={t_play:.1f}s, {pos_per_s:.0f} pos/s), "
                f"buffer: {dataset.raw_size}/{dataset.max_size}"
            )
            n_new = min(dataset._count - prev_count, dataset.max_size)
            if n_new > 0:
                # Token-count distribution over THIS generation's new samples
                # only (not the whole replay buffer — that would average over
                # many gens of stale data and barely move). Lets us see whether
                # SEQ_LEN is under-/over-provisioned: attention is L² so a cap
                # far above p99 wastes FLOPs, and max==cap means tokens are
                # being silently truncated.
                end = dataset._count % dataset.max_size
                start = (dataset._count - n_new) % dataset.max_size
                if end == 0:
                    gen_mask = dataset.tokens_mask[start:dataset.max_size]
                elif start < end:
                    gen_mask = dataset.tokens_mask[start:end]
                else:
                    gen_mask = np.concatenate([
                        dataset.tokens_mask[start:dataset.max_size],
                        dataset.tokens_mask[:end],
                    ])
                gen_counts = gen_mask.sum(axis=1)
                p50, p95, p99 = np.percentile(gen_counts, [50, 95, 99])
                cap = dataset.tokens_mask.shape[1]
                print(
                    f"  tokens/pos: mean={gen_counts.mean():.0f} "
                    f"p50={int(p50)} p95={int(p95)} p99={int(p99)} "
                    f"max={int(gen_counts.max())} (cap={cap})"
                )
            _print_sample_boards(results)
        finally:
            try:
                os.unlink(onnx_path)
            except OSError:
                pass

        # ---- 4. Train K epochs on the populated buffer -------------------
        if dataset.raw_size == 0:
            print("  No samples in buffer (all games timed out and were skipped)")
            continue

        t0 = time.time()
        with dataset.read_only_mode():
            for ep in range(cfg.epochs_per_gen):
                stats = trainer.train_epoch(
                    dataset, batch_size=cfg.training_batch_size,
                    num_workers=cfg.num_workers,
                )
                total_s  = f"{stats['total_loss']:.4f}"
                policy_s = f"{stats['policy_loss']:.4f}"
                value_s  = f"{stats['value_loss']:.4f}"
                aux_s    = f"{stats['aux_loss']:.4f}"
                qd = stats.get('qd_loss', 0); qe = stats.get('qe_loss', 0); mob = stats.get('mob_loss', 0)
                lr_v = getattr(trainer, "_current_lr", cfg.lr)
                print(
                    f"  Epoch {ep + 1}: loss={_cr(total_s)} "
                    f"(policy={_cy(policy_s)}, value={_cy(value_s)}, aux={_cy(aux_s)} "
                    f"[qd={qd:.4f} qe={qe:.4f} mob={mob:.4f}], "
                    f"lr={lr_v:.4f})"
                )
        t_train = time.time() - t0

        # ---- 5. Append CSV log row (one per generation) ------------------
        wins_w = sum(1 for r in results if r.outcome == "WhiteWins")
        wins_b = sum(1 for r in results if r.outcome == "BlackWins")
        draws_real = sum(1 for r in results if r.outcome == "Draw")
        draws_rep  = sum(1 for r in results if r.outcome == "DrawByRepetition")
        draws_to   = sum(1 for r in results if r.outcome == "InProgress")
        draws_tot  = draws_real + draws_rep + draws_to
        lengths = sorted(r.move_count for r in results)
        decisive = sorted(
            r.move_count for r in results
            if r.outcome in ("WhiteWins", "BlackWins")
        )
        avg_gl = sum(lengths) / len(lengths) if lengths else 0
        med_gl = lengths[len(lengths) // 2] if lengths else 0
        avg_dl = sum(decisive) / len(decisive) if decisive else 0
        med_dl = decisive[len(decisive) // 2] if decisive else 0
        with open(log_path, "a") as f:
            f.write(
                f"{gen},MCTS,{cfg.simulations},"
                f"{wins_w},{wins_b},"
                f"{draws_tot},{draws_rep},{draws_to},{draws_real},"
                f"0,{total_samples},{dataset.raw_size},"
                f"{stats.get('total_loss', 0):.6f},"
                f"{stats.get('policy_loss', 0):.6f},"
                f"{stats.get('value_loss', 0):.6f},"
                f"{stats.get('qd_loss', 0):.6f},"
                f"{lr_v:.8f},{t_play + t_train:.1f},,"
                f"{stats.get('qe_loss', 0):.6f},{stats.get('mob_loss', 0):.6f},"
                f"{avg_gl:.1f},{med_gl},{avg_dl:.1f},{med_dl},"
                f"{search_stats.top1_mean:.4f},{search_stats.top1_std:.4f},"
                f"{search_stats.depth_mean:.2f},{search_stats.depth_std:.2f},"
                f"{search_stats.valid_moves_mean:.1f},{search_stats.valid_moves_std:.1f}\n"
            )

        # ---- 6. Save checkpoint -----------------------------------------
        save_checkpoint(model, ckpt_path, generation=gen)
        print(f"  Model saved to {ckpt_path} (train={t_train:.1f}s)")

        if gen % cfg.checkpoint_every == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            archive_path = os.path.join(
                checkpoint_dir, f"{cfg.name}_gen{gen:05d}.pt"
            )
            save_checkpoint(model, archive_path, generation=gen)
            print(f"  Checkpoint archived to {archive_path}")

    dataset.close()
    save_checkpoint(model, ckpt_path, generation=gen)
    print(f"\nFinal checkpoint saved to {ckpt_path}")
