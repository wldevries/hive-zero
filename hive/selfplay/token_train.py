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

from ..nn.training import HiveDataset, Trainer
from ..nn.transformer import (
    HiveTransformer, create_model, export_onnx, save_checkpoint, load_checkpoint,
)
from .token_selfplay import GameResult, play_one_game
from .token_selfplay_concurrent import play_games_concurrent


@dataclass
class TrainConfig:
    name: str = "hive-token"                         # checkpoint dir = models/{name}/
    generations: Optional[int] = None                # None = infinite
    games_per_gen: int = 8
    simulations: int = 100
    play_batch: int = 8
    epochs_per_gen: int = 1
    training_batch_size: int = 64
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


def _ckpt_paths(name: str) -> tuple[str, str, str]:
    """Return (models_dir, latest_ckpt_path, replay_dir)."""
    models_dir = os.path.join("models", name)
    os.makedirs(models_dir, exist_ok=True)
    ckpt = os.path.join(models_dir, "latest.pt")
    replay = os.path.join(models_dir, "replay")
    return models_dir, ckpt, replay


def _load_or_init_model(cfg: TrainConfig, ckpt_path: str) -> tuple[HiveTransformer, int]:
    """Resume from `ckpt_path` if it exists, else build a fresh model."""
    if os.path.exists(ckpt_path):
        model, meta = load_checkpoint(ckpt_path)
        gen = int(meta.get("generation", 0))
        print(f"  Resumed gen={gen} from {ckpt_path}")
        return model, gen
    model = create_model(cfg.model_config)
    return model, 0


def _summarize_outcomes(outcomes: list[str]) -> str:
    def count(*keys):
        return sum(1 for o in outcomes if o in keys)
    return (
        f"W={count('WhiteWins')} "
        f"B={count('BlackWins')} "
        f"D={count('Draw', 'DrawByRepetition')} "
        f"T={count('InProgress')}"
    )


def _print_game_length_stats(results: list[GameResult]) -> None:
    """Game length stats over all games + decisive subset, mirroring the
    legacy CNN trainer's per-gen summary block."""
    if not results:
        return
    lengths = sorted(r.move_count for r in results)
    decisive = sorted(
        r.move_count for r in results
        if r.outcome in ("WhiteWins", "BlackWins")
    )
    print(
        f"  game length:  "
        f"min={lengths[0]}  avg={sum(lengths) / len(lengths):.0f}  "
        f"med={lengths[len(lengths) // 2]}  max={lengths[-1]}",
        end="",
    )
    if decisive:
        d_avg = sum(decisive) / len(decisive)
        d_med = decisive[len(decisive) // 2]
        print(f"   decisive: avg={d_avg:.0f}  med={d_med}")
    else:
        print()


def _print_sample_board(results: list[GameResult]) -> None:
    """Render the final boards of up to 3 representative games. Prefers
    decisive outcomes (one per side) and one draw/timeout for contrast,
    falling back to whatever's available. Multiple samples make it
    obvious whether self-play is actually producing varied games — when
    all games look identical (untrained-net first-gen artifact, or a
    bug like the move-order tie-break that converged everything to the
    same chain), every sample renders the same."""
    if not results:
        return
    seen_uhps: set[str] = set()
    picks: list[GameResult] = []
    # Try one of each outcome class to show variety.
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
    # If we still have room, fill with distinct-UHP games.
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

    # Count distinct game-strings so the user can see at a glance whether
    # self-play is degenerate (all games identical).
    distinct_uhps = len({r.final_uhp for r in results})
    print(f"  sample games ({distinct_uhps} distinct of {len(results)}):")
    for r in picks:
        label = f"{r.outcome} ({r.move_count} moves)"
        print(f"  · {label}")
        for line in r.final_board_render.splitlines():
            print(f"      {line}")


def run_training(cfg: TrainConfig) -> None:
    """Drive the play → train → export → repeat loop until `generations`
    are done or `time_limit_minutes` elapses."""
    models_dir, ckpt_path, replay_dir = _ckpt_paths(cfg.name)
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    model, gen = _load_or_init_model(cfg, ckpt_path)
    model.to(device)
    trainer = Trainer(
        model=model, device=device,
        lr=cfg.lr, weight_decay=cfg.weight_decay,
        grad_clip=cfg.grad_clip, optimizer=cfg.optimizer,
    )
    dataset = HiveDataset(max_size=cfg.buffer_size, buf_dir=replay_dir)
    dataset.augment_symmetry = cfg.augment_symmetry

    t_started = time.time()

    print(
        f"Token self-play training started:\n"
        f"  name={cfg.name}  device={device}  grid_size={cfg.grid_size}\n"
        f"  games/gen={cfg.games_per_gen}  sims={cfg.simulations}  "
        f"batch={cfg.training_batch_size}  epochs={cfg.epochs_per_gen}\n"
        f"  augment_symmetry={cfg.augment_symmetry}  buffer={cfg.buffer_size}"
    )

    while True:
        gen += 1
        if cfg.generations is not None and gen > cfg.generations:
            break
        if cfg.time_limit_minutes is not None:
            if (time.time() - t_started) / 60.0 >= cfg.time_limit_minutes:
                print(f"  Time limit ({cfg.time_limit_minutes:.1f} min) reached")
                break

        print(f"\n=== Generation {gen} ===")

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
            pbar = tqdm(
                total=cfg.games_per_gen * cfg.max_moves,
                desc=f"  gen {gen} selfplay",
                unit="ply",
                dynamic_ncols=True,
            )
            # Track previous total move-count snapshot so we can report
            # ply deltas to tqdm.
            prev_total_plies = [0]

            def _progress(round_idx: int, active: int, batch: int,
                          per_game: list[tuple[int, int]]):
                total_plies = sum(m for (m, _) in per_game)
                # `per_game` lists ACTIVE games only; finished games' ply
                # contributions stay at their last value. Approximate by
                # tracking the running sum we've already credited.
                delta = total_plies - prev_total_plies[0]
                if delta > 0:
                    pbar.update(delta)
                    prev_total_plies[0] = total_plies
                pbar.set_postfix_str(
                    f"round={round_idx} active={active}/{cfg.games_per_gen} "
                    f"batch={batch}",
                    refresh=True,
                )

            results = play_games_concurrent(
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
            )
            # Final tick to fill the bar in case some games didn't fully
            # update (the per-active-game snapshot drops finished games).
            total_plies_final = sum(r.move_count for r in results)
            if total_plies_final > prev_total_plies[0]:
                pbar.update(total_plies_final - prev_total_plies[0])
            pbar.close()
            t_play = time.time() - t0
            total_samples = sum(r.samples_written for r in results)
            outcomes = [r.outcome for r in results]
            print(
                f"  selfplay: {cfg.games_per_gen} games  "
                f"{_summarize_outcomes(outcomes)}  "
                f"+{total_samples} samples  "
                f"({t_play:.1f}s, {total_samples / max(t_play, 1e-3):.0f} samp/s)"
            )
            _print_game_length_stats(results)
            _print_sample_board(results)
            print(f"  buffer: {dataset.raw_size}/{dataset.max_size}")
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
                )
                print(
                    f"  train epoch {ep + 1}/{cfg.epochs_per_gen}: "
                    f"total={stats['total_loss']:.4f}  "
                    f"policy={stats['policy_loss']:.4f}  "
                    f"value={stats['value_loss']:.4f}  "
                    f"aux={stats['aux_loss']:.4f}  "
                    f"(qd={stats.get('qd_loss', 0):.4f} "
                    f"qe={stats.get('qe_loss', 0):.4f} "
                    f"mob={stats.get('mob_loss', 0):.4f})"
                )
        t_train = time.time() - t0
        print(f"  train: {t_train:.1f}s")

        # ---- 5. Save checkpoint -----------------------------------------
        if gen % cfg.checkpoint_every == 0:
            save_checkpoint(model, ckpt_path, generation=gen)
            print(f"  saved {ckpt_path}")

    dataset.close()
    save_checkpoint(model, ckpt_path, generation=gen)
    print(f"\nFinal checkpoint saved to {ckpt_path}")
