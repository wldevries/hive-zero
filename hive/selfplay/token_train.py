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

from ..nn.training import HiveDataset, Trainer
from ..nn.transformer import (
    HiveTransformer, create_model, export_onnx, save_checkpoint, load_checkpoint,
)
from .token_selfplay import play_one_game


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

            # ---- 3. Self-play N games -----------------------------------
            t0 = time.time()
            outcomes: list[str] = []
            total_samples = 0
            for game_idx in range(cfg.games_per_gen):
                result = play_one_game(
                    eval_fn=None,
                    ort_session=session,
                    dataset=dataset,
                    simulations=cfg.simulations,
                    play_batch=cfg.play_batch,
                    c_puct=cfg.c_puct,
                    dir_alpha=cfg.dir_alpha,
                    dir_epsilon=cfg.dir_epsilon,
                    max_moves=cfg.max_moves,
                    temperature_moves=cfg.temperature_moves,
                    temperature_start=cfg.temperature_start,
                    grid_size=cfg.grid_size,
                    tournament_mode=cfg.tournament_mode,
                    skip_timeout_data=cfg.skip_timeout_data,
                    rng_seed=gen * 10_000 + game_idx,
                )
                outcomes.append(result.outcome)
                total_samples += result.samples_written
            t_play = time.time() - t0
            print(
                f"  selfplay: {cfg.games_per_gen} games  "
                f"{_summarize_outcomes(outcomes)}  "
                f"+{total_samples} samples  "
                f"({t_play:.1f}s, {total_samples / max(t_play, 1e-3):.0f} samp/s)"
            )
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
                    f"aux={stats['aux_loss']:.4f}"
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
