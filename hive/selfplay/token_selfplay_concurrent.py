"""Python wrapper around the Rust concurrent token self-play runner.

The Rust function `engine_zero.play_games_concurrent_ort` plays K games
in lockstep with cross-game batching:

  - each round, every active game contributes up to `play_batch` MCTS
    leaves to one shared inference call
  - effective batch size = K × play_batch (much better GPU utilization
    than per-game `mcts_run_ort` which only batches one game's leaves)
  - the round-loop in Rust avoids per-batch Python/GIL trips

This wrapper:
  1. Calls the Rust runner
  2. Writes returned per-ply samples into a `HiveDataset`
  3. Returns the list of per-game outcomes for the trainer's logging
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from engine_zero import play_games_concurrent_ort

from ..nn.training import HiveDataset
from .token_selfplay import GameResult


def _outcome_white_value(outcome: str) -> Optional[float]:
    if outcome == "WhiteWins": return 1.0
    if outcome == "BlackWins": return -1.0
    if outcome in ("Draw", "DrawByRepetition"): return 0.0
    return None  # InProgress = hit max_moves cap


def play_games_concurrent(
    ort_session,
    dataset: HiveDataset,
    *,
    num_games: int,
    simulations: int = 100,
    play_batch: int = 8,
    c_puct: float = 1.5,
    draw_contempt: float = 0.0,
    dir_alpha: float = 0.3,
    dir_epsilon: float = 0.25,
    max_moves: int = 200,
    temperature_moves: int = 12,
    temperature_start: float = 1.0,
    temperature_end: float = 0.0,
    grid_size: int = 23,
    tournament_mode: bool = False,
    skip_timeout_data: bool = True,
    rng_seed: int = 0,
    progress_cb: Callable[[int, int, int, list[tuple[int, int]]], None] | None = None,
) -> list[GameResult]:
    """Drive `num_games` concurrent self-play games through the Rust ORT
    engine and append the per-ply samples to `dataset`. Returns a list of
    `GameResult` summaries in submission order."""
    raw_results = play_games_concurrent_ort(
        ort_session,
        num_games,
        simulations=simulations,
        play_batch=play_batch,
        c_puct=c_puct,
        draw_contempt=draw_contempt,
        dir_alpha=dir_alpha,
        dir_epsilon=dir_epsilon,
        max_moves=max_moves,
        temperature_moves=temperature_moves,
        temperature_start=temperature_start,
        temperature_end=temperature_end,
        grid_size=grid_size,
        tournament_mode=tournament_mode,
        rng_seed=int(rng_seed),
        progress_cb=progress_cb,
    )

    out: list[GameResult] = []
    for r in raw_results:
        outcome = r["outcome"]
        white_value = _outcome_white_value(outcome)
        samples_written = 0

        if white_value is None and skip_timeout_data:
            out.append(GameResult(
                outcome=outcome,
                move_count=int(r["move_count"]),
                samples_written=0,
                final_uhp=r["final_uhp"],
                final_board_render=r["final_board_render"],
            ))
            continue

        if white_value is None:
            # Timeout but caller wants the data — fall back to 0 since the
            # Rust runner doesn't currently surface the per-game heuristic
            # value. Single-game play_one_game does use heuristic_value;
            # we'll wire that across the FFI in a follow-up if it matters.
            white_value = 0.0

        for s in r["samples"]:
            mover_is_white = bool(s["mover_is_white"])
            z = white_value if mover_is_white else -white_value
            dataset.add_sample(
                tokens_categoricals=np.asarray(s["categoricals"]),
                tokens_positions=np.asarray(s["positions"]),
                tokens_flags=np.asarray(s["flags"]),
                tokens_mask=np.asarray(s["mask"]),
                move_src=np.asarray(s["move_src"]),
                move_dst=np.asarray(s["move_dst"]),
                move_probs=np.asarray(s["move_probs"]),
                value_target=float(z),
                root_q_target=float(s["root_q"]),
                value_only=False,
                policy_only=False,
            )
            samples_written += 1

        out.append(GameResult(
            outcome=outcome,
            move_count=int(r["move_count"]),
            samples_written=samples_written,
            final_uhp=r["final_uhp"],
            final_board_render=r["final_board_render"],
        ))

    return out
