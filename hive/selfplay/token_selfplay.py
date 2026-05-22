"""Token-based self-play for the Hive transformer architecture.

This is the minimal-viable Python self-play loop. Each ply runs MCTS via the
`HiveGame.mcts_run` PyO3 primitive (which drives a single-game token-based
MCTS through the new Python eval callback), records the (root tokens, visit
distribution) sample, plays one move, and continues until the game ends or a
move cap is hit. After the game finishes, the per-ply samples are written to
the replay buffer with appropriate value targets and value/policy masks.

Compared to the legacy CNN-side `RustParallelSelfPlay`, this module is
single-threaded and runs games sequentially. The CNN path's pipelined batch
inference layered onto Rayon-parallel game loops is a substantial speedup
that we'll port in a follow-up — for now the goal is end-to-end correctness
of the token architecture.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import torch

from engine_zero import HiveGame

from ..nn.training import HiveDataset, MAX_LEGAL_MOVES
from ..nn.transformer import (
    SEQ_LEN, F_FLAGS, BILINEAR_DIM, HiveTransformer,
)


# Type alias for the inference callback: takes the 4 token tensors,
# returns (policy, wdl, aux) numpy arrays in the FFI shape.
EvalFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]


@dataclass
class PlySample:
    """Per-ply data captured during MCTS, awaiting end-of-game value target."""
    tokens_categoricals: np.ndarray   # (L, 5) uint8
    tokens_positions: np.ndarray       # (L, 2) int8
    tokens_flags: np.ndarray           # (L, F) float32
    tokens_mask: np.ndarray            # (L,)   bool
    move_src: np.ndarray               # (K,)   uint8
    move_dst: np.ndarray               # (K,)   uint8
    move_probs: np.ndarray             # (K,)   float32
    root_q: float                      # search-improved Q from this ply's mover POV
    mover_is_white: bool               # for sign-flipping the value target
    value_only: bool = False           # fast-cap turn: mask policy loss


@dataclass
class GameResult:
    """Summary of one self-play game."""
    outcome: str           # "WhiteWins" / "BlackWins" / "Draw" / "DrawByRepetition" / "InProgress" (timeout)
    move_count: int
    samples_written: int
    final_uhp: str         # the game's final UHP GameString, for SGF export
    final_board_render: str = ""  # ANSI-coloured hex board snapshot at game end


def _sample_move_index(visit_probs: np.ndarray, temperature: float, rng: random.Random) -> int:
    """Sample a move index from the visit distribution with temperature.

    - `temperature == 0`: argmax (deterministic)
    - `temperature > 0`: sample proportional to `visits[i] ** (1 / T)`
    """
    if visit_probs.size == 0:
        raise ValueError("empty visit distribution")
    if temperature <= 0.0:
        return int(np.argmax(visit_probs))
    p = visit_probs.astype(np.float64)
    if temperature != 1.0:
        # visit_probs are already normalized — recover counts shape via **^(1/T).
        p = np.power(p, 1.0 / temperature)
    s = p.sum()
    if s <= 0.0:
        return int(np.argmax(visit_probs))
    p = p / s
    # Cumulative-prob sampling via the seeded rng for reproducibility.
    r = rng.random()
    acc = 0.0
    for i, v in enumerate(p):
        acc += v
        if r < acc:
            return i
    return int(p.size - 1)


def _outcome_to_value(outcome: str) -> Optional[float]:
    """Convert a game outcome string to White's terminal value in [-1, 1].
    Returns None for non-terminal outcomes (timeout)."""
    if outcome == "WhiteWins":
        return 1.0
    if outcome == "BlackWins":
        return -1.0
    if outcome in ("Draw", "DrawByRepetition"):
        return 0.0
    return None  # InProgress → timeout


def play_one_game(
    eval_fn: EvalFn | None,
    dataset: HiveDataset,
    *,
    ort_session=None,
    simulations: int = 64,
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
    rng_seed: int | None = None,
    on_move: Callable[[int, str, float], None] | None = None,
    playout_cap_p: float = 0.0,
    fast_cap: int = 20,
    random_opening_moves_min: int = 0,
    random_opening_moves_max: int = 0,
) -> GameResult:
    """Play one self-play game and append the per-ply samples to `dataset`.

    - For the first `temperature_moves` plies the chosen move is sampled
      proportional to the MCTS visit count (with `temperature_start`);
      after that, the visit-argmax move is played deterministically.
    - Dirichlet noise is mixed into the root prior every ply for exploration.
    - Timeout games (hit `max_moves` without a decisive outcome) are dropped
      from the replay buffer when `skip_timeout_data=True` — matches the
      `TimeoutTarget::Skip` strategy from the CNN era.
    - `playout_cap_p` > 0 enables KataGo-style playout cap randomization:
      each ply is full search with prob p, else `fast_cap` sims (value-only).
    - `random_opening_moves_min/max` play uniform-random moves at game start.

    Returns a `GameResult` summary for the caller's logging.
    """
    if (eval_fn is None) == (ort_session is None):
        raise ValueError("pass exactly one of `eval_fn` or `ort_session`")

    rng = random.Random(rng_seed)
    game = HiveGame(tournament_mode=tournament_mode, grid_size=grid_size)
    samples: list[PlySample] = []

    # Random opening moves: play uniformly at random before MCTS takes over.
    # valid_moves() returns (piece_str, from_pos, to_pos) tuples; use play_move().
    if random_opening_moves_max > 0:
        n_opening = rng.randint(random_opening_moves_min, random_opening_moves_max)
        for _ in range(n_opening):
            if game.is_game_over or game.move_count >= max_moves:
                break
            valid = game.valid_moves()
            if not valid:
                break
            piece_str, from_pos, to_pos = rng.choice(valid)
            game.play_move(piece_str, from_pos, to_pos)

    while not game.is_game_over and game.move_count < max_moves:
        # Empty valid_moves → forced pass (rare in Hive); skip the sample.
        if not game.valid_moves():
            game.play_pass()
            continue

        # Playout cap randomization: fast turns use fewer sims and no Dirichlet.
        if playout_cap_p > 0.0:
            is_fast = rng.random() >= playout_cap_p
        else:
            is_fast = False
        sims_this_ply = fast_cap if is_fast else simulations
        eff_dir_alpha = 0.0 if is_fast else dir_alpha
        eff_dir_epsilon = 0.0 if is_fast else dir_epsilon

        if ort_session is not None:
            cat, pos, flg, msk, msrc, mdst, mprob, root_q, _best_uhp = game.mcts_run_ort(
                ort_session,
                simulations=sims_this_ply,
                c_puct=c_puct,
                draw_contempt=draw_contempt,
                dir_alpha=eff_dir_alpha,
                dir_epsilon=eff_dir_epsilon,
                play_batch=play_batch,
            )
        else:
            cat, pos, flg, msk, msrc, mdst, mprob, root_q, _best_uhp = game.mcts_run(
                eval_fn,
                simulations=sims_this_ply,
                c_puct=c_puct,
                draw_contempt=draw_contempt,
                dir_alpha=eff_dir_alpha,
                dir_epsilon=eff_dir_epsilon,
                play_batch=play_batch,
            )
        cat = np.asarray(cat); pos = np.asarray(pos)
        flg = np.asarray(flg); msk = np.asarray(msk)
        msrc = np.asarray(msrc); mdst = np.asarray(mdst); mprob = np.asarray(mprob)

        mover_is_white = (game.turn_color == "w")
        samples.append(PlySample(
            tokens_categoricals=cat,
            tokens_positions=pos,
            tokens_flags=flg,
            tokens_mask=msk,
            move_src=msrc,
            move_dst=mdst,
            move_probs=mprob,
            root_q=float(root_q),
            mover_is_white=mover_is_white,
            value_only=is_fast,
        ))

        # Fast turns: argmax (no temperature). Full turns: temperature schedule.
        if is_fast:
            t = 0.0
        else:
            ply = game.move_count
            if temperature_moves <= 0:
                t = temperature_end
            elif ply >= temperature_moves:
                t = temperature_end
            else:
                frac = ply / max(1, temperature_moves)
                t = temperature_start + frac * (temperature_end - temperature_start)
        chosen_idx = _sample_move_index(mprob, t, rng)
        # Need the UHP string for the chosen move. mcts_run only returned
        # `best_uhp` for the argmax; reconstruct chosen move's UHP via the
        # token-slot lookup. encode_tokens_with_moves at the current state
        # gives us the slot → UHP mapping for *this* ply.
        _c, _p, _f, _m, move_list = game.encode_tokens_with_moves()
        chosen_mover = int(msrc[chosen_idx]); chosen_dest = int(mdst[chosen_idx])
        chosen_uhp = None
        for (mover_slot, dest_slot, piece_uhp, from_pos, to_pos) in move_list:
            if mover_slot == chosen_mover and dest_slot == chosen_dest:
                chosen_uhp = game.format_move_uhp(piece_uhp, from_pos, to_pos)
                break
        if chosen_uhp is None:
            raise RuntimeError(
                f"sampled move (mover={chosen_mover}, dest={chosen_dest}) not in legal list"
            )
        if not game.play_move_uhp(chosen_uhp):
            raise RuntimeError(f"failed to play sampled move {chosen_uhp!r}")

        if on_move is not None:
            on_move(game.move_count, chosen_uhp, float(root_q))

    # Determine value target. Real outcomes (decisive or draw) train with z;
    # timeouts are dropped if requested.
    final_state = game.state
    white_value = _outcome_to_value(final_state)
    if white_value is None and skip_timeout_data:
        return GameResult(
            outcome=final_state,
            move_count=game.move_count,
            samples_written=0,
            final_uhp=game.game_string,
            final_board_render=game.render_board(),
        )

    if white_value is None:
        # Timeout but caller wants the data — use heuristic value.
        wv, _ = game.heuristic_value()
        white_value = float(wv)

    written = 0
    for s in samples:
        # Value target is from the mover's POV at that ply.
        if s.mover_is_white:
            z = white_value
        else:
            z = -white_value
        # q-mix target: blend final outcome with the search-improved Q for
        # mid-game positions. Pure z for decisive outcomes is fine; for
        # timeouts the heuristic z is already a blend.
        dataset.add_sample(
            tokens_categoricals=s.tokens_categoricals,
            tokens_positions=s.tokens_positions,
            tokens_flags=s.tokens_flags,
            tokens_mask=s.tokens_mask,
            move_src=s.move_src,
            move_dst=s.move_dst,
            move_probs=s.move_probs,
            value_target=float(z),
            root_q_target=float(s.root_q),
            value_only=s.value_only,
            policy_only=False,
        )
        written += 1

    return GameResult(
        outcome=final_state,
        move_count=game.move_count,
        samples_written=written,
        final_uhp=game.game_string,
        final_board_render=game.render_board(),
    )


def make_torch_eval_fn(model: HiveTransformer, device: str = "cpu") -> EvalFn:
    """Wrap a HiveTransformer into the standard token eval callback shape."""
    model.eval()

    def eval_fn(categoricals, positions, flags, mask):
        cat = torch.from_numpy(np.asarray(categoricals)).to(device=device, dtype=torch.int64)
        pos = torch.from_numpy(np.asarray(positions)).to(device=device, dtype=torch.int64)
        flg = torch.from_numpy(np.asarray(flags)).to(device=device, dtype=torch.float32)
        msk = torch.from_numpy(np.asarray(mask)).to(device=device, dtype=torch.bool)
        with torch.no_grad():
            policy, wdl_logits, aux = model(cat, pos, flg, msk)
            wdl = torch.softmax(wdl_logits, dim=1)
        return (
            policy.float().cpu().numpy().astype(np.float32),
            wdl.cpu().numpy().astype(np.float32),
            aux.cpu().numpy().astype(np.float32),
        )

    return eval_fn
