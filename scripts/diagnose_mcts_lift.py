"""Diagnose whether MCTS at this checkpoint produces stronger play than raw policy.

Same network on both sides. One side uses MCTS at the configured sim count; the
other picks the highest-scoring legal move from the raw policy logits with no
search at all (mirroring the Rust MCTS prior scoring at
core-game/src/mcts/search.rs:333-346).

If MCTS does not clearly outperform raw policy at the current checkpoint, the
AlphaZero policy-improvement step has no signal — visit-distribution targets
are no better than the network's prior, and gradient toward them is noise.
That would explain post-pretraining degradation and from-scratch stagnation.

Usage:
    uv run python scripts/diagnose_mcts_lift.py models/hive_medium2/hive_medium2.pt
    uv run python scripts/diagnose_mcts_lift.py PATH --games 50 --simulations 800
"""

import argparse
import time

import numpy as np
import torch

from engine_zero import HiveGame
from hive.nn.model import load_checkpoint
from hive.encoding.move_encoder import (
    BILINEAR_DIM,
    q_section_offset,
    k_section_offset,
)


def make_eval_fn(model, device):
    """eval_fn for HiveGame.best_move — float32 numpy in/out, raw policy logits."""
    def eval_fn(board_batch, reserve_batch):
        bt = torch.from_numpy(np.asarray(board_batch, dtype=np.float32)).to(device)
        rv = torch.from_numpy(np.asarray(reserve_batch, dtype=np.float32)).to(device)
        with torch.no_grad():
            policy_logits, wdl_logits, _ = model(bt, rv)
            wdl = torch.softmax(wdl_logits, dim=1)
        return (
            policy_logits.float().cpu().numpy().astype(np.float32),
            wdl.float().cpu().numpy().astype(np.float32),
        )
    return eval_fn


def raw_policy_pick(game, model, device, grid_size):
    """Greedy argmax over legal moves using raw policy logits.

    Mirrors Rust MCTS's prior scoring exactly:
      - Placement (Single idx)         : score = logits[idx]
      - Movement  (Q·K dot product)    : score = sum_d Q[d, src] * K[d, dst] / sqrt(D)
    """
    board, reserve = game.encode_board()
    bt = torch.from_numpy(np.asarray(board, dtype=np.float32)).unsqueeze(0).to(device)
    rv = torch.from_numpy(np.asarray(reserve, dtype=np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, _, _ = model(bt, rv)
    logits = policy_logits[0].cpu().numpy()

    _, indexed_moves = game.get_legal_move_mask()
    if not indexed_moves:
        return None

    q_off = q_section_offset(grid_size)
    k_off = k_section_offset(grid_size)
    g2 = grid_size * grid_size
    inv_sqrt_d = 1.0 / float(np.sqrt(BILINEAR_DIM))

    Q = logits[q_off:q_off + BILINEAR_DIM * g2].reshape(BILINEAR_DIM, g2)
    K = logits[k_off:k_off + BILINEAR_DIM * g2].reshape(BILINEAR_DIM, g2)

    best_score = -np.inf
    best_move = None
    for primary, secondary, piece, frm, to in indexed_moves:
        if frm is None:
            score = float(logits[primary])
        else:
            score = float(np.dot(Q[:, primary], K[:, secondary])) * inv_sqrt_d
        if score > best_score:
            best_score = score
            best_move = (piece, frm, to)
    return best_move


def mcts_pick(game, eval_fn, sims, c_puct, draw_contempt):
    """800-sim MCTS via HiveGame.best_move; resolve UHP back to (piece, from, to)."""
    uhp = game.best_move(
        eval_fn,
        simulations=sims,
        c_puct=c_puct,
        draw_contempt=draw_contempt,
    )
    if uhp == "pass":
        return None
    for piece, frm, to in game.valid_moves():
        if game.format_move_uhp(piece, frm, to) == uhp:
            return (piece, frm, to)
    raise RuntimeError(f"MCTS returned UHP {uhp!r}; no matching legal move")


def play_one_game(model, device, sims, c_puct, draw_contempt, max_moves,
                  mcts_is_white, grid_size):
    game = HiveGame(grid_size=grid_size)
    eval_fn = make_eval_fn(model, device)

    while not game.is_game_over and game.move_count < max_moves:
        is_white_turn = game.turn_color == "w"
        is_mcts = (is_white_turn == mcts_is_white)

        if not game.valid_moves():
            game.play_pass()
            continue

        if is_mcts:
            mv = mcts_pick(game, eval_fn, sims, c_puct, draw_contempt)
        else:
            mv = raw_policy_pick(game, model, device, grid_size)

        if mv is None:
            game.play_pass()
        else:
            game.play_move(*mv)

    state = game.state
    if state == "WhiteWins":
        outcome = "MCTS" if mcts_is_white else "RAW"
    elif state == "BlackWins":
        outcome = "RAW" if mcts_is_white else "MCTS"
    else:
        outcome = "DRAW"
    return outcome, game.move_count, state


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint", help="path to .pt checkpoint")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--simulations", type=int, default=800)
    ap.add_argument("--c-puct", type=float, default=1.5)
    ap.add_argument("--draw-contempt", type=float, default=0.0)
    ap.add_argument("--max-moves", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("cuda requested but unavailable; falling back to cpu")
        args.device = "cpu"

    model, ckpt = load_checkpoint(args.checkpoint)
    model.to(args.device).eval()
    grid_size = ckpt.get("grid_size", 23)
    gen = ckpt.get("generation", "?")

    print(f"loaded {args.checkpoint} | gen={gen} | grid={grid_size} | device={args.device}")
    print(f"games={args.games}  sims={args.simulations}  max_moves={args.max_moves}  "
          f"c_puct={args.c_puct}  draw_contempt={args.draw_contempt}")
    print(f"MCTS({args.simulations} sims) vs raw policy (greedy argmax). Same network both sides.")
    print()

    mcts_wins = raw_wins = draws = 0
    state_counts = {}
    lengths = []
    t0 = time.time()

    for i in range(args.games):
        mcts_is_white = (i % 2 == 0)
        outcome, length, state = play_one_game(
            model, args.device, args.simulations, args.c_puct, args.draw_contempt,
            args.max_moves, mcts_is_white, grid_size,
        )
        if outcome == "MCTS":
            mcts_wins += 1
        elif outcome == "RAW":
            raw_wins += 1
        else:
            draws += 1
        state_counts[state] = state_counts.get(state, 0) + 1
        lengths.append(length)
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        print(f"  game {i+1:>3}/{args.games}  MCTS={'W' if mcts_is_white else 'B'}  "
              f"out={outcome:<4} len={length:>3} {state:<14} "
              f"  running MCTS-RAW-Draw {mcts_wins}-{raw_wins}-{draws}"
              f"  ({rate:.2f} g/s)")

    elapsed = time.time() - t0
    total = args.games
    decisive = mcts_wins + raw_wins
    mcts_decisive_share = (mcts_wins / decisive) if decisive else float("nan")
    score_mcts = (mcts_wins + 0.5 * draws) / total

    print()
    print(f"Results ({total} games, {elapsed:.0f}s, {total/elapsed:.2f} games/s)")
    print(f"  MCTS wins : {mcts_wins:>3}")
    print(f"  RAW  wins : {raw_wins:>3}")
    print(f"  Draws     : {draws:>3}  (states: {state_counts})")
    print(f"  MCTS score (W + 0.5*D) / N : {score_mcts:.1%}")
    if decisive:
        print(f"  MCTS share of decisive games: {mcts_decisive_share:.1%}")
    avg_len = sum(lengths) / len(lengths)
    med_len = sorted(lengths)[len(lengths) // 2]
    print(f"  game length: avg={avg_len:.1f}  med={med_len}")
    print()
    if score_mcts >= 0.65:
        print("  => MCTS clearly outperforms raw policy. Policy improvement signal exists;")
        print("    bottleneck for self-play is elsewhere (data quality, LR, exploration).")
    elif score_mcts >= 0.55:
        print("  => MCTS marginally better than raw policy. Improvement signal is weak;")
        print("    self-play will learn slowly. Consider more sims or better priors.")
    else:
        print("  => MCTS does NOT outperform raw policy at this checkpoint.")
        print("    Visit-distribution targets are no better than network priors --")
        print("    the AlphaZero policy-improvement step has nothing to teach.")
        print("    This explains both from-scratch stagnation and post-pretraining drift.")


if __name__ == "__main__":
    main()
