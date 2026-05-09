"""Replay a Hive selfplay SGF up to a chosen ply, then dump MCTS root-child stats
at that position so we can see why the trainer picked the move it did.

Originally built to diagnose `gen00013_game0001_white_58t.sgf` — black's last
move (`bG2 /bA3`) surrounds its own queen and is a guaranteed loss, but MCTS
picked it anyway. The trainer-recorded result is `WhiteWins`, confirming the
post-hoc terminal detection works. The question this script is meant to answer
is whether MCTS, given the same position and (close to the same) checkpoint,
also picks that move — and if so, what its visit / Q / prior look like.

Usage:
    uv run --extra ort-gpu python scripts/diagnose_selfloss.py \
        --sgf models/hive-small-heur/selfplay_sgf/gen00013.zip:gen00013_game0001_white_58t.sgf \
        --onnx models/hive-small-heur/hive-small-heur.onnx \
        --ply 57 \
        --simulations 800 --c-puct 1.25 --forced-playouts \
        --highlight "bG2 /bA3"

Pass `--dirichlet-alpha 0.3 --dirichlet-epsilon 0.25 --runs 20` to faithfully
reproduce the trainer's Dirichlet-noised search and aggregate over multiple runs.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from engine_zero import HiveGame, HiveOrtSession, parse_sgf_moves


TERMINAL_BADGE = {
    "ongoing": " ",
    "win": "W",       # winning terminal for the side to move (good)
    "loss": "L",      # losing terminal for the side to move (suicide!)
    "draw": "D",
}


def load_sgf(spec: str) -> str:
    """Accept either `path/to.sgf` or `path/to.zip:member.sgf`."""
    if ":" in spec and spec.lower().endswith(".sgf"):
        zip_path, _, member = spec.partition(":")
        with zipfile.ZipFile(zip_path) as z:
            with z.open(member) as f:
                return f.read().decode()
    return Path(spec).read_text()


def replay_to_ply(sgf_text: str, ply: int, grid_size: int) -> tuple[HiveGame, list[str]]:
    """Play `ply` moves from the SGF; return the game and the list of moves."""
    moves = parse_sgf_moves(sgf_text)
    if ply > len(moves):
        raise SystemExit(f"--ply {ply} exceeds SGF length {len(moves)}")
    g = HiveGame(grid_size=grid_size)
    for mv in moves[:ply]:
        g.play_move_uhp(mv)
    return g, moves


def fmt_row(rank: int, c, marker: str) -> str:
    badge = TERMINAL_BADGE.get(c.terminal, "?")
    return (
        f"{rank:>3} {marker:<2} [{badge}] {c.uhp_move:<22}"
        f" v={c.visits:>5}  Q={c.value:+.3f}  prior={c.prior:.3f}"
    )


def run_one(game: HiveGame, ort: HiveOrtSession, args, run_idx: int):
    stats = game.mcts_root_stats(
        ort,
        simulations=args.simulations,
        c_puct=args.c_puct,
        forced_playouts=args.forced_playouts,
        dir_alpha=args.dirichlet_alpha,
        dir_epsilon=args.dirichlet_epsilon,
        play_batch_size=args.play_batch_size,
    )

    children = sorted(stats.children, key=lambda c: -c.visits)
    print()
    print(f"=== Run {run_idx + 1} / {args.runs} ===")
    print(
        f"root visits={stats.root_visits}  root Q={stats.root_value:+.3f}  "
        f"children={len(children)}"
    )
    print()

    # Mirror Rust's tiered best_move: prefer proven wins for root, then
    # neutral (unproven / draw), then proven losses.
    def tier(c):
        return {"win": 0, "loss": 2, "draw": 1, "ongoing": 1}[c.terminal]

    if children:
        best_tier = min(tier(c) for c in children)
        candidates = [c for c in children if tier(c) == best_tier]
        best = max(candidates, key=lambda c: (c.visits, c.value))
    else:
        best = None

    print(
        f"{'#':>3} {'':<2} [T] {'move':<22}  visits   Q          prior"
    )
    print("-" * 66)
    for rank, c in enumerate(children, start=1):
        marker = ""
        if best is not None and c is best:
            marker = "*"  # what MCTS would actually play
        if args.highlight and c.uhp_move == args.highlight:
            marker += "!" if marker else "!"  # the move you're investigating
        print(fmt_row(rank, c, marker))

    # Summary line: did MCTS pick a self-loss?
    if best is not None:
        verdict = (
            "PICKED A LOSS" if best.terminal == "loss"
            else f"picked {best.terminal}"
        )
        print()
        print(
            f"-> best by visits: {best.uhp_move!r}  "
            f"v={best.visits}  Q={best.value:+.3f}  [{verdict}]"
        )

    # If user passed --highlight, also report what the search thinks of it
    if args.highlight:
        match = next((c for c in children if c.uhp_move == args.highlight), None)
        if match is None:
            print(f"   highlight {args.highlight!r}: NOT in root children list")
        else:
            print(
                f"   highlight {args.highlight!r}: "
                f"v={match.visits}  Q={match.value:+.3f}  prior={match.prior:.3f}  "
                f"terminal={match.terminal}"
            )

    return best, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sgf", required=True,
                   help="Path to .sgf, or zip.zip:member.sgf inside an archive")
    p.add_argument("--onnx", required=True,
                   help="Path to ONNX model (use one with CUDAExecutionProvider for speed)")
    p.add_argument("--ply", type=int, required=True,
                   help="Number of moves to replay before searching")
    p.add_argument("--simulations", type=int, default=800)
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--play-batch-size", type=int, default=16,
                   help="Match selfplay's --play-batch-sims (default 16)")
    p.add_argument("--forced-playouts", action="store_true",
                   help="Enable KataGo-style forced playouts (selection_k=0.5)")
    p.add_argument("--dirichlet-alpha", type=float, default=0.0)
    p.add_argument("--dirichlet-epsilon", type=float, default=0.0)
    p.add_argument("--runs", type=int, default=1,
                   help="Repeat the search N times (useful with Dirichlet noise)")
    p.add_argument("--highlight", type=str, default=None,
                   help="UHP move string to call out in the per-run output")
    p.add_argument("--top", type=int, default=0,
                   help="Truncate child table to top N visits per run (0 = no limit)")
    p.add_argument("--grid-size", type=int, default=23,
                   help="NN encoding grid size (must match the model's training grid)")
    args = p.parse_args()

    sgf_text = load_sgf(args.sgf)
    game, all_moves = replay_to_ply(sgf_text, args.ply, args.grid_size)
    actual_next = all_moves[args.ply] if args.ply < len(all_moves) else None

    print(f"Replayed {args.ply} moves; turn {game.turn_color} to move "
          f"(turn_number={game.turn_number}, move_count={game.move_count})")
    if actual_next is not None:
        print(f"Move actually played in the SGF at this ply: {actual_next!r}")
    print(f"Loading ONNX session: {args.onnx}")

    ort = HiveOrtSession(args.onnx)

    # Aggregate counters across runs
    picks_by_terminal: dict[str, int] = {"ongoing": 0, "win": 0, "loss": 0, "draw": 0}
    highlight_pick_count = 0

    for run_idx in range(args.runs):
        # Re-replay so each run starts from the same fresh state (no warm tree)
        if run_idx > 0:
            game, _ = replay_to_ply(sgf_text, args.ply, args.grid_size)

        best, stats = run_one(game, ort, args, run_idx)
        if best is not None:
            picks_by_terminal[best.terminal] = picks_by_terminal.get(best.terminal, 0) + 1
            if args.highlight and best.uhp_move == args.highlight:
                highlight_pick_count += 1

    if args.runs > 1:
        print()
        print("=== Aggregate ===")
        for kind, n in picks_by_terminal.items():
            if n:
                print(f"  picked {kind:<8}: {n}/{args.runs}")
        if args.highlight:
            print(f"  picked highlight {args.highlight!r}: "
                  f"{highlight_pick_count}/{args.runs}")


if __name__ == "__main__":
    sys.exit(main() or 0)
