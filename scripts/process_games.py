#!/usr/bin/env python3
"""
Replay all Hive SGF games to determine outcomes, then compute ELO ratings.

Outputs:
  games/game_outcomes.csv  - per-game: zip, sgf, p0, p1, game_type, move_count, result
  games/player_elo.csv     - per-player: elo, games, wins, losses, draws
"""

import csv
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from hive.sgf import game_type, parse_moves
from hive_engine import RustGame

GAMES_DIR = Path(__file__).parent.parent / "games"
OUTCOMES_CSV = GAMES_DIR / "game_outcomes.csv"
ELO_CSV = GAMES_DIR / "player_elo.csv"

K = 32
DEFAULT_ELO = 1500.0


def iter_sgf_contents():
    """Yield (zip_name, sgf_name, content) for every SGF file, in date order."""
    for zp in sorted(GAMES_DIR.rglob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            for name in sorted(z.namelist()):
                if name.endswith(".sgf"):
                    content = z.read(name).decode("iso-8859-1", errors="ignore")
                    yield zp.name, name, content

    for sgf in sorted(GAMES_DIR.rglob("*.sgf")):
        yield sgf.parent.name, sgf.name, sgf.read_text(encoding="iso-8859-1", errors="ignore")


def replay_game(moves: list[str]) -> str | None:
    """Replay UHP move strings through a fresh RustGame.

    Returns the final game state string ("WhiteWins", "BlackWins", "Draw",
    "InProgress"), or None if a move was rejected by the engine.
    """
    game = RustGame()
    try:
        for move_str in moves:
            if game.is_game_over:
                break
            if not game.play_move_uhp(move_str):
                return None
    except BaseException:
        return None
    return game.state


def result_from_metadata(content: str, p0: str, p1: str) -> str:
    """Determine result from SGF metadata (RE field or Resign action).

    Fallback for games that don't end by queen surrounding (e.g. resign).
    Returns 'p0_wins', 'p1_wins', 'draw', or 'unknown'.
    """
    # Newer format: explicit RE field containing winner's name
    m = re.search(r'RE\[([^\]]+)\]', content)
    if m:
        result = m.group(1)
        if p0 in result:
            return "p0_wins"
        if p1 in result:
            return "p1_wins"
        if "draw" in result.lower():
            return "draw"

    # Older format: resign action — the resigning player loses
    resign = re.search(r'P([01])\[\d+ [Rr]esign', content)
    if resign:
        return "p1_wins" if resign.group(1) == "0" else "p0_wins"

    accept_draw = re.search(r'P([01])\[\d+ [Aa]ccept[Dd]raw', content)
    if accept_draw:
        return "draw"

    return "unknown"


def expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))


def update_elo(elo: dict[str, float], p0: str, p1: str, result: str) -> None:
    """Update ELO ratings in-place. result must be 'p0_wins', 'p1_wins', or 'draw'."""
    ra = elo.get(p0, DEFAULT_ELO)
    rb = elo.get(p1, DEFAULT_ELO)
    ea = expected_score(ra, rb)
    eb = 1.0 - ea
    if result == "p0_wins":
        sa, sb = 1.0, 0.0
    elif result == "p1_wins":
        sa, sb = 0.0, 1.0
    else:
        sa, sb = 0.5, 0.5
    elo[p0] = ra + K * (sa - ea)
    elo[p1] = rb + K * (sb - eb)


def main():
    outcomes: list[tuple] = []
    elo: dict[str, float] = {}
    stats: dict[str, dict] = {}

    total = 0
    errors = 0
    determined = 0

    print("Replaying games...", flush=True)
    for zip_name, sgf_name, content in iter_sgf_contents():
        total += 1
        if total % 10000 == 0:
            print(f"  {total} games replayed ({determined} determined, {errors} errors)...",
                  flush=True)

        m0 = re.search(r'P0\[id "([^"]+)"', content)
        m1 = re.search(r'P1\[id "([^"]+)"', content)
        p0 = m0.group(1) if m0 else "unknown"
        p1 = m1.group(1) if m1 else "unknown"
        gtype = game_type(content)

        moves = list(parse_moves(content))
        move_count = len(moves)

        final_state = replay_game(moves)

        if final_state is None:
            # Replay error — move rejected by Rust engine, try metadata fallback
            result = result_from_metadata(content, p0, p1)
            if result == "unknown":
                errors += 1
            else:
                determined += 1
        elif final_state == "WhiteWins":
            result = "p0_wins"   # White moves first = P0
            determined += 1
        elif final_state == "BlackWins":
            result = "p1_wins"
            determined += 1
        elif final_state == "Draw":
            result = "draw"
            determined += 1
        else:
            # InProgress — game didn't reach terminal state (e.g. resign, incomplete)
            result = result_from_metadata(content, p0, p1)
            if result != "unknown":
                determined += 1

        outcomes.append((zip_name, sgf_name, p0, p1, gtype, move_count, result))

        # Initialise per-player stats
        for p in (p0, p1):
            if p not in stats:
                stats[p] = {"games": 0, "wins": 0, "losses": 0, "draws": 0}
        stats[p0]["games"] += 1
        stats[p1]["games"] += 1

        if result in ("p0_wins", "p1_wins", "draw"):
            if result == "p0_wins":
                stats[p0]["wins"] += 1
                stats[p1]["losses"] += 1
            elif result == "p1_wins":
                stats[p1]["wins"] += 1
                stats[p0]["losses"] += 1
            else:
                stats[p0]["draws"] += 1
                stats[p1]["draws"] += 1
            update_elo(elo, p0, p1, result)

    print(f"\nDone: {total} games total, {determined} outcomes determined, "
          f"{errors} parse errors, {total - determined - errors} unknown")

    # Write game outcomes CSV
    with open(OUTCOMES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["zip_file", "sgf_name", "p0", "p1", "game_type", "move_count", "result"])
        w.writerows(outcomes)
    print(f"Wrote {OUTCOMES_CSV}")

    # Write player ELO CSV (sorted by ELO descending)
    with open(ELO_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["player", "elo", "games", "wins", "losses", "draws"])
        ranked = sorted(stats.items(), key=lambda x: -elo.get(x[0], DEFAULT_ELO))
        for player, s in ranked:
            e = elo.get(player, DEFAULT_ELO)
            w.writerow([player, f"{e:.1f}", s["games"], s["wins"], s["losses"], s["draws"]])
    print(f"Wrote {ELO_CSV}")


if __name__ == "__main__":
    main()
