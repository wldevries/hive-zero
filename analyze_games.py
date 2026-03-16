#!/usr/bin/env python3
"""Analyze Hive SGF games across all zip archives and loose folders."""

import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hive.sgf import game_type

BOTS = {"Dumbot", "WeakBot", "SmartBot", "BestBot"}
GAMES_DIR = Path(__file__).parent / "games"


def iter_sgf_contents():
    """Yield (source, sgf_content) for every SGF file found under games/."""
    # Zip files
    for zp in sorted(GAMES_DIR.rglob("*.zip")):
        with zipfile.ZipFile(zp) as z:
            for name in z.namelist():
                if name.endswith(".sgf"):
                    yield zp.name, z.read(name).decode("iso-8859-1", errors="ignore")

    # Loose SGF files in subdirectories (e.g. games-Mar-16-2026/)
    for sgf in sorted(GAMES_DIR.rglob("*.sgf")):
        yield sgf.parent.name, sgf.read_text(encoding="iso-8859-1", errors="ignore")


def classify(p0: str, p1: str) -> str:
    b0, b1 = p0 in BOTS, p1 in BOTS
    g0, g1 = "guest" in p0.lower(), "guest" in p1.lower()
    if not b0 and not g0 and not b1 and not g1:
        return "human vs human"
    if (b0 or g0) and (b1 or g1):
        return "bot/guest vs bot/guest"
    return "human vs bot/guest"


def winner(content: str, p0: str, p1: str) -> str | None:
    """Return the winning player name, or None for draw/unknown."""
    # Newer format (2025+): explicit RE field containing winner's name
    m = re.search(r'RE\[([^\]]+)\]', content)
    if m:
        result = m.group(1)
        if p0 in result:
            return p0
        if p1 in result:
            return p1
        if "draw" in result.lower():
            return None

    # Older format: resign/acceptdraw action in move list
    # e.g. "; P0[88 Resign]" or "; P1[44 AcceptDraw]"
    resign = re.search(r'P([01])\[\d+ [Rr]esign', content)
    if resign:
        # The player who resigned loses — return the other player
        loser_idx = resign.group(1)
        return p1 if loser_idx == "0" else p0

    draw = re.search(r'P([01])\[\d+ [Aa]ccept[Dd]raw', content)
    if draw:
        return None

    return None  # normal win — would need board replay to determine


def main():
    players = Counter()
    wins: Counter = Counter()
    losses: Counter = Counter()
    match_types = Counter()
    game_types = Counter()
    total = 0

    for _source, content in iter_sgf_contents():
        m0 = re.search(r'P0\[id "([^"]+)"', content)
        m1 = re.search(r'P1\[id "([^"]+)"', content)
        p0 = m0.group(1) if m0 else "unknown"
        p1 = m1.group(1) if m1 else "unknown"

        players[p0] += 1
        players[p1] += 1
        match_types[classify(p0, p1)] += 1
        game_types[game_type(content)] += 1
        total += 1

        w = winner(content, p0, p1)
        if w == p0:
            wins[p0] += 1
            losses[p1] += 1
        elif w == p1:
            wins[p1] += 1
            losses[p0] += 1

    print(f"Total games: {total}\n")

    print("── Game type ──────────────────────────")
    for k, v in game_types.most_common():
        print(f"  {v:5d}  {k}")

    print("\n── Match type ─────────────────────────")
    for k, v in match_types.most_common():
        print(f"  {v:5d}  {k}")

    print("\n── Bots ───────────────────────────────")
    for name, count in players.most_common():
        if name in BOTS:
            w, l = wins[name], losses[name]
            ratio = f"{w/(w+l)*100:.0f}%" if (w + l) else "  -"
            print(f"  {count:5d}  {name:<12}  {w}W {l}L  {ratio}")

    print("\n── Top 20 humans ──────────────────────")
    human_players = [(n, c) for n, c in players.most_common() if n not in BOTS]
    for name, count in human_players[:20]:
        w, l = wins[name], losses[name]
        ratio = f"{w/(w+l)*100:.0f}%" if (w + l) else "  -"
        tag = " [guest]" if "guest" in name.lower() else ""
        print(f"  {count:5d}  {name:<14}{tag}  {w}W {l}L  {ratio}")


if __name__ == "__main__":
    main()
