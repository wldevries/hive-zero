#!/usr/bin/env python3
"""CLI: parse a Boardspace Hive SGF file and render each board state."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hive.core.render import render_board
from hive.sgf import game_type, parse_moves, read_sgf
from hive_engine import RustGame


def render_game(sgf_path: str):
    import re
    content = read_sgf(sgf_path)

    m = re.search(r'RE\[([^\]]+)\]', content)
    result = m.group(1) if m else 'Unknown'
    m = re.search(r'P0\[id "([^"]+)"', content)
    white = m.group(1) if m else 'White'
    m = re.search(r'P1\[id "([^"]+)"', content)
    black = m.group(1) if m else 'Black'

    print(f"\n{'='*60}")
    print(f"Game: {Path(sgf_path).name}  [{game_type(content)}]")
    print(f"White (P0): {white}  |  Black (P1): {black}")
    print(f"Result: {result}")
    print(f"{'='*60}")

    game = RustGame()
    errors = 0

    for i, move_str in enumerate(parse_moves(content)):
        color = 'White' if i % 2 == 0 else 'Black'
        print(f"\n--- Move {i + 1} ({color}): {move_str} ---")
        if not game.play_move_uhp(move_str):
            print(f"  ERROR: move rejected by engine: {move_str!r}")
            errors += 1
            if errors >= 3:
                print("  Too many errors, stopping.")
                break
            continue
        print(render_board(game))

    print(f"\nFinal state: {game.state}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        games_dir = Path(__file__).parent.parent / 'games'
        sgf_files = sorted(games_dir.rglob('*.sgf'))
        if not sgf_files:
            print("No SGF files found.")
            sys.exit(1)
        render_game(str(sgf_files[0]))
    else:
        for path in sys.argv[1:]:
            render_game(path)
