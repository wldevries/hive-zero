#!/usr/bin/env python3
"""Download Hive/Yinsh/Zertz game zips from boardspace.net into games/<game>/boardspace/."""

import re
import urllib.request
from pathlib import Path

GAMES = {
    "hive": "https://www.boardspace.net/hive/hivegames",
    "yinsh": "https://www.boardspace.net/yinsh/yinshgames",
    "zertz": "https://www.boardspace.net/zertz/games",
}
GAMES_ROOT = Path(__file__).parent.parent / "games"

# Subdirectories to skip — not game archives
SKIP_DIRS = {"rankings/", "test/", "tutorials/"}


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8", errors="ignore")


def subdir_links(index_html: str) -> list[str]:
    """Return all subdirectory hrefs (ending in /) from an index page."""
    return re.findall(r'href="([^"/][^"]*/?)"', index_html)


def zip_links(index_html: str) -> list[str]:
    """Return all .zip hrefs from an index page."""
    return re.findall(r'href="([^"]+\.zip)"', index_html)


def download(url: str, dest: Path):
    print(f"  downloading {dest.name} ... ", end="", flush=True)
    urllib.request.urlretrieve(url, dest)
    size_kb = dest.stat().st_size // 1024
    print(f"{size_kb}K")


def fetch_game(game: str, base_url: str) -> int:
    dest = GAMES_ROOT / game / "boardspace"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"=== {game} ===")
    print(f"Fetching index: {base_url}/")
    try:
        top_html = fetch(f"{base_url}/")
    except Exception as e:
        print(f"  ERROR: {e}\n")
        return 0

    dirs = [d for d in subdir_links(top_html) if d.endswith("/") and d not in SKIP_DIRS]
    print(f"Found {len(dirs)} subdirectories\n")

    total_new = 0
    for subdir in sorted(dirs):
        url = f"{base_url}/{subdir}"
        try:
            html = fetch(url)
        except Exception as e:
            print(f"{subdir}: skipped ({e})")
            continue

        zips = zip_links(html)
        if not zips:
            continue

        local_dir = dest / subdir.rstrip("/")
        local_dir.mkdir(exist_ok=True)
        existing = {p.name for p in local_dir.glob("*.zip")}
        new = [z for z in zips if z not in existing]
        print(f"{subdir}: {len(zips)} zips, {len(new)} new")

        for name in new:
            try:
                download(f"{url}{name}", local_dir / name)
                total_new += 1
            except Exception as e:
                print(f"  ERROR {name}: {e}")

    print(f"-> {total_new} new files in {dest}\n")
    return total_new


def main():
    grand_total = 0
    for game, base_url in GAMES.items():
        grand_total += fetch_game(game, base_url)
    print(f"Done. {grand_total} new files downloaded total.")


if __name__ == "__main__":
    main()
