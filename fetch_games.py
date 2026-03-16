#!/usr/bin/env python3
"""Download Hive game zips from boardspace.net into games/boardspace/."""

import re
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://www.boardspace.net/hive/hivegames"
DEST = Path(__file__).parent / "games" / "boardspace"

YEARS = range(2006, 2027)


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8", errors="ignore")


def zip_links(index_html: str) -> list[str]:
    return re.findall(r'href="(games-[^"]+\.zip)"', index_html)


def download(url: str, dest: Path):
    print(f"  downloading {dest.name} ... ", end="", flush=True)
    urllib.request.urlretrieve(url, dest)
    size_kb = dest.stat().st_size // 1024
    print(f"{size_kb}K")


def main():
    DEST.mkdir(parents=True, exist_ok=True)

    total_new = 0

    for year in YEARS:
        url = f"{BASE_URL}/archive-{year}/"
        try:
            html = fetch(url)
        except Exception as e:
            print(f"archive-{year}: skipped ({e})")
            continue

        zips = zip_links(html)
        if not zips:
            continue

        year_dir = DEST / f"archive-{year}"
        year_dir.mkdir(exist_ok=True)
        existing = {p.name for p in year_dir.glob("*.zip")}
        new = [z for z in zips if z not in existing]
        print(f"archive-{year}: {len(zips)} zips, {len(new)} new")

        for name in new:
            try:
                download(f"{url}{name}", year_dir / name)
                total_new += 1
            except Exception as e:
                print(f"  ERROR {name}: {e}")

    print(f"\nDone. {total_new} new files downloaded to {DEST}")


if __name__ == "__main__":
    main()
