#!/usr/bin/env python3
#!/usr/bin/env python3
"""Download Hive game zips from boardspace.net into games/hive/boardspace/."""

import re
import urllib.request
from pathlib import Path

BASE_URL = "https://www.boardspace.net/hive/hivegames"
DEST = Path(__file__).parent.parent / "games" / "hive" / "boardspace"

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


def main():
    DEST.mkdir(parents=True, exist_ok=True)

    # Discover all subdirectories from the top-level index
    print(f"Fetching index: {BASE_URL}/")
    top_html = fetch(f"{BASE_URL}/")
    dirs = [d for d in subdir_links(top_html) if d.endswith("/") and d not in SKIP_DIRS]
    print(f"Found {len(dirs)} subdirectories\n")

    total_new = 0

    for subdir in sorted(dirs):
        url = f"{BASE_URL}/{subdir}"
        try:
            html = fetch(url)
        except Exception as e:
            print(f"{subdir}: skipped ({e})")
            continue

        zips = zip_links(html)
        if not zips:
            continue

        local_dir = DEST / subdir.rstrip("/")
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

    print(f"\nDone. {total_new} new files downloaded to {DEST}")


if __name__ == "__main__":
    main()
