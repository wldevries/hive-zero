"""Boardspace SGF file parser — extracts UHP move strings from Hive SGF files."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path


def game_type(content: str) -> str:
    """Return 'base' or 'expansion' based on the SU field."""
    m = re.search(r'SU\[([^\]]+)\]', content)
    if not m:
        return 'base'
    su = m.group(1).lower()
    return 'expansion' if re.search(r'hive-[a-z]+', su) else 'base'


def parse_moves(content: str):
    """Yield UHP move strings from Boardspace SGF content.

    Handles Pick/Pickb + Dropb pairs, combined Move/movedone lines, and pass.
    Expansion pieces (M, L, P) are passed through as-is.
    """
    coord_stack: dict[str, list[str]] = defaultdict(list)  # 'C-R' -> [bottom..top]
    piece_coords: dict[str, str] = {}  # piece -> 'C-R'
    move_count = 0
    pending_old_coords: str | None = None  # set by Pickb, consumed by Dropb

    action_pat = re.compile(r';\s*P[01]\[\d+\s+(.*?)\](?:TM\[\d+\])?', re.MULTILINE)

    for m in action_pat.finditer(content):
        action = m.group(1).strip()
        low = action.lower()

        if low.startswith('done') or low.startswith('start'):
            pending_old_coords = None
            continue

        # Pick from reserve: "Pick W 4 wS1"
        if re.match(r'Pick [BW] \d+ \S+', action):
            pending_old_coords = None
            continue

        # Pick from board: "Pickb N 13 wS1"
        pb = re.match(r'Pickb ([A-Z]) (\d+) (\S+)', action)
        if pb:
            col, row = pb.group(1), pb.group(2)
            pending_old_coords = f"{col}-{row}"
            continue

        # Drop: "Dropb wS1 N 13 ."
        db = re.match(r'Dropb (\S+) ([A-Z]) (\d+) (.+)', action)
        if db:
            piece, col, row = db.group(1), db.group(2), db.group(3)
            ref_raw = db.group(4).strip()
            new_coords = f"{col}-{row}"
            old_coords, pending_old_coords = pending_old_coords, None

            uhp_ref = _resolve_ref(ref_raw, new_coords, piece, move_count,
                                   coord_stack, piece_coords)
            yield piece if uhp_ref == '' else f"{piece} {uhp_ref}"
            _update_coords(coord_stack, piece_coords, piece, old_coords, new_coords)
            move_count += 1
            continue

        # Move (combined): "Move B bS1 M 12 /wS1"
        mv = re.match(r'Move [BW] (\S+) ([A-Z]) (\d+) (.+)', action)
        if mv:
            piece, col, row = mv.group(1), mv.group(2), mv.group(3)
            ref_raw = mv.group(4).strip()
            new_coords = f"{col}-{row}"
            old_coords = piece_coords.get(piece)

            uhp_ref = _resolve_ref(ref_raw, new_coords, piece, move_count,
                                   coord_stack, piece_coords)
            yield piece if uhp_ref == '' else f"{piece} {uhp_ref}"
            _update_coords(coord_stack, piece_coords, piece, old_coords, new_coords)
            move_count += 1
            continue

        # movedone (alternate combined format): "movedone B bS1 M 12 /wS1"
        md = re.match(r'movedone [BW] (\S+) ([A-Z]) (\d+) (.+)', action)
        if md:
            piece, col, row = md.group(1), md.group(2), md.group(3)
            ref_raw = md.group(4).strip()
            new_coords = f"{col}-{row}"
            old_coords = piece_coords.get(piece)

            uhp_ref = _resolve_ref(ref_raw, new_coords, piece, move_count,
                                   coord_stack, piece_coords)
            yield piece if uhp_ref == '' else f"{piece} {uhp_ref}"
            _update_coords(coord_stack, piece_coords, piece, old_coords, new_coords)
            move_count += 1
            continue

        if low == 'pass':
            yield 'pass'
            move_count += 1
            continue


# ---- internal helpers ----

def _resolve_ref(ref_raw: str, new_coords: str, piece: str, move_count: int,
                 coord_stack, piece_coords) -> str:
    ref = re.sub(r'\\\\', r'\\', ref_raw)

    if ref == '.':
        if move_count == 0:
            return ''  # first placement — no reference needed
        stack = coord_stack.get(new_coords, [])
        if stack:
            return stack[-1]  # beetle stacking onto occupied hex
        old_coords = piece_coords.get(piece)
        if old_coords:
            return _drop_down_ref(old_coords, new_coords, piece)
        return ''

    if ref.endswith('.') and len(ref) > 1:
        return ref[:-1]  # "bQ." notation — strip dot, becomes pure piece name

    return ref


def _drop_down_ref(old_coords: str, new_coords: str, piece: str) -> str:
    """UHP ref for a beetle dropping from a stack to an adjacent empty hex."""
    cx, cy_s = old_coords.split('-')
    dx, dy_s = new_coords.split('-')
    cy, dy = int(cy_s), int(dy_s)
    if cy == dy:
        return f"-{piece}" if ord(cx) > ord(dx) else f"{piece}-"
    if cx == dx:
        return f"{piece}\\" if cy > dy else f"\\{piece}"
    return f"/{piece}" if cy > dy else f"{piece}/"


def _update_coords(coord_stack, piece_coords, piece, old_coords, new_coords):
    if old_coords and piece in coord_stack.get(old_coords, []):
        coord_stack[old_coords].remove(piece)
    coord_stack[new_coords].append(piece)
    piece_coords[piece] = new_coords


def read_sgf(path: str | Path) -> str:
    return Path(path).read_text(encoding='iso-8859-1')
