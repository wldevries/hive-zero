"""Boardspace SGF file parser — extracts UHP move strings from Hive SGF files."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path


_EXPANSION_PIECE_PAT = re.compile(r'\b[wb][MLP]\d*\b')

def game_type(content: str) -> str:
    """Return 'base' or 'expansion'.

    Checks the SU field first, then falls back to scanning move actions for
    expansion piece names (M=Mosquito, L=Ladybug, P=Pillbug) — some older
    files omit the expansion tag from SU but still use expansion pieces.
    """
    m = re.search(r'SU\[([^\]]+)\]', content)
    if m and re.search(r'hive-[a-z]+', m.group(1).lower()):
        return 'expansion'
    if _EXPANSION_PIECE_PAT.search(content):
        return 'expansion'
    return 'base'


def parse_moves(content: str):
    """Yield UHP move strings from Boardspace SGF content.

    Handles both old format (lowercase commands, no color prefix on pieces) and
    new format (uppercase commands, color-prefixed pieces like wB1/bA1).
    Handles Pick/Pickb + Dropb pairs, combined Move/movedone lines, and pass.
    Expansion pieces (M, L, P) are passed through as-is.

    Mid-turn undo: a player may drop a piece, pick it back up, then drop it
    at a different position before confirming with "done". Only the final drop
    (the one before "done") is yielded.
    """
    coord_stack: dict[str, list[str]] = defaultdict(list)  # 'C-R' -> [bottom..top]
    piece_coords: dict[str, str] = {}  # piece -> 'C-R'
    move_count = 0
    pending_old_coords: str | None = None  # set by Pickb, consumed by Dropb
    # Buffered move waiting for "done" confirmation: (piece, uhp_ref, old_coords, new_coords)
    pending_move: tuple | None = None

    # Capture player index (0=white, 1=black) for inferring color in old format
    action_pat = re.compile(r';\s*P([01])\[\d+\s+(.*?)\](?:TM\[\d+\])?', re.MULTILINE)

    for m in action_pat.finditer(content):
        player_color = 'w' if m.group(1) == '0' else 'b'
        action = m.group(2).strip()
        low = action.lower()

        if low.startswith('done') or low.startswith('start'):
            # Finalise the buffered move for this turn.
            if pending_move is not None:
                piece, uhp_ref, old_coords, new_coords = pending_move
                yield piece if uhp_ref == '' else f"{piece} {uhp_ref}"
                _update_coords(coord_stack, piece_coords, piece, old_coords, new_coords)
                move_count += 1
                pending_move = None
            pending_old_coords = None
            continue

        # Pick from reserve: "Pick W 4 wS1" (new) or "pick b 1 A1" (old)
        if re.match(r'pick\s+[BWbw]\s+\d+\s+\S+', action, re.IGNORECASE):
            pending_old_coords = None
            continue

        # Pick from board: "Pickb N 13 wS1" (new) or "pickb P 13 G1" (old)
        pb = re.match(r'pickb\s+([A-Za-z])\s+(\d+)\s+(\S+)', action, re.IGNORECASE)
        if pb:
            col, row = pb.group(1).upper(), pb.group(2)
            picked_coords = f"{col}-{row}"

            if pending_move is not None and picked_coords == pending_move[3]:
                # Player is picking up the piece they just dropped this turn (undo).
                # Cancel the buffered move and restore the original from-position.
                pending_old_coords = pending_move[2]  # original from (None = was in reserve)
                pending_move = None
            else:
                pending_old_coords = picked_coords
            continue

        # Drop: "Dropb wS1 N 13 ." (new) or "dropb A1 O 13 wB1-" (old)
        db = re.match(r'dropb\s+(\S+)\s+([A-Za-z])\s+(\d+)\s+(.+)', action, re.IGNORECASE)
        if db:
            piece = _ensure_color(db.group(1), player_color)
            col, row = db.group(2).upper(), db.group(3)
            ref_raw = db.group(4).strip()
            new_coords = f"{col}-{row}"
            old_coords, pending_old_coords = pending_old_coords, None

            uhp_ref = _resolve_ref(ref_raw, new_coords, piece, move_count,
                                   coord_stack, piece_coords)
            # Buffer until "done" — do not yield or update coords yet.
            pending_move = (piece, uhp_ref, old_coords, new_coords)
            continue

        # movedone (alternate combined format): "movedone B bS1 M 12 /wS1"
        # Must be checked before Move since 'movedone' starts with 'move'
        md = re.match(r'movedone\s+([BWbw])\s+(\S+)\s+([A-Za-z])\s+(\d+)\s+(.+)', action, re.IGNORECASE)
        if md:
            color = 'w' if md.group(1).upper() == 'W' else 'b'
            piece = _ensure_color(md.group(2), color)
            col, row = md.group(3).upper(), md.group(4)
            ref_raw = md.group(5).strip()
            new_coords = f"{col}-{row}"
            old_coords = piece_coords.get(piece)

            uhp_ref = _resolve_ref(ref_raw, new_coords, piece, move_count,
                                   coord_stack, piece_coords)
            yield piece if uhp_ref == '' else f"{piece} {uhp_ref}"
            _update_coords(coord_stack, piece_coords, piece, old_coords, new_coords)
            move_count += 1
            continue

        # Move (combined): "Move B bS1 M 12 /wS1" (new) or "move W B1 N 13 ." (old)
        mv = re.match(r'move\s+([BWbw])\s+(\S+)\s+([A-Za-z])\s+(\d+)\s+(.+)', action, re.IGNORECASE)
        if mv:
            color = 'w' if mv.group(1).upper() == 'W' else 'b'
            piece = _ensure_color(mv.group(2), color)
            col, row = mv.group(3).upper(), mv.group(4)
            ref_raw = mv.group(5).strip()
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

def _ensure_color(piece: str, color: str) -> str:
    """Add color prefix and normalize piece type letter to uppercase.

    Handles three formats:
      New, correct:   'bA1', 'wQ'   — color prefix + uppercase type → return as-is
      New, lowercase: 'ba1', 'wq'   — color prefix + lowercase type → uppercase type
      Old, no prefix: 'A1', 'a1', 'b1', 'q' — no color prefix → add color + uppercase
    """
    if len(piece) >= 2 and piece[0] in ('w', 'b') and piece[1].isalpha():
        # Has color prefix — normalize type letter to uppercase.
        return piece[0] + piece[1:].upper()
    # No color prefix (e.g. 'A1', 'a1', 'b1', 'q') — add color and uppercase.
    return color + piece.upper()


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
