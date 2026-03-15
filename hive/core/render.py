"""Console hex board renderer for Hive."""

from __future__ import annotations

from .hex import Hex
from .board import Board
from .pieces import PieceColor


def render_board(board: Board) -> str:
    r"""Render a board to a string using compact flat-top hex tiles.

    Uses 4-char wide hexes with 2-char piece labels (e.g. wQ, bA).
    Stacked hexes show a reference number in the bottom row, with
    a legend below the board listing the full stack top-to-bottom.
    """
    top_pieces = board.all_top_pieces()
    if not top_pieces:
        return "  (empty board)"

    # Build label map and identify stacks
    labels: dict[Hex, str] = {}
    stacks: dict[Hex, list] = {}  # hex -> full stack (only for stacked positions)
    for pos, piece in top_pieces:
        labels[pos] = f"{piece.color.value}{piece.piece_type.value}"
        stack = board.stack_at(pos)
        if len(stack) > 1:
            stacks[pos] = stack

    # Assign reference numbers to stacks
    stack_refs: dict[Hex, int] = {}
    for i, pos in enumerate(sorted(stacks.keys(), key=lambda h: (h.r, h.q)), 1):
        stack_refs[pos] = i

    # Screen coordinates for flat-top hex at (q, r):
    #   screen_x = q * 3  (each hex is 4 wide, overlapping by 1)
    #   screen_y = r * 2 + q  (axial to screen: diagonal q axis)
    positions = {}
    for pos in labels:
        sx = pos.q * 3
        sy = pos.r * 2 + pos.q
        positions[pos] = (sx, sy)

    if not positions:
        return "  (empty board)"

    # Find bounds and normalize to 0-based
    min_sx = min(sx for sx, sy in positions.values())
    max_sx = max(sx for sx, sy in positions.values())
    min_sy = min(sy for sx, sy in positions.values())
    max_sy = max(sy for sx, sy in positions.values())

    for pos in list(positions):
        sx, sy = positions[pos]
        positions[pos] = (sx - min_sx, sy - min_sy)

    # Canvas dimensions
    width = (max_sx - min_sx) + 5
    height = (max_sy - min_sy) + 3

    # Build canvas
    canvas = [[' '] * width for _ in range(height)]

    # ANSI colors
    WHITE_FG = '\033[97m'   # bright white
    BLACK_FG = '\033[93m'   # yellow for black pieces (visible on dark bg)
    DIM = '\033[2m'         # dim for stack reference
    RESET = '\033[0m'

    color_labels: dict[tuple[int, int], str] = {}  # (row, col) -> colored string

    # Draw non-stacked hexes first, then stacked ones on top
    sorted_positions = sorted(positions.items(), key=lambda x: x[0] in stacks)

    for pos, (sx, sy) in sorted_positions:
        label = labels[pos]
        has_stack = pos in stacks

        # Top: __ at (sy, sx+1..sx+2)
        canvas[sy][sx + 1] = '_'
        canvas[sy][sx + 2] = '_'

        # Middle: /XX\ at (sy+1, sx..sx+3)
        canvas[sy + 1][sx] = '/'
        canvas[sy + 1][sx + 1] = label[0]
        canvas[sy + 1][sx + 2] = label[1]
        canvas[sy + 1][sx + 3] = '\\'

        # Bottom row
        canvas[sy + 2][sx] = '\\'
        canvas[sy + 2][sx + 3] = '/'
        if has_stack:
            ref = str(stack_refs[pos])
            # Right-align single digit, use both chars for 2-digit
            if len(ref) == 1:
                canvas[sy + 2][sx + 1] = '#'
                canvas[sy + 2][sx + 2] = ref[0]
            else:
                canvas[sy + 2][sx + 1] = ref[0]
                canvas[sy + 2][sx + 2] = ref[1]
        else:
            canvas[sy + 2][sx + 1] = '_'
            canvas[sy + 2][sx + 2] = '_'

        # Record colored top label
        piece = dict(top_pieces)[pos]
        color = WHITE_FG if piece.color == PieceColor.WHITE else BLACK_FG
        color_labels[(sy + 1, sx + 1)] = f"{color}{label}{RESET}"

        # Record colored stack ref
        if has_stack:
            ref = str(stack_refs[pos])
            ref_text = f"#{ref}" if len(ref) == 1 else ref
            color_labels[(sy + 2, sx + 1)] = f"{DIM}{ref_text}{RESET}"

    # Build output string, replacing label chars with colored versions
    lines = []
    for row_idx, row in enumerate(canvas):
        line = ''
        col = 0
        while col < len(row):
            if (row_idx, col) in color_labels:
                line += color_labels[(row_idx, col)]
                col += 2  # skip both label chars
            else:
                line += row[col]
                col += 1
        lines.append(line.rstrip())

    # Remove leading/trailing blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    # Add stack legend
    if stack_refs:
        lines.append('')
        for pos in sorted(stack_refs.keys(), key=lambda h: stack_refs[h]):
            ref = stack_refs[pos]
            stack = stacks[pos]
            # Show top to bottom
            parts = []
            for piece in reversed(stack):
                pc = WHITE_FG if piece.color == PieceColor.WHITE else BLACK_FG
                parts.append(f"{pc}{piece}{RESET}")
            lines.append(f"  #{ref}: {' > '.join(parts)}")

    return '\n'.join(lines)
