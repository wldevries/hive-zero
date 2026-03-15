"""Console hex board renderer for Hive."""

from __future__ import annotations

from .hex import Hex
from .board import Board
from .pieces import PieceColor


def render_board(board: Board) -> str:
    r"""Render a board to a string using compact flat-top hex tiles.

    Uses 4-char wide hexes with 2-char piece labels (e.g. wQ, bA).
    Stacked pieces show the piece underneath in the bottom row:
       __
      /wB\   <- top piece (beetle)
      \wQ/   <- piece underneath (queen)
    """
    top_pieces = board.all_top_pieces()
    if not top_pieces:
        return "  (empty board)"

    # Build maps of hex -> top label and hex -> bottom label (for stacks)
    labels: dict[Hex, str] = {}
    bottom_labels: dict[Hex, str] = {}
    for pos, piece in top_pieces:
        label = f"{piece.color.value}{piece.piece_type.value}"
        labels[pos] = label
        stack = board.stack_at(pos)
        if len(stack) > 1:
            below = stack[-2]  # piece just below top
            bottom_labels[pos] = f"{below.color.value}{below.piece_type.value}"

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

    # Find bounds
    min_sx = min(sx for sx, sy in positions.values())
    max_sx = max(sx for sx, sy in positions.values())
    min_sy = min(sy for sx, sy in positions.values())
    max_sy = max(sy for sx, sy in positions.values())

    # Normalize to 0-based
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
    DIM = '\033[2m'         # dim for bottom piece in stack
    RESET = '\033[0m'

    color_labels: dict[tuple[int, int], str] = {}  # (row, col) -> colored string

    # Draw non-stacked hexes first, then stacked ones on top
    sorted_positions = sorted(positions.items(), key=lambda x: x[0] in bottom_labels)

    for pos, (sx, sy) in sorted_positions:
        label = labels[pos]
        has_stack = pos in bottom_labels

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
            bl = bottom_labels[pos]
            canvas[sy + 2][sx + 1] = bl[0]
            canvas[sy + 2][sx + 2] = bl[1]
        else:
            canvas[sy + 2][sx + 1] = '_'
            canvas[sy + 2][sx + 2] = '_'

        # Record colored top label
        piece = dict(top_pieces)[pos]
        color = WHITE_FG if piece.color == PieceColor.WHITE else BLACK_FG
        color_labels[(sy + 1, sx + 1)] = f"{color}{label}{RESET}"

        # Record colored bottom label for stacks
        if has_stack:
            bl = bottom_labels[pos]
            stack = board.stack_at(pos)
            below = stack[-2]
            bcolor = WHITE_FG if below.color == PieceColor.WHITE else BLACK_FG
            color_labels[(sy + 2, sx + 1)] = f"{DIM}{bcolor}{bl}{RESET}"

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

    return '\n'.join(lines)
