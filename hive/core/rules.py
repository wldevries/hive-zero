"""Movement rules for each Hive piece type."""

from __future__ import annotations
from .hex import Hex, DIRECTIONS
from .board import Board
from .pieces import Piece, PieceType, PieceColor


def get_moves(piece: Piece, board: Board) -> set[Hex]:
    """Get all legal destination hexes for a piece already on the board."""
    pos = board.piece_position(piece)
    if pos is None:
        return set()

    # Piece must be on top of its stack to move
    if board.top_piece(pos) != piece:
        return set()

    # One Hive Rule: removing this piece must not disconnect the hive
    # (Beetles on top of stacks don't break connectivity when they move)
    if board.stack_height(pos) == 1:
        if not board.is_connected(exclude=piece):
            return set()

    move_fn = _MOVE_FUNCTIONS.get(piece.piece_type)
    if move_fn is None:
        return set()

    return move_fn(piece, pos, board)


def get_placements(color: PieceColor, board: Board) -> set[Hex]:
    """Get all legal placement positions for a player.

    A new piece must be placed adjacent to at least one friendly piece
    and NOT adjacent to any enemy piece. Exception: on the second move
    of the game (total move 2), the constraint about enemy adjacency
    is relaxed.
    """
    my_pieces = board.pieces_on_board(color)
    opp_color = PieceColor.BLACK if color == PieceColor.WHITE else PieceColor.WHITE
    opp_pieces = board.pieces_on_board(opp_color)

    # First piece: place at origin
    if not my_pieces and not opp_pieces:
        return {Hex(0, 0)}

    # Second piece (first move by opponent): adjacent to the first piece
    if not my_pieces and opp_pieces:
        result = set()
        for p in opp_pieces:
            pos = board.piece_position(p)
            result.update(board.empty_neighbors(pos))
        return result

    # Normal placement: adjacent to friendly, not adjacent to enemy
    friendly_positions = {board.piece_position(p) for p in my_pieces}
    enemy_positions = {board.piece_position(p) for p in opp_pieces}

    # All empty hexes adjacent to friendly pieces
    candidates = set()
    for fpos in friendly_positions:
        candidates.update(board.empty_neighbors(fpos))

    # Filter out those adjacent to enemy pieces
    valid = set()
    for cand in candidates:
        adjacent_to_enemy = False
        for n in cand.neighbors():
            top = board.top_piece(n)
            if top is not None and top.color == opp_color:
                adjacent_to_enemy = True
                break
        if not adjacent_to_enemy:
            valid.add(cand)

    return valid


# ---- Piece-specific movement ----

def _queen_moves(piece: Piece, pos: Hex, board: Board) -> set[Hex]:
    """Queen: moves exactly 1 space by sliding."""
    board.remove_piece(piece)
    moves = set()
    for n in pos.neighbors():
        if n not in board.occupied and board.can_slide(pos, n):
            # Must remain adjacent to hive after moving
            if any(adj in board.occupied for adj in n.neighbors()):
                moves.add(n)
    board.place_piece(piece, pos)
    return moves


def _beetle_moves(piece: Piece, pos: Hex, board: Board) -> set[Hex]:
    """Beetle: moves 1 space, can climb on top of the hive."""
    board.remove_piece(piece)
    moves = set()
    for n in pos.neighbors():
        # Beetle can move to occupied or empty adjacent hexes
        src_height = board.stack_height(pos)  # height after removing beetle
        dst_height = board.stack_height(n)

        # If both source and dest are ground level, normal slide rules apply
        if src_height == 0 and dst_height == 0:
            if board.can_slide(pos, n):
                # Must stay adjacent to hive
                if any(adj in board.occupied for adj in n.neighbors()):
                    moves.add(n)
        else:
            # Moving up, down, or across the top of the hive
            # Gate rule for elevated movement: need to check if the beetle
            # can physically pass between the two common neighbors
            # At height, the beetle can always move (no gate blocking at elevation)
            # Actually: gate blocking applies based on max height of common neighbors
            # vs max(src_height, dst_height). Simplified: if beetle or target is elevated,
            # gate doesn't block.
            if dst_height > 0 or any(adj in board.occupied for adj in n.neighbors()):
                moves.add(n)

    board.place_piece(piece, pos)
    return moves


def _grasshopper_moves(piece: Piece, pos: Hex, board: Board) -> set[Hex]:
    """Grasshopper: jumps in a straight line over at least one piece."""
    board.remove_piece(piece)
    moves = set()
    for d in DIRECTIONS:
        # Must jump over at least one piece
        current = pos + d
        if current not in board.occupied:
            continue
        # Keep going until we find an empty hex
        while current in board.occupied:
            current = current + d
        moves.add(current)
    board.place_piece(piece, pos)
    return moves


def _spider_moves(piece: Piece, pos: Hex, board: Board) -> set[Hex]:
    """Spider: moves exactly 3 spaces by sliding along the hive."""
    board.remove_piece(piece)
    moves = _walk_exactly_n(pos, 3, board)
    board.place_piece(piece, pos)
    return moves


def _ant_moves(piece: Piece, pos: Hex, board: Board) -> set[Hex]:
    """Ant: slides any number of spaces around the hive perimeter."""
    board.remove_piece(piece)
    moves = _walk_any(pos, board)
    board.place_piece(piece, pos)
    return moves


def _walk_exactly_n(start: Hex, n: int, board: Board) -> set[Hex]:
    """Find all hexes reachable by sliding exactly n steps along the hive edge."""
    # BFS tracking (position, steps_taken), visited per step to allow revisiting at different steps
    results = set()
    # State: (position, steps_taken), track visited as set of (pos, step) to prevent loops
    frontier = [(start, 0, frozenset([start]))]

    while frontier:
        pos, steps, visited = frontier.pop()
        if steps == n:
            if pos != start:
                results.add(pos)
            continue

        for neighbor in pos.neighbors():
            if neighbor in board.occupied:
                continue
            if neighbor in visited:
                continue
            # Must be able to slide there
            if not board.can_slide(pos, neighbor):
                continue
            # Must remain adjacent to the hive
            if not any(adj in board.occupied for adj in neighbor.neighbors()):
                continue
            frontier.append((neighbor, steps + 1, visited | {neighbor}))

    return results


def _walk_any(start: Hex, board: Board) -> set[Hex]:
    """Find all hexes reachable by sliding any number of steps (ant movement)."""
    visited = {start}
    queue = [start]
    results = set()

    while queue:
        pos = queue.pop()
        for neighbor in pos.neighbors():
            if neighbor in board.occupied:
                continue
            if neighbor in visited:
                continue
            if not board.can_slide(pos, neighbor):
                continue
            if not any(adj in board.occupied for adj in neighbor.neighbors()):
                continue
            visited.add(neighbor)
            results.add(neighbor)
            queue.append(neighbor)

    return results


_MOVE_FUNCTIONS = {
    PieceType.QUEEN: _queen_moves,
    PieceType.BEETLE: _beetle_moves,
    PieceType.GRASSHOPPER: _grasshopper_moves,
    PieceType.SPIDER: _spider_moves,
    PieceType.ANT: _ant_moves,
}
