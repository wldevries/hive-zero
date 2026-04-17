"""Interactive human vs AI play mode for Zertz."""

import random
import re
import sys

import numpy as np
import torch


def make_eval_fn(model, device):
    """Return a callable that runs the ZertzNet on a board tensor batch."""
    model.eval()

    def eval_fn(board_np, reserve_np):
        board = torch.from_numpy(np.array(board_np)).float().to(device)
        reserve = torch.from_numpy(np.array(reserve_np)).float().to(device)
        with torch.no_grad():
            place, cap_dir, value = model(board, reserve)
        return (
            place.cpu().numpy(),
            cap_dir.cpu().numpy(),
            value.squeeze(-1).cpu().numpy(),
        )

    return eval_fn


_COORD_RE = re.compile(r"^[A-G][1-7]$")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _parse_wgb_counts(line):
    """Parse W/G/B counts from a board status line."""
    clean = _ANSI_RE.sub("", line)
    m = re.search(r"W=(\d+)\s+G=(\d+)\s+B=(\d+)", clean)
    if not m:
        return None
    return {"W": int(m.group(1)), "G": int(m.group(2)), "B": int(m.group(3))}


def _get_resource_state(board_str):
    """Extract supply + captures from board_str."""
    supply = None
    captures = {0: None, 1: None}
    for line in board_str.splitlines():
        if line.startswith("Supply:"):
            supply = _parse_wgb_counts(line)
        elif line.startswith("Captures P1:"):
            captures[0] = _parse_wgb_counts(line)
        elif line.startswith("Captures P2:"):
            captures[1] = _parse_wgb_counts(line)
    return supply, captures


def _coord_on_board(coord):
    """Return True if coordinate is a valid Zertz board cell."""
    if not _COORD_RE.match(coord):
        return False
    col = ord(coord[0]) - ord("A") - 3
    row = int(coord[1])
    r = 4 - row
    q = col
    return abs(q) <= 3 and abs(r) <= 3 and abs(q + r) <= 3


def _build_placement_index(moves):
    """Index legal placement moves for detailed validation feedback."""
    valid_set = set(moves)
    colors_available = set()
    placeonly = set()  # (color, place)
    remove_by = {}  # (color, place) -> set(remove coords)
    places_any_color = set()  # place coords that are legal for at least one color
    removable_coords = set()  # coords that appear as legal remove targets

    for mv in moves:
        parts = mv.split()
        if not parts:
            continue
        if parts[0] not in ("W", "G", "B"):
            continue
        color = parts[0]
        colors_available.add(color)

        if len(parts) == 2:
            place = parts[1]
            placeonly.add((color, place))
            places_any_color.add(place)
        elif len(parts) == 3:
            place, remove = parts[1], parts[2]
            places_any_color.add(place)
            removable_coords.add(remove)
            remove_by.setdefault((color, place), set()).add(remove)

    return {
        "valid_set": valid_set,
        "colors_available": colors_available,
        "placeonly": placeonly,
        "remove_by": remove_by,
        "places_any_color": places_any_color,
        "removable_coords": removable_coords,
    }


def parse_placement(s, moves, board_str, current_player):
    """Parse placement input and return (move_str_or_None, feedback_message)."""
    parts = s.strip().upper().split()
    if not parts:
        return None, "Enter a move."

    if parts[0] not in ("W", "G", "B"):
        return None, "Invalid color — start with W, G, or B."

    if len(parts) not in (2, 3):
        return None, "Invalid format — use: W/G/B <place> [remove]."

    color = parts[0]
    place = parts[1]
    idx = _build_placement_index(moves)
    supply, captures = _get_resource_state(board_str)

    if color not in idx["colors_available"]:
        if supply and all(v == 0 for v in supply.values()):
            cap_counts = captures.get(current_player) or {}
            cap_n = cap_counts.get(color, 0)
            if cap_n == 0:
                return (
                    None,
                    f"Supply is empty — you must place from your captures, and you have no {color} captures.",
                )
            return (
                None,
                f"Supply is empty — you place from captures. You have {cap_n} {color}, but {color} cannot be legally placed now.",
            )

        if supply and supply.get(color, 0) == 0:
            return None, f"No {color} marbles left in supply."

        return None, f"No {color} marbles available to place."

    if not _COORD_RE.match(place):
        return None, f"Invalid place coordinate '{place}' — use A1..G7."
    if not _coord_on_board(place):
        return None, f"{place} is not on the board (no ring there)."

    key = (color, place)
    has_placeonly = key in idx["placeonly"]
    legal_removes = idx["remove_by"].get(key, set())

    if not has_placeonly and not legal_removes:
        if place not in idx["places_any_color"]:
            return None, f"Cannot place at {place}: no ring to place on there."
        return None, f"Cannot place {color} at {place}."

    if len(parts) == 2:
        if legal_removes:
            return (
                None,
                f"Placing at {place} requires removing a ring; add a remove coordinate.",
            )
        move_str = f"{color} {place}"
        if move_str in idx["valid_set"]:
            return move_str, None
        return None, "Not a legal move."

    remove = parts[2]
    if not _COORD_RE.match(remove):
        return None, f"Invalid remove coordinate '{remove}' — use A1..G7."
    if not _coord_on_board(remove):
        return None, f"{remove} is not on the board (no ring there)."
    if remove == place:
        return (
            None,
            f"Cannot remove {remove}: you cannot remove the ring you just placed on.",
        )

    move_str = f"{color} {place} {remove}"
    if move_str in idx["valid_set"]:
        return move_str, None

    if has_placeonly and not legal_removes:
        return (
            None,
            f"No ring can be removed after placing at {place}; omit the remove coordinate.",
        )

    if remove not in idx["removable_coords"]:
        return None, f"Cannot remove {remove}: there is no removable ring there."

    return None, f"Cannot remove {remove} after placing at {place}."


def run(model_path=None, device="cuda", simulations=200, human_color=None):
    import engine_zero  # the compiled Rust extension

    # Load model if provided
    eval_fn = None
    if model_path:
        from zertz.nn.model import load_checkpoint

        model, _ = load_checkpoint(model_path)
        model = model.to(device)
        eval_fn = make_eval_fn(model, device)

    game = engine_zero.ZertzGame()

    # Decide who plays as whom
    if human_color is None:
        human_player = random.randint(0, 1)
    else:
        human_player = 0 if human_color.lower() in ("p1", "1") else 1

    player_names = ["P1", "P2"]
    print(
        f"\nYou are {player_names[human_player]}. AI is {player_names[1 - human_player]}."
    )
    if eval_fn is None:
        print("No model loaded — AI will play randomly.")
    print()

    move_count = 0
    while game.outcome() == "ongoing":
        print(game.board_str())
        current = game.next_player()
        moves = game.valid_moves()

        if not moves:
            print("No legal moves — game over.")
            break

        is_capture_turn = moves[0].startswith("CAP")

        if current == human_player:
            # Human's turn
            if is_capture_turn:
                print(f"Your turn ({player_names[current]}) — captures are mandatory:")
                for i, mv in enumerate(moves):
                    print(f"  {i + 1}) {mv}")
                while True:
                    try:
                        choice = input("Pick a number: ").strip()
                        idx = int(choice) - 1
                        if 0 <= idx < len(moves):
                            chosen = moves[idx]
                            break
                        print(f"Enter a number between 1 and {len(moves)}.")
                    except (ValueError, EOFError):
                        print("Enter a number.")
            else:
                # Placement turn
                board_now = game.board_str()
                supply_line = _parse_supply(board_now)
                print(f"Your turn ({player_names[current]}) — place a marble.")
                print(f"  Supply: {supply_line}")
                supply_counts, capture_counts = _get_resource_state(board_now)
                if supply_counts and all(v == 0 for v in supply_counts.values()):
                    mine = capture_counts.get(current) or {"W": 0, "G": 0, "B": 0}
                    print(
                        "  Supply is empty — you must place from your captures "
                        f"(W={mine['W']} G={mine['G']} B={mine['B']})."
                    )
                print("  Input format: W/G/B <place> <remove>   e.g. W D4 D3")
                print("  (omit <remove> if no ring can be removed)")
                while True:
                    try:
                        raw = input("Move: ").strip()
                    except EOFError:
                        sys.exit(0)
                    if not raw:
                        continue
                    mv, feedback = parse_placement(raw, moves, board_now, current)
                    if mv is not None:
                        chosen = mv
                        break
                    print(f"  {feedback}")
        else:
            # AI's turn
            print(f"AI ({player_names[current]}) is thinking...", end="", flush=True)
            if eval_fn is not None:
                chosen = game.best_move(eval_fn, simulations=simulations)
            else:
                chosen = random.choice(moves)
            print(f"\rAI ({player_names[current]}) plays: {chosen}          ")

        game.play(chosen)
        move_count += 1

    print()
    print(game.board_str())
    outcome = game.outcome()
    if outcome == "p1":
        winner = "P1"
    elif outcome == "p2":
        winner = "P2"
    elif outcome == "draw":
        winner = None
    else:
        winner = None

    if winner:
        you_won = (winner == "P1" and human_player == 0) or (
            winner == "P2" and human_player == 1
        )
        if you_won:
            print(f"{winner} wins — you win! Congratulations!")
        else:
            print(f"{winner} wins — AI wins.")
    else:
        print("Draw.")


def _parse_supply(board_str):
    """Extract the supply line from the board string for display."""
    for line in board_str.splitlines():
        if line.startswith("Supply:"):
            # Strip ANSI codes for terminal display (they're already colored)
            return line
    return ""
