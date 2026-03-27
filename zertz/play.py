"""Interactive human vs AI play mode for Zertz."""

import random
import sys
import torch
import numpy as np


def make_eval_fn(model, device):
    """Return a callable that runs the ZertzNet on a board tensor batch."""
    model.eval()

    def eval_fn(board_np):
        # board_np is (N, C, H, W) numpy array from Rust
        t = torch.from_numpy(np.array(board_np)).float().to(device)
        with torch.no_grad():
            policy_logits, value = model(t)
        policy = torch.softmax(policy_logits, dim=-1).cpu().numpy()
        value = value.squeeze(-1).cpu().numpy()
        return policy, value

    return eval_fn


def parse_placement(s, valid_set):
    """Parse 'W D4 D3' or 'G A4' and return the move string if valid."""
    parts = s.strip().split()
    if not parts:
        return None
    parts[0] = parts[0].upper()
    if parts[0] not in ("W", "G", "B"):
        return None
    move_str = " ".join(parts)
    if move_str in valid_set:
        return move_str
    return None


def run(model_path=None, device="cuda", simulations=200, human_color=None):
    import hive_engine  # the compiled Rust extension

    # Load model if provided
    eval_fn = None
    if model_path:
        from zertz.nn.model import load_checkpoint
        model, _ = load_checkpoint(model_path)
        model = model.to(device)
        eval_fn = make_eval_fn(model, device)

    game = hive_engine.ZertzGame()

    # Decide who plays as whom
    if human_color is None:
        human_player = random.randint(0, 1)
    else:
        human_player = 0 if human_color.lower() in ("p1", "1") else 1

    player_names = ["P1", "P2"]
    print(f"\nYou are {player_names[human_player]}. AI is {player_names[1 - human_player]}.")
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
                valid_set = set(moves)
                supply_line = _parse_supply(game.board_str())
                print(f"Your turn ({player_names[current]}) — place a marble.")
                print(f"  Supply: {supply_line}")
                print("  Input format: W/G/B <place> <remove>   e.g. W D4 D3")
                print("  (omit <remove> if no ring can be removed)")
                while True:
                    try:
                        raw = input("Move: ").strip()
                    except EOFError:
                        sys.exit(0)
                    if not raw:
                        continue
                    mv = parse_placement(raw, valid_set)
                    if mv is not None:
                        chosen = mv
                        break
                    # Show close matches to help the user
                    parts = raw.strip().upper().split()
                    if parts and parts[0] in ("W", "G", "B"):
                        prefix = parts[0]
                        hints = [m for m in moves if m.startswith(prefix)][:6]
                        if hints:
                            print(f"  Not a legal move. Legal moves starting with {prefix}:")
                            for h in hints:
                                print(f"    {h}")
                    else:
                        print("  Invalid — start with W, G, or B.")
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
        you_won = (winner == "P1" and human_player == 0) or (winner == "P2" and human_player == 1)
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
