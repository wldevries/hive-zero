"""Interactive human vs AI play mode for Tic-Tac-Toe."""

import random
import sys

import numpy as np
import torch

from tictactoe.nn.model import load_checkpoint


CELL_CHARS = {0: '\u00b7', 1: 'X', 2: 'O'}


def _render_board(board):
    """Render a 9-element board list. 0=empty, 1=X, 2=O."""
    lines = []
    lines.append("  1 2 3")
    for row in range(3):
        cells = " ".join(CELL_CHARS[board[row * 3 + col]] for col in range(3))
        lines.append(f"{row + 1} {cells}")
    return "\n".join(lines)


def run(model_path=None, device="cuda", simulations=800, human_color=None):
    from engine_zero import TTTGame

    if model_path is None:
        print("No model specified. Use --model to load a trained model.")
        sys.exit(1)

    model, ckpt = load_checkpoint(model_path)
    model.to(device)
    model.eval()
    gen = ckpt.get("generation", "?")
    blocks = len(model.res_blocks)
    ch = model.input_conv.out_channels
    print(f"Loaded model: {model_path} (gen {gen}, {blocks}b/{ch}ch, {simulations} sims)")

    def eval_fn(board_tensor_np):
        bt = torch.from_numpy(np.array(board_tensor_np)).to(device, dtype=torch.float32)
        with torch.no_grad():
            policy, value = model(bt)
        return (
            policy.float().cpu().numpy(),
            value.float().cpu().numpy().squeeze(1),
        )

    # Determine human color
    if human_color is None:
        human_player = random.choice([1, 2])
    elif human_color.lower() in ("x", "1", "p1"):
        human_player = 1
    else:
        human_player = 2

    human_mark = "X" if human_player == 1 else "O"
    ai_player = 3 - human_player
    ai_mark = "X" if ai_player == 1 else "O"
    print(f"You are {human_mark}, AI is {ai_mark}")
    print(f"Enter moves as 'row col' (1-3 each), e.g. '1 1' for top-left\n")

    game = TTTGame()

    while True:
        board = game.board()
        print(_render_board(board))
        outcome = game.outcome()
        if outcome is not None:
            if outcome == 0:
                print("\nDraw!")
            elif outcome == human_player:
                print(f"\nYou win!")
            else:
                print(f"\nAI wins!")
            break

        current = game.current_player()

        if current == human_player:
            while True:
                try:
                    line = input(f"\nYour move ({human_mark}): ").strip()
                    if line.lower() in ("q", "quit", "exit"):
                        print("Goodbye!")
                        return
                    parts = line.split()
                    if len(parts) != 2:
                        print("Enter row and column (1-3), e.g. '1 1'")
                        continue
                    row, col = int(parts[0]) - 1, int(parts[1]) - 1
                    if not (0 <= row < 3 and 0 <= col < 3):
                        print("Row and column must be 1-3")
                        continue
                    idx = row * 3 + col
                    if board[idx] != 0:
                        print("That cell is already taken")
                        continue
                    break
                except (ValueError, EOFError):
                    print("Enter row and column (1-3), e.g. '1 1'")
                    continue
            game.play_move(idx)
        else:
            move, value = game.best_move(eval_fn, simulations=simulations, c_puct=1.5)
            row, col = divmod(int(move), 3)
            print(f"\nAI plays ({ai_mark}): {row + 1} {col + 1}  (value: {value:+.3f})")
            game.play_move(int(move))

    # Ask to play again
    try:
        again = input("\nPlay again? (y/n): ").strip().lower()
        if again in ("y", "yes"):
            run(model_path=model_path, device=device, simulations=simulations, human_color=human_color)
    except EOFError:
        pass
