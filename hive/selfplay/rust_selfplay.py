"""Rust-accelerated self-play for training.

Uses RustSelfPlaySession for the full game loop in Rust,
with Python only handling NN inference and training.
"""

from __future__ import annotations
import numpy as np

from ..encoding.move_encoder import POLICY_SIZE


class RustParallelSelfPlay:
    """Self-play using Rust game loop with Python NN inference callback.

    The entire game loop (MCTS, move selection, training data collection)
    runs in Rust. Python only provides the eval_fn for GPU inference.
    """

    def __init__(self, model, device: str = "cpu",
                 simulations: int = 100, max_moves: int = 200,
                 temperature: float = 1.0, temp_threshold: int = 30,
                 resign_threshold: float = -0.97, resign_moves: int = 5,
                 resign_min_moves: int = 20,
                 calibration_frac: float = 0.1,
                 playout_cap_p: float = 0.0,
                 fast_cap: int = 20,
                 leaf_batch_size: int = 1,
                 random_opening_moves: int = 0,
                 **kwargs):
        self.model = model
        self.device = device
        self.simulations = simulations
        self.max_moves = max_moves
        self.temperature = temperature
        self.temp_threshold = temp_threshold
        self.resign_threshold = resign_threshold
        self.resign_moves = resign_moves
        self.resign_min_moves = resign_min_moves
        self.calibration_frac = calibration_frac
        self.playout_cap_p = playout_cap_p
        self.fast_cap = fast_cap
        self.leaf_batch_size = leaf_batch_size
        self.random_opening_moves = random_opening_moves

    def _eval_fn(self):
        """Return a callable for Rust's GPU inference callback."""
        import torch
        model = self.model
        device = self.device

        def eval_fn(board_batch, reserve_batch):
            board_4d = np.asarray(board_batch)
            reserves = np.asarray(reserve_batch)
            if model is None:
                n = board_4d.shape[0]
                return (np.ones((n, POLICY_SIZE), dtype=np.float32) / POLICY_SIZE,
                        np.zeros(n, dtype=np.float32))
            bt = torch.tensor(board_4d).to(device)
            rv = torch.tensor(reserves).to(device)
            with torch.no_grad():
                policy_logits, values = model(bt, rv)
            policy = torch.softmax(policy_logits, dim=1).cpu().numpy()
            vals = values.cpu().numpy().flatten()
            return policy.astype(np.float32), vals.astype(np.float32)

        return eval_fn

    def play_games(self, num_games: int, opening_sequences: list[list[str]] | None = None):
        """Play num_games entirely in Rust. Returns SelfPlayResult.

        opening_sequences: per-game UHP move lists to replay before MCTS.
            Empty inner list (or None) means use random_opening_moves for that game.
        """
        from hive_engine import RustSelfPlaySession

        session = RustSelfPlaySession(
            num_games=num_games,
            simulations=self.simulations,
            max_moves=self.max_moves,
            temperature=self.temperature,
            temp_threshold=self.temp_threshold,
            playout_cap_p=self.playout_cap_p,
            fast_cap=self.fast_cap,
            c_puct=1.5,
            leaf_batch_size=self.leaf_batch_size,
            resign_threshold=self.resign_threshold,
            resign_moves=self.resign_moves,
            resign_min_moves=self.resign_min_moves,
            calibration_frac=self.calibration_frac,
            random_opening_moves=self.random_opening_moves,
        )

        from tqdm import tqdm

        pbar = tqdm(total=self.max_moves, unit="turn", desc="  Self-play", leave=False)

        def progress(finished, total, active, moves, resigned, max_turn=0):
            advance = max_turn - pbar.n
            if advance > 0:
                pbar.update(advance)
            pbar.set_postfix(active=f"{active}/{total}",
                             resigned=resigned if resigned else None)

        result = session.play_games(self._eval_fn(), progress,
                                    opening_sequences=opening_sequences)
        pbar.update(pbar.total - pbar.n)
        pbar.close()
        return result


def _render_boards_horizontally(board_strings: list[str], labels: list[str] | None = None, sep: str = "   ") -> str:
    """Render multiple board strings side-by-side, handling ANSI escape codes."""
    import re
    _ANSI = re.compile(r'\033\[[0-9;]*m')

    def visual_len(s: str) -> int:
        return len(_ANSI.sub('', s))

    boards_lines = [b.split('\n') for b in board_strings]
    board_widths = [max((visual_len(line) for line in lines), default=0) for lines in boards_lines]

    if labels:
        board_widths = [max(w, len(labels[i])) for i, w in enumerate(board_widths)]

    max_height = max(len(lines) for lines in boards_lines)
    all_lines = []

    if labels:
        all_lines.append(sep.join(lbl.ljust(board_widths[i]) for i, lbl in enumerate(labels)))

    for row in range(max_height):
        parts = []
        for bi, lines in enumerate(boards_lines):
            if row < len(lines):
                line = lines[row]
                padding = board_widths[bi] - visual_len(line)
                parts.append(line + ' ' * padding)
            else:
                parts.append(' ' * board_widths[bi])
        all_lines.append(sep.join(parts))

    return '\n'.join(all_lines)
