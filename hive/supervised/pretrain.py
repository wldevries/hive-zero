"""Supervised pre-training from high-ELO human game records.

Converts Boardspace SGF archives into (board, reserve, policy, value) samples
and trains HiveNet via imitation learning before self-play begins.

Policy target : one-hot on the move actually played.
Value target  : +1 win / 0 draw / −1 loss from the current player's POV.
"""

from __future__ import annotations

import csv

from ..training_log import LOG_HEADER
import os
import random
import time
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import colorama
colorama.init()
_R = colorama.Style.RESET_ALL
_B = colorama.Style.BRIGHT
_cr = lambda v: f"{colorama.Fore.RED}{_B}{v}{_R}"       # total loss
_cy = lambda v: f"{colorama.Fore.YELLOW}{_B}{v}{_R}"   # policy / value loss
_cc = lambda v: f"{colorama.Fore.CYAN}{_B}{v}{_R}"     # chunk / epoch labels

from hive_engine import parse_sgf_moves as parse_moves
from ..uhp import normalize_piece as _normalize_piece

# Heavy imports (torch, hive_engine) are deferred to class/function bodies
# so that `main.py` can be imported without torch installed or the Rust
# extension built.
_NUM_CHANNELS: int | None = None
_GRID_SIZE: int | None = None
_POLICY_SIZE: int | None = None


def _load_encoding_consts() -> tuple[int, int, int]:
    global _NUM_CHANNELS, _GRID_SIZE, _POLICY_SIZE
    if _NUM_CHANNELS is None:
        from ..encoding.board_encoder import NUM_CHANNELS, GRID_SIZE
        from ..encoding.move_encoder import POLICY_SIZE
        _NUM_CHANNELS, _GRID_SIZE, _POLICY_SIZE = NUM_CHANNELS, GRID_SIZE, POLICY_SIZE
    return _NUM_CHANNELS, _GRID_SIZE, _POLICY_SIZE


# ---------------------------------------------------------------------------
# UHP move parser (operates on hive_engine.RustGame state)
# ---------------------------------------------------------------------------

# Maps (prefix, suffix) direction notation to axial (dq, dr)
# Flat-top hexagons, same convention as hive/core/hex.py DIRECTIONS.
_DIRS: dict[tuple[str, str], tuple[int, int]] = {
    ("", "-"):  (1,  0),   # E   – suffix -
    ("", "/"):  (1, -1),   # NE  – suffix /
    ("\\", ""): (0, -1),   # NW  – prefix \
    ("-", ""):  (-1, 0),   # W   – prefix -
    ("/", ""):  (-1, 1),   # SW  – prefix /
    ("", "\\"): (0,  1),   # SE  – suffix \
}


def parse_uhp_move(
    game,
    move_str: str,
) -> tuple[str, Optional[tuple[int, int]], tuple[int, int]]:
    """Parse a UHP move string given the current RustGame state.

    Returns (piece_str, from_pos_or_None, to_pos).
    Raises ValueError on any parse failure.
    """
    parts = move_str.strip().split()
    if not parts:
        raise ValueError("empty move string")

    # Normalize piece name to UHP form used by the Rust engine (e.g. wQ1 → wQ).
    piece_str = _normalize_piece(parts[0])

    # Build piece-name → (q, r) map covering ALL pieces, including those buried
    # under beetle stacks (all_top_pieces only returns the top of each stack).
    piece_pos: dict[str, tuple[int, int]] = {}
    for (q, r), _ in game.all_top_pieces():
        for p in game.stack_at(q, r):
            piece_pos[p] = (q, r)

    # from_pos: None when placing from reserve, (q,r) when moving on board.
    from_pos: Optional[tuple[int, int]] = piece_pos.get(piece_str)

    if len(parts) == 1:
        # First placement of the game: no reference piece, origin cell.
        return (piece_str, None, (0, 0))

    pos_str = parts[1]
    prefix = suffix = ""
    ref_str = pos_str

    if pos_str[0] in ("-", "/", "\\"):
        prefix = pos_str[0]
        ref_str = pos_str[1:]
    elif pos_str[-1] in ("-", "/", "\\"):
        suffix = pos_str[-1]
        ref_str = pos_str[:-1]
    # else: no direction → beetle climbing on top of ref piece

    # Normalize reference piece name to UHP form (e.g. wQ1 → wQ).
    ref_str = _normalize_piece(ref_str)

    ref_pos = piece_pos.get(ref_str)
    if ref_pos is None:
        raise ValueError(f"reference piece {ref_str!r} not found on board")

    if not prefix and not suffix:
        # Beetle stacking: land on the same cell as the reference piece.
        to_pos = ref_pos
    else:
        delta = _DIRS.get((prefix, suffix))
        if delta is None:
            raise ValueError(
                f"unknown direction notation: prefix={prefix!r} suffix={suffix!r}"
            )
        to_pos = (ref_pos[0] + delta[0], ref_pos[1] + delta[1])

    return (piece_str, from_pos, to_pos)


# ---------------------------------------------------------------------------
# Piece validation
# ---------------------------------------------------------------------------

import re as _re
_BASE_PIECE_RE = _re.compile(r'^[wb][QBASG][123]?$')


def _is_base_piece(piece_str: str) -> bool:
    """Return True iff piece_str is a valid base-game piece (e.g. wQ, bA2)."""
    return bool(_BASE_PIECE_RE.match(piece_str))



# ---------------------------------------------------------------------------
# Single-game converter
# ---------------------------------------------------------------------------

def game_to_samples(
    sgf_content: str,
    result: str,
    verbose: bool = False,
    game_name: str = "",
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Convert one SGF game to a list of training samples.

    Args:
        sgf_content: Raw SGF text (iso-8859-1 decoded).
        result: 'p0_wins', 'p1_wins', or 'draw'.

    Returns:
        List of (board_tensor, reserve_vector, policy_target, value_target).
          board_tensor  : shape (NUM_CHANNELS, GRID_SIZE, GRID_SIZE), float32
          reserve_vector: shape (RESERVE_SIZE,), float32
          policy_target : shape (POLICY_SIZE,), float32 one-hot
          value_target  : float, ∈ {-1.0, 0.0, +1.0}
    """
    import hive_engine
    NUM_CHANNELS, GRID_SIZE, POLICY_SIZE = _load_encoding_consts()

    if result == "p0_wins":
        outcome = {"w": 1.0, "b": -1.0}
    elif result == "p1_wins":
        outcome = {"w": -1.0, "b": 1.0}
    else:
        outcome = {"w": 0.0, "b": 0.0}

    game = hive_engine.RustGame()
    samples: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []

    for move_str in parse_moves(sgf_content):
        if game.is_game_over:
            break

        if move_str == "pass":
            game.play_pass()
            continue

        # Encode the position *before* the move is played.
        board_arr, reserve_arr = game.encode_board()
        board = np.array(board_arr, dtype=np.float32).reshape(
            NUM_CHANNELS, GRID_SIZE, GRID_SIZE
        )
        reserve = np.array(reserve_arr, dtype=np.float32)
        value = outcome[game.turn_color]

        # Parse the played move into (piece, from, to).
        try:
            piece_str, from_pos, to_pos = parse_uhp_move(game, move_str)
        except ValueError as e:
            if verbose:
                print(f"  [skip] {game_name}: parse error on {move_str!r}: {e}")
            break

        # Reject non-base pieces in Python before hitting Rust.
        if not _is_base_piece(piece_str):
            if verbose:
                print(f"  [skip] {game_name}: non-base piece {piece_str!r} in {move_str!r}")
            break

        move_idx = game.encode_move(piece_str, from_pos, to_pos)
        if move_idx < 0:
            if verbose:
                print(f"  [skip] {game_name}: move outside encoding grid: {move_str!r}")
            break

        policy = np.zeros(POLICY_SIZE, dtype=np.float32)
        policy[move_idx] = 1.0

        samples.append((board, reserve, policy, value))

        # Advance the game state.
        game.play_move(piece_str, from_pos, to_pos)

    return samples


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_filtered_games(
    games_csv: str,
    elo_csv: str,
    min_elo: float = 1600.0,
    min_games: int = 20,
    exclude_players: set[str] | None = None,
) -> list[tuple[str, str, str]]:
    """Return a filtered list of (zip_file, sgf_name, result) tuples.

    Filters: base games, both players ELO ≥ min_elo with ≥ min_games played,
    outcome determined.
    """
    qualified: set[str] = set()
    with open(elo_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["games"]) >= min_games and float(row["elo"]) >= min_elo:
                qualified.add(row["player"])

    if exclude_players is None:
        exclude_players = set()

    games: list[tuple[str, str, str]] = []
    with open(games_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row["game_type"] == "base"
                and row["result"] in ("p0_wins", "p1_wins", "draw")
                and row["p0"] in qualified
                and row["p1"] in qualified
                and row["p0"] not in exclude_players
                and row["p1"] not in exclude_players
            ):
                games.append((row["zip_file"], row["sgf_name"], row["result"]))

    return games


def build_zip_index(boardspace_dir: str) -> dict[str, str]:
    """Walk boardspace_dir and return {zip_filename: full_path}."""
    index: dict[str, str] = {}
    for root, _, files in os.walk(boardspace_dir):
        for fname in files:
            if fname.endswith(".zip"):
                index[fname] = os.path.join(root, fname)
    return index


# ---------------------------------------------------------------------------
# Pre-trainer
# ---------------------------------------------------------------------------

class Pretrainer:
    """Supervised pre-training from Boardspace human game archives."""

    def __init__(
        self,
        model_path: str = "model.pt",
        device: str = "cuda",
        num_blocks: int = 6,
        channels: int = 64,
        lr: float = 0.005,
    ):
        from ..nn.model import create_model, load_checkpoint, save_checkpoint
        from ..nn.training import Trainer

        self.model_path = model_path
        self.device = device
        self._save_checkpoint = save_checkpoint

        if os.path.exists(model_path):
            self.model, ckpt = load_checkpoint(model_path)
            it = ckpt.get("iteration", 0)
            blocks = len(self.model.res_blocks)
            ch = self.model.input_conv.out_channels
            params = sum(p.numel() for p in self.model.parameters())
            print(
                f"Resumed from {model_path} "
                f"(iteration {it}, {blocks}b×{ch}ch, {params/1e6:.2f}M params)"
            )
        else:
            self.model = create_model(num_blocks, channels)
            params = sum(p.numel() for p in self.model.parameters())
            print(
                f"Created new model ({num_blocks} blocks, {channels} channels, "
                f"{params/1e6:.2f}M params)"
            )

        self.model.to(device)
        self.trainer = Trainer(self.model, device=device, lr=lr)

    def run(
        self,
        games: list[tuple[str, str, str]],
        zip_index: dict[str, str],
        num_epochs: int = 3,
        batch_size: int = 512,
        buffer_size: int = 100_000,
        epochs_per_chunk: int = 3,
        checkpoint_dir: str = "checkpoints",
        verbose_samples: bool = False,
    ) -> None:
        """Pre-train the model.

        Args:
            games: Filtered list of (zip_file, sgf_name, result).
            zip_index: Mapping from zip filename to full filesystem path.
            num_epochs: Full passes over the game list.
            batch_size: SGD mini-batch size.
            buffer_size: Max positions held in the training buffer before
                training and clearing.
            epochs_per_chunk: SGD epochs run each time the buffer is trained.
            checkpoint_dir: Directory for per-epoch checkpoints.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        log_path = f"{model_name}_log.csv"
        header = LOG_HEADER
        needs_header = True
        if os.path.exists(log_path):
            with open(log_path) as f:
                first = f.readline()
            needs_header = not first.startswith("iter,")
        log = open(log_path, "a", buffering=1)
        if needs_header:
            log.write(header)

        total_games = len(games)
        print(f"Dataset: {total_games} games | buffer: {buffer_size} | "
              f"epochs: {num_epochs} | SGD epochs/chunk: {epochs_per_chunk}")

        from ..nn.training import HiveDataset
        dataset = HiveDataset(max_size=buffer_size)
        chunk_idx = 0
        total_positions = 0
        total_errors = 0

        for epoch in range(1, num_epochs + 1):
            random.shuffle(games)
            positions_this_epoch = 0
            errors_this_epoch = 0

            print(f"\n=== Pre-train epoch {epoch}/{num_epochs} ===")
            epoch_start = time.time()
            losses: dict = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

            games_done = 0
            for zip_file, sgf_name, result in games:
                games_done += 1
                zip_path = zip_index.get(zip_file)

                if zip_path is None:
                    errors_this_epoch += 1
                else:
                    try:
                        with zipfile.ZipFile(zip_path, "r") as zf:
                            content = zf.read(sgf_name).decode("iso-8859-1")
                        samples = game_to_samples(content, result, verbose=verbose_samples, game_name=sgf_name)
                    except Exception:
                        errors_this_epoch += 1
                        samples = []

                    for board, reserve, policy, value in samples:
                        dataset.add_sample(board, reserve, policy, float(value))
                        total_positions += 1
                        positions_this_epoch += 1

                # Loading progress (overwrite line with \r).
                if games_done % 100 == 0 or len(dataset) >= buffer_size:
                    elapsed_load = time.time() - epoch_start
                    print(
                        f"\r  loading  games={games_done}/{total_games} "
                        f"buf={len(dataset)}/{buffer_size} "
                        f"[{elapsed_load:.1f}s]",
                        end="", flush=True,
                    )

                # Train when the buffer is full (or at end of epoch).
                buffer_full = len(dataset) >= buffer_size
                last_game = games_done == total_games
                if (buffer_full or last_game) and len(dataset) > 0:
                        print()  # newline after \r loading line
                        chunk_idx += 1
                        chunk_start = time.time()
                        for mini_epoch in range(1, epochs_per_chunk + 1):
                            losses = self.trainer.train_epoch(dataset, batch_size=batch_size)
                            elapsed = time.time() - chunk_start
                            tl = f"{losses['total_loss']:.4f}"
                            pl = f"{losses['policy_loss']:.4f}"
                            vl = f"{losses['value_loss']:.4f}"
                            print(
                                f"  chunk={_cc(chunk_idx)} mini={mini_epoch}/{epochs_per_chunk} "
                                f"loss={_cr(tl)} (pol={_cy(pl)} val={_cy(vl)}) [{elapsed:.1f}s]"
                            )
                        chunk_elapsed = time.time() - chunk_start
                        chunk_positions = len(dataset)
                        dataset.clear()

                        lr = self.trainer._current_lr
                        tl = f"{losses['total_loss']:.4f}"
                        pl = f"{losses['policy_loss']:.4f}"
                        vl = f"{losses['value_loss']:.4f}"
                        print(
                            f"  epoch={_cc(epoch)} games={games_done}/{total_games} "
                            f"pos={total_positions} chunk={_cc(chunk_idx)} "
                            f"loss={_cr(tl)} (policy={_cy(pl)} value={_cy(vl)}) "
                            f"lr={lr} [{chunk_elapsed:.1f}s]"
                        )
                        log.write(
                            f"{chunk_idx},pretrain,0,0,0,0,0,{total_positions},{chunk_positions},"
                            f"{losses['total_loss']:.6f},{losses['policy_loss']:.6f},"
                            f"{losses['value_loss']:.6f},{losses.get('qd_loss', 0):.6f},"
                            f"{lr:.8f},{chunk_elapsed:.1f},"
                            f"epoch={epoch},"
                            f"{losses.get('qe_loss', 0):.6f},{losses.get('mob_loss', 0):.6f}\n"
                        )


            total_errors += errors_this_epoch
            elapsed = time.time() - epoch_start
            print(
                f"  Epoch {epoch} done in {elapsed:.0f}s. "
                f"Positions this epoch: {positions_this_epoch}. "
                f"Errors: {errors_this_epoch}."
            )

            # Save model.pt and a numbered epoch checkpoint.
            epoch_losses = {
                "policy_loss": losses.get("policy_loss", 0),
                "value_loss": losses.get("value_loss", 0),
            }
            self._save_checkpoint(self.model, self.model_path, epoch, epoch_losses)
            ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_epoch{epoch}.pt")
            self._save_checkpoint(self.model, ckpt_path, epoch, epoch_losses)
            print(f"  Model saved → {self.model_path}  |  Checkpoint → {ckpt_path}")

        log.close()
        print(
            f"\nPre-training complete. "
            f"Total positions: {total_positions}. "
            f"Total errors: {total_errors}. "
            f"Model saved → {self.model_path}"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Supervised pre-training from human games")
    parser.add_argument("--games-csv", default="games/game_outcomes.csv")
    parser.add_argument("--elo-csv", default="games/player_elo.csv")
    parser.add_argument("--boardspace-dir", default="games/boardspace")
    parser.add_argument("--min-elo", type=float, default=1600.0)
    parser.add_argument("--min-games", type=int, default=20)
    parser.add_argument("--model", default="model.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--epochs-per-chunk", type=int, default=3)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    print("Loading filtered game list...")
    games = load_filtered_games(
        args.games_csv, args.elo_csv,
        min_elo=args.min_elo, min_games=args.min_games,
    )
    print(f"  {len(games)} qualifying games (ELO≥{args.min_elo}, games≥{args.min_games})")

    print("Indexing zip archives...")
    zip_index = build_zip_index(args.boardspace_dir)
    print(f"  {len(zip_index)} zip files found")

    pretrainer = Pretrainer(
        model_path=args.model,
        device=args.device,
        num_blocks=args.blocks,
        channels=args.channels,
        lr=args.lr,
    )
    pretrainer.run(
        games=games,
        zip_index=zip_index,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        epochs_per_chunk=args.epochs_per_chunk,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
