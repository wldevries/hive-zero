"""Supervised pre-training from high-ELO human game records.

Converts Boardspace SGF archives into (board, reserve, policy, value) samples
and trains HiveNet via imitation learning before self-play begins.

Policy target : one-hot on the move actually played.
Value target  : +1 win / 0 draw / −1 loss from the current player's POV.
"""

from __future__ import annotations

import csv

from hive.selfplay.selfplay import LOG_HEADER
import os
import random
import time
import zipfile
from typing import Optional

import numpy as np
import colorama
colorama.init()
_R = colorama.Style.RESET_ALL
_B = colorama.Style.BRIGHT
_cr = lambda v: f"{colorama.Fore.RED}{_B}{v}{_R}"       # total loss
_cy = lambda v: f"{colorama.Fore.YELLOW}{_B}{v}{_R}"   # policy / value loss
_cc = lambda v: f"{colorama.Fore.CYAN}{_B}{v}{_R}"     # chunk / epoch labels

from engine_zero import parse_sgf_moves as parse_moves
from ..uhp import normalize_piece as _normalize_piece

# Heavy imports (torch, engine_zero) are deferred to class/function bodies
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
    return _NUM_CHANNELS, _GRID_SIZE, _POLICY_SIZE  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# UHP move parser (operates on engine_zero.HiveGame state)
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
    """Parse a UHP move string given the current HiveGame state.

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

def _result_from_game_state(state: str) -> str:
    """Derive 'p0_wins'/'p1_wins'/'draw'/'unknown' from a UHP game state string.

    UHP state format: e.g. 'Base+WhiteWins', 'Base+BlackWins', 'Base+Draw'.
    White = p0 (first player).
    """
    if "WhiteWins" in state:
        return "p0_wins"
    elif "BlackWins" in state:
        return "p1_wins"
    elif "Draw" in state:
        return "draw"
    return "unknown"


def game_to_samples(
    sgf_content: str,
    result: Optional[str] = None,
    verbose: bool = False,
    game_name: str = "",
    grid_size: int = 23,
) -> list[tuple]:
    """Convert one SGF game to a list of training samples.

    The per-position policy target is a sparse joint distribution over ALL
    legal moves at that position: the played move gets prob 1.0, every other
    legal move gets prob 0.0. Having the other legal moves present forms the
    softmax denominator at train time and matches what MCTS sees at inference.

    Args:
        sgf_content: Raw SGF text (iso-8859-1 decoded).
        result: 'p0_wins', 'p1_wins', 'draw', or None to infer from the SGF.
        grid_size: NN encoding grid size.

    Returns:
        List of (board_tensor, reserve_vector,
                 place_idx, place_probs,
                 movement_src, movement_dst, movement_probs,
                 value_target).
          board_tensor   : shape (NUM_CHANNELS, grid_size, grid_size), float32
          reserve_vector : shape (RESERVE_SIZE,), float32
          place_idx      : list[int] of flat placement indices (into [0, 5*G²))
          place_probs    : list[float] with 1.0 on the played placement, 0 elsewhere
          movement_src   : list[int] of legal src cells
          movement_dst   : list[int] of legal dst cells
          movement_probs : list[float] with 1.0 on the played movement pair, 0 elsewhere
          value_target   : float, ∈ {-1.0, 0.0, +1.0}
    """
    import engine_zero
    NUM_CHANNELS, _, _ = _load_encoding_consts()

    known_result = result
    game = engine_zero.HiveGame(grid_size=grid_size)
    pending: list[tuple] = []

    for move_str in parse_moves(sgf_content):
        if game.is_game_over:
            break

        if move_str == "pass":
            game.play_pass()
            continue

        # Encode the position *before* the move is played.
        board_arr, reserve_arr = game.encode_board()
        board = np.array(board_arr, dtype=np.float32).reshape(
            NUM_CHANNELS, grid_size, grid_size
        )
        reserve = np.array(reserve_arr, dtype=np.float32)
        turn_color = game.turn_color  # 'w' or 'b'

        # Parse the played move into (piece, from, to).
        try:
            piece_str, from_pos, to_pos = parse_uhp_move(game, move_str)
        except ValueError as e:
            if verbose:
                print(f"  [skip] {game_name}: parse error on {move_str!r}: {e}")
            return []

        # Reject non-base pieces in Python before hitting Rust.
        if not _is_base_piece(piece_str):
            if verbose:
                print(f"  [skip] {game_name}: non-base piece {piece_str!r} in {move_str!r}")
            return []

        primary_idx, secondary_idx = game.encode_move(piece_str, from_pos, to_pos)
        if primary_idx < 0:
            if verbose:
                print(f"  [skip] {game_name}: move outside encoding grid: {move_str!r}")
            return []

        # Enumerate all legal moves at the current position, split by kind.
        _mask, legal_moves = game.get_legal_move_mask()
        place_idx: list[int] = []
        mv_src: list[int] = []
        mv_dst: list[int] = []
        for entry in legal_moves:
            primary, secondary = entry[0], entry[1]
            if secondary is None:
                place_idx.append(int(primary))
            else:
                mv_src.append(int(primary))
                mv_dst.append(int(secondary))

        place_probs = [0.0] * len(place_idx)
        mv_probs = [0.0] * len(mv_src)

        if from_pos is None:
            try:
                i = place_idx.index(int(primary_idx))
            except ValueError:
                if verbose:
                    print(f"  [skip] {game_name}: played placement not in legal list: {move_str!r}")
                return []
            place_probs[i] = 1.0
        else:
            found = False
            for i, (s, d) in enumerate(zip(mv_src, mv_dst)):
                if s == int(primary_idx) and d == int(secondary_idx):
                    mv_probs[i] = 1.0
                    found = True
                    break
            if not found:
                if verbose:
                    print(f"  [skip] {game_name}: played movement not in legal list: {move_str!r}")
                return []

        pending.append((board, reserve, place_idx, place_probs,
                        mv_src, mv_dst, mv_probs, turn_color))

        # Advance the game state.
        game.play_move(piece_str, from_pos, to_pos)

    # Determine result from final game state when not supplied by caller.
    if known_result is None:
        known_result = _result_from_game_state(game.state)

    if known_result == "p0_wins":
        outcome = {"w": 1.0, "b": -1.0}
    elif known_result == "p1_wins":
        outcome = {"w": -1.0, "b": 1.0}
    else:
        outcome = {"w": 0.0, "b": 0.0}

    return [
        (board, reserve, place_idx, place_probs,
         mv_src, mv_dst, mv_probs, outcome[turn_color])
        for board, reserve, place_idx, place_probs,
            mv_src, mv_dst, mv_probs, turn_color in pending
    ]


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
    decisive outcome only (p0_wins or p1_wins — draws and unknowns excluded).
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
                and row["result"] in ("p0_wins", "p1_wins")
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
        num_attention_layers: int = 0,
        lr: float = 0.005,
        grid_size: int = 23,
    ):
        from ..nn.model import create_model, load_checkpoint, save_checkpoint, export_onnx
        from ..nn.training import Trainer

        self.model_path = model_path
        self.device = device
        self._save_checkpoint = save_checkpoint
        self._export_onnx = export_onnx

        if os.path.exists(model_path):
            self.model, ckpt = load_checkpoint(model_path)
            it = ckpt.get("generation", 0)
            blocks = len(self.model.res_blocks)
            ch = self.model.input_conv.out_channels
            attn = len(self.model.attention_layers)
            gs = self.model.grid_size
            params = sum(p.numel() for p in self.model.parameters())
            print(
                f"Resumed from {model_path} "
                f"(iteration {it}, {blocks}b×{ch}ch, {attn} attn, grid {gs}x{gs}, {params/1e6:.2f}M params)"
            )
            grid_size = gs
        else:
            self.model = create_model(num_blocks, channels, grid_size=grid_size,
                                      num_attention_layers=num_attention_layers)
            params = sum(p.numel() for p in self.model.parameters())
            print(
                f"Created new model ({num_blocks} blocks, {channels} channels, "
                f"{num_attention_layers} attn layers, grid {grid_size}x{grid_size}, {params/1e6:.2f}M params)"
            )
        self.grid_size = grid_size

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
        augment_symmetry: bool = True,
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
            augment_symmetry: Apply D6 hex symmetry augmentation (12x) during training.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        log_path = f"{model_name}_log.csv"
        needs_header = True
        if os.path.exists(log_path):
            with open(log_path) as f:
                first = f.readline()
            needs_header = not first.startswith("iter,")
        if needs_header:
            with open(log_path, "a") as f:
                f.write(LOG_HEADER)

        total_games = len(games)
        print(f"Dataset: {total_games} games | buffer: {buffer_size} | "
              f"epochs: {num_epochs} | SGD epochs/chunk: {epochs_per_chunk} | "
              f"symmetry_aug: {augment_symmetry}")

        from ..nn.training import HiveDataset
        dataset = HiveDataset(max_size=buffer_size, grid_size=self.grid_size)
        dataset.augment_symmetry = augment_symmetry
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
                        samples = game_to_samples(content, result, verbose=verbose_samples, game_name=sgf_name, grid_size=self.grid_size)
                    except Exception:
                        errors_this_epoch += 1
                        samples = []

                    for (board, reserve, place_idx, place_probs,
                         mv_src, mv_dst, mv_probs_per, value) in samples:
                        dataset.add_sample(board, reserve,
                                           place_idx, place_probs,
                                           mv_src, mv_dst, mv_probs_per,
                                           float(value))
                        total_positions += 1
                        positions_this_epoch += 1

                # Loading progress (overwrite line with \r).
                if games_done % 100 == 0 or dataset._size >= buffer_size:
                    elapsed_load = time.time() - epoch_start
                    print(
                        f"\r  loading  games={games_done}/{total_games} "
                        f"buf={dataset._size}/{buffer_size} "
                        f"[{elapsed_load:.1f}s]",
                        end="", flush=True,
                    )

                # Train when the buffer is full (or at end of epoch).
                buffer_full = dataset._size >= buffer_size
                last_game = games_done == total_games
                if (buffer_full or last_game) and dataset._size > 0:
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
                        chunk_positions = dataset._size
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
                        with open(log_path, "a") as log:
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
            onnx_path = self.model_path.rsplit(".", 1)[0] + ".onnx"
            self._export_onnx(self.model, onnx_path)
            ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_epoch{epoch}.pt")
            self._save_checkpoint(self.model, ckpt_path, epoch, epoch_losses)
            print(f"  Model saved → {self.model_path}  |  Checkpoint → {ckpt_path}")

        print(
            f"\nPre-training complete. "
            f"Total positions: {total_positions}. "
            f"Total errors: {total_errors}. "
            f"Model saved → {self.model_path}"
        )


