"""Supervised pre-training for YINSH from Boardspace human game archives.

Converts SGF games into (board, reserve, policy_target, value) samples and
trains YinshNet via imitation learning before self-play begins.

Per-position policy target  : 1.0 at every legal move's policy index, plus an
                              extra +1.0 boost on the indices of the move that
                              was actually played (handles ClaimRow's joint
                              `Sum(row, ring)` factorization by writing both
                              cells). After normalization in the trainer the
                              played move dominates while the rest of the legal
                              set still keeps its `target > 0` legal-mask role.
Value target                : +1 win / 0 draw / −1 loss from the position-side
                              player's POV.
"""

from __future__ import annotations

import csv
import os
import random
import time
import zipfile
from typing import Optional

import colorama
import numpy as np

from yinsh.selfplay.selfplay import LOG_HEADER

colorama.init()
_R = colorama.Style.RESET_ALL
_B = colorama.Style.BRIGHT
_cr = lambda v: f"{colorama.Fore.RED}{_B}{v}{_R}"      # total loss
_cy = lambda v: f"{colorama.Fore.YELLOW}{_B}{v}{_R}"   # policy / value loss
_cc = lambda v: f"{colorama.Fore.CYAN}{_B}{v}{_R}"     # chunk / epoch labels


# Heavy imports are deferred so this module is cheap to import.
_NUM_CHANNELS: int | None = None
_GRID_SIZE: int | None = None
_POLICY_SIZE: int | None = None
_RESERVE_SIZE: int | None = None


def _load_encoding_consts() -> tuple[int, int, int, int]:
    global _NUM_CHANNELS, _GRID_SIZE, _POLICY_SIZE, _RESERVE_SIZE
    if _NUM_CHANNELS is None:
        from yinsh.nn.model import (
            NUM_CHANNELS,
            GRID_SIZE,
            POLICY_SIZE,
            RESERVE_SIZE,
        )
        _NUM_CHANNELS, _GRID_SIZE, _POLICY_SIZE, _RESERVE_SIZE = (
            NUM_CHANNELS, GRID_SIZE, POLICY_SIZE, RESERVE_SIZE,
        )
    return _NUM_CHANNELS, _GRID_SIZE, _POLICY_SIZE, _RESERVE_SIZE  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Single-game converter
# ---------------------------------------------------------------------------

def _result_from_sgf(content: str, p0: str, p1: str) -> str:
    """Best-effort outcome inference from the SGF `RE[...]` field."""
    re_start = content.find("RE[")
    if re_start < 0:
        return "unknown"
    re_end = content.find("]", re_start + 3)
    if re_end < 0:
        return "unknown"
    re_val = content[re_start + 3:re_end].lower()
    if p0 and p0.lower() in re_val:
        return "p0_wins"
    if p1 and p1.lower() in re_val:
        return "p1_wins"
    if "draw" in re_val:
        return "draw"
    return "unknown"


def game_to_samples(
    sgf_content: str,
    result: str,
    verbose: bool = False,
    game_name: str = "",
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, int]]:
    """Convert one SGF game into pretrain samples.

    Returns list of `(board, reserve, policy_target, value, phase_flag)`:
      board         : (NUM_CHANNELS, GRID_SIZE, GRID_SIZE) float32
      reserve       : (RESERVE_SIZE,) float32
      policy_target : (POLICY_SIZE,) float32 — legal-mask (1.0 each) + played-move boost
      value         : ∈ {-1.0, 0.0, +1.0} from the to-move player's POV
      phase_flag    : 0=Setup, 1=Normal, 2=ClaimRow (Yinsh dataset's phase column)
    """
    import engine_zero
    NUM_CHANNELS, GRID_SIZE, POLICY_SIZE, RESERVE_SIZE = _load_encoding_consts()

    if result not in ("p0_wins", "p1_wins", "draw"):
        return []

    try:
        moves = engine_zero.parse_yinsh_sgf_moves(sgf_content)
    except Exception as e:
        if verbose:
            print(f"  [skip] {game_name}: parse error: {e}")
        return []

    if not moves:
        return []

    game = engine_zero.YinshGame()
    pending: list[tuple[np.ndarray, np.ndarray, np.ndarray, str, int]] = []

    phase_to_flag = {"setup": 0, "normal": 1, "claim_row": 2}

    for move_str in moves:
        if game.outcome() != "ongoing":
            break

        # Encode position before the move.
        board_arr, reserve_arr = game.encode()
        board = np.asarray(board_arr, dtype=np.float32).reshape(
            NUM_CHANNELS, GRID_SIZE, GRID_SIZE
        )
        reserve = np.asarray(reserve_arr, dtype=np.float32)
        side_to_move = game.current_player()  # 'white' or 'black'
        phase_flag = phase_to_flag.get(game.phase(), 1)

        # Build policy target: legal mask (1.0 per legal index) + +1.0 boost on the
        # played move's indices. ClaimRow encodes to two indices (Sum); both get
        # the boost so the joint score sees correctly ranked logits.
        try:
            played_indices = game.encode_move(move_str)
        except Exception as e:
            if verbose:
                print(f"  [skip] {game_name}: encode_move({move_str!r}) failed: {e}")
            return []

        legal_mask = np.asarray(game.legal_policy_mask(), dtype=np.float32)
        # Verify the played move is actually legal in the model's encoding —
        # if any of its indices isn't in the legal mask, skip the game (mismatch
        # between SGF and engine rules).
        if any(legal_mask[i] == 0.0 for i in played_indices):
            if verbose:
                print(f"  [skip] {game_name}: played move {move_str!r} not in legal set")
            return []

        target = legal_mask.copy()
        for i in played_indices:
            target[i] += 1.0

        pending.append((board, reserve, target, side_to_move, phase_flag))

        try:
            game.play(move_str)
        except Exception as e:
            if verbose:
                print(f"  [skip] {game_name}: play({move_str!r}) failed: {e}")
            return []

    if result == "p0_wins":
        outcome = {"white": 1.0, "black": -1.0}
    elif result == "p1_wins":
        outcome = {"white": -1.0, "black": 1.0}
    else:
        outcome = {"white": 0.0, "black": 0.0}

    return [
        (board, reserve, target, outcome[side], phase_flag)
        for board, reserve, target, side, phase_flag in pending
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
) -> list[tuple[str, str, str, str, str]]:
    """Return a filtered list of (zip_file_basename, sgf_name, result, p0, p1)
    from a Boardspace YINSH `game_outcomes.csv`.

    Filters: both players' ELO ≥ min_elo with ≥ min_games played, decisive
    outcome only (p0_wins or p1_wins). The CSV's `zip_file` column may be path-
    style (`games/yinsh/boardspace\\archive-2024\\games-Apr-17-2024.zip`) or a
    bare filename — we normalize to the basename so `build_zip_index`'s lookup
    works either way.
    """
    qualified: set[str] = set()
    with open(elo_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["games"]) >= min_games and float(row["elo"]) >= min_elo:
                qualified.add(row["player"])

    if exclude_players is None:
        exclude_players = set()

    games: list[tuple[str, str, str, str, str]] = []
    with open(games_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row["result"] in ("p0_wins", "p1_wins")
                and row["p0"] in qualified
                and row["p1"] in qualified
                and row["p0"] not in exclude_players
                and row["p1"] not in exclude_players
            ):
                zip_basename = os.path.basename(row["zip_file"].replace("\\", "/"))
                games.append((zip_basename, row["sgf_name"], row["result"], row["p0"], row["p1"]))

    return games


def build_zip_index(boardspace_dir: str) -> dict[str, str]:
    """Walk boardspace_dir and return {zip_filename_basename: full_path}."""
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
    """Supervised pre-training from Boardspace YINSH archives."""

    def __init__(
        self,
        model_path: str = "yinsh.pt",
        device: str = "cuda",
        model_config: dict | None = None,
        lr: float = 0.005,
        optimizer: str = "adamw",
        weight_decay: float = 1e-2,
    ):
        from yinsh.nn.model import (
            create_model,
            load_checkpoint,
            save_checkpoint,
            export_onnx,
            _describe_trunk,
        )
        from yinsh.nn.training import Trainer

        self.model_path = model_path
        self.device = device
        self._save_checkpoint = save_checkpoint
        self._export_onnx = export_onnx

        ckpt: dict = {}
        if os.path.exists(model_path):
            self.model, ckpt = load_checkpoint(model_path)
            it = ckpt.get("generation", 0)
            ch = self.model.input_conv.out_channels
            trunk_desc = _describe_trunk(self.model.trunk_spec)
            params = sum(p.numel() for p in self.model.parameters())
            print(
                f"Resumed from {model_path} "
                f"(iteration {it}, {ch}ch, {trunk_desc}, {params/1e6:.2f}M params)"
            )
        else:
            self.model = create_model(model_config)
            ch = self.model.input_conv.out_channels
            trunk_desc = _describe_trunk(self.model.trunk_spec)
            params = sum(p.numel() for p in self.model.parameters())
            print(
                f"Created new model ({ch}ch, {trunk_desc}, {params/1e6:.2f}M params)"
            )

        self.model.to(device)
        self.trainer = Trainer(self.model, device=device, lr=lr,
                               optimizer=optimizer, weight_decay=weight_decay)
        print(f"  Optimizer: {optimizer} (lr={lr}, weight_decay={weight_decay})")
        opt_state = ckpt.get("optimizer_state_dict")
        if opt_state is not None:
            try:
                self.trainer.optimizer.load_state_dict(opt_state)
                print("  Restored optimizer state (momentum buffers)")
            except (ValueError, KeyError) as e:
                print(f"  Skipped optimizer state (incompatible: {e})")

    def run(
        self,
        games: list[tuple[str, str, str, str, str]],
        zip_index: dict[str, str],
        num_epochs: int = 3,
        batch_size: int = 256,
        buffer_size: int = 100_000,
        epochs_per_chunk: int = 3,
        checkpoint_dir: str = "checkpoints",
        verbose_samples: bool = False,
        augment_symmetry: bool = True,
        value_loss_scale: float = 1.0,
    ) -> None:
        """Pre-train the YINSH model.

        Args mirror Hive's `Pretrainer.run`. The training buffer is the same
        `YinshDataset` used by self-play, so per-position aug + the on-disk
        replay buffer code paths are reused unchanged.
        """
        from yinsh.nn.training import YinshDataset

        os.makedirs(checkpoint_dir, exist_ok=True)
        model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        log_path = f"{model_name}_log.csv"
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write(LOG_HEADER)

        total_games = len(games)
        print(
            f"Dataset: {total_games} games | buffer: {buffer_size} | "
            f"epochs: {num_epochs} | epochs/chunk: {epochs_per_chunk} | "
            f"symmetry_aug: {augment_symmetry}"
        )

        dataset = YinshDataset(max_size=buffer_size)
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
            chunk_buf: list[tuple] = []  # accumulate (b, r, p, v, phase) tuples

            for zip_file, sgf_name, result, _p0, _p1 in games:
                games_done += 1
                zip_path = zip_index.get(zip_file)
                if zip_path is None:
                    errors_this_epoch += 1
                else:
                    try:
                        with zipfile.ZipFile(zip_path, "r") as zf:
                            content = zf.read(sgf_name).decode("iso-8859-1")
                        samples = game_to_samples(
                            content, result,
                            verbose=verbose_samples, game_name=sgf_name,
                        )
                    except Exception:
                        errors_this_epoch += 1
                        samples = []
                    chunk_buf.extend(samples)
                    positions_this_epoch += len(samples)
                    total_positions += len(samples)

                if games_done % 100 == 0 or len(chunk_buf) >= buffer_size:
                    elapsed_load = time.time() - epoch_start
                    print(
                        f"\r  loading  games={games_done}/{total_games} "
                        f"buf={len(chunk_buf)}/{buffer_size} [{elapsed_load:.1f}s]",
                        end="", flush=True,
                    )

                buffer_full = len(chunk_buf) >= buffer_size
                last_game = games_done == total_games
                if (buffer_full or last_game) and chunk_buf:
                    print()  # newline after \r loading line
                    chunk_idx += 1
                    chunk_start = time.time()

                    # Push the chunk into the dataset.
                    self._push_to_dataset(dataset, chunk_buf, generation=chunk_idx)
                    chunk_positions = len(chunk_buf)
                    chunk_buf = []

                    for mini_epoch in range(1, epochs_per_chunk + 1):
                        losses = self.trainer.train_epoch(
                            dataset, batch_size=batch_size, value_loss_scale=value_loss_scale,
                        )
                        elapsed = time.time() - chunk_start
                        tl = f"{losses['total_loss']:.4f}"
                        pl = f"{losses['policy_loss']:.4f}"
                        vl = f"{losses['value_loss']:.4f}"
                        print(
                            f"  chunk={_cc(chunk_idx)} mini={mini_epoch}/{epochs_per_chunk} "
                            f"loss={_cr(tl)} (pol={_cy(pl)} val={_cy(vl)}) [{elapsed:.1f}s]"
                        )

                    chunk_elapsed = time.time() - chunk_start
                    dataset.clear()

                    lr = self.trainer._current_lr
                    with open(log_path, "a") as log:
                        log.write(
                            # gen,epoch,simulations,games,positions,buffer,
                            # wins_p1,wins_p2,draws,timeouts,
                            # avg_game_len,med_game_len,min_game_len,max_game_len,
                            # loss,policy_loss,value_loss,
                            # lr,duration_s,
                            # mcts_top1_mean,mcts_top1_std,mcts_depth_mean,mcts_depth_std,mcts_moves_mean,mcts_moves_std,
                            # comment
                            f"{chunk_idx},pretrain,0,{games_done},{total_positions},{chunk_positions},"
                            f"0,0,0,0,"
                            f"0,0,0,0,"
                            f"{losses['total_loss']:.6f},{losses['policy_loss']:.6f},{losses['value_loss']:.6f},"
                            f"{lr:.8f},{chunk_elapsed:.1f},"
                            f"0,0,0,0,0,0,"
                            f"epoch={epoch}\n"
                        )

            total_errors += errors_this_epoch
            elapsed = time.time() - epoch_start
            print(
                f"  Epoch {epoch} done in {elapsed:.0f}s. "
                f"Positions this epoch: {positions_this_epoch}. "
                f"Errors: {errors_this_epoch}."
            )

            epoch_losses = {
                "policy_loss": losses.get("policy_loss", 0),
                "value_loss": losses.get("value_loss", 0),
            }
            self._save_checkpoint(
                self.model, self.model_path, epoch, epoch_losses,
                optimizer=self.trainer.optimizer,
            )
            try:
                onnx_path = self.model_path.rsplit(".", 1)[0] + ".onnx"
                self._export_onnx(self.model, onnx_path)
            except Exception as e:
                print(f"  ONNX export failed (non-fatal): {e}")
            ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_epoch{epoch}.pt")
            self._save_checkpoint(
                self.model, ckpt_path, epoch, epoch_losses,
                optimizer=self.trainer.optimizer,
            )
            print(f"  Model saved -> {self.model_path}  |  Checkpoint -> {ckpt_path}")

        print(
            f"\nPre-training complete. "
            f"Total positions: {total_positions}. "
            f"Total errors: {total_errors}. "
            f"Model saved -> {self.model_path}"
        )

    @staticmethod
    def _push_to_dataset(dataset, samples: list[tuple], generation: int) -> None:
        """Stack a sample list and push to YinshDataset.add_batch."""
        if not samples:
            return
        n = len(samples)
        boards = np.stack([s[0] for s in samples])
        reserves = np.stack([s[1] for s in samples])
        policies = np.stack([s[2] for s in samples])
        values = np.array([s[3] for s in samples], dtype=np.float32)
        phases = [s[4] for s in samples]
        # All samples come from real human moves — no value-only (fast-cap) turns.
        value_only = [False] * n
        dataset.add_batch(
            board_tensors=boards,
            reserve_vectors=reserves,
            policy_targets=policies,
            value_targets=values,
            value_only=value_only,
            phase_flags=phases,
            generation=generation,
        )
