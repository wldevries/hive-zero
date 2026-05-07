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
import math
import os
import random
import time
import zipfile
from typing import Optional

import colorama
import numpy as np

colorama.init()
_R = colorama.Style.RESET_ALL
_B = colorama.Style.BRIGHT
_cr = lambda v: f"{colorama.Fore.RED}{_B}{v}{_R}"      # total loss
_cy = lambda v: f"{colorama.Fore.YELLOW}{_B}{v}{_R}"   # policy / value loss
_cc = lambda v: f"{colorama.Fore.CYAN}{_B}{v}{_R}"     # chunk / epoch labels

PRETRAIN_LOG_HEADER = (
    "chunk,chunk_in_epoch,chunk_games,mini_epoch,epoch,"
    "games_done,total_games,positions,buffer_positions,"
    "train_total,train_policy,train_value,"
    "val_total,val_policy,val_value,"
    "val_top1,val_uniform_ce,"
    "lr,duration_s,best_val\n"
)

# Slight over-estimate of average positions per Yinsh game (typical is ~66).
# Used only to compute the chunk count up front so chunks are roughly equal —
# overshooting yields fewer-but-larger chunks, which is the safe direction.
EST_POSITIONS_PER_GAME = 80


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
    played_move_boost: float = 10.0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, float, int]]:
    """Convert one SGF game into pretrain samples.

    Returns list of `(board, reserve, policy_target, value, phase_flag)`:
      board         : (NUM_CHANNELS, GRID_SIZE, GRID_SIZE) float32
      reserve       : (RESERVE_SIZE,) float32
      policy_target : (POLICY_SIZE,) float32 — legal-mask (1.0 each) + played-move boost
      value         : ∈ {-1.0, 0.0, +1.0} from the to-move player's POV
      phase_flag    : 0=Setup, 1=Normal, 2=ClaimRow (Yinsh dataset's phase column)

    `played_move_boost` is added on top of the legal-mask 1.0 at the played
    move's policy index(es). With ~36 legal moves typical, boost=10 puts ~24%
    of the post-softmax mass on the played move (vs ~5% at boost=1).
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
            target[i] += played_move_boost

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
        warmup_steps: int = 0,
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
                               optimizer=optimizer, weight_decay=weight_decay,
                               warmup_steps=warmup_steps)
        warmup_note = f", warmup_steps={warmup_steps}" if warmup_steps > 0 else ""
        print(f"  Optimizer: {optimizer} (lr={lr}, weight_decay={weight_decay}{warmup_note})")
        opt_state = ckpt.get("optimizer_state_dict")
        if opt_state is not None:
            try:
                self.trainer.optimizer.load_state_dict(opt_state)
                for pg in self.trainer.optimizer.param_groups:
                    pg["lr"] = lr
                    pg["weight_decay"] = weight_decay
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
        val_frac: float = 0.1,
        played_move_boost: float = 10.0,
    ) -> None:
        """Pre-train the YINSH model.

        Args mirror Hive's `Pretrainer.run`. The training buffer is the same
        `YinshDataset` used by self-play, so per-position aug + the on-disk
        replay buffer code paths are reused unchanged. With `val_frac > 0`,
        ~val_frac of games are held out by stable hash; held-out loss is
        evaluated after each chunk and the best-by-val-loss checkpoint is
        saved alongside the per-epoch checkpoints as `<model>.best.pt`.
        """
        from yinsh.nn.training import YinshDataset
        from shared.val_split import split_train_val

        os.makedirs(checkpoint_dir, exist_ok=True)
        model_name = os.path.splitext(os.path.basename(self.model_path))[0]
        log_path = f"{model_name}_pretrain_log.csv"
        # Rotate if an old-schema log exists, so new rows aren't appended onto
        # a header that has fewer columns.
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                existing_header = f.readline()
            if existing_header != PRETRAIN_LOG_HEADER:
                backup = f"{log_path}.{int(time.time())}.bak"
                os.rename(log_path, backup)
                print(f"  Rotated pretrain log with old schema -> {backup}")
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write(PRETRAIN_LOG_HEADER)

        train_games, val_games = split_train_val(games, val_frac, sgf_name_index=1)
        total_games = len(train_games)

        # Plan chunks up front so each chunk is roughly equal in size — this
        # avoids the catastrophic-overfit "tail chunk" the position-based flush
        # used to produce when total positions wasn't a multiple of buffer_size.
        est_positions = total_games * EST_POSITIONS_PER_GAME
        chunks_per_epoch = max(1, math.ceil(est_positions / buffer_size))
        games_per_chunk = math.ceil(total_games / chunks_per_epoch)

        print(
            f"Dataset: {total_games} train + {len(val_games)} val games | "
            f"buffer: {buffer_size} | epochs: {num_epochs} | "
            f"epochs/chunk: {epochs_per_chunk} | symmetry_aug: {augment_symmetry}"
        )
        print(
            f"  Chunking: {chunks_per_epoch} chunk(s)/epoch × ~{games_per_chunk} games "
            f"(~{games_per_chunk * EST_POSITIONS_PER_GAME // 1000}k pos est)"
        )

        val_dataset = self._build_val_dataset(
            val_games, zip_index, verbose_samples, played_move_boost,
        )

        dataset = YinshDataset(max_size=buffer_size)
        dataset.augment_symmetry = augment_symmetry
        chunk_idx = 0
        total_positions = 0
        total_errors = 0
        best_val_total = float("inf")
        best_path = self.model_path.rsplit(".", 1)[0] + ".best.pt"

        for epoch in range(1, num_epochs + 1):
            random.shuffle(train_games)
            positions_this_epoch = 0
            errors_this_epoch = 0

            print(f"\n=== Pre-train epoch {epoch}/{num_epochs} ===")
            epoch_start = time.time()
            losses: dict = {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

            games_done = 0
            chunk_in_epoch = 0
            chunk_start_game = 0
            chunk_buf: list[tuple] = []  # accumulate (b, r, p, v, phase) tuples

            for zip_file, sgf_name, result, _p0, _p1 in train_games:
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
                            played_move_boost=played_move_boost,
                        )
                    except Exception:
                        errors_this_epoch += 1
                        samples = []
                    chunk_buf.extend(samples)
                    positions_this_epoch += len(samples)
                    total_positions += len(samples)

                if games_done % 100 == 0:
                    elapsed_load = time.time() - epoch_start
                    print(
                        f"\r  loading  games={games_done}/{total_games} "
                        f"buf={len(chunk_buf)}/{buffer_size} [{elapsed_load:.1f}s]",
                        end="", flush=True,
                    )

                chunk_full = games_done % games_per_chunk == 0
                last_game = games_done == total_games
                if (chunk_full or last_game) and chunk_buf:
                    print()  # newline after \r loading line
                    chunk_idx += 1
                    chunk_in_epoch += 1
                    chunk_games = games_done - chunk_start_game
                    chunk_start_game = games_done
                    chunk_start = time.time()

                    # Push the chunk into the dataset.
                    self._push_to_dataset(dataset, chunk_buf, generation=chunk_idx)
                    chunk_positions = len(chunk_buf)
                    chunk_buf = []

                    # Train all mini-epochs, capturing per-mini-epoch losses +
                    # wall-clock so we can log each as its own CSV row.
                    mini_records: list[tuple[dict, float]] = []
                    last_t = chunk_start
                    for mini_epoch in range(1, epochs_per_chunk + 1):
                        losses = self.trainer.train_epoch(
                            dataset, batch_size=batch_size, value_loss_scale=value_loss_scale,
                        )
                        now = time.time()
                        mini_duration = now - last_t
                        last_t = now
                        elapsed = now - chunk_start
                        mini_records.append((losses, mini_duration))
                        tl = f"{losses['total_loss']:.4f}"
                        pl = f"{losses['policy_loss']:.4f}"
                        vl = f"{losses['value_loss']:.4f}"
                        print(
                            f"  chunk={_cc(chunk_idx)} mini={mini_epoch}/{epochs_per_chunk} "
                            f"loss={_cr(tl)} (pol={_cy(pl)} val={_cy(vl)}) [{elapsed:.1f}s]"
                        )

                    dataset.clear()

                    val_losses: dict | None = None
                    is_best = False
                    if val_dataset is not None and len(val_dataset) > 0:
                        val_losses = self.trainer.validate(
                            val_dataset, batch_size=batch_size, value_loss_scale=value_loss_scale,
                        )
                        vt = f"{val_losses['total_loss']:.4f}"
                        vp = f"{val_losses['policy_loss']:.4f}"
                        vv = f"{val_losses['value_loss']:.4f}"
                        t1 = f"{val_losses['top1_acc']*100:.1f}%"
                        uce = f"{val_losses['uniform_ce']:.3f}"
                        print(
                            f"  chunk={_cc(chunk_idx)} VAL "
                            f"loss={_cr(vt)} (pol={_cy(vp)} val={_cy(vv)}) "
                            f"top1={_cc(t1)} uniform_ce={uce}"
                        )
                        if val_losses["total_loss"] < best_val_total:
                            best_val_total = val_losses["total_loss"]
                            is_best = True
                            self._save_checkpoint(
                                self.model, best_path, epoch,
                                {**val_losses, "val_best": True},
                                optimizer=self.trainer.optimizer,
                            )
                            print(f"  new best val={best_val_total:.4f} -> {best_path}")

                    lr = self.trainer._current_lr

                    # One CSV row per mini-epoch. val_* + best_val are written
                    # only on the last mini-epoch row (where val ran); other
                    # rows leave those columns blank so analytics tools read
                    # them as NaN rather than a misleading 0.
                    with open(log_path, "a") as log:
                        for mi, (mlosses, mdur) in enumerate(mini_records, start=1):
                            is_last_mini = mi == len(mini_records)
                            if is_last_mini and val_losses is not None:
                                v_total = f"{val_losses['total_loss']:.6f}"
                                v_pol = f"{val_losses['policy_loss']:.6f}"
                                v_val = f"{val_losses['value_loss']:.6f}"
                                v_top1 = f"{val_losses['top1_acc']:.6f}"
                                v_uce = f"{val_losses['uniform_ce']:.6f}"
                                v_best = str(int(is_best))
                            else:
                                v_total = v_pol = v_val = v_top1 = v_uce = v_best = ""
                            log.write(
                                f"{chunk_idx},{chunk_in_epoch},{chunk_games},{mi},{epoch},"
                                f"{games_done},{total_games},"
                                f"{total_positions},{chunk_positions},"
                                f"{mlosses['total_loss']:.6f},"
                                f"{mlosses['policy_loss']:.6f},"
                                f"{mlosses['value_loss']:.6f},"
                                f"{v_total},{v_pol},{v_val},{v_top1},{v_uce},"
                                f"{lr:.8f},{mdur:.1f},{v_best}\n"
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

    def _build_val_dataset(
        self,
        val_games: list[tuple[str, str, str, str, str]],
        zip_index: dict[str, str],
        verbose_samples: bool,
        played_move_boost: float = 10.0,
    ):
        """Build an in-memory val YinshDataset from `val_games`. Returns None if empty."""
        from yinsh.nn.training import YinshDataset
        if not val_games:
            return None
        print(f"  Loading val set ({len(val_games)} games)...")
        all_samples: list[tuple] = []
        load_start = time.time()
        for zip_file, sgf_name, result, _p0, _p1 in val_games:
            zip_path = zip_index.get(zip_file)
            if zip_path is None:
                continue
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    content = zf.read(sgf_name).decode("iso-8859-1")
                samples = game_to_samples(
                    content, result, verbose=verbose_samples, game_name=sgf_name,
                    played_move_boost=played_move_boost,
                )
            except Exception:
                samples = []
            all_samples.extend(samples)
        if not all_samples:
            return None
        # Size dataset to fit; never auto-clears, never grows further.
        ds = YinshDataset(max_size=max(len(all_samples), 1024))
        ds.augment_symmetry = False  # val should be deterministic
        self._push_to_dataset(ds, all_samples, generation=0)
        elapsed = time.time() - load_start
        print(f"  Val set: {len(all_samples)} positions [{elapsed:.1f}s]")
        return ds

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
