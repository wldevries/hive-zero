"""Training loop for YinshNet — single flat 59-channel policy head, MSE value.

The dataset optionally augments samples with the subset of D6 hex symmetries
that map every Yinsh valid cell to another valid cell (computed in Rust by
`yinsh_valid_d6_indices`). Two channel groups need direction permutation:
  - RemoveRow channels (1-3): 3 row directions, via `yinsh_d6_dir_permutations`
  - MoveRing channels (5-58): 6 movement directions, via `yinsh_d6_movement_dir_permutations`
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .model import (
    GRID_SIZE,
    NUM_CHANNELS,
    POLICY_CHANNELS,
    POLICY_SIZE,
    RESERVE_SIZE,
    YinshNet,
    create_model,
)

_GS = GRID_SIZE * GRID_SIZE  # 121

# Layout of the 59 policy channels (must match move_encoding.rs).
_CH_PLACE_RING = 0
_CH_REMOVE_ROW_BASE = 1   # channels 1, 2, 3
_CH_REMOVE_RING = 4
_CH_MOVE_BASE = 5          # channels 5..58: 6 dirs × 9 distances
_MAX_RING_DIST = 9
_NUM_DIRS = 6

_GRID_PERM_CACHE: list | None = None
_DIR_PERM_CACHE: list | None = None
_MOVE_DIR_PERM_CACHE: list | None = None
_VALID_SYM_CACHE: list | None = None


def _load_grid_perms() -> list[np.ndarray]:
    global _GRID_PERM_CACHE
    if _GRID_PERM_CACHE is None:
        from engine_zero import yinsh_d6_grid_permutations
        _GRID_PERM_CACHE = [np.asarray(p, dtype=np.int64) for p in yinsh_d6_grid_permutations()]
    return _GRID_PERM_CACHE


def _load_dir_perms() -> list[np.ndarray]:
    global _DIR_PERM_CACHE
    if _DIR_PERM_CACHE is None:
        from engine_zero import yinsh_d6_dir_permutations
        _DIR_PERM_CACHE = [np.asarray(p, dtype=np.int64) for p in yinsh_d6_dir_permutations()]
    return _DIR_PERM_CACHE


def _load_move_dir_perms() -> list[np.ndarray]:
    global _MOVE_DIR_PERM_CACHE
    if _MOVE_DIR_PERM_CACHE is None:
        from engine_zero import yinsh_d6_movement_dir_permutations
        _MOVE_DIR_PERM_CACHE = [np.asarray(p, dtype=np.int64) for p in yinsh_d6_movement_dir_permutations()]
    return _MOVE_DIR_PERM_CACHE


def _load_valid_syms() -> list[int]:
    global _VALID_SYM_CACHE
    if _VALID_SYM_CACHE is None:
        from engine_zero import yinsh_valid_d6_indices
        _VALID_SYM_CACHE = list(yinsh_valid_d6_indices())
    return _VALID_SYM_CACHE


def _apply_symmetry(
    board: np.ndarray,
    policy: np.ndarray,
    sym: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply D6 transform `sym` to (board[9, 11, 11], policy[7139]).

    Spatial channels are gathered using the grid permutation. Two channel groups
    need direction permutation:
      - RemoveRow (ch 1-3): via row-direction table
      - MoveRing  (ch 5-58): 6-direction blocks of 9 distances each
    """
    if sym == 0:
        return board, policy

    grid_perm = _load_grid_perms()[sym]
    dir_perm = _load_dir_perms()[sym]
    move_dir_perm = _load_move_dir_perms()[sym]

    # Board: pad with one zero column so out-of-board positions gather to 0.
    flat = board.reshape(NUM_CHANNELS, _GS)
    padded = np.concatenate([flat, np.zeros((NUM_CHANNELS, 1), dtype=np.float32)], axis=1)
    new_board = padded[:, grid_perm].reshape(NUM_CHANNELS, GRID_SIZE, GRID_SIZE)

    # Policy: spatial gather first, then channel permutation for direction-sensitive groups.
    pol_flat = policy.reshape(POLICY_CHANNELS, _GS)
    pol_padded = np.concatenate([pol_flat, np.zeros((POLICY_CHANNELS, 1), dtype=np.float32)], axis=1)
    spatially_perm = pol_padded[:, grid_perm]  # (59, 121)

    new_policy = np.empty_like(spatially_perm)
    # PlaceRing (ch 0) and RemoveRing (ch 4): spatial only.
    new_policy[_CH_PLACE_RING] = spatially_perm[_CH_PLACE_RING]
    new_policy[_CH_REMOVE_RING] = spatially_perm[_CH_REMOVE_RING]
    # RemoveRow (ch 1-3): permute direction channels via row-dir table.
    for d in range(3):
        old_d = int(dir_perm[d])
        new_policy[_CH_REMOVE_ROW_BASE + d] = spatially_perm[_CH_REMOVE_ROW_BASE + old_d]
    # MoveRing (ch 5-58): permute the 6 direction-blocks of 9 distances.
    for new_d in range(_NUM_DIRS):
        old_d = int(move_dir_perm[new_d])
        new_base = _CH_MOVE_BASE + new_d * _MAX_RING_DIST
        old_base = _CH_MOVE_BASE + old_d * _MAX_RING_DIST
        new_policy[new_base : new_base + _MAX_RING_DIST] = spatially_perm[old_base : old_base + _MAX_RING_DIST]

    return new_board, new_policy.reshape(POLICY_SIZE)


class YinshDataset(Dataset):
    """Ring-buffer replay dataset for Yinsh self-play positions.

    When `buf_dir` is given, all arrays are backed by memory-mapped files so the
    buffer survives process restarts.  On resume the existing files are opened
    read-write and `_count`/`_size` are restored from `meta.npz`.
    """

    def __init__(self, max_size: int = 100_000, buf_dir: str | None = None):
        self.max_size = max_size
        self._buf_dir = buf_dir
        self._meta_path = os.path.join(buf_dir, "meta.npz") if buf_dir else None
        self.augment_symmetry = False

        resuming = buf_dir is not None and os.path.exists(self._meta_path)
        if resuming:
            meta = np.load(self._meta_path)
            stored_max = int(meta["max_size"])
            if stored_max != max_size:
                raise ValueError(
                    f"Buffer max_size mismatch: stored {stored_max} vs requested {max_size}"
                )
            self._count = int(meta["count"])
            self._size = int(meta["size"])
        else:
            self._count = 0
            self._size = 0

        file_mode = "r+" if resuming else "w+"

        def _arr(name, shape, dtype):
            if buf_dir is None:
                return np.zeros(shape, dtype=dtype)
            os.makedirs(buf_dir, exist_ok=True)
            path = os.path.join(buf_dir, f"{name}.bin")
            return np.memmap(path, dtype=dtype, mode=file_mode, shape=shape)

        self.board_tensors = _arr(
            "board_tensors", (max_size, NUM_CHANNELS, GRID_SIZE, GRID_SIZE), np.float32
        )
        self.reserve_vectors = _arr("reserve_vectors", (max_size, RESERVE_SIZE), np.float32)
        self.policy_targets = _arr("policy_targets", (max_size, POLICY_SIZE), np.float32)
        self.value_targets = _arr("value_targets", (max_size,), np.float32)
        self.value_only = _arr("value_only", (max_size,), np.bool_)
        self.phase_flags = _arr("phase_flags", (max_size,), np.uint8)
        self.generations = _arr("generations", (max_size,), np.int32)

        if resuming:
            print(
                f"  Replay buffer resumed: {self._size} samples "
                f"(_count={self._count}) from {buf_dir}"
            )

    def flush(self):
        """Flush all memmap arrays to disk. No-op for in-memory datasets."""
        for arr in (
            self.board_tensors, self.reserve_vectors, self.policy_targets,
            self.value_targets, self.value_only, self.phase_flags, self.generations,
        ):
            if isinstance(arr, np.memmap):
                arr.flush()

    def _save_meta(self):
        self.flush()
        np.savez(
            self._meta_path,
            count=self._count,
            size=self._size,
            max_size=self.max_size,
        )

    def add_batch(
        self,
        board_tensors: np.ndarray,
        reserve_vectors: np.ndarray,
        policy_targets: np.ndarray,
        value_targets: np.ndarray,
        value_only: list[bool],
        phase_flags: list[int] | np.ndarray,
        generation: int = 0,
    ):
        n = board_tensors.shape[0]
        boards = board_tensors.reshape(n, NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
        vo_arr = np.asarray(value_only, dtype=np.bool_)
        pf_arr = np.asarray(phase_flags, dtype=np.uint8)
        gen_arr = np.full(n, generation, dtype=np.int32)

        start = self._count % self.max_size
        end = start + n
        if end <= self.max_size:
            self.board_tensors[start:end] = boards
            self.reserve_vectors[start:end] = reserve_vectors
            self.policy_targets[start:end] = policy_targets
            self.value_targets[start:end] = value_targets
            self.value_only[start:end] = vo_arr
            self.phase_flags[start:end] = pf_arr
            self.generations[start:end] = gen_arr
        else:
            first = self.max_size - start
            for arr, data in [
                (self.board_tensors, boards),
                (self.reserve_vectors, reserve_vectors),
                (self.policy_targets, policy_targets),
                (self.value_targets, value_targets),
                (self.value_only, vo_arr),
                (self.phase_flags, pf_arr),
                (self.generations, gen_arr),
            ]:
                arr[start:] = data[:first]
                arr[: n - first] = data[first:]

        self._count += n
        self._size = min(self._size + n, self.max_size)
        if self._meta_path is not None:
            self._save_meta()

    def clear(self):
        self._count = 0
        self._size = 0
        if self._meta_path is not None:
            self._save_meta()

    def __len__(self):
        if self.augment_symmetry:
            return self._size * len(_load_valid_syms())
        return self._size

    def __getitem__(self, idx):
        if self.augment_symmetry:
            valid_syms = _load_valid_syms()
            sym = valid_syms[idx % len(valid_syms)]
            base_idx = idx // len(valid_syms)
        else:
            sym = 0
            base_idx = idx

        board = self.board_tensors[base_idx].copy()
        policy = self.policy_targets[base_idx].copy()
        if sym != 0:
            board, policy = _apply_symmetry(board, policy, sym)

        return (
            torch.from_numpy(board),
            torch.from_numpy(self.reserve_vectors[base_idx].copy()),
            torch.from_numpy(policy),
            torch.tensor(self.value_targets[base_idx], dtype=torch.float32),
            torch.tensor(bool(self.value_only[base_idx]), dtype=torch.bool),
        )


def _policy_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample CE: -sum(p * log_softmax(logits)). Targets sum to ~1 already."""
    t_sum = target.sum(dim=1, keepdim=True).clamp(min=1e-8)
    t_norm = target / t_sum
    return -(t_norm * torch.log_softmax(logits, dim=1)).sum(dim=1)


class Trainer:
    """SGD+momentum trainer for YinshNet."""

    def __init__(
        self,
        model: Optional[YinshNet] = None,
        weight_decay: float = 1e-4,
        device: str = "cpu",
        lr: float = 0.02,
    ):
        self.device = torch.device(device)
        self.model = model or create_model()
        self.model.to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    @property
    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def train_epoch(
        self,
        dataset: YinshDataset,
        batch_size: int = 256,
        value_loss_scale: float = 1.0,
    ) -> dict:
        self.model.train()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        device_type = self.device.type

        for board, reserve, policy_target, value_target, value_only_mask in tqdm(
            loader, desc="  Training", leave=False, unit="batch"
        ):
            board = board.to(self.device)
            reserve = reserve.to(self.device)
            policy_target = policy_target.to(self.device)
            value_target = value_target.to(self.device).unsqueeze(1)
            value_only_mask = value_only_mask.to(self.device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                policy_logits, value = self.model(board, reserve)

            # Policy loss: skip value-only (fast-cap) turns.
            policy_active = ~value_only_mask
            if policy_active.any():
                p_loss_per = _policy_ce(
                    policy_logits[policy_active], policy_target[policy_active]
                )
                policy_loss = p_loss_per.mean()
            else:
                policy_loss = torch.tensor(0.0, device=self.device)

            value_loss = ((value.squeeze(1) - value_target.squeeze(1)) ** 2).mean()

            loss = policy_loss + value_loss_scale * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_loss += float(loss.item())
            num_batches += 1

        if num_batches == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        return {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "total_loss": total_loss / num_batches,
        }
