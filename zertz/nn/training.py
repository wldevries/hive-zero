"""Training loop for ZertzNet with factorized conv1x1 policy heads."""

from __future__ import annotations
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional
from tqdm import tqdm

from .model import ZertzNet, create_model, NUM_CHANNELS, GRID_SIZE, POLICY_SIZE, RESERVE_SIZE
from shared.replay_buffer import handle_buffer_size_mismatch

_BOARD_SIZE = 37  # number of valid hex cells on the Zertz board
_NUM_DIRS = 6
_RADIUS = 3
_GS = GRID_SIZE * GRID_SIZE  # 49

# Inverse gather permutations for direction channels under each D6 symmetry.
# _D6_DIR_PERMS[sym][new_d] = old_d: "to build new channel new_d, gather from old channel old_d"
_D6_DIR_PERMS = [
    [0, 1, 2, 3, 4, 5],  # sym 0: identity
    [5, 0, 1, 2, 3, 4],  # sym 1: rot 1
    [4, 5, 0, 1, 2, 3],  # sym 2: rot 2
    [3, 4, 5, 0, 1, 2],  # sym 3: rot 3 (180°)
    [2, 3, 4, 5, 0, 1],  # sym 4: rot 4
    [1, 2, 3, 4, 5, 0],  # sym 5: rot 5
    [0, 5, 4, 3, 2, 1],  # sym 6: mirror (self-inverse)
    [1, 0, 5, 4, 3, 2],  # sym 7: mirror+rot1 (self-inverse)
    [2, 1, 0, 5, 4, 3],  # sym 8: mirror+rot2 (self-inverse)
    [3, 2, 1, 0, 5, 4],  # sym 9: mirror+rot3 (self-inverse)
    [4, 3, 2, 1, 0, 5],  # sym 10: mirror+rot4 (self-inverse)
    [5, 4, 3, 2, 1, 0],  # sym 11: mirror+rot5 (self-inverse)
]

_GRID_PERM_CACHE: list | None = None
_HEX_PERM_CACHE: list | None = None


def _load_grid_perms() -> list:
    global _GRID_PERM_CACHE
    if _GRID_PERM_CACHE is None:
        from engine_zero import zertz_d6_grid_permutations
        _GRID_PERM_CACHE = [np.array(p) for p in zertz_d6_grid_permutations()]
    return _GRID_PERM_CACHE


def _load_hex_perms() -> list:
    global _HEX_PERM_CACHE
    if _HEX_PERM_CACHE is None:
        from engine_zero import zertz_d6_hex_permutations
        _HEX_PERM_CACHE = [np.array(p) for p in zertz_d6_hex_permutations()]
    return _HEX_PERM_CACHE


class ZertzDataset(Dataset):
    """Ring-buffer replay dataset for Zertz self-play positions.

    When `buf_dir` is given, all arrays are stored in a single HDF5 file
    (`replay.h5`) so the buffer survives process restarts. `_count`/`_size`
    live as HDF5 attributes in the same file — no separate metadata file.
    """

    def __init__(self, max_size: int = 50_000, buf_dir: str | None = None):
        self.max_size = max_size
        self._h5file = None
        self.augment_symmetry = False

        def _zeros(shape, dtype):
            return np.zeros(shape, dtype=dtype)

        if buf_dir is not None:
            import h5py
            os.makedirs(buf_dir, exist_ok=True)
            h5path = os.path.join(buf_dir, "replay.h5")
            resuming = os.path.exists(h5path)
            self._h5file = h5py.File(h5path, "r+" if resuming else "w")

            if resuming:
                stored_max = int(self._h5file.attrs["max_size"])
                if stored_max != max_size:
                    max_size, self._h5file = handle_buffer_size_mismatch(
                        self._h5file, h5path, max_size
                    )
                    self.max_size = max_size
                stored_policy_size = int(self._h5file["policy_targets"].shape[1]) if "policy_targets" in self._h5file else 4440
                if stored_policy_size != POLICY_SIZE:
                    raise ValueError(
                        f"Replay buffer policy format changed: stored {stored_policy_size} vs current {POLICY_SIZE}. "
                        f"Delete {h5path} to start fresh with the new format."
                    )
                self._count = int(self._h5file.attrs["count"])
                self._size = int(self._h5file.attrs["size"])
                print(f"  Replay buffer resumed: {self._size} samples from {h5path}")
            else:
                self._count = 0
                self._size = 0
                self._h5file.attrs["max_size"] = max_size
                self._h5file.attrs["count"] = 0
                self._h5file.attrs["size"] = 0

            def _ds(name, shape, dtype):
                if name in self._h5file:
                    return self._h5file[name]
                return self._h5file.create_dataset(name, shape=shape, dtype=dtype, track_order=False)
        else:
            self._count = 0
            self._size = 0
            _ds = lambda name, shape, dtype: _zeros(shape, dtype)

        self.board_tensors    = _ds("board_tensors",    (max_size, NUM_CHANNELS, GRID_SIZE, GRID_SIZE), np.float32)
        self.reserve_vectors  = _ds("reserve_vectors",  (max_size, RESERVE_SIZE),                       np.float32)
        self.policy_targets   = _ds("policy_targets",   (max_size, POLICY_SIZE),                        np.float32)
        self.value_targets    = _ds("value_targets",    (max_size,),                                    np.float32)
        self.value_only       = _ds("value_only",       (max_size,),                                    np.bool_)
        self.capture_turn     = _ds("capture_turn",     (max_size,),                                    np.bool_)
        self.mid_capture_turn = _ds("mid_capture_turn", (max_size,),                                    np.bool_)

        if self._h5file is not None:
            self._h5file.flush()

    def close(self):
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None

    def __del__(self):
        self.close()

    def add_batch(self, board_tensors: np.ndarray, reserve_vectors: np.ndarray,
                  policy_targets: np.ndarray, value_targets: np.ndarray,
                  value_only: list[bool],
                  capture_turn: list[bool] | None = None,
                  mid_capture_turn: list[bool] | None = None):
        n = board_tensors.shape[0]
        boards   = board_tensors.reshape(n, NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
        vo_arr   = np.asarray(value_only, dtype=np.bool_)
        ct_arr   = np.asarray(capture_turn, dtype=np.bool_) if capture_turn is not None else np.zeros(n, dtype=np.bool_)
        mct_arr  = np.asarray(mid_capture_turn, dtype=np.bool_) if mid_capture_turn is not None else np.zeros(n, dtype=np.bool_)

        pairs = [
            (self.board_tensors,    boards),
            (self.reserve_vectors,  reserve_vectors),
            (self.policy_targets,   policy_targets),
            (self.value_targets,    value_targets),
            (self.value_only,       vo_arr),
            (self.capture_turn,     ct_arr),
            (self.mid_capture_turn, mct_arr),
        ]

        start = self._count % self.max_size
        end   = start + n
        if end <= self.max_size:
            for arr, data in pairs:
                arr[start:end] = data
        else:
            first = self.max_size - start
            for arr, data in pairs:
                arr[start:]     = data[:first]
                arr[:n - first] = data[first:]

        self._count += n
        self._size = min(self._size + n, self.max_size)

        if self._h5file is not None:
            self._h5file.attrs["count"] = self._count
            self._h5file.attrs["size"]  = self._size
            self._h5file.flush()

    def clear(self):
        self._count = 0
        self._size = 0
        if self._h5file is not None:
            self._h5file.attrs["count"] = 0
            self._h5file.attrs["size"]  = 0
            self._h5file.flush()

    def __len__(self):
        return self._size * 12 if self.augment_symmetry else self._size

    def __getitem__(self, idx):
        if self.augment_symmetry:
            sym = idx % 12
            base_idx = idx // 12
        else:
            sym = 0
            base_idx = idx

        board = self.board_tensors[base_idx].copy()
        policy = self.policy_targets[base_idx].copy()

        if sym != 0:
            grid_perm = _load_grid_perms()[sym]

            # Board: permute each channel's 7×7 grid cells (sentinel 49 → zeros)
            bf = board.reshape(NUM_CHANNELS, _GS)
            padded = np.concatenate([bf, np.zeros((NUM_CHANNELS, 1), dtype=np.float32)], axis=1)
            board = padded[:, grid_perm].reshape(NUM_CHANNELS, GRID_SIZE, GRID_SIZE)

            # Policy [10, 49]: permute grid cells, then reorder direction channels (4-9)
            p = policy.reshape(10, _GS)
            padded_p = np.concatenate([p, np.zeros((10, 1), dtype=np.float32)], axis=1)
            p_perm = padded_p[:, grid_perm]  # (10, 49)
            dir_perm = _D6_DIR_PERMS[sym]
            cap_perm = p_perm[4:][dir_perm]  # reorder direction channels
            policy = np.concatenate([p_perm[:4].reshape(-1), cap_perm.reshape(-1)])

        return (
            torch.from_numpy(board),
            torch.from_numpy(self.reserve_vectors[base_idx].copy()),
            torch.from_numpy(policy),
            torch.tensor(self.value_targets[base_idx], dtype=torch.float32),
            torch.tensor(self.value_only[base_idx], dtype=torch.bool),
            torch.tensor(self.capture_turn[base_idx], dtype=torch.bool),
            torch.tensor(self.mid_capture_turn[base_idx], dtype=torch.bool),
        )


def _marginalize_policy(flat_policy: torch.Tensor):
    """Split flat policy[B, 490] into per-head targets.

    Layout: [place_W(49), place_G(49), place_B(49), remove(49),
             cap_E(49), cap_NE(49), cap_NW(49), cap_W(49), cap_SW(49), cap_SE(49)]

    Returns:
        place_cp: [B, 3*49] color/position targets (channels 0-2)
        place_rm: [B, 49] remove-ring targets (channel 3)
        cap_dir:  [B, 6*49] direction targets (6 channels × 49 grid cells)
    """
    place_cp = flat_policy[:, :3 * _GS]           # [B, 147]
    place_rm = flat_policy[:, 3 * _GS:4 * _GS]    # [B, 49]
    cap_dir  = flat_policy[:, 4 * _GS:]            # [B, 294]
    return place_cp, place_rm, cap_dir


def _head_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample cross-entropy: -sum(normalized_target * log_softmax(logits), dim=1).

    Normalizes target to sum to 1 per sample. Returns [N] losses.
    """
    t_sum = target.sum(dim=1, keepdim=True).clamp(min=1e-8)
    t_norm = target / t_sum
    return -(t_norm * torch.log_softmax(logits, dim=1)).sum(dim=1)


class Trainer:
    """Trains ZertzNet on self-play data using SGD with momentum."""

    def __init__(self, model: Optional[ZertzNet] = None,
                 weight_decay: float = 1e-4, device: str = "cpu",
                 lr: float = 0.02):
        self.device = torch.device(device)
        self.model = model or create_model()
        self.model.to(self.device)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=lr,
            momentum=0.9, weight_decay=weight_decay,
        )
        self._compiled = torch.compile(self.model, dynamic=True)

    @property
    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def train_epoch(self, dataset: ZertzDataset, batch_size: int = 256, value_loss_scale: float = 1.0) -> dict:
        """Train one epoch with factorized policy heads. Returns loss dict."""
        self.model.train()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        total_place_value_loss = 0.0
        total_capture_value_loss = 0.0
        total_place_policy_loss = 0.0
        total_capture_policy_loss = 0.0
        place_value_batches = 0
        capture_value_batches = 0
        place_policy_batches = 0
        capture_policy_batches = 0
        num_batches = 0

        device_type = self.device.type
        for board, reserve, policy_target, value_target, vo_mask, cap_mask, mid_cap_mask in tqdm(
                loader, desc="  Training", leave=False, unit="batch"):
            board = board.to(self.device)
            reserve = reserve.to(self.device)
            policy_target = policy_target.to(self.device)
            value_target = value_target.to(self.device).unsqueeze(1)
            vo_mask = vo_mask.to(self.device)
            cap_mask = cap_mask.to(self.device)
            mid_cap_mask = mid_cap_mask.to(self.device)  # kept for value logging split

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                place_logits, cap_dir_logits, value = self._compiled(board, reserve)

            # Convert flat policy to per-head targets
            place_cp_t, place_rm_t, cap_dir_t = _marginalize_policy(policy_target)

            # --- Policy loss ---
            policy_mask = (~vo_mask).float()
            policy_loss = torch.tensor(0.0, device=self.device)

            # Place turns: train place color/position (ch 0-2) and remove (ch 3) heads
            place_active = ~cap_mask & ~vo_mask
            if place_active.any():
                pw = policy_mask[place_active]
                n_place = place_active.sum().item()
                pp = torch.tensor(0.0, device=self.device)

                # Color/position head (channels 0-2, flat 3*49=147)
                cp_logits = place_logits[place_active, :3 * _GS]
                cp_target = place_cp_t[place_active]
                has_cp = cp_target.sum(dim=1) > 0
                if has_cp.any():
                    cp_loss = _head_ce(cp_logits[has_cp], cp_target[has_cp])
                    policy_loss = policy_loss + (cp_loss * pw[has_cp]).mean()
                    pp = pp + (cp_loss * pw[has_cp]).sum() / n_place

                # Remove head (channel 3, flat 49)
                rm_logits = place_logits[place_active, 3 * _GS:]
                rm_target = place_rm_t[place_active]
                has_rm = rm_target.sum(dim=1) > 0
                if has_rm.any():
                    rm_loss = _head_ce(rm_logits[has_rm], rm_target[has_rm])
                    policy_loss = policy_loss + (rm_loss * pw[has_rm]).mean()
                    pp = pp + (rm_loss * pw[has_rm]).sum() / n_place

                total_place_policy_loss += pp.item()
                place_policy_batches += 1

            # Capture turns (first-hop and mid-capture): train cap_dir head uniformly
            cap_active = cap_mask & ~vo_mask
            if cap_active.any():
                cw = policy_mask[cap_active]
                n_cap = cap_active.sum().item()
                c_loss = _head_ce(cap_dir_logits[cap_active], cap_dir_t[cap_active])
                policy_loss = policy_loss + (c_loss * cw).mean()
                total_capture_policy_loss += ((c_loss * cw).sum() / n_cap).item()
                capture_policy_batches += 1

            # --- Value loss ---
            per_sample_value = (value.squeeze(1) - value_target.squeeze(1)) ** 2
            value_loss = per_sample_value.mean()

            # --- Split logging ---
            place_mask = ~cap_mask
            if place_mask.any():
                total_place_value_loss += per_sample_value[place_mask].mean().item()
                place_value_batches += 1
            if cap_mask.any():
                total_capture_value_loss += per_sample_value[cap_mask].mean().item()
                capture_value_batches += 1

            loss = policy_loss + value_loss_scale * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0,
                    "place_value_loss": 0.0, "capture_value_loss": 0.0,
                    "place_policy_loss": 0.0, "capture_policy_loss": 0.0}

        return {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "total_loss": total_loss / num_batches,
            "place_value_loss": total_place_value_loss / place_value_batches if place_value_batches else 0.0,
            "capture_value_loss": total_capture_value_loss / capture_value_batches if capture_value_batches else 0.0,
            "place_policy_loss": total_place_policy_loss / place_policy_batches if place_policy_batches else 0.0,
            "capture_policy_loss": total_capture_policy_loss / capture_policy_batches if capture_policy_batches else 0.0,
        }
