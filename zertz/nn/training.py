"""Training loop for ZertzNet with factorized conv1x1 policy heads."""

from __future__ import annotations
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional
from tqdm import tqdm

from .model import ZertzNet, create_model, NUM_CHANNELS, GRID_SIZE, POLICY_SIZE, RESERVE_SIZE

_BOARD_SIZE = 37
_PLACE_ONLY_OFFSET = 3 * _BOARD_SIZE * _BOARD_SIZE  # 4107
_CAPTURE_OFFSET = _PLACE_ONLY_OFFSET + 3 * _BOARD_SIZE  # 4218
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


def _build_hex_to_grid() -> np.ndarray:
    """Build lookup: hex_index (0..36) → flat grid index (row*7+col) in 7x7 grid."""
    table = np.zeros(_BOARD_SIZE, dtype=np.int64)
    idx = 0
    for r in range(-_RADIUS, _RADIUS + 1):
        q_min = max(-_RADIUS, -_RADIUS - r)
        q_max = min(_RADIUS, _RADIUS - r)
        for q in range(q_min, q_max + 1):
            row = r + _RADIUS
            col = q - q_min
            table[idx] = row * GRID_SIZE + col
            idx += 1
    return table


_HEX_TO_GRID = _build_hex_to_grid()


class ZertzDataset(Dataset):
    """Ring-buffer replay dataset for Zertz self-play positions."""

    def __init__(self, max_size: int = 50_000):
        self.max_size = max_size
        self._count = 0
        self._size = 0
        self.board_tensors = np.zeros((max_size, NUM_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.reserve_vectors = np.zeros((max_size, RESERVE_SIZE), dtype=np.float32)
        self.policy_targets = np.zeros((max_size, POLICY_SIZE), dtype=np.float32)
        self.value_targets = np.zeros(max_size, dtype=np.float32)
        self.value_only = np.zeros(max_size, dtype=np.bool_)
        self.capture_turn = np.zeros(max_size, dtype=np.bool_)
        self.mid_capture_turn = np.zeros(max_size, dtype=np.bool_)
        self.augment_symmetry = False

    def add_batch(self, board_tensors: np.ndarray, reserve_vectors: np.ndarray,
                  policy_targets: np.ndarray, value_targets: np.ndarray,
                  value_only: list[bool],
                  capture_turn: list[bool] | None = None,
                  mid_capture_turn: list[bool] | None = None):
        n = board_tensors.shape[0]
        boards = board_tensors.reshape(n, NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
        for i in range(n):
            idx = self._count % self.max_size
            self.board_tensors[idx] = boards[i]
            self.reserve_vectors[idx] = reserve_vectors[i]
            self.policy_targets[idx] = policy_targets[i]
            self.value_targets[idx] = value_targets[i]
            self.value_only[idx] = value_only[i]
            self.capture_turn[idx] = capture_turn[i] if capture_turn is not None else False
            self.mid_capture_turn[idx] = mid_capture_turn[i] if mid_capture_turn is not None else False
            self._count += 1
            self._size = min(self._size + 1, self.max_size)

    def clear(self):
        self._count = 0
        self._size = 0

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
            hex_perm = _load_hex_perms()[sym]

            gc = GRID_SIZE * GRID_SIZE
            bf = board.reshape(NUM_CHANNELS, gc)
            padded = np.concatenate([bf, np.zeros((NUM_CHANNELS, 1), dtype=np.float32)], axis=1)
            board = padded[:, grid_perm].reshape(NUM_CHANNELS, GRID_SIZE, GRID_SIZE)

            place = policy[:_PLACE_ONLY_OFFSET].reshape(3, _BOARD_SIZE, _BOARD_SIZE)
            place_only = policy[_PLACE_ONLY_OFFSET:_CAPTURE_OFFSET].reshape(3, _BOARD_SIZE)
            capture = policy[_CAPTURE_OFFSET:].reshape(_NUM_DIRS, _BOARD_SIZE)

            new_place = place[:, hex_perm, :][:, :, hex_perm]
            new_place_only = place_only[:, hex_perm]
            dir_perm = _D6_DIR_PERMS[sym]
            new_capture = capture[dir_perm, :][:, hex_perm]

            policy = np.concatenate([
                new_place.reshape(-1),
                new_place_only.reshape(-1),
                new_capture.reshape(-1),
            ])

        return (
            torch.from_numpy(board),
            torch.from_numpy(self.reserve_vectors[base_idx].copy()),
            torch.from_numpy(policy),
            torch.tensor(self.value_targets[base_idx], dtype=torch.float32),
            torch.tensor(self.value_only[base_idx], dtype=torch.bool),
            torch.tensor(self.capture_turn[base_idx], dtype=torch.bool),
            torch.tensor(self.mid_capture_turn[base_idx], dtype=torch.bool),
        )


def _marginalize_policy(flat_policy: torch.Tensor, gi: torch.Tensor):
    """Convert flat policy[B, 4440] to per-head targets on the 7x7 grid.

    Returns:
        place_cp: [B, 3*49] color/position targets (channels 0-2)
        place_rm: [B, 49] remove-ring targets (channel 3)
        cap_dir:  [B, 6*49] direction targets (6 channels × 49 grid cells)
    """
    B = flat_policy.shape[0]
    device = flat_policy.device

    # Place region [0, 4107): color(3) * 37 * 37, indexed as [color, place_at, remove]
    place_probs = flat_policy[:, :_PLACE_ONLY_OFFSET].reshape(B, 3, _BOARD_SIZE, _BOARD_SIZE)
    color_place_hex = place_probs.sum(dim=3)  # [B, 3, 37] marginalize over remove
    remove_hex = place_probs.sum(dim=(1, 2))  # [B, 37] marginalize over color+place

    # PlaceOnly [4107, 4218): color(3) * 37
    place_only = flat_policy[:, _PLACE_ONLY_OFFSET:_CAPTURE_OFFSET].reshape(B, 3, _BOARD_SIZE)
    color_place_hex = color_place_hex + place_only

    # Scatter hex→grid for place heads
    gi_exp = gi.unsqueeze(0).expand(B, -1)  # [B, 37]
    place_cp = torch.zeros(B, 3, _GS, device=device)
    for c in range(3):
        place_cp[:, c].scatter_add_(1, gi_exp, color_place_hex[:, c])
    place_cp = place_cp.reshape(B, 3 * _GS)

    place_rm = torch.zeros(B, _GS, device=device)
    place_rm.scatter_add_(1, gi_exp, remove_hex)

    # Capture region [4218, 4440): direction(6) * 37 + from(37)
    cap_probs = flat_policy[:, _CAPTURE_OFFSET:].reshape(B, _NUM_DIRS, _BOARD_SIZE)  # [B, 6, 37]
    cap_dir = torch.zeros(B, _NUM_DIRS, _GS, device=device)
    for d in range(_NUM_DIRS):
        cap_dir[:, d].scatter_add_(1, gi_exp, cap_probs[:, d])
    cap_dir = cap_dir.reshape(B, _NUM_DIRS * _GS)  # [B, 294]

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
        self._hex_to_grid = torch.from_numpy(_HEX_TO_GRID).to(self.device)

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

        gi = self._hex_to_grid
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
                place_logits, cap_dir_logits, value = self.model(board, reserve)

            # Convert flat policy to per-head targets
            place_cp_t, place_rm_t, cap_dir_t = _marginalize_policy(policy_target, gi)

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
