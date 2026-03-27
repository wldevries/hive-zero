"""Training loop for ZertzNet."""

from __future__ import annotations
import random
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

_GRID_PERM_CACHE: list | None = None
_HEX_PERM_CACHE: list | None = None


def _load_grid_perms() -> list:
    global _GRID_PERM_CACHE
    if _GRID_PERM_CACHE is None:
        from hive_engine import zertz_d6_grid_permutations
        _GRID_PERM_CACHE = [np.array(p) for p in zertz_d6_grid_permutations()]
    return _GRID_PERM_CACHE


def _load_hex_perms() -> list:
    global _HEX_PERM_CACHE
    if _HEX_PERM_CACHE is None:
        from hive_engine import zertz_d6_hex_permutations
        _HEX_PERM_CACHE = [np.array(p) for p in zertz_d6_hex_permutations()]
    return _HEX_PERM_CACHE


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
        self.weights = np.ones(max_size, dtype=np.float32)
        self.value_only = np.zeros(max_size, dtype=np.bool_)
        self.augment_symmetry = False

    def add_batch(self, board_tensors: np.ndarray, reserve_vectors: np.ndarray,
                  policy_targets: np.ndarray, value_targets: np.ndarray,
                  weights: np.ndarray, value_only: list[bool]):
        n = board_tensors.shape[0]
        boards = board_tensors.reshape(n, NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
        for i in range(n):
            idx = self._count % self.max_size
            self.board_tensors[idx] = boards[i]
            self.reserve_vectors[idx] = reserve_vectors[i]
            self.policy_targets[idx] = policy_targets[i]
            self.value_targets[idx] = value_targets[i]
            self.weights[idx] = weights[i]
            self.value_only[idx] = value_only[i]
            self._count += 1
            self._size = min(self._size + 1, self.max_size)

    def clear(self):
        self._count = 0
        self._size = 0

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        board = self.board_tensors[idx].copy()
        policy = self.policy_targets[idx].copy()

        if self.augment_symmetry:
            sym = random.randint(0, 11)
            if sym != 0:
                grid_perm = _load_grid_perms()[sym]
                hex_perm = _load_hex_perms()[sym]

                gc = GRID_SIZE * GRID_SIZE
                bf = board.reshape(NUM_CHANNELS, gc)
                padded = np.concatenate([bf, np.zeros((NUM_CHANNELS, 1), dtype=np.float32)], axis=1)
                board = padded[:, grid_perm].reshape(NUM_CHANNELS, GRID_SIZE, GRID_SIZE)

                place = policy[:_PLACE_ONLY_OFFSET].reshape(3, _BOARD_SIZE, _BOARD_SIZE)
                place_only = policy[_PLACE_ONLY_OFFSET:_CAPTURE_OFFSET].reshape(3, _BOARD_SIZE)
                capture = policy[_CAPTURE_OFFSET:].reshape(_BOARD_SIZE, _BOARD_SIZE)

                new_place = place[:, hex_perm, :][:, :, hex_perm]
                new_place_only = place_only[:, hex_perm]
                new_capture = capture[hex_perm, :][:, hex_perm]

                policy = np.concatenate([
                    new_place.reshape(-1),
                    new_place_only.reshape(-1),
                    new_capture.reshape(-1),
                ])

        return (
            torch.from_numpy(board),
            torch.from_numpy(self.reserve_vectors[idx].copy()),
            torch.from_numpy(policy),
            torch.tensor(self.value_targets[idx], dtype=torch.float32),
            torch.tensor(self.weights[idx], dtype=torch.float32),
            torch.tensor(self.value_only[idx], dtype=torch.bool),
        )


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

    @property
    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def train_epoch(self, dataset: ZertzDataset, batch_size: int = 256) -> dict:
        """Train one epoch. Returns loss dict."""
        self.model.train()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0

        device_type = self.device.type
        for board, reserve, policy_target, value_target, weight, vo_mask in tqdm(
                loader, desc="  Training", leave=False, unit="batch"):
            board = board.to(self.device)
            reserve = reserve.to(self.device)
            policy_target = policy_target.to(self.device)
            value_target = value_target.to(self.device).unsqueeze(1)
            weight = weight.to(self.device)
            vo_mask = vo_mask.to(self.device)

            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                policy_logits, value = self.model(board, reserve)

            log_probs = torch.log_softmax(policy_logits, dim=1)
            per_sample_policy = -(policy_target * log_probs).sum(dim=1)
            policy_weight = weight * (~vo_mask).float()
            policy_loss = (per_sample_policy * policy_weight).mean()

            per_sample_value = (value.squeeze(1) - value_target.squeeze(1)) ** 2
            value_loss = (per_sample_value * weight).mean()

            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "total_loss": 0.0}

        return {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "total_loss": total_loss / num_batches,
        }
