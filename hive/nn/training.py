"""Training loop for the Hive neural network."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional

from .model import HiveNet, create_model, save_model
from ..encoding.board_encoder import NUM_CHANNELS, GRID_SIZE, RESERVE_SIZE
from ..encoding.move_encoder import POLICY_SIZE


class HiveDataset(Dataset):
    """Dataset of (board_tensor, reserve_vector, policy_target, value_target) tuples.

    Acts as a replay buffer with a max capacity using pre-allocated numpy arrays
    and a ring buffer to avoid GC pressure from thousands of individual arrays.
    """

    def __init__(self, max_size: int = 50_000):
        """Args:
            max_size: Maximum number of samples to keep.
        """
        self.max_size = max_size
        self._count = 0  # total samples added (for ring buffer index)
        self._size = 0   # current number of valid samples
        # Pre-allocate contiguous arrays
        self.board_tensors = np.zeros((max_size, NUM_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.reserve_vectors = np.zeros((max_size, RESERVE_SIZE), dtype=np.float32)
        self.policy_targets = np.zeros((max_size, POLICY_SIZE), dtype=np.float32)
        self.value_targets = np.zeros(max_size, dtype=np.float32)
        self.weights = np.ones(max_size, dtype=np.float32)
        self.value_only = np.zeros(max_size, dtype=np.bool_)

    def add_sample(self, board_tensor: np.ndarray, reserve_vector: np.ndarray,
                   policy_target: np.ndarray, value_target: float,
                   weight: float = 1.0, value_only: bool = False):
        idx = self._count % self.max_size
        self.board_tensors[idx] = board_tensor
        self.reserve_vectors[idx] = reserve_vector
        self.policy_targets[idx] = policy_target
        self.value_targets[idx] = value_target
        self.weights[idx] = weight
        self.value_only[idx] = value_only
        self._count += 1
        self._size = min(self._size + 1, self.max_size)

    def clear(self):
        self._count = 0
        self._size = 0

    def __len__(self):
        return self._size

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.board_tensors[idx].copy()),
            torch.from_numpy(self.reserve_vectors[idx].copy()),
            torch.from_numpy(self.policy_targets[idx].copy()),
            torch.tensor(self.value_targets[idx], dtype=torch.float32),
            torch.tensor(self.weights[idx], dtype=torch.float32),
            torch.tensor(self.value_only[idx], dtype=torch.bool),
        )


class Trainer:
    """Trains the HiveNet model on self-play data.

    Uses SGD with momentum and cosine annealing with warm restarts:
      - LR decays from lr_max to lr_min following a cosine curve
      - Every T_0 iterations, LR jumps back to lr_max (warm restart)
      - Restarts help escape sharp local minima for better generalization
    """

    def __init__(self, model: Optional[HiveNet] = None,
                 weight_decay: float = 1e-4, device: str = "cpu",
                 lr_max: float = 0.05, lr_min: float = 1e-5,
                 t_0: int = 30, t_mult: int = 1):
        self.device = torch.device(device)
        self.model = model or create_model()
        self.model.to(self.device)
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.t_0 = t_0
        self.t_mult = t_mult
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=lr_max,
            momentum=0.9, weight_decay=weight_decay,
        )
        self._current_lr = lr_max
        self._last_restart = 0
        self._current_t = t_0

    def fast_forward_lr(self, iteration: int):
        """Fast-forward cosine schedule state to given iteration (for resume)."""
        # Replay restart boundaries to find current cycle state
        last_restart = 0
        current_t = self.t_0
        while last_restart + current_t <= iteration:
            last_restart += current_t
            current_t = current_t * self.t_mult
        self._last_restart = last_restart
        self._current_t = current_t
        self.update_lr(iteration)

    def update_lr(self, iteration: int):
        """Update learning rate using cosine annealing with warm restarts."""
        import math
        # How far into the current restart cycle
        elapsed = iteration - self._last_restart
        if elapsed >= self._current_t:
            # Warm restart
            self._last_restart = iteration
            self._current_t = self._current_t * self.t_mult
            elapsed = 0
        # Cosine decay within current cycle
        target_lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + math.cos(math.pi * elapsed / self._current_t)
        )
        if abs(target_lr - self._current_lr) > 1e-8:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = target_lr
            if elapsed == 0 and iteration > 0:
                print(f"  LR warm restart: {self._current_lr:.6f} -> {target_lr:.6f}")
            self._current_lr = target_lr

    def train_epoch(self, dataset: HiveDataset, batch_size: int = 64) -> dict:
        """Train one epoch. Returns loss dict."""
        self.model.train()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0

        for board, reserve, policy_target, value_target, weight, vo_mask in loader:
            board = board.to(self.device)
            reserve = reserve.to(self.device)
            policy_target = policy_target.to(self.device)
            value_target = value_target.to(self.device).unsqueeze(1)
            weight = weight.to(self.device)
            vo_mask = vo_mask.to(self.device)

            policy_logits, value = self.model(board, reserve)

            # Policy loss: weighted cross-entropy, masked for value-only samples
            log_probs = torch.log_softmax(policy_logits, dim=1)
            per_sample_policy = -(policy_target * log_probs).sum(dim=1)
            policy_weight = weight * (~vo_mask).float()
            policy_loss = (per_sample_policy * policy_weight).mean()

            # Value loss: weighted MSE (all samples contribute)
            per_sample_value = (value.squeeze(1) - value_target.squeeze(1)) ** 2
            value_loss = (per_sample_value * weight).mean()

            # Combined loss
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            return {"policy_loss": 0, "value_loss": 0, "total_loss": 0}

        return {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "total_loss": total_loss / num_batches,
        }
