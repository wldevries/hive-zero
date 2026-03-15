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
from ..encoding.symmetry import apply_symmetry


class HiveDataset(Dataset):
    """Dataset of (board_tensor, reserve_vector, policy_target, value_target) tuples.

    Acts as a replay buffer with a max capacity. When full, oldest samples
    are discarded.
    """

    def __init__(self, max_size: int = 0, augment: bool = True):
        """Args:
            max_size: Maximum number of samples to keep. 0 = unlimited.
            augment: Apply random symmetry augmentation (12 hex symmetries).
        """
        from collections import deque
        self.max_size = max_size
        self.augment = augment
        maxlen = max_size if max_size > 0 else None
        self.board_tensors: deque[np.ndarray] = deque(maxlen=maxlen)
        self.reserve_vectors: deque[np.ndarray] = deque(maxlen=maxlen)
        self.policy_targets: deque[np.ndarray] = deque(maxlen=maxlen)
        self.value_targets: deque[float] = deque(maxlen=maxlen)

    def add_sample(self, board_tensor: np.ndarray, reserve_vector: np.ndarray,
                   policy_target: np.ndarray, value_target: float):
        self.board_tensors.append(board_tensor)
        self.reserve_vectors.append(reserve_vector)
        self.policy_targets.append(policy_target)
        self.value_targets.append(value_target)

    def clear(self):
        self.board_tensors.clear()
        self.reserve_vectors.clear()
        self.policy_targets.clear()
        self.value_targets.clear()

    def __len__(self):
        return len(self.board_tensors)

    def __getitem__(self, idx):
        board = self.board_tensors[idx]
        policy = self.policy_targets[idx]

        if self.augment:
            sym_idx = np.random.randint(0, 12)
            board, policy = apply_symmetry(board, policy, sym_idx)

        return (
            torch.tensor(board),
            torch.tensor(self.reserve_vectors[idx]),
            torch.tensor(policy),
            torch.tensor(self.value_targets[idx], dtype=torch.float32),
        )


class Trainer:
    """Trains the HiveNet model on self-play data.

    Uses SGD with momentum and a stepped LR schedule following AlphaZero:
      - Start at lr=0.05
      - Drop by 10x at iteration milestones (default: 30, 60, 90)
    """

    # Default LR schedule: (iteration_threshold, lr)
    # Applied in order — last matching threshold wins.
    DEFAULT_LR_SCHEDULE = [
        (0, 0.05),
        (30, 0.005),
        (60, 0.0005),
        (90, 0.00005),
    ]

    def __init__(self, model: Optional[HiveNet] = None,
                 weight_decay: float = 1e-4, device: str = "cpu",
                 lr_schedule: list[tuple[int, float]] | None = None):
        self.device = torch.device(device)
        self.model = model or create_model()
        self.model.to(self.device)
        self.lr_schedule = lr_schedule or self.DEFAULT_LR_SCHEDULE
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=self.lr_schedule[0][1],
            momentum=0.9, weight_decay=weight_decay,
        )
        self._current_lr = self.lr_schedule[0][1]

    def update_lr(self, iteration: int):
        """Update learning rate based on iteration and schedule."""
        target_lr = self.lr_schedule[0][1]
        for threshold, lr in self.lr_schedule:
            if iteration >= threshold:
                target_lr = lr
        if target_lr != self._current_lr:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = target_lr
            print(f"  LR updated: {self._current_lr} -> {target_lr}")
            self._current_lr = target_lr

    def train_epoch(self, dataset: HiveDataset, batch_size: int = 64) -> dict:
        """Train one epoch. Returns loss dict."""
        self.model.train()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0

        for board, reserve, policy_target, value_target in loader:
            board = board.to(self.device)
            reserve = reserve.to(self.device)
            policy_target = policy_target.to(self.device)
            value_target = value_target.to(self.device).unsqueeze(1)

            policy_logits, value = self.model(board, reserve)

            # Policy loss: cross-entropy with target distribution
            log_probs = torch.log_softmax(policy_logits, dim=1)
            policy_loss = -(policy_target * log_probs).sum(dim=1).mean()

            # Value loss: MSE
            value_loss = nn.functional.mse_loss(value, value_target)

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
