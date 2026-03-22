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
        self.policy_only = np.zeros(max_size, dtype=np.bool_)
        self.my_queen_danger = np.zeros(max_size, dtype=np.float32)
        self.opp_queen_danger = np.zeros(max_size, dtype=np.float32)

    def add_sample(self, board_tensor: np.ndarray, reserve_vector: np.ndarray,
                   policy_target: np.ndarray, value_target: float,
                   weight: float = 1.0, value_only: bool = False, policy_only: bool = False,
                   my_queen_danger: float = 0.0, opp_queen_danger: float = 0.0):
        idx = self._count % self.max_size
        self.board_tensors[idx] = board_tensor
        self.reserve_vectors[idx] = reserve_vector
        self.policy_targets[idx] = policy_target
        self.value_targets[idx] = value_target
        self.weights[idx] = weight
        self.value_only[idx] = value_only
        self.policy_only[idx] = policy_only
        self.my_queen_danger[idx] = my_queen_danger
        self.opp_queen_danger[idx] = opp_queen_danger
        self._count += 1
        self._size = min(self._size + 1, self.max_size)

    def add_batch(self, board_tensors: np.ndarray, reserve_vectors: np.ndarray,
                  policy_targets: np.ndarray, value_targets: np.ndarray,
                  weights: np.ndarray, value_only: list[bool], policy_only: list[bool],
                  my_queen_danger: np.ndarray | None = None,
                  opp_queen_danger: np.ndarray | None = None):
        """Bulk insert from contiguous arrays. Much faster than per-sample add."""
        n = board_tensors.shape[0]
        boards_flat = board_tensors.reshape(n, NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
        for i in range(n):
            idx = self._count % self.max_size
            self.board_tensors[idx] = boards_flat[i]
            self.reserve_vectors[idx] = reserve_vectors[i]
            self.policy_targets[idx] = policy_targets[i]
            self.value_targets[idx] = value_targets[i]
            self.weights[idx] = weights[i]
            self.value_only[idx] = value_only[i]
            self.policy_only[idx] = policy_only[i]
            self.my_queen_danger[idx] = my_queen_danger[i] if my_queen_danger is not None else 0.0
            self.opp_queen_danger[idx] = opp_queen_danger[i] if opp_queen_danger is not None else 0.0
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
            torch.tensor(self.policy_only[idx], dtype=torch.bool),
            torch.tensor(self.my_queen_danger[idx], dtype=torch.float32),
            torch.tensor(self.opp_queen_danger[idx], dtype=torch.float32),
        )


class Trainer:
    """Trains the HiveNet model on self-play data using SGD with momentum."""

    def __init__(self, model: Optional[HiveNet] = None,
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

    def train_epoch(self, dataset: HiveDataset, batch_size: int = 64) -> dict:
        """Train one epoch. Returns loss dict."""
        self.model.train()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_qd_loss = 0.0
        total_loss = 0.0
        num_batches = 0

        for board, reserve, policy_target, value_target, weight, vo_mask, po_mask, mqd_target, oqd_target in loader:
            board = board.to(self.device)
            reserve = reserve.to(self.device)
            policy_target = policy_target.to(self.device)
            value_target = value_target.to(self.device).unsqueeze(1)
            weight = weight.to(self.device)
            vo_mask = vo_mask.to(self.device)
            po_mask = po_mask.to(self.device)
            mqd_target = mqd_target.to(self.device).unsqueeze(1)
            oqd_target = oqd_target.to(self.device).unsqueeze(1)

            policy_logits, value, my_qd, opp_qd = self.model(board, reserve)

            # Policy loss: weighted cross-entropy, masked for value-only samples
            log_probs = torch.log_softmax(policy_logits, dim=1)
            per_sample_policy = -(policy_target * log_probs).sum(dim=1)
            policy_weight = weight * (~vo_mask).float()
            policy_loss = (per_sample_policy * policy_weight).mean()

            # Value loss: weighted MSE, masked for policy-only samples (zero-heuristic draws)
            per_sample_value = (value.squeeze(1) - value_target.squeeze(1)) ** 2
            value_weight = weight * (~po_mask).float()
            value_loss = (per_sample_value * value_weight).mean()

            # Auxiliary queen danger loss: MSE on both heads, always active
            qd_loss = 0.15 * ((my_qd - mqd_target) ** 2).mean() + \
                      0.15 * ((opp_qd - oqd_target) ** 2).mean()

            # Combined loss
            loss = policy_loss + value_loss + qd_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_qd_loss += qd_loss.item()
            total_loss += loss.item()
            num_batches += 1

        if num_batches == 0:
            return {"policy_loss": 0, "value_loss": 0, "qd_loss": 0, "total_loss": 0}

        return {
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
            "qd_loss": total_qd_loss / num_batches,
            "total_loss": total_loss / num_batches,
        }
