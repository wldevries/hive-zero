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
    """Dataset of (board_tensor, reserve_vector, policy_target, value_target) tuples."""

    def __init__(self):
        self.board_tensors: list[np.ndarray] = []
        self.reserve_vectors: list[np.ndarray] = []
        self.policy_targets: list[np.ndarray] = []
        self.value_targets: list[float] = []

    def add_sample(self, board_tensor: np.ndarray, reserve_vector: np.ndarray,
                   policy_target: np.ndarray, value_target: float):
        self.board_tensors.append(board_tensor)
        self.reserve_vectors.append(reserve_vector)
        self.policy_targets.append(policy_target)
        self.value_targets.append(value_target)

    def __len__(self):
        return len(self.board_tensors)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.board_tensors[idx]),
            torch.tensor(self.reserve_vectors[idx]),
            torch.tensor(self.policy_targets[idx]),
            torch.tensor(self.value_targets[idx], dtype=torch.float32),
        )


class Trainer:
    """Trains the HiveNet model on self-play data."""

    def __init__(self, model: Optional[HiveNet] = None, lr: float = 0.001,
                 weight_decay: float = 1e-4, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model or create_model()
        self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

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
