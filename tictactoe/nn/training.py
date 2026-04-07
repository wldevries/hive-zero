"""Training loop for TicTacToeNet."""

from __future__ import annotations
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from .model import NUM_CHANNELS, GRID_SIZE, POLICY_SIZE


class TicTacToeDataset(Dataset):
    """Ring-buffer replay dataset for tic-tac-toe self-play positions."""

    def __init__(self, max_size: int = 50_000):
        self.max_size = max_size
        self._count = 0
        self._size = 0
        self.board_tensors = np.zeros((max_size, NUM_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.policy_targets = np.zeros((max_size, POLICY_SIZE), dtype=np.float32)
        self.value_targets = np.zeros(max_size, dtype=np.float32)
        self.weights = np.ones(max_size, dtype=np.float32)
        self.value_only = np.zeros(max_size, dtype=np.bool_)

    def add_batch(self, board_tensors: np.ndarray, policy_targets: np.ndarray,
                  value_targets: np.ndarray, weights: np.ndarray,
                  value_only: list[bool]):
        n = board_tensors.shape[0]
        boards = board_tensors.reshape(n, NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
        for i in range(n):
            idx = self._count % self.max_size
            self.board_tensors[idx] = boards[i]
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
        board = torch.from_numpy(self.board_tensors[idx].copy())
        policy = torch.from_numpy(self.policy_targets[idx].copy())
        value = torch.tensor(self.value_targets[idx], dtype=torch.float32)
        weight = torch.tensor(self.weights[idx], dtype=torch.float32)
        vo = self.value_only[idx]
        return board, policy, value, weight, vo


class Trainer:
    """SGD trainer for TicTacToeNet."""

    def __init__(self, model, device: str = "cpu", lr: float = 0.02):
        self.model = model
        self.device = torch.device(device)
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    @property
    def _current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def train_epoch(self, dataset: TicTacToeDataset, batch_size: int = 256,
                    value_loss_scale: float = 1.0) -> dict:
        self.model.train()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=0, drop_last=False)

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_samples = 0

        for board, policy_target, value_target, weight, value_only in loader:
            board = board.to(self.device)
            policy_target = policy_target.to(self.device)
            value_target = value_target.to(self.device)
            weight = weight.to(self.device)
            value_only_mask = torch.as_tensor(value_only, dtype=torch.bool).to(self.device)

            policy_logits, value = self.model(board)
            value = value.squeeze(1)

            # Policy loss: cross-entropy (masked for value-only turns)
            log_probs = torch.log_softmax(policy_logits, dim=1)
            policy_loss_per = -(policy_target * log_probs).sum(dim=1)
            policy_mask = ~value_only_mask
            if policy_mask.any():
                policy_loss = (policy_loss_per * weight * policy_mask.float()).sum() / (weight * policy_mask.float()).sum()
            else:
                policy_loss = torch.tensor(0.0, device=self.device)

            # Value loss: MSE
            value_loss = (weight * (value - value_target) ** 2).mean()

            loss = policy_loss + value_loss_scale * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            bs = board.size(0)
            total_loss += loss.item() * bs
            total_policy_loss += policy_loss.item() * bs
            total_value_loss += value_loss.item() * bs
            total_samples += bs

        return {
            "total_loss": total_loss / max(total_samples, 1),
            "policy_loss": total_policy_loss / max(total_samples, 1),
            "value_loss": total_value_loss / max(total_samples, 1),
        }
