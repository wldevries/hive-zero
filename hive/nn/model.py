"""AlphaZero-style neural network for Hive."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoding.board_encoder import NUM_CHANNELS, GRID_SIZE, RESERVE_SIZE
from ..encoding.move_encoder import POLICY_SIZE


class ResBlock(nn.Module):
    """Residual block with two convolutions and batch norm."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class HiveNet(nn.Module):
    """AlphaZero-style network with policy and value heads.

    Architecture:
        - Input convolution
        - Residual tower (num_blocks blocks)
        - Policy head: conv -> flatten -> linear -> POLICY_SIZE
        - Value head: conv -> flatten -> concat reserve -> linear -> tanh
    """

    def __init__(self, num_blocks: int = 10, channels: int = 128):
        super().__init__()

        # Input convolution
        self.input_conv = nn.Conv2d(NUM_CHANNELS, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * GRID_SIZE * GRID_SIZE, POLICY_SIZE)

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(GRID_SIZE * GRID_SIZE + RESERVE_SIZE, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, board_tensor: torch.Tensor, reserve_vector: torch.Tensor):
        """Forward pass.

        Args:
            board_tensor: (batch, NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
            reserve_vector: (batch, RESERVE_SIZE)

        Returns:
            policy_logits: (batch, POLICY_SIZE)
            value: (batch, 1) in [-1, 1]
        """
        # Shared trunk
        x = F.relu(self.input_bn(self.input_conv(board_tensor)))
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = torch.cat([v, reserve_vector], dim=1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


def create_model(num_blocks: int = 10, channels: int = 128) -> HiveNet:
    """Create a new HiveNet model."""
    return HiveNet(num_blocks=num_blocks, channels=channels)


def save_model(model: HiveNet, path: str):
    torch.save(model.state_dict(), path)


def load_model(path: str, num_blocks: int = 10, channels: int = 128) -> HiveNet:
    model = HiveNet(num_blocks=num_blocks, channels=channels)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model
