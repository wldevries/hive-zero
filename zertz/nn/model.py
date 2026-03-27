"""ZertzNet: AlphaZero-style network for Zertz."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.nn.resblock import ResBlock

# From Rust: hive_engine.ZERTZ_NUM_CHANNELS, ZERTZ_GRID_SIZE, ZERTZ_POLICY_SIZE
NUM_CHANNELS = 14
GRID_SIZE = 7
POLICY_SIZE = 5587


class ZertzNet(nn.Module):
    """AlphaZero-style network for Zertz.

    Input:  (batch, 14, 7, 7) board tensor — no separate reserve vector
    Output: policy_logits (batch, 5587), value (batch, 1) in [-1, 1]

    Policy head is a linear layer over the flattened trunk output (not spatial
    conv like Hive) because Zertz moves don't decompose neatly into per-cell
    channels.
    """

    def __init__(self, num_blocks: int = 6, channels: int = 64):
        super().__init__()
        self.game = "zertz"

        self.input_conv = nn.Conv2d(NUM_CHANNELS, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        trunk_flat = channels * GRID_SIZE * GRID_SIZE  # e.g. 64*7*7 = 3136

        # Policy head: flatten trunk → linear → POLICY_SIZE
        self.policy_fc = nn.Linear(trunk_flat, POLICY_SIZE)

        # Value head: 1x1 conv → flatten → FC(256) → tanh
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(GRID_SIZE * GRID_SIZE, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, board_tensor: torch.Tensor):
        """Forward pass.

        Args:
            board_tensor: (batch, 14, 7, 7)

        Returns:
            policy_logits: (batch, 5587)
            value: (batch, 1) in [-1, 1]
        """
        x = F.relu(self.input_bn(self.input_conv(board_tensor)))
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        p = x.view(x.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


def create_model(num_blocks: int = 6, channels: int = 64) -> ZertzNet:
    return ZertzNet(num_blocks=num_blocks, channels=channels)


def save_checkpoint(model: ZertzNet, path: str, iteration: int = 0,
                    metadata: dict | None = None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "game": "zertz",
        "num_blocks": len(model.res_blocks),
        "channels": model.input_conv.out_channels,
        "iteration": iteration,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)


def load_checkpoint(path: str) -> tuple[ZertzNet, dict]:
    checkpoint = torch.load(path, weights_only=False)
    num_blocks = checkpoint.get("num_blocks", 6)
    channels = checkpoint.get("channels", 64)
    model = ZertzNet(num_blocks=num_blocks, channels=channels)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    compatible = {k: v for k, v in state_dict.items()
                  if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(compatible, strict=False)
    if "model_state_dict" not in checkpoint:
        checkpoint = {"iteration": 0, "metadata": {}}
    model.eval()
    return model, checkpoint
