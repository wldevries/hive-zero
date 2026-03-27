"""ZertzNet: AlphaZero-style network for Zertz."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.nn.resblock import ResBlock

# From Rust: hive_engine.ZERTZ_NUM_CHANNELS, ZERTZ_GRID_SIZE, ZERTZ_POLICY_SIZE
NUM_CHANNELS = 4
GRID_SIZE = 7
POLICY_SIZE = 5587
RESERVE_SIZE = 9  # [supply_W, supply_G, supply_B, cur_cap_W/G/B, opp_cap_W/G/B]


class ZertzNet(nn.Module):
    """AlphaZero-style network for Zertz.

    Input:  (batch, 4, 7, 7) board tensor + (batch, 9) reserve vector
    Output: policy_logits (batch, 5587), value (batch, 1) in [-1, 1]

    Reserve vector: [supply_W, supply_G, supply_B, cur_cap_W/G/B, opp_cap_W/G/B]
    all normalized to [0, 1] by initial supply per color.

    Policy head is a linear layer over the flattened trunk output.
    Value head concatenates the reserve vector before its FC layers.
    """

    def __init__(self, num_blocks: int = 6, channels: int = 64):
        super().__init__()
        self.game = "zertz"

        self.input_conv = nn.Conv2d(NUM_CHANNELS, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        trunk_flat = channels * GRID_SIZE * GRID_SIZE

        # Policy head: flatten trunk → concat reserve → linear → POLICY_SIZE
        self.policy_fc = nn.Linear(trunk_flat + RESERVE_SIZE, POLICY_SIZE)

        # Value head: 1x1 conv → flatten → concat reserve → FC(256) → tanh
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(GRID_SIZE * GRID_SIZE + RESERVE_SIZE, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, board_tensor: torch.Tensor, reserve_vector: torch.Tensor):
        """Forward pass.

        Args:
            board_tensor: (batch, 4, 7, 7)
            reserve_vector: (batch, 9)

        Returns:
            policy_logits: (batch, 5587)
            value: (batch, 1) in [-1, 1]
        """
        x = F.relu(self.input_bn(self.input_conv(board_tensor)))
        for block in self.res_blocks:
            x = block(x)

        # Policy head: concat reserve so move choice can condition on capture progress
        p = x.view(x.size(0), -1)
        p = torch.cat([p, reserve_vector], dim=1)
        policy_logits = self.policy_fc(p)

        # Value head: squeeze to 1 channel, flatten, concat reserve
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = torch.cat([v, reserve_vector], dim=1)
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
