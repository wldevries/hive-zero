"""YinshNet: AlphaZero-style network for YINSH.

Policy layout (59 channels × 11×11 grid):
  ch 0:    PlaceRing destination
  ch 1-3:  RemoveRow start, dir 0/1/2
  ch 4:    RemoveRing target
  ch 5-58: MoveRing — channel = 5 + dir_idx*9 + (dist-1), value at source cell

All moves use `PolicyIndex::Single`; Rust MCTS reads the logit at the source cell
in the appropriate channel. No summing or bilinear arithmetic needed.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.nn.resblock import ResBlock

# Mirror Rust constants — must match yinsh_game::board_encoding / move_encoding.
NUM_CHANNELS = 9
GRID_SIZE = 11
RESERVE_SIZE = 6
POLICY_CHANNELS = 59  # 5 single-move channels + 6 dirs × 9 distances
POLICY_SIZE = POLICY_CHANNELS * GRID_SIZE * GRID_SIZE  # 7139


class YinshNet(nn.Module):
    """Trunk → flat policy head + value head.

    Input:  board (B, 9, 11, 11), reserve (B, 6)
    Output: policy (B, 7139), value (B, 1) in [-1, 1]
    """

    def __init__(self, num_blocks: int = 8, channels: int = 96):
        super().__init__()
        self.game = "yinsh"

        self.input_conv = nn.Conv2d(
            NUM_CHANNELS + RESERVE_SIZE, channels, 3, padding=1, bias=False
        )
        self.input_bn = nn.BatchNorm2d(channels)
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        # Single conv1x1 policy head producing 59 channels (matches the Rust
        # `move_encoding` layout: place_ring, remove_row×3, remove_ring,
        # then 54 ring-movement channels = 6 dirs × 9 distances).
        self.policy_conv = nn.Conv2d(channels, POLICY_CHANNELS, 1)

        # Value head: conv1x1 → flatten → FC256 → tanh.
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(GRID_SIZE * GRID_SIZE, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, board: torch.Tensor, reserve: torch.Tensor):
        # Broadcast the 6-element reserve vector spatially so the trunk sees
        # global state in every residual block.
        r = reserve.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, GRID_SIZE, GRID_SIZE)
        x = torch.cat([board, r], dim=1)
        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            x = block(x)

        policy_logits = self.policy_conv(x).reshape(x.size(0), -1)  # (B, 847)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value


def create_model(num_blocks: int = 8, channels: int = 96) -> YinshNet:
    return YinshNet(num_blocks=num_blocks, channels=channels)


def save_checkpoint(
    model: YinshNet, path: str, generation: int = 0, metadata: dict | None = None
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "game": "yinsh",
            "num_blocks": len(model.res_blocks),
            "channels": model.input_conv.out_channels,
            "generation": generation,
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(path: str) -> tuple[YinshNet, dict]:
    ckpt = torch.load(path, weights_only=False)
    num_blocks = ckpt.get("num_blocks", 8)
    channels = ckpt.get("channels", 96)
    model = YinshNet(num_blocks=num_blocks, channels=channels)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model_state = model.state_dict()
    compatible = {
        k: v for k, v in state_dict.items()
        if k in model_state and v.shape == model_state[k].shape
    }
    model.load_state_dict(compatible, strict=False)
    if "model_state_dict" not in ckpt:
        ckpt = {"generation": 0, "metadata": {}}
    if "generation" not in ckpt and "iteration" in ckpt:
        ckpt["generation"] = ckpt.pop("iteration")
    model.eval()
    return model, ckpt


def export_onnx(model: YinshNet, path: str):
    """Export to ONNX for Rust-native inference via the `ort` crate."""
    import logging

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    dummy_board = torch.zeros(1, NUM_CHANNELS, GRID_SIZE, GRID_SIZE, device=device)
    dummy_reserve = torch.zeros(1, RESERVE_SIZE, device=device)
    onnx_logger = logging.getLogger("onnxscript")
    prev_level = onnx_logger.level
    onnx_logger.setLevel(logging.WARNING)
    torch.onnx.export(
        model,
        (dummy_board, dummy_reserve),
        path,
        input_names=["board", "reserve"],
        output_names=["policy", "value"],
        dynamic_shapes=({0: "batch_board"}, {0: "batch_reserve"}),
        dynamo=True,
        verbose=False,
        opset_version=21,
    )
    onnx_logger.setLevel(prev_level)
    if was_training:
        model.train()
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  ONNX exported: {path} ({size_mb:.1f} MB)")
