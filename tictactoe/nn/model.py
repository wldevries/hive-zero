"""TicTacToeNet: minimal AlphaZero-style network for 3x3 tic-tac-toe."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.nn.resblock import ResBlock

NUM_CHANNELS = 2
GRID_SIZE = 3
POLICY_SIZE = 9
RESERVE_SIZE = 0


class TicTacToeNet(nn.Module):
    """AlphaZero-style network for tic-tac-toe.

    Input:  (batch, 2, 3, 3) board tensor
    Output: policy_logits (batch, 9), value (batch, 1) in [-1, 1]
    """

    def __init__(self, num_blocks: int = 2, channels: int = 32):
        super().__init__()
        self.game = "tictactoe"

        self.input_conv = nn.Conv2d(NUM_CHANNELS, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        # Policy head: conv1x1 -> flatten to 9
        self.policy_conv = nn.Conv2d(channels, 1, 1)

        # Value head: conv1x1 -> flatten -> FC -> tanh
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(GRID_SIZE * GRID_SIZE, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, board_tensor: torch.Tensor, reserve_vector: torch.Tensor | None = None):
        x = F.relu(self.input_bn(self.input_conv(board_tensor)))
        for block in self.res_blocks:
            x = block(x)

        # Policy
        policy_logits = self.policy_conv(x).view(x.size(0), -1)  # (B, 9)

        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)  # (B, 9)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # (B, 1)

        return policy_logits, v


def create_model(num_blocks: int = 2, channels: int = 32) -> TicTacToeNet:
    return TicTacToeNet(num_blocks=num_blocks, channels=channels)


def save_checkpoint(model: TicTacToeNet, path: str, generation: int = 0, metadata: dict | None = None):
    torch.save({
        "model_state_dict": model.state_dict(),
        "game": "tictactoe",
        "num_blocks": len(model.res_blocks),
        "channels": model.input_conv.out_channels,
        "generation": generation,
        "metadata": metadata or {},
    }, path)


def load_checkpoint(path: str) -> tuple[TicTacToeNet, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = create_model(
        num_blocks=ckpt["num_blocks"],
        channels=ckpt["channels"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def export_onnx(model: TicTacToeNet, path: str):
    model.eval()
    dummy_board = torch.zeros(1, NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
    torch.onnx.export(
        model,
        (dummy_board,),
        path,
        input_names=["board"],
        output_names=["policy", "value"],
        dynamic_axes={
            "board": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
    )
