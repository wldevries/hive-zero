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

from shared.nn.gpba import GlobalPoolBias
from shared.nn.resblock import ResBlock

# Mirror Rust constants — must match yinsh_game::board_encoding / move_encoding.
NUM_CHANNELS = 9
GRID_SIZE = 11
RESERVE_SIZE = 6
POLICY_CHANNELS = 59  # 5 single-move channels + 6 dirs × 9 distances
POLICY_SIZE = POLICY_CHANNELS * GRID_SIZE * GRID_SIZE  # 7139


def _describe_trunk(trunk_spec: list[dict]) -> str:
    """Human-readable trunk summary, e.g. '8xres -> gpba -> 4xres'."""
    parts = []
    for spec in trunk_spec:
        t = spec["type"]
        c = spec.get("count", 1)
        parts.append(f"{c}x{t}" if c > 1 else t)
    return " -> ".join(parts)


class YinshNet(nn.Module):
    """Trunk → flat policy head + WDL value head.

    Input:  board (B, 9, 11, 11), reserve (B, 6)
    Output: policy (B, 7139), wdl_logits (B, 3) raw — callers softmax as needed.

    The trunk is a sequence of named layer types controlled by trunk_spec:
        "res"  — ResBlock (standard 2-conv residual)
        "gpba" — GlobalPoolBias (KataGo-style global context injection)

    Each spec entry has a "type" field and an optional "count" (default 1).
    Example: [{"type": "res", "count": 8}, {"type": "gpba"}, {"type": "res", "count": 4}]
    """

    def __init__(self, channels: int = 96, trunk: list[dict] | None = None):
        super().__init__()
        self.game = "yinsh"

        trunk_spec = trunk or [{"type": "res", "count": 8}]
        self.trunk_spec = trunk_spec

        self.input_conv = nn.Conv2d(
            NUM_CHANNELS + RESERVE_SIZE, channels, 3, padding=1, bias=False
        )
        self.input_bn = nn.BatchNorm2d(channels)

        self.trunk = nn.ModuleList()
        for spec in trunk_spec:
            layer_type = spec["type"]
            count = spec.get("count", 1)
            for _ in range(count):
                if layer_type == "res":
                    self.trunk.append(ResBlock(channels))
                elif layer_type == "gpba":
                    self.trunk.append(GlobalPoolBias(channels))
                else:
                    raise ValueError(f"Unknown trunk layer type: {layer_type!r}")

        # Single conv1x1 policy head producing 59 channels (matches the Rust
        # `move_encoding` layout: place_ring, remove_row×3, remove_ring,
        # then 54 ring-movement channels = 6 dirs × 9 distances).
        self.policy_conv = nn.Conv2d(channels, POLICY_CHANNELS, 1)

        # WDL value head: conv1x1 → flatten → FC256 → 3 logits (Win, Draw, Loss).
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(GRID_SIZE * GRID_SIZE, 256)
        self.value_fc2 = nn.Linear(256, 3)

    def forward(self, board: torch.Tensor, reserve: torch.Tensor):
        # Broadcast the 6-element reserve vector spatially so the trunk sees
        # global state in every residual block.
        r = reserve.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, GRID_SIZE, GRID_SIZE)
        x = torch.cat([board, r], dim=1)
        x = F.relu(self.input_bn(self.input_conv(x)))
        for layer in self.trunk:
            x = layer(x)

        policy_logits = self.policy_conv(x).reshape(x.size(0), -1)  # (B, 7139)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        wdl_logits = self.value_fc2(v)  # (B, 3) raw logits

        return policy_logits, wdl_logits


def create_model(model_config: dict | None = None) -> YinshNet:
    cfg = model_config or {}
    trunk = cfg.get("trunk")
    # Backwards compat: a bare "num_blocks" maps to a single res-block run.
    if trunk is None and "num_blocks" in cfg:
        trunk = [{"type": "res", "count": cfg["num_blocks"]}]
    return YinshNet(
        channels=cfg.get("channels", 96),
        trunk=trunk,
    )


def save_checkpoint(
    model: YinshNet, path: str, generation: int = 0, metadata: dict | None = None
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "game": "yinsh",
            "channels": model.input_conv.out_channels,
            "trunk": model.trunk_spec,
            "generation": generation,
            "metadata": metadata or {},
        },
        path,
    )


def load_checkpoint(path: str) -> tuple[YinshNet, dict]:
    ckpt = torch.load(path, weights_only=False)
    channels = ckpt.get("channels", 96)
    trunk = ckpt.get("trunk")
    # Backwards compat: older checkpoints stored "num_blocks" instead of a trunk spec.
    if trunk is None and "num_blocks" in ckpt:
        trunk = [{"type": "res", "count": ckpt["num_blocks"]}]
    model = YinshNet(channels=channels, trunk=trunk)
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


class _OnnxExportWrapper(nn.Module):
    """Wraps YinshNet for ONNX export, softmaxing WDL logits to probabilities."""
    def __init__(self, model: "YinshNet"):
        super().__init__()
        self.model = model

    def forward(self, board: torch.Tensor, reserve: torch.Tensor):
        policy, wdl_logits = self.model(board, reserve)
        wdl = F.softmax(wdl_logits, dim=1)
        return policy, wdl


def export_onnx(model: YinshNet, path: str):
    """Export to ONNX for Rust-native inference via the `ort` crate."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    dummy_board = torch.zeros(1, NUM_CHANNELS, GRID_SIZE, GRID_SIZE, device=device)
    dummy_reserve = torch.zeros(1, RESERVE_SIZE, device=device)
    wrapper = _OnnxExportWrapper(model)
    torch.onnx.export(
        wrapper,
        (dummy_board, dummy_reserve),
        path,
        input_names=["board", "reserve"],
        output_names=["policy", "wdl"],
        dynamic_axes={"board": {0: "batch"}, "reserve": {0: "batch"},
                      "policy": {0: "batch"}, "wdl": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    if was_training:
        model.train()
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  ONNX exported: {path} ({size_mb:.1f} MB)")
