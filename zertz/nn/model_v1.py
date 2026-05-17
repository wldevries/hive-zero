"""ZertzNetV1: pre-split-ply Zertz model, kept for cross-version battle.

This is the architecture that produced `zertz-medium` and other pre-refactor
checkpoints — 6-channel board encoding (no mid_placement flag) and a single
conv1x1 per policy head (no per-head pre-projection). V1 was trained on the
joint placement move-space; its `place[3, :]` channel is a marginal over
"which ring to remove" conditioned on a joint placement turn.

Only used when battling an old checkpoint against the new split-ply model
via `ZertzSelfPlaySession.play_battle_versioned(style1="joint", ...)`.
Training, ONNX export, and the production self-play loop all use the new
`ZertzNet` in `model.py`.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.nn.gpba import GlobalPoolBias
from shared.nn.resblock import ResBlock
from shared.nn.seresblock import SEResBlock

# V1 board encoding — 6 channels (no mid_placement flag).
NUM_CHANNELS = 6
GRID_SIZE = 7
POLICY_SIZE = 490
RESERVE_SIZE = 22

PLACE_HEAD_CHANNELS = 4  # ch 0-2: place W/G/B, ch 3: remove ring
PLACE_HEAD_SIZE = PLACE_HEAD_CHANNELS * GRID_SIZE * GRID_SIZE  # 196
NUM_DIR_CHANNELS = 6
CAP_HEAD_SIZE = NUM_DIR_CHANNELS * GRID_SIZE * GRID_SIZE  # 294


class ZertzNetV1(nn.Module):
    """Pre-split-ply ZertzNet. Single conv1x1 per policy head, 6-channel input."""

    def __init__(self, channels: int = 64, trunk: list[dict] | None = None):
        super().__init__()
        self.game = "zertz"

        trunk_spec = trunk or [{"type": "res", "count": 6}]
        self.trunk_spec = trunk_spec

        self.input_conv = nn.Conv2d(NUM_CHANNELS + RESERVE_SIZE, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        self.trunk = nn.ModuleList()
        for spec in trunk_spec:
            layer_type = spec["type"]
            count = spec.get("count", 1)
            for _ in range(count):
                if layer_type == "res":
                    self.trunk.append(ResBlock(channels))
                elif layer_type == "seres":
                    self.trunk.append(SEResBlock(channels))
                elif layer_type == "gpba":
                    self.trunk.append(GlobalPoolBias(channels))
                else:
                    raise ValueError(f"Unknown trunk layer type: {layer_type!r}")

        # V1 had a single conv1x1 per head — no per-head pre-projection.
        self.policy_place = nn.Conv2d(channels, PLACE_HEAD_CHANNELS, 1)
        self.policy_cap_dir = nn.Conv2d(channels, NUM_DIR_CHANNELS, 1)

        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(GRID_SIZE * GRID_SIZE, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, board_tensor: torch.Tensor, reserve_vector: torch.Tensor):
        r = reserve_vector.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, GRID_SIZE, GRID_SIZE)
        x = torch.cat([board_tensor, r], dim=1)  # (B, 6+22, 7, 7)

        x = F.relu(self.input_bn(self.input_conv(x)))
        for layer in self.trunk:
            x = layer(x)

        place_logits = self.policy_place(x).view(x.size(0), -1)      # (B, 196)
        cap_dir_logits = self.policy_cap_dir(x).view(x.size(0), -1)  # (B, 294)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return place_logits, cap_dir_logits, value


def is_v1_state_dict(state_dict: dict) -> bool:
    """Detect a V1 checkpoint by the **absence** of post-refactor keys.

    `ZertzNet` introduces `policy_place_pre_conv.*` (the per-head pre-
    projection conv); V1 has none of those. Inspecting the keys avoids
    instantiating either model class just to feature-test.
    """
    return not any(k.startswith("policy_place_pre_conv") for k in state_dict.keys())


def create_v1_model(model_config: dict | None = None) -> ZertzNetV1:
    cfg = model_config or {}
    trunk = cfg.get("trunk")
    if trunk is None and "num_blocks" in cfg:
        trunk = [{"type": "res", "count": cfg["num_blocks"]}]
    return ZertzNetV1(channels=cfg.get("channels", 64), trunk=trunk)


def load_v1_checkpoint(path: str) -> tuple[ZertzNetV1, dict]:
    """Load a V1 checkpoint into a `ZertzNetV1`. The caller is expected to
    have verified the checkpoint is V1 via `is_v1_state_dict` first — this
    function does not autodetect."""
    checkpoint = torch.load(path, weights_only=False)
    channels = checkpoint.get("channels", 64)
    trunk = checkpoint.get("trunk")
    if trunk is None and "num_blocks" in checkpoint:
        trunk = [{"type": "res", "count": checkpoint["num_blocks"]}]
    model = ZertzNetV1(channels=channels, trunk=trunk)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    compatible = {k: v for k, v in state_dict.items()
                  if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(compatible, strict=False)
    if "model_state_dict" not in checkpoint:
        checkpoint = {"generation": 0, "metadata": {}}
    if "generation" not in checkpoint and "iteration" in checkpoint:
        checkpoint["generation"] = checkpoint.pop("iteration")
    model.eval()
    return model, checkpoint
