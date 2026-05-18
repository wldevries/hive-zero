"""ZertzNet: AlphaZero-style network for Zertz with factorized conv1x1 policy heads."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.nn.gpba import GlobalPoolBias
from shared.nn.resblock import ResBlock
from shared.nn.seresblock import SEResBlock

# From Rust board encoding
NUM_CHANNELS = 7  # 0-3 marbles/empty, 4 capture-turn flag, 5 mid-capture src, 6 mid-placement flag
GRID_SIZE = 7
POLICY_SIZE = 490  # flat training storage: place_W/G/B/remove[4*49] + cap_dir[6*49]
RESERVE_SIZE = 22

# Policy head sizes (must match Rust PLACE_HEAD_SIZE, CAP_HEAD_SIZE)
PLACE_HEAD_CHANNELS = 4  # ch 0-2: place W/G/B, ch 3: remove ring
PLACE_HEAD_SIZE = PLACE_HEAD_CHANNELS * GRID_SIZE * GRID_SIZE  # 196
NUM_DIR_CHANNELS = 6  # one per hex direction (E, NE, NW, W, SW, SE)
CAP_HEAD_SIZE = NUM_DIR_CHANNELS * GRID_SIZE * GRID_SIZE  # 294


def _describe_trunk(trunk_spec: list[dict]) -> str:
    """Human-readable trunk summary, e.g. '6xres -> gpba -> 6xres'."""
    parts = []
    for spec in trunk_spec:
        t = spec["type"]
        c = spec.get("count", 1)
        parts.append(f"{c}x{t}" if c > 1 else t)
    return " -> ".join(parts)


class ZertzNet(nn.Module):
    """AlphaZero-style network for Zertz with direction-based capture policy head.

    Input:  (batch, 7, 7, 7) board tensor + (batch, 22) reserve vector
    Output: place_logits (batch, 4*7*7),
            cap_dir_logits (batch, 6*7*7),
            value (batch, 1) in [-1, 1]

    Reserve vector is broadcast spatially and concatenated with the board
    tensor before the trunk, so the trunk sees both board state and global
    context (supply, captures) through all residual blocks.

    The trunk is a sequence of named layer types controlled by trunk_spec:
        "res"   — ResBlock (standard 2-conv residual)
        "seres" — SEResBlock (residual with Squeeze-and-Excitation channel gating)
        "gpba"  — GlobalPoolBias (KataGo-style global context injection)

    Each spec entry has a "type" field and an optional "count" (default 1).
    Example: [{"type": "seres", "count": 6}, {"type": "gpba"}, {"type": "seres", "count": 2}]

    Policy heads — each gets its own per-cell MLP (Conv1×1 + BN + ReLU)
    before the final projection. Post split-ply refactor, marble placement,
    ring removal, and capture-direction are three independent MCTS decision
    types: forcing them to share a single projection layer makes them fight
    over features instead of specialising. Mirrors Hive's place/Q/K pattern.
    The marble-placement (3ch) and ring-removal (1ch) projections concat to
    reproduce the existing 4-channel place_logits layout — Rust untouched.
    """

    def __init__(self, channels: int = 64, trunk: list[dict] | None = None):
        super().__init__()
        self.game = "zertz"

        trunk_spec = trunk or [{"type": "res", "count": 6}]
        self.trunk_spec = trunk_spec

        # Input conv takes board channels + broadcast reserve
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

        # Per-head pre-projection: Conv1×1(C→C) + BN + ReLU lets each policy
        # head specialise on different feature semantics from the trunk.
        self.policy_place_pre_conv  = nn.Conv2d(channels, channels, 1, bias=False)
        self.policy_place_pre_bn    = nn.BatchNorm2d(channels)
        self.policy_remove_pre_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.policy_remove_pre_bn   = nn.BatchNorm2d(channels)
        self.policy_cap_pre_conv    = nn.Conv2d(channels, channels, 1, bias=False)
        self.policy_cap_pre_bn      = nn.BatchNorm2d(channels)

        # Final projections — three independent heads. place_marble + remove_ring
        # concatenate to a 4-channel block matching the Rust place_logits layout
        # (ch 0-2 = W/G/B, ch 3 = remove).
        self.policy_place_marble = nn.Conv2d(channels, 3, 1)
        self.policy_remove_ring  = nn.Conv2d(channels, 1, 1)
        self.policy_cap_dir      = nn.Conv2d(channels, NUM_DIR_CHANNELS, 1)

        # Value head: conv1x1 → flatten → FC(256) → tanh
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(GRID_SIZE * GRID_SIZE, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, board_tensor: torch.Tensor, reserve_vector: torch.Tensor):
        """Forward pass.

        Returns:
            place_logits: (batch, 4*7*7) flattened placement head
            cap_dir_logits: (batch, 6*7*7) flattened direction head
            value: (batch, 1) in [-1, 1]
        """
        # Broadcast reserve spatially and concat with board tensor
        r = reserve_vector.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, GRID_SIZE, GRID_SIZE)
        x = torch.cat([board_tensor, r], dim=1)  # (B, 7+22, 7, 7)

        # Trunk
        x = F.relu(self.input_bn(self.input_conv(x)))
        for layer in self.trunk:
            x = layer(x)

        # Per-head pre-projection — each policy head sees specialised features.
        place_p  = F.relu(self.policy_place_pre_bn(self.policy_place_pre_conv(x)))
        remove_p = F.relu(self.policy_remove_pre_bn(self.policy_remove_pre_conv(x)))
        cap_p    = F.relu(self.policy_cap_pre_bn(self.policy_cap_pre_conv(x)))

        # Final projections. Concat place_marble (3ch) + remove_ring (1ch) into
        # a 4-channel block so the Rust-facing place_logits keeps its layout
        # (ch 0-2 = W/G/B, ch 3 = remove).
        place_marble = self.policy_place_marble(place_p)   # (B, 3, 7, 7)
        remove_ring  = self.policy_remove_ring(remove_p)    # (B, 1, 7, 7)
        place_logits = torch.cat([place_marble, remove_ring], dim=1).view(x.size(0), -1)  # (B, 196)
        cap_dir_logits = self.policy_cap_dir(cap_p).view(x.size(0), -1)                    # (B, 294)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return place_logits, cap_dir_logits, value


def create_model(model_config: dict | None = None) -> ZertzNet:
    cfg = model_config or {}
    trunk = cfg.get("trunk")
    # Backwards compat: a bare "num_blocks" maps to a single res-block run.
    if trunk is None and "num_blocks" in cfg:
        trunk = [{"type": "res", "count": cfg["num_blocks"]}]
    return ZertzNet(
        channels=cfg.get("channels", 64),
        trunk=trunk,
    )


def save_checkpoint(model: ZertzNet, path: str, generation: int = 0,
                    metadata: dict | None = None,
                    optimizer: "torch.optim.Optimizer | None" = None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "game": "zertz",
        "channels": model.input_conv.out_channels,
        "trunk": model.trunk_spec,
        "generation": generation,
        "metadata": metadata or {},
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def export_onnx(model: ZertzNet, path: str):
    """Export model to ONNX format for Rust-native inference via ort.

    Inputs: board_tensor (B, 7, 7, 7), reserve (B, 22)
    Outputs: place (B, 196), cap_dir (B, 294), value (B, 1)
    """
    import os
    was_training = model.training
    model.eval()
    dummy_board = torch.zeros(1, NUM_CHANNELS, GRID_SIZE, GRID_SIZE).cuda()
    dummy_reserve = torch.zeros(1, RESERVE_SIZE).cuda()
    input_names = ["board", "reserve"]
    output_names = ["place", "cap_dir", "value"]
    torch.onnx.export(
        model,
        (dummy_board, dummy_reserve),
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={"board": {0: "batch"}, "reserve": {0: "batch"},
                      "place": {0: "batch"}, "cap_dir": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    if was_training:
        model.train()
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  ONNX exported: {path} ({size_mb:.1f} MB)")


def load_checkpoint(path: str) -> tuple[ZertzNet, dict]:
    checkpoint = torch.load(path, weights_only=False)
    channels = checkpoint.get("channels", 64)
    trunk = checkpoint.get("trunk")
    # Backwards compat: older checkpoints stored "num_blocks" instead of a trunk spec.
    if trunk is None and "num_blocks" in checkpoint:
        trunk = [{"type": "res", "count": checkpoint["num_blocks"]}]
    model = ZertzNet(channels=channels, trunk=trunk)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    compatible = {k: v for k, v in state_dict.items()
                  if k in model_state and v.shape == model_state[k].shape}
    model.load_state_dict(compatible, strict=False)
    if "model_state_dict" not in checkpoint:
        checkpoint = {"generation": 0, "metadata": {}}
    # Backwards compat: old checkpoints use "iteration"
    if "generation" not in checkpoint and "iteration" in checkpoint:
        checkpoint["generation"] = checkpoint.pop("iteration")
    model.eval()
    return model, checkpoint


