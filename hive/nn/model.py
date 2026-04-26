"""AlphaZero-style neural network for Hive."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.nn.attention import SpatialAttention
from shared.nn.gpba import GlobalPoolBias
from shared.nn.resblock import ResBlock
from ..encoding.board_encoder import NUM_CHANNELS, DEFAULT_GRID_SIZE, RESERVE_SIZE
from ..encoding.move_encoder import NUM_POLICY_CHANNELS, NUM_PLACE_CHANNELS, BILINEAR_DIM


def _describe_trunk(trunk_spec: list[dict]) -> str:
    """Human-readable trunk summary, e.g. '5×res → gpba → 5×res → gpba → attn'."""
    parts = []
    for spec in trunk_spec:
        t = spec["type"]
        c = spec.get("count", 1)
        parts.append(f"{c}x{t}" if c > 1 else t)
    return " -> ".join(parts)


class HiveNet(nn.Module):
    """AlphaZero-style network with configurable trunk, bilinear Q·K policy, value, and auxiliary heads.

    The trunk is a sequence of named layer types controlled by trunk_spec:
        "res"  — ResBlock (standard 2-conv residual)
        "gpba" — GlobalPoolBias (KataGo-style global context injection)
        "attn" — SpatialAttention (multi-head self-attention with adaLN-Zero)

    Each spec entry has a "type" field and an optional "count" (default 1).
    Example: [{"type": "res", "count": 5}, {"type": "gpba"}, {"type": "res", "count": 5}]

    Reserve vector is broadcast spatially and concatenated before the trunk.
    SpatialAttention layers additionally receive it as an adaLN-Zero conditioning signal.

    Policy heads (concatenated flat output):
        place_head: Conv1×1(C → 5, G, G)   placement logits per piece type
        q_head:     Conv1×1(C → D, G, G)   Q embeddings for movement source
        k_head:     Conv1×1(C → D, G, G)   K embeddings for movement destination
      Output: (batch, (5+2D)×G²) = [place | Q | K]
      Movement prior: Q[src] · K[dst] / sqrt(D)

    Value head:  conv → flatten → FC(256) → softmax(3)  [W, D, L]
    Auxiliary head: conv → flatten → FC(64) → sigmoid(6)
        [my_qd, opp_qd, my_queen_escape, opp_queen_escape, my_mobility, opp_mobility]
    """

    def __init__(self, channels: int = 64, grid_size: int = DEFAULT_GRID_SIZE,
                 trunk: list[dict] | None = None,
                 bilinear_dim: int = BILINEAR_DIM):
        super().__init__()
        self.game = "hive"
        if grid_size > DEFAULT_GRID_SIZE:
            raise ValueError(f"grid_size {grid_size} exceeds max board size {DEFAULT_GRID_SIZE}")
        if grid_size % 2 == 0:
            raise ValueError(f"grid_size must be odd, got {grid_size}")
        self.grid_size = grid_size
        self.bilinear_dim = bilinear_dim

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
                elif layer_type == "gpba":
                    self.trunk.append(GlobalPoolBias(channels))
                elif layer_type == "attn":
                    self.trunk.append(SpatialAttention(channels, cond_dim=RESERVE_SIZE))
                else:
                    raise ValueError(f"Unknown trunk layer type: {layer_type!r}")

        self.num_policy_channels = NUM_POLICY_CHANNELS
        self.policy_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(channels)
        self.policy_place = nn.Conv2d(channels, NUM_PLACE_CHANNELS, 1)
        self.policy_q     = nn.Conv2d(channels, bilinear_dim, 1)
        self.policy_k     = nn.Conv2d(channels, bilinear_dim, 1)

        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(grid_size * grid_size, 256)
        self.value_fc2 = nn.Linear(256, 3)

        self.qd_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.qd_bn = nn.BatchNorm2d(1)
        self.qd_fc1 = nn.Linear(grid_size * grid_size, 64)
        self.qd_fc2 = nn.Linear(64, 6)

    def forward(self, board_tensor: torch.Tensor, reserve_vector: torch.Tensor):
        """
        Args:
            board_tensor:  (batch, NUM_CHANNELS, G, G)
            reserve_vector: (batch, RESERVE_SIZE)

        Returns:
            policy_logits: (batch, (5+2D)×G²) — [place | Q | K]
            wdl_logits:    (batch, 3) raw logits — callers softmax as needed.
                           Training uses log_softmax for stability; the ONNX
                           wrapper softmaxes before export so ORT consumers
                           (Rust) still see probabilities.
            aux:           (batch, 6) sigmoid
        """
        g = board_tensor.size(-1)
        r = reserve_vector.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, g, g)
        x = torch.cat([board_tensor, r], dim=1)

        x = F.relu(self.input_bn(self.input_conv(x)))
        for layer in self.trunk:
            if isinstance(layer, SpatialAttention):
                x = layer(x, reserve_vector)
            else:
                x = layer(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        place = self.policy_place(p).flatten(1)
        q     = self.policy_q(p).flatten(1)
        k     = self.policy_k(p).flatten(1)
        policy_logits = torch.cat([place, q, k], dim=1)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        wdl_logits = self.value_fc2(v)

        qd = F.relu(self.qd_bn(self.qd_conv(x)))
        qd = qd.flatten(1)
        qd = F.relu(self.qd_fc1(qd))
        aux = torch.sigmoid(self.qd_fc2(qd))

        return policy_logits, wdl_logits, aux


def create_model(model_config: dict | None = None) -> HiveNet:
    """Create a new HiveNet from a model config dict."""
    cfg = model_config or {}
    return HiveNet(
        channels=cfg.get("channels", 64),
        grid_size=cfg.get("grid_size", DEFAULT_GRID_SIZE),
        trunk=cfg.get("trunk"),
        bilinear_dim=cfg.get("bilinear_dim", BILINEAR_DIM),
    )


def save_checkpoint(model: HiveNet, path: str, generation: int = 0,
                    metadata: dict | None = None):
    """Save model with architecture config and training metadata."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "game": "hive",
        "channels": model.input_conv.out_channels,
        "grid_size": model.grid_size,
        "trunk": model.trunk_spec,
        "bilinear_dim": model.bilinear_dim,
        "generation": generation,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)


class _OnnxExportWrapper(nn.Module):
    """Wraps HiveNet for ONNX export, passing WDL (B, 3) through directly."""
    def __init__(self, model: "HiveNet"):
        super().__init__()
        self.model = model

    def forward(self, board: torch.Tensor, reserve: torch.Tensor):
        policy, wdl_logits, aux = self.model(board, reserve)
        wdl = F.softmax(wdl_logits, dim=1)
        return policy, wdl, aux


def export_onnx(model: HiveNet, path: str, batch_size: int | None = None):
    """Export model to ONNX for Rust-native ORT inference.

    Inputs:  board_tensor (B, NUM_CHANNELS, G, G), reserve (B, RESERVE_SIZE)
    Outputs: policy (B, (5+2D)×G²), wdl (B, 3) [P(win), P(draw), P(loss)], aux (B, 6)
    Contempt is applied in the Rust caller as W - L - contempt * D.
    """
    import os
    g = model.grid_size
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    b = batch_size or 1
    dummy_board = torch.zeros(b, NUM_CHANNELS, g, g, device=device)
    dummy_reserve = torch.zeros(b, RESERVE_SIZE, device=device)
    wrapper = _OnnxExportWrapper(model).eval()
    dynamic_axes = None if batch_size else {
        "board": {0: "batch"}, "reserve": {0: "batch"},
        "policy": {0: "batch"}, "wdl": {0: "batch"}, "aux": {0: "batch"},
    }
    torch.onnx.export(
        wrapper,
        (dummy_board, dummy_reserve),
        path,
        input_names=["board", "reserve"],
        output_names=["policy", "wdl", "aux"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )
    if was_training:
        model.train()
    size_mb = os.path.getsize(path) / (1024 * 1024)
    batch_label = f" (batch={batch_size}, static)" if batch_size else ""
    print(f"  ONNX exported: {path} ({size_mb:.1f} MB){batch_label}")


def load_checkpoint(path: str) -> tuple[HiveNet, dict]:
    """Load model from checkpoint. Returns (model, checkpoint_dict)."""
    checkpoint = torch.load(path, weights_only=False)
    channels = checkpoint.get("channels", 64)
    grid_size = checkpoint.get("grid_size", DEFAULT_GRID_SIZE)
    trunk = checkpoint.get("trunk")
    bilinear_dim = checkpoint.get("bilinear_dim", BILINEAR_DIM)
    model = HiveNet(channels=channels, grid_size=grid_size, trunk=trunk, bilinear_dim=bilinear_dim)

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
