"""AlphaZero-style neural network for Hive."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.nn.resblock import ResBlock
from ..encoding.board_encoder import NUM_CHANNELS, DEFAULT_GRID_SIZE, RESERVE_SIZE
from ..encoding.move_encoder import NUM_POLICY_CHANNELS, policy_size


class HiveNet(nn.Module):
    """AlphaZero-style network with policy, value, and auxiliary heads.

    Reserve vector is broadcast spatially and concatenated with the board
    tensor before the trunk, so the trunk sees both board state and global
    context (reserves) through all residual blocks.

    Architecture:
        - Broadcast reserve + concat with board tensor
        - Input convolution
        - Residual tower (num_blocks blocks)
        - Policy head: conv -> flatten -> POLICY_SIZE
        - Value head: conv(1x1) -> flatten -> FC(256) -> tanh
        - Auxiliary head: conv(1x1) -> flatten -> FC(64) -> sigmoid
          (my_qd, opp_qd, my_queen_escape, opp_queen_escape, my_mobility, opp_mobility)
    """

    def __init__(self, num_blocks: int = 10, channels: int = 128,
                 grid_size: int = DEFAULT_GRID_SIZE):
        super().__init__()
        self.game = "hive"
        if grid_size > DEFAULT_GRID_SIZE:
            raise ValueError(f"grid_size {grid_size} exceeds max board size {DEFAULT_GRID_SIZE}")
        if grid_size % 2 == 0:
            raise ValueError(f"grid_size must be odd, got {grid_size}")
        self.grid_size = grid_size

        # Input convolution (board channels + broadcast reserve)
        self.input_conv = nn.Conv2d(NUM_CHANNELS + RESERVE_SIZE, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        # Policy head - convolutional to avoid giant linear layer
        # Outputs per-cell logits: one channel per piece (11 pieces per player).
        # Channel = piece index (0=Queen, 1-2=Spider, 3-4=Beetle, 5-7=Grasshopper, 8-10=Ant).
        # Covers both placement and movement — no separate direction channels.
        self.num_policy_channels = NUM_POLICY_CHANNELS
        self.policy_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(channels)
        self.policy_out = nn.Conv2d(channels, self.num_policy_channels, 1)

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(grid_size * grid_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Auxiliary head (own pathway from trunk)
        # Outputs: [my_qd, opp_qd, my_queen_escape, opp_queen_escape, my_mobility, opp_mobility]
        self.qd_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.qd_bn = nn.BatchNorm2d(1)
        self.qd_fc1 = nn.Linear(grid_size * grid_size, 64)
        self.qd_fc2 = nn.Linear(64, 6)

    def forward(self, board_tensor: torch.Tensor, reserve_vector: torch.Tensor):
        """Forward pass.

        Args:
            board_tensor: (batch, NUM_CHANNELS, GRID_SIZE, GRID_SIZE)
            reserve_vector: (batch, RESERVE_SIZE)

        Returns:
            policy_logits: (batch, POLICY_SIZE)
            value: (batch, 1) in [-1, 1]
            aux: (batch, 6) in [0, 1] — [my_qd, opp_qd, my_qe, opp_qe, my_mob, opp_mob]
        """
        # Broadcast reserve spatially and concat with board tensor
        g = board_tensor.size(-1)
        r = reserve_vector.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, g, g)
        x = torch.cat([board_tensor, r], dim=1)  # (B, 39+10, G, G)

        # Shared trunk
        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy head - outputs (batch, num_policy_channels, GRID_SIZE, GRID_SIZE)
        # Then flattened to (batch, num_policy_channels * GRID_SIZE * GRID_SIZE)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = self.policy_out(p)
        policy_logits = p.view(p.size(0), -1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        # Auxiliary head (own pathway from trunk)
        qd = F.relu(self.qd_bn(self.qd_conv(x)))
        qd = qd.view(qd.size(0), -1)
        qd = F.relu(self.qd_fc1(qd))
        aux = torch.sigmoid(self.qd_fc2(qd))

        return policy_logits, value, aux


def create_model(num_blocks: int = 10, channels: int = 128,
                 grid_size: int = DEFAULT_GRID_SIZE) -> HiveNet:
    """Create a new HiveNet model."""
    return HiveNet(num_blocks=num_blocks, channels=channels, grid_size=grid_size)


def save_checkpoint(model: HiveNet, path: str, generation: int = 0,
                    metadata: dict | None = None):
    """Save model with training metadata."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "game": "hive",
        "num_blocks": model.res_blocks.__len__(),
        "channels": model.input_conv.out_channels,
        "grid_size": model.grid_size,
        "generation": generation,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)


def export_onnx(model: HiveNet, path: str):
    """Export model to ONNX format for Rust-native inference via ort.

    Inputs: board_tensor (B, 39, G, G), reserve (B, 10)
    Outputs: policy (B, 11*G*G), value (B, 1), aux (B, 6)
    """
    import os
    g = model.grid_size
    was_training = model.training
    model.eval()
    dummy_board = torch.zeros(1, NUM_CHANNELS, g, g)
    dummy_reserve = torch.zeros(1, RESERVE_SIZE)
    torch.onnx.export(
        model,
        (dummy_board, dummy_reserve),
        path,
        input_names=["board", "reserve"],
        output_names=["policy", "value", "aux"],
        dynamic_axes={
            "board": {0: "batch"},
            "reserve": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
            "aux": {0: "batch"},
        },
        opset_version=17,
    )
    if was_training:
        model.train()
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  ONNX exported: {path} ({size_mb:.1f} MB)")


def load_checkpoint(path: str) -> tuple[HiveNet, dict]:
    """Load model from checkpoint. Returns (model, checkpoint_dict)."""
    checkpoint = torch.load(path, weights_only=False)
    num_blocks = checkpoint.get("num_blocks", 10)
    channels = checkpoint.get("channels", 128)
    grid_size = checkpoint.get("grid_size", DEFAULT_GRID_SIZE)
    model = HiveNet(num_blocks=num_blocks, channels=channels, grid_size=grid_size)

    # Support both checkpoint format and raw state_dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Filter out keys with shape mismatches (e.g. aux head grew from 2 to 6 outputs)
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


# Keep simple aliases for backward compat
def save_model(model: HiveNet, path: str):
    save_checkpoint(model, path)


def load_model(path: str, num_blocks: int = 10, channels: int = 128) -> HiveNet:
    model, _ = load_checkpoint(path)
    return model
