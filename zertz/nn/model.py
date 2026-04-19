"""ZertzNet: AlphaZero-style network for Zertz with factorized conv1x1 policy heads."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.nn.resblock import ResBlock

# From Rust board encoding
NUM_CHANNELS = 6
GRID_SIZE = 7
POLICY_SIZE = 490  # flat training storage: place_W/G/B/remove[4*49] + cap_dir[6*49]
RESERVE_SIZE = 22

# Policy head sizes (must match Rust PLACE_HEAD_SIZE, CAP_HEAD_SIZE)
PLACE_HEAD_CHANNELS = 4  # ch 0-2: place W/G/B, ch 3: remove ring
PLACE_HEAD_SIZE = PLACE_HEAD_CHANNELS * GRID_SIZE * GRID_SIZE  # 196
NUM_DIR_CHANNELS = 6  # one per hex direction (E, NE, NW, W, SW, SE)
CAP_HEAD_SIZE = NUM_DIR_CHANNELS * GRID_SIZE * GRID_SIZE  # 294


class ZertzNet(nn.Module):
    """AlphaZero-style network for Zertz with direction-based capture policy head.

    Input:  (batch, 6, 7, 7) board tensor + (batch, 22) reserve vector
    Output: place_logits (batch, 4*7*7),
            cap_dir_logits (batch, 6*7*7),
            value (batch, 1) in [-1, 1]

    Reserve vector is broadcast spatially and concatenated with the board
    tensor before the trunk, so the trunk sees both board state and global
    context (supply, captures) through all residual blocks.

    Policy heads are conv1x1 over the trunk (no flatten/FC).
    Rust MCTS scores each capture as cap_dir[direction][source_pos].
    """

    def __init__(self, num_blocks: int = 6, channels: int = 64):
        super().__init__()
        self.game = "zertz"

        # Input conv takes board channels + broadcast reserve
        self.input_conv = nn.Conv2d(NUM_CHANNELS + RESERVE_SIZE, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)

        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_blocks)])

        # Policy heads (conv1x1 over trunk)
        self.policy_place = nn.Conv2d(channels, PLACE_HEAD_CHANNELS, 1)
        self.policy_cap_dir = nn.Conv2d(channels, NUM_DIR_CHANNELS, 1)

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
        x = torch.cat([board_tensor, r], dim=1)  # (B, 6+22, 7, 7)

        # Trunk
        x = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy heads (conv1x1 over trunk, then flatten for Rust)
        place_logits = self.policy_place(x).view(x.size(0), -1)    # (B, 196)
        cap_dir_logits = self.policy_cap_dir(x).view(x.size(0), -1) # (B, 294)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return place_logits, cap_dir_logits, value


def create_model(num_blocks: int = 6, channels: int = 64) -> ZertzNet:
    return ZertzNet(num_blocks=num_blocks, channels=channels)


def save_checkpoint(model: ZertzNet, path: str, generation: int = 0,
                    metadata: dict | None = None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "game": "zertz",
        "num_blocks": len(model.res_blocks),
        "channels": model.input_conv.out_channels,
        "generation": generation,
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)


def export_onnx(model: ZertzNet, path: str):
    """Export model to ONNX format for Rust-native inference via ort.

    Inputs: board_tensor (B, 6, 7, 7), reserve (B, 22)
    Outputs: place (B, 196), cap_dir (B, 294), value (B, 1)
    """
    import os
    was_training = model.training
    model.eval()
    dummy_board = torch.zeros(1, NUM_CHANNELS, GRID_SIZE, GRID_SIZE).cuda()
    dummy_reserve = torch.zeros(1, RESERVE_SIZE).cuda()
    input_names = ["board", "reserve"]
    output_names = ["place", "cap_dir", "value"]
    import logging
    _onnx_logger = logging.getLogger("onnxscript")
    _prev_level = _onnx_logger.level
    _onnx_logger.setLevel(logging.WARNING)
    torch.onnx.export(
        model,
        (dummy_board, dummy_reserve),
        path,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=(
            {0: "batch_board"},
            {0: "batch_reserve"}
        ),
        dynamo=True,
        verbose=False,
        opset_version=21,
    )
    _onnx_logger.setLevel(_prev_level)
    if was_training:
        model.train()
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  ONNX exported: {path} ({size_mb:.1f} MB)")


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
        checkpoint = {"generation": 0, "metadata": {}}
    # Backwards compat: old checkpoints use "iteration"
    if "generation" not in checkpoint and "iteration" in checkpoint:
        checkpoint["generation"] = checkpoint.pop("iteration")
    model.eval()
    return model, checkpoint
