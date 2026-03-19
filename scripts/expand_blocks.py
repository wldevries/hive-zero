"""Expand a trained model by appending identity-initialized residual blocks.

The new blocks are zero-initialized at bn2.weight (gamma), so they act as
identity functions at the start. The network initially behaves identically
to the original, then training gradually activates the new capacity.

Usage:
    uv run python scripts/expand_blocks.py --input model.pt --output model_10b.pt --add-blocks 4
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from hive.nn.model import HiveNet, load_checkpoint, save_checkpoint


def expand_blocks(src_path: str, dst_path: str, add_blocks: int):
    model, ckpt = load_checkpoint(src_path)
    old_blocks = len(model.res_blocks)
    channels = model.input_conv.out_channels
    iteration = ckpt.get("iteration", 0)

    print(f"Loaded: {old_blocks} blocks, {channels} channels, iteration {iteration}")

    new_model = HiveNet(num_blocks=old_blocks + add_blocks, channels=channels)

    # Copy all existing weights
    new_model.input_conv.load_state_dict(model.input_conv.state_dict())
    new_model.input_bn.load_state_dict(model.input_bn.state_dict())
    for i, block in enumerate(model.res_blocks):
        new_model.res_blocks[i].load_state_dict(block.state_dict())
    new_model.policy_conv.load_state_dict(model.policy_conv.state_dict())
    new_model.policy_bn.load_state_dict(model.policy_bn.state_dict())
    new_model.policy_out.load_state_dict(model.policy_out.state_dict())
    new_model.value_conv.load_state_dict(model.value_conv.state_dict())
    new_model.value_bn.load_state_dict(model.value_bn.state_dict())
    new_model.value_fc1.load_state_dict(model.value_fc1.state_dict())
    new_model.value_fc2.load_state_dict(model.value_fc2.state_dict())

    # Zero-initialize bn2.weight (gamma) on new blocks → identity at start
    for i in range(old_blocks, old_blocks + add_blocks):
        torch.nn.init.zeros_(new_model.res_blocks[i].bn2.weight)

    params = sum(p.numel() for p in new_model.parameters())
    print(f"Expanded: {old_blocks + add_blocks} blocks, {channels} channels, {params/1e6:.2f}M params")

    save_checkpoint(new_model, dst_path, iteration, ckpt.get("metadata", {}))
    print(f"Saved to {dst_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source checkpoint")
    parser.add_argument("--output", required=True, help="Destination checkpoint")
    parser.add_argument("--add-blocks", type=int, default=4, help="Blocks to append (default: 4)")
    args = parser.parse_args()
    expand_blocks(args.input, args.output, args.add_blocks)
