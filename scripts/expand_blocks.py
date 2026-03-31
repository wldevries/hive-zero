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


def expand_hive(src_path, dst_path, add_blocks):
    from hive.nn.model import HiveNet, load_checkpoint, save_checkpoint

    model, ckpt = load_checkpoint(src_path)
    old_blocks = len(model.res_blocks)
    channels = model.input_conv.out_channels
    grid_size = model.grid_size
    iteration = ckpt.get("generation", 0)

    print(f"Loaded: {old_blocks} blocks, {channels} channels, grid={grid_size}, generation {iteration}")

    new_model = HiveNet(num_blocks=old_blocks + add_blocks, channels=channels, grid_size=grid_size)

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
    new_model.qd_conv.load_state_dict(model.qd_conv.state_dict())
    new_model.qd_bn.load_state_dict(model.qd_bn.state_dict())
    new_model.qd_fc1.load_state_dict(model.qd_fc1.state_dict())
    new_model.qd_fc2.load_state_dict(model.qd_fc2.state_dict())

    # Zero-initialize bn2.weight (gamma) on new blocks → identity at start
    for i in range(old_blocks, old_blocks + add_blocks):
        torch.nn.init.zeros_(new_model.res_blocks[i].bn2.weight)

    params = sum(p.numel() for p in new_model.parameters())
    print(f"Expanded: {old_blocks + add_blocks} blocks, {channels} channels, {params/1e6:.2f}M params")
    save_checkpoint(new_model, dst_path, iteration, ckpt.get("metadata", {}))
    print(f"Saved to {dst_path}")


def expand_zertz(src_path, dst_path, add_blocks):
    from zertz.nn.model import ZertzNet, load_checkpoint, save_checkpoint

    model, ckpt = load_checkpoint(src_path)
    old_blocks = len(model.res_blocks)
    channels = model.input_conv.out_channels
    iteration = ckpt.get("generation", 0)

    print(f"Loaded: {old_blocks} blocks, {channels} channels, generation {iteration}")

    new_model = ZertzNet(num_blocks=old_blocks + add_blocks, channels=channels)

    new_model.input_conv.load_state_dict(model.input_conv.state_dict())
    new_model.input_bn.load_state_dict(model.input_bn.state_dict())
    for i, block in enumerate(model.res_blocks):
        new_model.res_blocks[i].load_state_dict(block.state_dict())
    new_model.policy_fc.load_state_dict(model.policy_fc.state_dict())
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


def expand_blocks(src_path: str, dst_path: str, add_blocks: int):
    ckpt = torch.load(src_path, weights_only=False)
    game = ckpt.get("game", "hive")
    print(f"Game: {game}")
    if game == "zertz":
        expand_zertz(src_path, dst_path, add_blocks)
    else:
        expand_hive(src_path, dst_path, add_blocks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source checkpoint")
    parser.add_argument("--output", required=True, help="Destination checkpoint")
    parser.add_argument("--add-blocks", type=int, default=4, help="Blocks to append (default: 4)")
    args = parser.parse_args()
    expand_blocks(args.input, args.output, args.add_blocks)
