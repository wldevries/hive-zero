"""Widen a trained model by adding zero-initialized channels (Net2WiderNet).

Existing channel weights are preserved in the first `old_channels` slots.
New channels are zero-initialized so the network initially behaves
identically to the original, then training gradually activates the new
capacity.

For conv layers: new output channels get zero weights; new input channels
get zero weights. For BatchNorm: new channels get mean=0, var=1, gamma=1,
beta=0 (identity transform).

Usage:
    uv run python scripts/widen_channels.py --input model.pt --output model_wide.pt --channels 128
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from hive.nn.model import HiveNet, load_checkpoint, save_checkpoint


def _widen_conv(old_conv, new_conv):
    """Copy old conv weights into new (wider) conv, zero-padding new channels."""
    old_w = old_conv.weight.data  # (out_old, in_old, kH, kW)
    new_w = new_conv.weight.data  # (out_new, in_new, kH, kW)
    new_w.zero_()
    out_old, in_old = old_w.shape[:2]
    new_w[:out_old, :in_old] = old_w
    if old_conv.bias is not None:
        new_conv.bias.data.zero_()
        new_conv.bias.data[:out_old] = old_conv.bias.data


def _widen_bn(old_bn, new_bn):
    """Copy old BatchNorm params, init new channels to identity (mean=0, var=1, gamma=1, beta=0)."""
    old_ch = old_bn.num_features
    # gamma (weight)
    new_bn.weight.data.fill_(1.0)
    new_bn.weight.data[:old_ch] = old_bn.weight.data
    # beta (bias)
    new_bn.bias.data.zero_()
    new_bn.bias.data[:old_ch] = old_bn.bias.data
    # running stats
    new_bn.running_mean.zero_()
    new_bn.running_mean[:old_ch] = old_bn.running_mean
    new_bn.running_var.fill_(1.0)
    new_bn.running_var[:old_ch] = old_bn.running_var
    new_bn.num_batches_tracked.copy_(old_bn.num_batches_tracked)


def widen_channels(src_path: str, dst_path: str, new_channels: int):
    model, ckpt = load_checkpoint(src_path)
    old_channels = model.input_conv.out_channels
    num_blocks = len(model.res_blocks)
    iteration = ckpt.get("iteration", 0)

    if new_channels <= old_channels:
        print(f"Error: new channels ({new_channels}) must be > old channels ({old_channels})")
        sys.exit(1)

    print(f"Loaded: {num_blocks} blocks, {old_channels} channels, iteration {iteration}")

    new_model = HiveNet(num_blocks=num_blocks, channels=new_channels)

    # Input conv: (39, old) -> (39, new)
    _widen_conv(model.input_conv, new_model.input_conv)
    _widen_bn(model.input_bn, new_model.input_bn)

    # Residual blocks: (old, old) -> (new, new)
    for i in range(num_blocks):
        old_block = model.res_blocks[i]
        new_block = new_model.res_blocks[i]
        _widen_conv(old_block.conv1, new_block.conv1)
        _widen_bn(old_block.bn1, new_block.bn1)
        _widen_conv(old_block.conv2, new_block.conv2)
        _widen_bn(old_block.bn2, new_block.bn2)

    # Policy head: conv (old, old, 1) -> (new, new, 1), then (old, 11, 1) -> (new, 11, 1)
    _widen_conv(model.policy_conv, new_model.policy_conv)
    _widen_bn(model.policy_bn, new_model.policy_bn)
    # policy_out: input channels widen, output stays 11
    _widen_conv(model.policy_out, new_model.policy_out)

    # Value head: (old, 1, 1) -> (new, 1, 1), input channels widen
    _widen_conv(model.value_conv, new_model.value_conv)
    _widen_bn(model.value_bn, new_model.value_bn)
    # FC layers unchanged (input from 1-channel conv + reserve, not trunk channels)
    new_model.value_fc1.load_state_dict(model.value_fc1.state_dict())
    new_model.value_fc2.load_state_dict(model.value_fc2.state_dict())

    # Auxiliary head: (old, 1, 1) -> (new, 1, 1)
    _widen_conv(model.qd_conv, new_model.qd_conv)
    _widen_bn(model.qd_bn, new_model.qd_bn)
    new_model.qd_fc1.load_state_dict(model.qd_fc1.state_dict())
    new_model.qd_fc2.load_state_dict(model.qd_fc2.state_dict())

    # Verify: forward pass should produce identical output
    new_model.eval()
    model.eval()
    with torch.no_grad():
        dummy_board = torch.randn(1, 39, 23, 23)
        dummy_reserve = torch.randn(1, 10)
        old_policy, old_value, old_aux = model(dummy_board, dummy_reserve)
        new_policy, new_value, new_aux = new_model(dummy_board, dummy_reserve)
        policy_diff = (old_policy - new_policy).abs().max().item()
        value_diff = (old_value - new_value).abs().max().item()
        aux_diff = (old_aux - new_aux).abs().max().item()
        print(f"Verification (max abs diff): policy={policy_diff:.2e}, value={value_diff:.2e}, aux={aux_diff:.2e}")
        if policy_diff > 1e-4 or value_diff > 1e-4 or aux_diff > 1e-4:
            print("WARNING: output differs significantly — check widening logic!")

    params = sum(p.numel() for p in new_model.parameters())
    print(f"Widened: {num_blocks} blocks, {new_channels} channels, {params/1e6:.2f}M params")

    save_checkpoint(new_model, dst_path, iteration, ckpt.get("metadata", {}))
    print(f"Saved to {dst_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source checkpoint")
    parser.add_argument("--output", required=True, help="Destination checkpoint")
    parser.add_argument("--channels", type=int, required=True, help="New channel count (must be > current)")
    args = parser.parse_args()
    widen_channels(args.input, args.output, args.channels)
