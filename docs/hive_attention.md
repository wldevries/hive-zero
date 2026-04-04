# Attention Layers for Hive CNN

## Problem

Hive piece movement is topological, not spatial. An ant traverses the entire hive
perimeter in one move — two hexes 12 grid cells apart may be one move away, while
two hexes 2 cells apart may be unreachable. Spiders crawl exactly 3 steps along
the hive surface; grasshoppers jump over lines of pieces. None of these movement
patterns map to local convolutional filters.

A CNN with 6 residual blocks on a 15x15 grid has a receptive field large enough to
*see* the whole board, but it cannot easily *reason* about reachability along the
hive surface. Planning a queen surround requires understanding which pieces can
reach which hexes — a graph property the CNN's spatial inductive bias doesn't
capture.

Evidence: the network finishes games almost exclusively with ants, the one piece
type where "can reach any perimeter hex" is trivially true regardless of hive
shape. It never learns to coordinate spiders/beetles/grasshoppers for attacks
because it can't represent their reachability.

## Proposal: Self-Attention After CNN Trunk

Keep the existing CNN trunk (input conv + residual blocks) for local feature
extraction, then add 1-2 self-attention layers before the policy/value heads.

```
Input (24ch + 10 reserve, 15x15)
  │
  ├─ Input conv + BN + ReLU
  ├─ N residual blocks (spatial feature extraction)
  │
  ├─ 1-2 self-attention layers (global relationship reasoning)  ← NEW
  │
  ├─ Policy heads (place, src, dst)
  ├─ Value head
  └─ Auxiliary head
```

Each attention layer operates on the 15x15 = 225 spatial positions as tokens:

1. Reshape trunk output from (B, C, H, W) → (B, H*W, C)
2. Multi-head self-attention: each position attends to all others
3. Reshape back to (B, C, H, W) for downstream heads

### Why This Helps

- **Dynamic connectivity**: Attention weights are computed from the input, so the
  network can learn "this cell relates to that distant cell because they're
  connected along the hive perimeter" — different for every position.
- **No fixed locality**: Unlike conv filters, attention has no spatial bias. A
  piece 10 hexes away gets the same opportunity to influence a cell as an adjacent
  piece.
- **Compositional reasoning**: Multi-head attention can simultaneously track
  different relationships (piece reachability, queen proximity, hive boundary
  shape) in different heads.

### Why Not Pure Transformer

A pure Vision Transformer (no CNN) would work but:
- Loses the spatial inductive bias that *is* useful for local patterns (beetle
  stacking, adjacent pieces, gate detection)
- Needs more data/compute to learn what convolutions get for free
- Requires positional encoding (the CNN implicitly has this)

The hybrid approach gets the best of both: CNN handles local spatial patterns
efficiently, attention handles non-local relational reasoning.

## Architecture Details

### Attention Layer

```python
class SpatialAttention(nn.Module):
    """Multi-head self-attention over spatial positions."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape to sequence: (B, H*W, C)
        tokens = x.view(B, C, H * W).permute(0, 2, 1)

        # Self-attention with residual
        normed = self.norm(tokens)
        attn_out, _ = self.attn(normed, normed, normed)
        tokens = tokens + attn_out

        # FFN with residual
        tokens = tokens + self.ffn(self.norm2(tokens))

        # Reshape back: (B, C, H, W)
        return tokens.permute(0, 2, 1).view(B, C, H, W)
```

### Computational Cost

At 15x15 = 225 tokens with 96 channels:
- Attention matrix: 225 x 225 = 50,625 entries per head
- Per-layer FLOPs: ~2 * 225^2 * 96 ≈ 10M (tiny vs CNN trunk)
- Memory: 225x225 attention map per head per batch element

For comparison, one residual block (2x conv3x3 at 96ch on 15x15): ~60M FLOPs.
So 1-2 attention layers add ~15-30% compute on top of 6 residual blocks. Very
manageable.

### Positional Encoding

The CNN trunk output already encodes spatial position implicitly (each cell's
features reflect its receptive field context). No explicit positional encoding
should be needed — the attention layers receive position-aware features from
the trunk.

If needed, learnable 2D positional embeddings (15x15) can be added, but try
without first.

## Integration

### Minimal Changes

1. Add `SpatialAttention` module
2. In `HiveNet.forward()`, insert 1-2 attention layers between the trunk and
   the policy/value/aux heads
3. The attention layers use the same channel dimension as the trunk, so no
   dimension changes needed for the heads

### ONNX Export

`nn.MultiheadAttention` exports cleanly to ONNX. The reshape ops are standard.
No expected issues with ORT inference.

### Checkpoint Compatibility

New attention layer weights won't exist in old checkpoints. The existing
`load_checkpoint` already handles missing keys via `strict=False`, so old
checkpoints load fine — the attention layers just start randomly initialized.
This means you can resume from an existing trained model and fine-tune with
attention added.

## Alternatives Considered

### Squeeze-and-Excite (SE) Blocks

Channel-wise attention: global average pool → FC → sigmoid → scale channels.
Cheap, but only learns *which features matter globally*, not *which spatial
positions relate to each other*. Doesn't solve the reachability problem.

### Non-Local Blocks (Wang et al. 2018)

Essentially self-attention embedded in a CNN. Very similar to this proposal but
typically uses a single head and no FFN. The full transformer-style block
(multi-head + FFN + layer norm) is more expressive.

### Reachability Input Channels

Encode "cells reachable by my ants/spiders/etc" as additional input channels.
This gives the CNN the information directly but:
- Expensive to compute per position (ant reachability = flood fill along
  perimeter)
- Already partially computed for valid_moves in Rust
- Makes the input encoding much larger
- The network still can't reason about *opponent's* reachability without
  doubling the channels

Could be combined with attention as a complementary approach.

## Experiment Plan

1. **Baseline**: Current 6-block 96-channel CNN, 800 sims, note draw rate
2. **+1 attention**: Add 1 SpatialAttention layer after trunk, same hyperparams
3. **+2 attention**: Add 2 layers
4. **Metrics**: Draw rate, decisive game length, value loss convergence,
   win rate vs baseline checkpoint

Key question: does the network learn to coordinate non-ant pieces for queen
surrounds? Track which piece types deliver the final surrounding move in
decisive games.
