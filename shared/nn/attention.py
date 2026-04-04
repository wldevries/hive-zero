"""Spatial self-attention block with adaLN-Zero conditioning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """Multi-head self-attention over spatial positions with adaLN-Zero.

    Takes (B, C, H, W) feature maps, treats H*W positions as tokens,
    applies adaLN-Zero conditioned multi-head attention + FFN with residuals,
    and reshapes back to (B, C, H, W).

    adaLN-Zero: conditioning vector produces per-channel scale, shift, and gate
    for each normalization + residual. Gate is zero-initialized so the block
    starts as an identity function.

    Uses F.scaled_dot_product_attention directly instead of
    nn.MultiheadAttention for clean ONNX dynamo export.
    """

    def __init__(self, channels: int, cond_dim: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.LayerNorm(channels, elementwise_affine=False)
        # QKV projection (combined for efficiency)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.out_proj = nn.Linear(channels, channels)

        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.norm2 = nn.LayerNorm(channels, elementwise_affine=False)

        # adaLN-Zero: conditioning -> (scale1, shift1, gate1, scale2, shift2, gate2)
        self.adaln = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * channels),
        )
        # Zero-init the final linear so gates start at 0 (identity)
        nn.init.zeros_(self.adaln[1].weight)
        nn.init.zeros_(self.adaln[1].bias)

    def forward(self, x, cond):
        """
        Args:
            x: (B, C, H, W) feature map
            cond: (B, cond_dim) conditioning vector
        """
        B, C, H, W = x.shape
        # Reshape to sequence: (B, H*W, C)
        tokens = x.view(B, C, H * W).permute(0, 2, 1)

        # Compute adaLN parameters from conditioning
        adaln_params = self.adaln(cond)  # (B, 6*C)
        scale1, shift1, gate1, scale2, shift2, gate2 = adaln_params.chunk(6, dim=-1)
        # Each is (B, C) — will broadcast over sequence dim

        # Self-attention with adaLN-Zero
        normed = self.norm(tokens) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        qkv = self.qkv(normed).reshape(B, H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, seq, head_dim)
        q, k, v = qkv.unbind(0)
        attn_out = F.scaled_dot_product_attention(q, k, v)  # (B, num_heads, seq, head_dim)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, H * W, C)
        attn_out = self.out_proj(attn_out)
        tokens = tokens + gate1.unsqueeze(1) * attn_out

        # FFN with adaLN-Zero
        normed2 = self.norm2(tokens) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        tokens = tokens + gate2.unsqueeze(1) * self.ffn(normed2)

        # Reshape back: (B, C, H, W)
        return tokens.permute(0, 2, 1).view(B, C, H, W)
