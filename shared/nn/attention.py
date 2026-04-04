"""Spatial self-attention block shared across game networks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """Multi-head self-attention over spatial positions.

    Takes (B, C, H, W) feature maps, treats H*W positions as tokens,
    applies pre-norm multi-head attention + FFN with residuals, and
    reshapes back to (B, C, H, W).

    Uses F.scaled_dot_product_attention directly instead of
    nn.MultiheadAttention for clean ONNX dynamo export.
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.LayerNorm(channels)
        # QKV projection (combined for efficiency)
        self.qkv = nn.Linear(channels, 3 * channels)
        self.out_proj = nn.Linear(channels, channels)

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
        # QKV: (B, seq, 3*C) -> 3x (B, num_heads, seq, head_dim)
        qkv = self.qkv(normed).reshape(B, H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, seq, head_dim)
        q, k, v = qkv.unbind(0)
        attn_out = F.scaled_dot_product_attention(q, k, v)  # (B, num_heads, seq, head_dim)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, H * W, C)
        attn_out = self.out_proj(attn_out)
        tokens = tokens + attn_out

        # FFN with residual
        tokens = tokens + self.ffn(self.norm2(tokens))

        # Reshape back: (B, C, H, W)
        return tokens.permute(0, 2, 1).view(B, C, H, W)
