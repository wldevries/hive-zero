"""Spatial self-attention block shared across game networks."""

import torch.nn as nn


class SpatialAttention(nn.Module):
    """Multi-head self-attention over spatial positions.

    Takes (B, C, H, W) feature maps, treats H*W positions as tokens,
    applies pre-norm multi-head attention + FFN with residuals, and
    reshapes back to (B, C, H, W).
    """

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
