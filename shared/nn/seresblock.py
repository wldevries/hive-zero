"""Residual block with Squeeze-and-Excitation channel attention."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEResBlock(nn.Module):
    """Residual block with two convolutions, batch norm, and SE gating.

    SE block reweights channels based on a global summary of the feature
    map: global avg pool -> bottleneck FC -> ReLU -> FC -> sigmoid -> scale
    by 2 -> multiply. The final FC is zero-initialized and the sigmoid is
    scaled by 2, so at init `s = 2 * sigmoid(0) = 1.0` and the SE branch is
    a true identity. It only learns to deviate from uniform gating as
    training progresses (KataGo-style; without the 2x the branch would
    halve every feature at init).
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        squeezed = max(channels // reduction, 4)
        self.se_fc1 = nn.Linear(channels, squeezed)
        self.se_fc2 = nn.Linear(squeezed, channels)
        nn.init.zeros_(self.se_fc2.weight)
        nn.init.zeros_(self.se_fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        # SE gating. adaptive_avg_pool2d(x, 1) exports cleanly to ONNX as
        # GlobalAveragePool, same reasoning as in GlobalPoolBias.
        s = F.adaptive_avg_pool2d(x, 1).flatten(1)   # (B, C)
        s = F.relu(self.se_fc1(s))                   # (B, squeezed)
        s = 2.0 * torch.sigmoid(self.se_fc2(s))      # (B, C), in [0, 2], identity at init
        x = x * s.unsqueeze(-1).unsqueeze(-1)

        x = F.relu(x + residual)
        return x
