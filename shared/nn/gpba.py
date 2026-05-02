"""Global Pooling Bias Accumulator (GPBA) — KataGo-style global context injection."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalPoolBias(nn.Module):
    """Injects global context into every spatial cell via mean+max pooling.

    Computes mean and max over H×W, passes through two FC layers, and adds the
    result as a per-channel bias broadcast to every position. Gives every cell
    access to whole-board feature statistics without the O(H²W²) cost of
    attention. The final FC layer is zero-initialized so the block starts as an
    identity and only learns to contribute as training progresses.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.fc1 = nn.Linear(2 * channels, channels)
        self.fc2 = nn.Linear(channels, channels)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W). Adaptive pool with output_size=1 exports to ONNX
        # as GlobalAveragePool / GlobalMaxPool, which run on ORT's CUDA EP
        # at any opset. Equivalent .mean / .amax over the spatial dims
        # export to ReduceMean / ReduceMax, which switched to axes-as-input
        # form at opset 18 and force a CPU fallback under CUDA EP.
        mean = F.adaptive_avg_pool2d(x, 1).flatten(1)   # (B, C)
        max_ = F.adaptive_max_pool2d(x, 1).flatten(1)   # (B, C)
        g = torch.cat([mean, max_], dim=1)              # (B, 2C)
        g = F.relu(self.fc1(g))                         # (B, C)
        bias = self.fc2(g)                              # (B, C)
        return x + bias.unsqueeze(-1).unsqueeze(-1)
