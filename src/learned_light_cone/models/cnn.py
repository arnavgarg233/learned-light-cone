"""Local 1D CNN with circular padding.

With kernel size k and n layers the receptive field radius is n * (k - 1) / 2
cells per application, for any weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalCNN1d(nn.Module):
    def __init__(self, kernel: int = 3, width: int = 32, n_layers: int = 4):
        super().__init__()
        assert kernel % 2 == 1, "use odd kernel for symmetric stencil"
        self.kernel = kernel
        self.width = width
        self.n_layers = n_layers
        self.pad = (kernel - 1) // 2
        chans = [1] + [width] * (n_layers - 1) + [1]
        self.convs = nn.ModuleList(
            [nn.Conv1d(chans[i], chans[i + 1], kernel, padding=0) for i in range(n_layers)]
        )

    @property
    def receptive_radius(self) -> int:
        return self.n_layers * ((self.kernel - 1) // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N) -> (B, N)
        x = x.unsqueeze(1)  # (B, 1, N)
        for i, conv in enumerate(self.convs):
            x = F.pad(x, (self.pad, self.pad), mode="circular")
            x = conv(x)
            if i < self.n_layers - 1:
                x = F.gelu(x)
        return x.squeeze(1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
