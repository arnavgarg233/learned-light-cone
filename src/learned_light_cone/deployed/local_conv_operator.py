"""Finite-convolution reference operator on the 0.25 degree grid.

A residual stack of 3x3 convolutions with no global operations. Its receptive
field per step is the sum of kernel half-widths times dilation, so the response
to a localized perturbation has bounded radius. Longitude uses circular padding
and latitude uses replicate padding.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class PeriodicConv2d(nn.Module):
    """3x3 conv: circular pad in lon (dim -1), replicate pad in lat (dim -2)."""

    def __init__(self, cin, cout, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=0, dilation=dilation)
        self.d = dilation

    def forward(self, x):
        x = F.pad(x, (self.d, self.d, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.d, self.d), mode="replicate")
        return self.conv(x)


class LocalConvNet(nn.Module):
    """Residual local CNN. Per-step receptive radius is finite and tunable via
    n_layers and dilation -- the causal anchor."""

    def __init__(self, in_channels=26, hidden=64, n_layers=6, dilation=1):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, hidden, 1)
        self.layers = nn.ModuleList(
            [PeriodicConv2d(hidden, hidden, dilation=dilation) for _ in range(n_layers)]
        )
        self.act = nn.GELU()
        self.proj = nn.Conv2d(hidden, in_channels, 1)
        self.n_layers, self.dilation = n_layers, dilation

    def receptive_radius_km(self, grid_deg=0.25):
        """Half-width of receptive field in km along a meridian (great circle)."""
        # each 3x3 conv with dilation d reaches d grid cells each side
        cells = self.n_layers * self.dilation
        km_per_cell = grid_deg / 360.0 * (2 * 3.14159265 * 6371.0)
        return cells * km_per_cell

    def forward(self, x):
        z = self.act(self.lift(x))
        for lyr in self.layers:
            z = z + self.act(lyr(z))
        return x + self.proj(z)  # residual next-state
