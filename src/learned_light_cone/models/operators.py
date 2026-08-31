"""Multi-field 1D Fourier neural operator for coupled states such as (u, u_t)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fno import SpectralConv1d


class FNOMulti(nn.Module):
    """1D FNO on a multi-field state: (B, n_fields, N) -> (B, n_fields, N).

    Same spectral-conv design as FNO1d, but the lift/projection are multi-channel
    so the operator can learn the inter-field coupling of the wave map (u <-> v).
    """

    def __init__(self, modes: int = 16, width: int = 32, n_layers: int = 4, n_fields: int = 2):
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.n_fields = n_fields
        # lift the n_fields physical channels into the lifted feature space
        self.fc0 = nn.Linear(n_fields, width)
        self.specs = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, n_fields)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_fields, N) -> (B, n_fields, N)
        x = x.permute(0, 2, 1)  # (B, N, n_fields)
        x = self.fc0(x)  # (B, N, width)
        x = x.permute(0, 2, 1)  # (B, width, N)
        for i, (spec, w) in enumerate(zip(self.specs, self.ws)):
            x = spec(x) + w(x)
            if i < self.n_layers - 1:
                x = F.gelu(x)
        x = x.permute(0, 2, 1)  # (B, N, width)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)  # (B, N, n_fields)
        return x.permute(0, 2, 1)  # (B, n_fields, N)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
