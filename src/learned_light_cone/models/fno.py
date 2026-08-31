"""1D Fourier neural operator without a grid-coordinate channel.

Omitting the coordinate input keeps the model translation equivariant, so the
measured response does not depend on where the perturbation is placed.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes = modes
        scale = 1.0 / (in_ch * out_ch)
        self.weight = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, N)
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, N//2+1)
        m = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.out_ch, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :m] = torch.einsum("bix,iox->box", x_ft[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(out_ft, n=N, dim=-1)


class FNO1d(nn.Module):
    def __init__(self, modes: int = 16, width: int = 32, n_layers: int = 4):
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.fc0 = nn.Linear(1, width)
        self.specs = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, N) -> (B, N)
        x = x.unsqueeze(-1)  # (B, N, 1)
        x = self.fc0(x)  # (B, N, width)
        x = x.permute(0, 2, 1)  # (B, width, N)
        for i, (spec, w) in enumerate(zip(self.specs, self.ws)):
            x = spec(x) + w(x)
            if i < self.n_layers - 1:
                x = F.gelu(x)
        x = x.permute(0, 2, 1)  # (B, N, width)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)  # (B, N, 1)
        return x.squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
