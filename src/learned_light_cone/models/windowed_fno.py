"""1D Fourier neural operator with a selectable spectral window.

`rect` is the standard hard cutoff. `raised_cosine` and `gaussian` taper the retained
band towards zero at the edge, which steepens the real-space kernel tail.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_window(modes: int, kind: str = "rect", gaussian_sigma_frac: float = 0.5) -> torch.Tensor:
    """Fixed real spectral window w_k of length `modes`, w_k in [0, 1].

    The window multiplies the retained mode amplitudes (mode index k = 0..modes-1).
    All windows are anchored so the DC / lowest modes are essentially unattenuated
    (w_0 ~ 1) and only the rolloff near the band edge k -> modes-1 differs:

      "rect"          : flat 1 everywhere -> the hard cutoff (CONTROL). Real-space
                        kernel = sinc, polynomial 1/r tail.
      "raised_cosine" : Hann-style half-cosine taper
                            w_k = 0.5 * (1 + cos(pi * k / (modes-1))).
                        C^1 at the band edge (w and w' -> 0), so the kernel tail
                        decays faster than any fixed polynomial set by the cutoff
                        order; empirically exponential-like.
      "gaussian"      : w_k = exp(-0.5 * (k / (sigma))^2), sigma = sigma_frac*modes.
                        C^infty, the canonical exponential-tail taper (a Gaussian
                        in frequency is a Gaussian in space).

    Returns a 1D float tensor of length `modes`.
    """
    if modes <= 0:
        return torch.ones(0, dtype=torch.float32)
    k = torch.arange(modes, dtype=torch.float64)
    kind = kind.lower()
    if kind == "rect":
        w = torch.ones(modes, dtype=torch.float64)
    elif kind in ("raised_cosine", "hann", "raised-cosine"):
        if modes == 1:
            w = torch.ones(1, dtype=torch.float64)
        else:
            # half of a Hann window: 1 at k=0, 0 at k=modes-1, C^1 at the edge.
            w = 0.5 * (1.0 + torch.cos(math.pi * k / (modes - 1)))
    elif kind == "gaussian":
        sigma = max(gaussian_sigma_frac * modes, 1e-6)
        w = torch.exp(-0.5 * (k / sigma) ** 2)
    else:
        raise ValueError(
            f"unknown window kind {kind!r}; expected one of rect, raised_cosine, gaussian"
        )
    return w.to(torch.float32)


class WindowedSpectralConv1d(nn.Module):
    """SpectralConv1d with a fixed (non-learned) spectral window on the K modes.

    Identical in structure and parameter count to src.models.fno.SpectralConv1d;
    the only addition is a non-trainable buffer `window` of shape (modes,) that
    multiplies the per-mode output amplitudes after the learned einsum.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        modes: int,
        window: str = "rect",
        gaussian_sigma_frac: float = 0.5,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.modes = modes
        self.window_kind = window
        scale = 1.0 / (in_ch * out_ch)
        # SAME learned weights as the rectangular FNO (same shape, same count).
        self.weight = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes, dtype=torch.cfloat))
        # FIXED non-learned window buffer (zero parameters added).
        w = make_window(modes, window, gaussian_sigma_frac)  # (modes,)
        self.register_buffer("window", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, N)
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, N//2+1)
        m = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(B, self.out_ch, x_ft.shape[-1], dtype=torch.cfloat, device=x.device)
        # learned per-mode transfer R_phi applied to the retained modes ...
        prod = torch.einsum("bix,iox->box", x_ft[:, :, :m], self.weight[:, :, :m])
        # ... then the FIXED spectral window taper (real, per-mode) folded in:
        #     G_eff(k) = window[k] * R_phi(k).  window is all-ones for "rect".
        prod = prod * self.window[:m].to(prod.dtype).view(1, 1, m)
        out_ft[:, :, :m] = prod
        return torch.fft.irfft(out_ft, n=N, dim=-1)


class WindowedFNO1d(nn.Module):
    """FNO1d whose spectral convs taper the retained K modes by a fixed window.

    Drop-in twin of src.models.fno.FNO1d: same modes/width/n_layers, same MLP
    head, same parameter count. `window` selects the taper applied (identically)
    in every spectral conv layer. window="rect" reproduces the standard FNO.
    """

    def __init__(
        self,
        modes: int = 16,
        width: int = 32,
        n_layers: int = 4,
        window: str = "rect",
        gaussian_sigma_frac: float = 0.5,
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.window = window
        self.gaussian_sigma_frac = gaussian_sigma_frac
        self.fc0 = nn.Linear(1, width)
        self.specs = nn.ModuleList(
            [
                WindowedSpectralConv1d(
                    width, width, modes, window=window, gaussian_sigma_frac=gaussian_sigma_frac
                )
                for _ in range(n_layers)
            ]
        )
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
