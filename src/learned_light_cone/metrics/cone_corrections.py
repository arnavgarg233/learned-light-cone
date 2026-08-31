"""Reference-corrected cone metrics.

1. Band-limited shift reference: an exactly causal operator restricted to the same
   retained modes already has a truncation ring, so excess is measured against it
   and not against the untruncated roll.
2. Directional cone: for one-way advection, upstream, downstream and
   beyond-front energy are reported separately.
3. Fit quality: fit_front_speed returns R^2 and the intercept so that a fit on a
   saturated response can be rejected.
"""

from __future__ import annotations

import numpy as np
import torch

from .cone import signed_periodic_distance


def bandlimited_shift_step(m: int, modes: int, N: int):
    """Step fn of an ideal, perfectly causal, band-limited shift operator:
    truncate to the lowest `modes` rFFT modes, then roll by m. This is the
    zero-learning null whose leakage is pure spectral-truncation ring."""

    @torch.no_grad()
    def step(x: torch.Tensor) -> torch.Tensor:
        xf = torch.fft.rfft(x, dim=-1)
        if modes < xf.shape[-1]:
            xf[..., modes:] = 0
        xb = torch.fft.irfft(xf, n=N, dim=-1)
        return torch.roll(xb, m, dims=-1)

    return step


def directional_leakage(R: np.ndarray, x0: int, m: float, r0: int) -> dict:
    """Split out-of-cone energy by direction, per step k=1..K.

    Returns dict of (K,) arrays:
      upstream  : energy at signed d < -r0           (physically forbidden for +m)
      downstream_ahead : energy at d > m*k + r0       (ahead of the causal front)
      total_out : upstream + downstream_ahead          (symmetric-cone leakage)
    """
    K = R.shape[0] - 1
    N = R.shape[-1]
    d = signed_periodic_distance(N, x0)  # signed
    up = np.zeros(K)
    ahead = np.zeros(K)
    tot = np.zeros(K)
    for k in range(1, K + 1):
        e = R[k].astype(np.float64)
        s = e.sum()
        if s <= 0:
            continue
        up[k - 1] = e[d < -r0].sum() / s
        ahead[k - 1] = e[d > (m * k + r0)].sum() / s
        tot[k - 1] = up[k - 1] + ahead[k - 1]
    return dict(upstream=up, downstream_ahead=ahead, total_out=tot)


def fit_front_speed_r2(R: np.ndarray, x0: int, q: float = 0.95, kmin: int = 1):
    """Like fit_front_speed but also returns R^2 of the linear fit so a speed can
    be gated. Returns (c_hat, r0, r2, radii)."""
    from .cone import quantile_front

    K = R.shape[0] - 1
    ks = np.arange(kmin, K + 1)
    radii = np.array([quantile_front(R[k], x0, q) for k in ks], dtype=np.float64)
    A = np.vstack([np.ones_like(ks, dtype=np.float64), ks.astype(np.float64)]).T
    coef, *_ = np.linalg.lstsq(A, radii, rcond=None)
    r0_fit, c_hat = coef
    pred = A @ coef
    ss_res = float(((radii - pred) ** 2).sum())
    ss_tot = float(((radii - radii.mean()) ** 2).sum()) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    return float(c_hat), float(r0_fit), float(r2), radii
