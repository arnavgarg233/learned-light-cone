"""Jacobian magnitude baselines: Frobenius norm (Hutchinson) and spectral radius.

Both reuse the finite-difference Jacobian-vector product of the cone probe,
J v ~ (F(u + eps v) - F(u)) / eps, so magnitude and geometry are measured with
the same protocol.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def _jvp(step_fn, u: torch.Tensor, v: torch.Tensor, eps: float) -> torch.Tensor:
    """Finite-difference J v for a one-step map. u, v: (B, ...) ; returns (B, ...)."""
    return (step_fn(u + eps * v) - step_fn(u)) / eps


@torch.no_grad()
def jacobian_frobenius(
    step_fn, base: torch.Tensor, eps: float = 1e-3, n_probe: int = 16, seed: int = 0
) -> np.ndarray:
    """Hutchinson estimate of ||J||_F per base state.

    For v with iid unit-variance entries, E[||J v||^2] = tr(J^T J) = ||J||_F^2.
    base: (B, N) (or (B, C, N)). Returns (B,) array of ||J||_F estimates.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = base.shape[0]
    acc = torch.zeros(B, device=base.device)
    for _ in range(n_probe):
        v = torch.randn(base.shape, generator=g).to(base.device)
        Jv = _jvp(step_fn, base, v, eps)
        acc = acc + (Jv.reshape(B, -1) ** 2).sum(dim=-1)
    fro2 = acc / n_probe
    return torch.sqrt(torch.clamp(fro2, min=0.0)).cpu().numpy()


@torch.no_grad()
def jacobian_spectral_radius(
    step_fn, base: torch.Tensor, eps: float = 1e-3, n_iter: int = 30, seed: int = 0
) -> np.ndarray:
    """Power iteration for the dominant eigenvalue magnitude |lambda_max(J)| per
    base state, using only JVPs: repeatedly w = J v, lambda ~ ||w||/||v||,
    v <- w/||w||. This is the contraction/expansion (rho(J) <= 1) quantity used
    by Jacobian-regularization stability work. base: (B, N) or (B, C, N).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    B = base.shape[0]
    v = torch.randn(base.shape, generator=g).to(base.device)
    v = v / (v.reshape(B, -1).norm(dim=-1).reshape((-1,) + (1,) * (v.ndim - 1)) + 1e-30)
    lam = torch.zeros(B, device=base.device)
    for _ in range(n_iter):
        w = _jvp(step_fn, base, v, eps)
        wn = w.reshape(B, -1).norm(dim=-1)
        vn = v.reshape(B, -1).norm(dim=-1)
        lam = wn / (vn + 1e-30)
        shape = (-1,) + (1,) * (w.ndim - 1)
        v = w / (wn.reshape(shape) + 1e-30)
    return lam.cpu().numpy()


def spectral_gain_from_G(G: np.ndarray) -> dict:
    """Spectral-gain baseline (McCabe-style) from a learned transfer G(k).

    Returns max amplification max_k |G(k)| (a spectral-radius-in-frequency
    proxy) and the high-frequency mean gain. |G(k)| > 1 anywhere flags spectral
    blow-up risk. G is the complex per-mode gain from learned_light_cone.metrics.spectral.
    """
    mag = np.abs(np.asarray(G))
    n = len(mag)
    hi = mag[n // 2 :] if n > 1 else mag
    return dict(
        max_gain=float(mag.max()),
        hf_mean_gain=float(hi.mean()),
        n_modes_amplified=int((mag > 1.0 + 1e-6).sum()),
    )
