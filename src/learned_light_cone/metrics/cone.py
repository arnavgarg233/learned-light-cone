"""Learned-light-cone measurement.

Given a one-step map F (a model or the exact solver), apply a localized
perturbation to a base state, roll both forward k steps, and measure the
response field R_k(x) = ((F^k(u+eps v) - F^k(u)) / eps)^2 ~ |D F^k[u] v|^2.

Two readouts (deliberately redundant; they fail differently):
  - front speed  c_hat  via the quantile-energy radius r_q(k)  (primary)
  - out-of-cone leakage mass Lambda_k = energy beyond the causal front
    (cleaner: identically 0 for a perfectly causal operator)

All leakage is reported baseline-corrected against the numerical solver run
through the identical protocol:  Lambda_corr = Lambda_model - Lambda_solver.
"""

from __future__ import annotations

import numpy as np
import torch


def signed_periodic_distance(N: int, x0: int) -> np.ndarray:
    idx = np.arange(N)
    return (idx - x0 + N // 2) % N - N // 2


@torch.no_grad()
def signed_perturbation_probe(
    step_fn, base: torch.Tensor, bump: torch.Tensor, eps: float, K: int
) -> torch.Tensor:
    """Return signed response of shape (B, K+1, N): (pert_traj - base_traj) / eps.

    base, bump: (B, N). step_fn maps (M, N) -> (M, N).
    """
    B = base.shape[0]
    pert = base + eps * bump
    s = torch.cat([base, pert], dim=0)
    states = [s]
    for _ in range(K):
        s = step_fn(s)
        states.append(s)
    traj = torch.stack(states, dim=1)  # (2B, K+1, N)
    base_traj, pert_traj = traj[:B], traj[B:]
    return (pert_traj - base_traj) / eps


@torch.no_grad()
def perturbation_probe(
    step_fn, base: torch.Tensor, bump: torch.Tensor, eps: float, K: int
) -> torch.Tensor:
    """Return R of shape (B, K+1, N): squared normalized response per step.

    base, bump: (B, N). step_fn maps (M, N) -> (M, N). Base and perturbed
    trajectories are rolled together in one batch for efficiency.
    """
    diff = signed_perturbation_probe(step_fn, base, bump, eps, K)
    return diff**2


def quantile_front(R_k: np.ndarray, x0: int, q: float = 0.95) -> float:
    """Radius (cells) containing fraction q of response energy at one step."""
    N = R_k.shape[-1]
    d = np.abs(signed_periodic_distance(N, x0))
    e = R_k.astype(np.float64)
    total = e.sum()
    if total <= 0:
        return 0.0
    order = np.argsort(d, kind="stable")
    csum = np.cumsum(e[order]) / total
    j = int(np.searchsorted(csum, q))
    j = min(j, len(order) - 1)
    return float(d[order][j])


def fit_front_speed(R: np.ndarray, x0: int, q: float = 0.95, kmin: int = 1):
    """Fit r_q(k) ~ r0 + c_hat * k over steps k = kmin..K. Returns (c_hat, r0, radii)."""
    K = R.shape[0] - 1
    ks = np.arange(kmin, K + 1)
    radii = np.array([quantile_front(R[k], x0, q) for k in ks], dtype=np.float64)
    A = np.vstack([np.ones_like(ks, dtype=np.float64), ks.astype(np.float64)]).T
    coef, *_ = np.linalg.lstsq(A, radii, rcond=None)
    r0, c_hat = coef
    return float(c_hat), float(r0), radii


def threshold_front(R_k: np.ndarray, x0: int, floor_frac: float = 1e-2) -> float:
    """Largest radius whose energy exceeds floor_frac * peak (robustness check)."""
    N = R_k.shape[-1]
    d = np.abs(signed_periodic_distance(N, x0))
    e = R_k.astype(np.float64)
    peak = e.max()
    if peak <= 0:
        return 0.0
    mask = e >= floor_frac * peak
    return float(d[mask].max()) if mask.any() else 0.0


def leakage_mass(R: np.ndarray, x0: int, m: float, r0: int) -> np.ndarray:
    """Out-of-cone energy fraction per step.

    Causal cone at step k: |d| <= m*k + r0 (max-speed cone dilated by the
    perturbation's half-support r0, so a rigid shift stays inside -> 0 leakage).
    Returns Lambda_k for k = 1..K (length K).
    """
    K = R.shape[0] - 1
    N = R.shape[-1]
    d = np.abs(signed_periodic_distance(N, x0))
    out = np.zeros(K, dtype=np.float64)
    for k in range(1, K + 1):
        e = R[k].astype(np.float64)
        total = e.sum()
        if total <= 0:
            out[k - 1] = 0.0
            continue
        outside = d > (m * k + r0)
        out[k - 1] = e[outside].sum() / total
    return out
