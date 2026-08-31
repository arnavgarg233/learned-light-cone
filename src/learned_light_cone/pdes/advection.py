"""1D linear advection, u_t + c u_x = 0, on a periodic domain.

The time step is chosen so that c*dt = m*dx for an integer m. The exact solution
operator is then a roll by m cells and has zero response outside the cone.
"""

from __future__ import annotations

import numpy as np


def make_grid(N: int, L: float = 1.0):
    dx = L / N
    x = np.arange(N) * dx
    return x, dx


def band_limited_ic(
    rng: np.random.Generator,
    N: int,
    n_samples: int,
    J: int = 12,
    p: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """Random band-limited initial conditions.

    u0(x) = sum_{k=1}^{J} a_k sin(2 pi k x + phi_k),   a_k ~ N(0, k^{-p}).

    Returns array of shape (n_samples, N), each row standardized to zero mean
    and unit std when normalize=True.
    """
    x, _ = make_grid(N)
    ks = np.arange(1, J + 1)
    # amplitude std decays as k^{-p/2} so that variance ~ k^{-p}
    amp_std = ks.astype(np.float64) ** (-p / 2.0)
    out = np.zeros((n_samples, N), dtype=np.float64)
    for s in range(n_samples):
        a = rng.normal(0.0, amp_std)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=J)
        u = np.zeros(N)
        for j, k in enumerate(ks):
            u += a[j] * np.sin(2.0 * np.pi * k * x + phi[j])
        out[s] = u
    if normalize:
        out = out - out.mean(axis=1, keepdims=True)
        std = out.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        out = out / std
    return out


def exact_step(u: np.ndarray, m: int) -> np.ndarray:
    """One exact solver step: shift right by m cells (u_new[i] = u_old[i-m])."""
    return np.roll(u, m, axis=-1)


def exact_rollout(u0: np.ndarray, m: int, K: int) -> np.ndarray:
    """Return stacked exact states for k = 0..K, shape (K+1, ..., N)."""
    states = [np.asarray(u0)]
    u = np.asarray(u0)
    for _ in range(K):
        u = exact_step(u, m)
        states.append(u)
    return np.stack(states, axis=0)


def make_local_bump(
    N: int, x0: int, sigma: float = 1.0, width_cells: int | None = None
) -> np.ndarray:
    """Compact perturbation centered at cell x0.

    Default: Gaussian bump of scale `sigma` cells, support radius ~ 3*sigma.
    `width_cells` (if given) instead makes a flat box of that many cells.
    The bump is L2-normalized. The half-support radius (in cells) is returned
    by `bump_radius` and is used to dilate the causal cone so a rigid shift of
    the bump stays inside the cone (Lambda_solver = 0).
    """
    idx = np.arange(N)
    # signed periodic distance from x0
    d = (idx - x0 + N // 2) % N - N // 2
    if width_cells is not None:
        v = (np.abs(d) <= (width_cells - 1) / 2.0).astype(np.float64)
    else:
        v = np.exp(-0.5 * (d / sigma) ** 2)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v


def bump_radius(sigma: float = 1.0, width_cells: int | None = None) -> int:
    """Half-support radius (cells) used to dilate the causal cone."""
    if width_cells is not None:
        return int(np.ceil((width_cells - 1) / 2.0))
    return int(np.ceil(3.0 * sigma))
