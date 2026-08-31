"""1D inviscid Burgers, u_t + (u^2 / 2)_x = 0, on a periodic domain.

The characteristic speed is the local state u, so the cone depends on the
solution. Integrated with a Rusanov flux and CFL-limited substeps.
"""

from __future__ import annotations

import numpy as np

from .advection import make_grid

# Rusanov is monotone for CFL <= 1; we keep a comfortable margin so that a single
# coarse step stays stable even as the solution steepens toward a shock.
CFL_TARGET = 0.4


def burgers_ic(
    rng: np.random.Generator,
    N: int,
    n_samples: int,
    J: int = 6,
    p: float = 1.0,
    amp: float = 0.5,
) -> np.ndarray:
    """Smooth band-limited initial conditions of moderate amplitude.

    u0(x) = amp * sum_{k=1}^{J} a_k sin(2 pi k x + phi_k),  a_k ~ N(0, k^{-p}),
    each row standardized to unit std *before* the amplitude scale, so that
    `amp` is the rms magnitude of the field. The amplitude is chosen so that
    characteristics cross and shocks form within a handful of coarse steps over
    the rollout horizon (too small -> stays smooth, no nonlinearity; too large
    -> instant shock, nothing to learn). A low band limit J keeps the initial
    field smooth so the FV solver is accurate before the shock forms.

    Returns array of shape (n_samples, N).
    """
    x, _ = make_grid(N)
    ks = np.arange(1, J + 1)
    amp_std = ks.astype(np.float64) ** (-p / 2.0)
    out = np.zeros((n_samples, N), dtype=np.float64)
    for s in range(n_samples):
        a = rng.normal(0.0, amp_std)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=J)
        u = np.zeros(N)
        for j, k in enumerate(ks):
            u += a[j] * np.sin(2.0 * np.pi * k * x + phi[j])
        u = u - u.mean()
        std = u.std()
        if std == 0:
            std = 1.0
        out[s] = amp * (u / std)
    return out


def _rusanov_flux(u: np.ndarray, dx: float, dt_sub: float) -> np.ndarray:
    """One forward-Euler sub-step with the periodic Rusanov (local LF) flux.

    u: (..., N). Returns the updated state of the same shape. Operates on the
    last axis so a whole batch is advanced at once.
    """
    f = 0.5 * u * u  # physical flux f(u) = u^2/2
    uR = np.roll(u, -1, axis=-1)  # u_{i+1}
    fR = np.roll(f, -1, axis=-1)  # f(u_{i+1})
    # local max wave speed at the i+1/2 interface: max(|u_i|, |u_{i+1}|)
    a = np.maximum(np.abs(u), np.abs(uR))
    # numerical flux at interface i+1/2  (between cell i and i+1)
    flux_R = 0.5 * (f + fR) - 0.5 * a * (uR - u)
    flux_L = np.roll(flux_R, 1, axis=-1)  # interface i-1/2 = flux at (i-1)+1/2
    return u - (dt_sub / dx) * (flux_R - flux_L)


def _n_substeps(u: np.ndarray, dt: float, dx: float, cfl_target: float = CFL_TARGET) -> int:
    """Number of equal sub-steps so every sub-step obeys CFL <= cfl_target.

    The CFL number for Rusanov is max|u| * dt_sub / dx. With dt_sub = dt / n we
    need n >= max|u| * dt / (cfl_target * dx). Computed from the *current* state
    (its max characteristic speed) so it adapts as the solution steepens.
    """
    smax = float(np.max(np.abs(u)))
    if smax <= 0.0:
        return 1
    n = int(np.ceil(smax * dt / (cfl_target * dx)))
    return max(1, n)


def burgers_step(
    u: np.ndarray,
    dt: float,
    dx: float,
    substeps: int | None = None,
    cfl_target: float = CFL_TARGET,
) -> np.ndarray:
    """One coarse step of length dt: the map the network learns, u -> u_next.

    Internally sub-stepped with the Rusanov flux so each sub-step satisfies the
    CFL condition. If `substeps` is None it is chosen adaptively from max|u| of
    the input so the step is stable even near shock formation; passing an
    explicit `substeps` (e.g. a fixed number reused for a whole dataset) makes
    the map deterministic and exactly reproducible as a training target.

    u: (N,) or (B, N). Returns the same shape.
    """
    u = np.asarray(u, dtype=np.float64)
    n = substeps if substeps is not None else _n_substeps(u, dt, dx, cfl_target)
    dt_sub = dt / n
    out = u
    for _ in range(n):
        out = _rusanov_flux(out, dx, dt_sub)
    return out


def burgers_rollout(
    u0: np.ndarray,
    dt: float,
    dx: float,
    K: int,
    substeps: int | None = None,
    cfl_target: float = CFL_TARGET,
) -> np.ndarray:
    """Stacked coarse states for k = 0..K, shape (K+1, ..., N).

    Sub-stepping is recomputed per coarse step (when substeps is None) so the
    rollout stays stable across the horizon as the shock develops.
    """
    u0 = np.asarray(u0, dtype=np.float64)
    states = [u0]
    u = u0
    for _ in range(K):
        u = burgers_step(u, dt, dx, substeps=substeps, cfl_target=cfl_target)
        states.append(u)
    return np.stack(states, axis=0)


def cmax(u: np.ndarray) -> np.ndarray | float:
    """Per-sample max characteristic speed max_x |u(x)| (the cone-speed bound).

    For Burgers the characteristic speed is f'(u) = u, so the fastest signal in
    a state travels at max|u|. After k coarse steps the true causal cone radius
    (in cells) is ~ k * dt * cmax(u) / dx, evaluated on the *base* state used in
    the perturbation probe. Returns a scalar for a single (N,) state or an array
    of length B for a (B, N) batch.
    """
    u = np.asarray(u, dtype=np.float64)
    return np.max(np.abs(u), axis=-1)


def burgers_dt(N: int, dt_frac: float = 1.0, L: float = 1.0) -> float:
    """A convenience coarse timestep dt = dt_frac * dx.

    With dt = dx, a unit-speed characteristic advances ~1 cell per coarse step,
    matching the cells/step convention used by the cone metrics. The actual cone
    speed in cells/step for a state u is then dt_frac * cmax(u).
    """
    dx = L / N
    return dt_frac * dx
