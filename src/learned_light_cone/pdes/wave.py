"""1D wave equation, u_tt = c^2 u_xx, on a periodic domain with state (u, u_t).

Each Fourier mode is rotated exactly. With c*dt = m*dx the response to an impulse
is confined to |x - x0| <= m*k.
"""

from __future__ import annotations

import numpy as np
import torch


def _omega_theta(N: int, c: float, dt: float, L: float = 1.0, device="cpu"):
    """Per-(rfft)-mode angular frequency omega_j and phase theta_j = omega_j*dt."""
    j = torch.arange(N // 2 + 1, dtype=torch.float64, device=device)
    k = 2.0 * np.pi * j / L  # physical wavenumber of mode j
    omega = c * k  # non-dispersive
    theta = omega * dt
    return omega, theta


def wave_dt(N: int, c: float, m: int, L: float = 1.0) -> float:
    """Timestep giving the clean integer-shift two-branch cone: c*dt = m*dx."""
    dx = L / N
    return m * dx / c


def band_limited_ic_wave(
    rng, N: int, n_samples: int, J: int = 12, p: float = 1.0, zero_velocity: bool = True
):
    """Initial (u0, v0). With zero_velocity=True a u-bump splits symmetrically
    into +/- c branches (cleanest light cone). u0 standardized to unit std."""
    from .advection import band_limited_ic

    u0 = band_limited_ic(rng, N, n_samples, J, p, normalize=True)
    if zero_velocity:
        v0 = np.zeros_like(u0)
    else:
        v0 = band_limited_ic(rng, N, n_samples, J, p, normalize=True)
    return u0, v0


@torch.no_grad()
def exact_step_wave(
    u: torch.Tensor, v: torch.Tensor, theta: torch.Tensor, omega: torch.Tensor, dt: float
):
    """One exact spectral step. u,v: (..., N) real. theta,omega: (N//2+1,)."""
    N = u.shape[-1]
    uh = torch.fft.rfft(u.to(torch.float64), dim=-1)
    vh = torch.fft.rfft(v.to(torch.float64), dim=-1)
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    # sin(theta)/omega with the omega->0 limit (= dt) for the j=0 mode
    sinc_over_omega = torch.where(
        omega > 0, sin / torch.clamp(omega, min=1e-30), torch.full_like(omega, dt)
    )
    uh_new = uh * cos + vh * sinc_over_omega
    vh_new = -uh * (omega * sin) + vh * cos
    u_new = torch.fft.irfft(uh_new, n=N, dim=-1)
    v_new = torch.fft.irfft(vh_new, n=N, dim=-1)
    return u_new.to(u.dtype), v_new.to(v.dtype)


def make_wave_stepper(N: int, c: float, m: int, device="cpu", L: float = 1.0):
    """Return (step_fn, dt). step_fn maps stacked state (B, 2, N) -> (B, 2, N),
    channel 0 = u, channel 1 = v. Suitable as the exact-solver baseline."""
    dt = wave_dt(N, c, m, L)
    omega, theta = _omega_theta(N, c, dt, L, device=device)

    @torch.no_grad()
    def step(s: torch.Tensor) -> torch.Tensor:
        u, v = s[:, 0], s[:, 1]
        u2, v2 = exact_step_wave(u, v, theta, omega, dt)
        return torch.stack([u2, v2], dim=1)

    return step, dt


def exact_rollout_wave(u0, v0, N, c, m, K, device="cpu", L: float = 1.0):
    """Stacked exact states for k=0..K. Returns (K+1, B, 2, N)."""
    step, _ = make_wave_stepper(N, c, m, device=device, L=L)
    u0 = torch.as_tensor(u0, dtype=torch.float32, device=device)
    v0 = torch.as_tensor(v0, dtype=torch.float32, device=device)
    if u0.ndim == 1:
        u0 = u0.unsqueeze(0)
        v0 = v0.unsqueeze(0)
    s = torch.stack([u0, v0], dim=1)
    states = [s]
    for _ in range(K):
        s = step(s)
        states.append(s)
    return torch.stack(states, dim=0)
