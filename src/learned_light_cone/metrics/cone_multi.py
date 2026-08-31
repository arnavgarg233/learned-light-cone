"""Cone probe for multi-field states such as the wave equation (u, u_t).

Perturbs one field and reads the response in another. With c*dt = m*dx the exact
response is confined to |x - x0| <= m*k, so the scalar leakage and front-speed
readouts in metrics.cone apply to the returned field unchanged.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def signed_perturbation_probe_field(
    step_fn,
    base: torch.Tensor,
    bump: torch.Tensor,
    eps: float,
    K: int,
    perturb_field: int = 0,
    read_field: int = 0,
) -> torch.Tensor:
    """Multi-field signed perturbation response of shape (B, K+1, N).

    base : (B, n_fields, N) base states.
    bump : (B, N)           localized perturbation applied to channel
                            `perturb_field` only.
    step_fn : maps (M, n_fields, N) -> (M, n_fields, N) (a model or exact solver).
    """
    B = base.shape[0]
    pert = base.clone()
    pert[:, perturb_field] = pert[:, perturb_field] + eps * bump
    s = torch.cat([base, pert], dim=0)  # (2B, n_fields, N)
    states = [s]
    for _ in range(K):
        s = step_fn(s)
        states.append(s)
    traj = torch.stack(states, dim=1)  # (2B, K+1, n_fields, N)
    base_traj, pert_traj = traj[:B], traj[B:]
    diff = (pert_traj[:, :, read_field] - base_traj[:, :, read_field]) / eps
    return diff


@torch.no_grad()
def perturbation_probe_field(
    step_fn,
    base: torch.Tensor,
    bump: torch.Tensor,
    eps: float,
    K: int,
    perturb_field: int = 0,
    read_field: int = 0,
) -> torch.Tensor:
    """Multi-field perturbation response R of shape (B, K+1, N).

    The base and perturbed trajectories are rolled together in one batch and
    R_k(x) = ((F^k(s + eps*v) - F^k(s)) / eps)^2 ~ |D F^k[s] v|^2 is read off the
    `read_field` channel. With perturb_field == read_field == 0 this is the
    u-impulse -> u-response probe whose causal support is the symmetric cone
    |d| <= m*k.
    """
    diff = signed_perturbation_probe_field(step_fn, base, bump, eps, K, perturb_field, read_field)
    return diff**2
