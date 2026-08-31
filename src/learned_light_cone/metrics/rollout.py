"""Rollout error and time-to-failure under a higher-wavenumber test distribution."""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def model_rollout(model, u0: np.ndarray, K: int, device: torch.device) -> np.ndarray:
    """Autoregressively apply the model K times. Returns (K+1, B, N)."""
    model.eval()
    u = torch.as_tensor(u0, dtype=torch.float32, device=device)
    if u.ndim == 1:
        u = u.unsqueeze(0)
    states = [u]
    for _ in range(K):
        u = model(u)
        states.append(u)
    return torch.stack(states, dim=0).cpu().numpy()


def rollout_rmse_curve(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-step RMSE averaged over batch and space. pred,true: (K+1, B, N)."""
    err = np.sqrt(((pred - true) ** 2).mean(axis=(1, 2)))
    return err


def time_to_failure(rmse_curve: np.ndarray, eta: float) -> int:
    """First step H where RMSE exceeds eta (len-1 of curve if it never does)."""
    K = len(rmse_curve) - 1
    above = np.where(rmse_curve > eta)[0]
    return int(above[0]) if len(above) else K
