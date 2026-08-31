"""One-step training: learn the map u(t) -> u(t+dt) from (input, shifted) pairs."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def train_one_step(
    model: nn.Module,
    inputs: np.ndarray,  # (Ntrain, N)
    targets: np.ndarray,  # (Ntrain, N)
    device: torch.device,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_inputs: np.ndarray | None = None,
    val_targets: np.ndarray | None = None,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    torch.manual_seed(seed)
    model = model.to(device)
    X = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    Y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()
    n = X.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    history = {"train": [], "val": []}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        running = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            pred = model(X[idx])
            loss = loss_fn(pred, Y[idx])
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        sched.step()
        history["train"].append(running / n)
        if val_inputs is not None and (ep % 25 == 0 or ep == epochs - 1):
            v = evaluate_one_step(model, val_inputs, val_targets, device)
            history["val"].append((ep, v))
            if verbose:
                print(f"  ep {ep:4d}  train {running / n:.3e}  val {v:.3e}")
    return history


@torch.no_grad()
def evaluate_one_step(
    model: nn.Module, inputs: np.ndarray, targets: np.ndarray, device: torch.device
) -> float:
    model.eval()
    X = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    Y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    pred = model(X)
    return torch.mean((pred - Y) ** 2).item()
