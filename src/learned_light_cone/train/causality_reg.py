"""Training with a differentiable out-of-cone penalty.

The penalty weight changes the cone while the architecture and mode budget stay
fixed, which separates the effect of out-of-cone response from capacity.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _signed_periodic_distance_t(N: int, x0: int, device: torch.device) -> torch.Tensor:
    """Signed periodic distance from cell x0 as a torch tensor on `device`.

    Mirrors metrics.cone.signed_periodic_distance but stays in torch so the
    soft cone mask is part of the autograd graph's static (constant) inputs.
    """
    idx = torch.arange(N, device=device, dtype=torch.float32)
    return (idx - x0 + N // 2) % N - N // 2


def causality_penalty(
    model: nn.Module,
    base_batch: torch.Tensor,  # (B, N) base states, in-distribution
    bump_batch: torch.Tensor,  # (B, N) or (N,) localized L2-normalized perturbation(s)
    eps: float,
    K: int,
    m: float,
    r0: int,
    device: torch.device,
    x0: int | None = None,
    tau: float = 1.0,
) -> torch.Tensor:
    """Differentiable out-of-cone response-energy fraction, averaged over k.

    Physics: a maximally-causal operator can move a localized impulse no faster
    than the true characteristic speed, so after k steps all of the response
    energy must sit inside |d| <= m*k + r0 (the max-speed cone dilated by the
    bump's half-support so a rigid shift contributes zero). Energy outside that
    cone is "superluminal leakage" and is what we penalise.

    Unlike `perturbation_probe`, this keeps the autograd graph: base and
    perturbed states are rolled forward together through `model` with grad
    enabled, so d(penalty)/d(theta) flows back through every step.

    The cone indicator is replaced by a SOFT mask
        w_k(x) = sigmoid((|d(x)| - (m*k + r0)) / tau)  in [0, 1],
    ~1 well outside the cone, ~0 well inside, with a tau-wide transition. The
    per-step penalty is the soft-masked energy fraction
        L_k = sum_x w_k(x) R_k(x) / (sum_x R_k(x) + 1e-12),
    and we return mean_k L_k for k=1..K. A hard boolean mask would have zero
    gradient w.r.t. the boundary and would not push energy across it.

    base_batch/bump_batch are torch float tensors. The impulse may be shared
    (shape (N,)) or per-base (shape (B, N)). Returns a scalar tensor.
    """
    B, N = base_batch.shape
    base = base_batch.to(device)
    if bump_batch.ndim == 1:
        bump = bump_batch.to(device).unsqueeze(0).expand(B, -1)
    else:
        bump = bump_batch.to(device)
    if x0 is None:
        # impulse center: where the (mean) bump energy is maximal
        x0 = int(torch.argmax((bump**2).mean(dim=0)).item())

    d_abs = _signed_periodic_distance_t(N, x0, device).abs()  # (N,)

    pert = base + eps * bump
    s = torch.cat([base, pert], dim=0)  # (2B, N)
    total = base.new_zeros(())
    for k in range(1, K + 1):
        s = model(s)
        base_k, pert_k = s[:B], s[B:]
        R_k = ((pert_k - base_k) / eps) ** 2  # (B, N), graph intact
        # soft cone mask for this step (broadcast over batch)
        radius = m * k + r0
        w_k = torch.sigmoid((d_abs - radius) / tau)  # (N,)
        num = (w_k.unsqueeze(0) * R_k).sum(dim=1)  # (B,)
        den = R_k.sum(dim=1) + 1e-12  # (B,)
        total = total + (num / den).mean()
    return total / K


def train_with_causality(
    model: nn.Module,
    inputs: np.ndarray,  # (Ntrain, N)
    targets: np.ndarray,  # (Ntrain, N)
    device: torch.device,
    lam_reg: float,
    base_states: np.ndarray,  # (Bprobe, N) in-distribution states for the cone probe
    bump: np.ndarray,  # (N,) L2-normalized localized impulse
    m: float,  # true speed (cells/step) defining the cone slope
    r0: int,  # bump half-support, dilates the cone
    eps: float = 1e-3,
    K: int = 8,
    tau: float = 1.0,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_inputs: np.ndarray | None = None,
    val_targets: np.ndarray | None = None,
    x0: int | None = None,
    probe_batch: int = 32,
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """One-step MSE + lam_reg * causality_penalty, trained with Adam.

    This is the dose-controlled training loop for the causal intervention. It is
    deliberately written here (NOT in src/train/train_one_step.py, which must
    stay the unregularised baseline) so the only difference between the dose arms
    is this added penalty term, a clean matched-architecture comparison.

    The fit objective stays the same one-step MSE as the baseline; the cone term
    only changes WHERE in space the model is allowed to route perturbation
    energy. With lam_reg=0 this reproduces train_one_step's behaviour (up to the
    extra probe forward pass, which has no gradient effect when lam_reg=0, we
    skip it in that case for speed).

    Returns a history dict with keys "train" (total loss), "mse", "reg", "val".
    """
    torch.manual_seed(seed)
    model = model.to(device)
    X = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    Y = torch.as_tensor(targets, dtype=torch.float32, device=device)
    base_t = torch.as_tensor(base_states, dtype=torch.float32, device=device)
    bump_t = torch.as_tensor(bump, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    mse_fn = nn.MSELoss()
    n = X.shape[0]
    nprobe = base_t.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    history = {"train": [], "mse": [], "reg": [], "val": []}

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(device)
        run_loss = run_mse = run_reg = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            opt.zero_grad()
            pred = model(X[idx])
            mse = mse_fn(pred, Y[idx])
            if lam_reg > 0.0:
                # subsample probe bases each step for cheap, stochastic cone grad
                pj = torch.randint(0, nprobe, (min(probe_batch, nprobe),), generator=g).to(device)
                reg = causality_penalty(
                    model,
                    base_t[pj],
                    bump_t,
                    eps,
                    K,
                    m,
                    r0,
                    device,
                    x0=x0,
                    tau=tau,
                )
                loss = mse + lam_reg * reg
            else:
                reg = mse.new_zeros(())
                loss = mse
            loss.backward()
            opt.step()
            bs = len(idx)
            run_loss += loss.item() * bs
            run_mse += mse.item() * bs
            run_reg += float(reg.item()) * bs
        sched.step()
        history["train"].append(run_loss / n)
        history["mse"].append(run_mse / n)
        history["reg"].append(run_reg / n)
        if val_inputs is not None and (ep % 25 == 0 or ep == epochs - 1):
            model.eval()
            with torch.no_grad():
                Xv = torch.as_tensor(val_inputs, dtype=torch.float32, device=device)
                Yv = torch.as_tensor(val_targets, dtype=torch.float32, device=device)
                v = torch.mean((model(Xv) - Yv) ** 2).item()
            history["val"].append((ep, v))
            if verbose:
                print(
                    f"  ep {ep:4d}  loss {run_loss / n:.3e}  "
                    f"mse {run_mse / n:.3e}  reg {run_reg / n:.3e}  val {v:.3e}"
                )
    return history


def fourier_mode_mask(field: torch.Tensor, keep_modes: int) -> torch.Tensor:
    """Post-hoc high-k masking: keep only the lowest `keep_modes` rFFT modes.

    A blunt, non-trained lever for the intervention. Where the trained causality
    penalty tightens the cone by reshaping the operator, this directly amputates
    the high-wavenumber content of a *field* (e.g. a model's one-step output or
    a rollout state). Because the FNO's superluminal leakage is carried by the
    high-k tail of its spectral convolution, band-limiting the output is the
    crudest possible way to shrink the cone, useful as a control that verifies
    high-k content is the mechanism, not as the headline dose.

    field: (..., N) real torch tensor. Returns a real tensor of the same shape
    with rFFT modes >= keep_modes set to zero.
    """
    N = field.shape[-1]
    ft = torch.fft.rfft(field, dim=-1)  # (..., N//2+1)
    k = min(keep_modes, ft.shape[-1])
    if k < ft.shape[-1]:
        ft = ft.clone()
        ft[..., k:] = 0
    return torch.fft.irfft(ft, n=N, dim=-1)
