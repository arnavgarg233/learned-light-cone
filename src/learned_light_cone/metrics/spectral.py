"""Per-mode transfer function of a one-step operator, estimated by linear response.

A small plane wave and its quadrature are added to a base state, one step is
applied, and the response is projected back onto mode k to recover the complex
gain G(k). |G| is the per-mode growth factor and -arg G the phase advance. Exact
propagators for advection and the wave equation are provided for comparison.
"""

from __future__ import annotations

import numpy as np
import torch


def _plane_waves(N: int, ks) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cosine / sine plane-wave probes on the cell grid x_j = j / N.

    Returns (cos, sin, ks) where cos[i], sin[i] are shape (N,) arrays for the
    i-th requested wavenumber. We probe with BOTH quadratures because applying
    the operator mixes the real and imaginary parts of the mode (a phase shift),
    so a single cosine probe cannot resolve arg G on its own.
    """
    ks = np.asarray(list(ks), dtype=int)
    j = np.arange(N)
    phase = 2.0 * np.pi * np.outer(ks, j) / N  # (n_k, N)
    return np.cos(phase), np.sin(phase), ks


@torch.no_grad()
def learned_transfer(step_fn, base, eps: float, ks, device) -> dict:
    """Estimate the complex per-mode gain G_theta(k) of a one-step operator.

    For each wavenumber index k in `ks` we form the perturbed states
        base + eps * cos(2 pi k x / N)      (the "real" quadrature)
        base + eps * sin(2 pi k x / N)      (the "imag" quadrature)
    apply ONE step, take the eps-normalized differences
        dc = (F(base + eps cos_k) - F(base)) / eps   ~  DF[base] cos_k
        ds = (F(base + eps sin_k) - F(base)) / eps   ~  DF[base] sin_k
    and project onto mode k. Writing the response to the complex probe
    e^{i k x} = cos + i sin as (dc + i ds), its k-th Fourier coefficient is the
    learned gain G_theta(k): a clean linear operator returns G * e^{i k x}, whose
    projection onto e^{i k x} is exactly G.

    Parameters
    ----------
    step_fn : callable mapping a batch (M, N) -> (M, N) (model or solver).
    base    : (B, N) torch tensor of base states. The gain is averaged over the
              batch (for a linear shift-invariant map every base gives the same
              G; the spread across bases reports the linearization error).
    eps     : perturbation amplitude (small -> linear-response regime).
    ks      : iterable of integer wavenumber indices to probe (0 <= k <= N//2).
    device  : torch device.

    Returns
    -------
    dict with keys:
      ks        : (n_k,) int array of probed wavenumbers
      G         : (n_k,) complex array, batch-mean learned gain G_theta(k)
      G_per_base: (B, n_k) complex array, gain per base state
      G_std     : (n_k,) float array, std over bases of |G| (linearity diagnostic)
      eps       : the eps used
    """
    base = torch.as_tensor(base, dtype=torch.float32, device=device)
    if base.ndim == 1:
        base = base.unsqueeze(0)
    B, N = base.shape
    cos_np, sin_np, ks = _plane_waves(N, ks)
    n_k = len(ks)

    f0 = step_fn(base)  # (B, N)
    G_per_base = np.zeros((B, n_k), dtype=np.complex128)

    for i in range(n_k):
        cos_k = torch.as_tensor(cos_np[i], dtype=torch.float32, device=device)
        sin_k = torch.as_tensor(sin_np[i], dtype=torch.float32, device=device)
        cos_b = cos_k.unsqueeze(0).expand(B, -1)
        sin_b = sin_k.unsqueeze(0).expand(B, -1)
        dc = (step_fn(base + eps * cos_b) - f0) / eps  # (B, N) ~ DF cos_k
        ds = (step_fn(base + eps * sin_b) - f0) / eps  # (B, N) ~ DF sin_k
        # complex response to e^{i k x} = cos + i sin, projected onto mode k:
        resp = (dc + 1j * ds).cpu().numpy().astype(np.complex128)  # (B, N)
        proj = (resp * np.conj(cos_np[i] + 1j * sin_np[i])).sum(axis=1)  # <resp, e^{ikx}>
        norm = float((cos_np[i] ** 2 + sin_np[i] ** 2).sum())  # = ||e^{ikx}||^2 (real)
        G_per_base[:, i] = proj / (norm + 1e-30)

    G = G_per_base.mean(axis=0)
    G_std = np.abs(G_per_base).std(axis=0)
    return dict(ks=ks, G=G, G_per_base=G_per_base, G_std=G_std, eps=float(eps))


def dispersion_and_growth(G: np.ndarray, dt: float):
    """Per-mode angular frequency and growth rate from a complex gain G(k).

    A one-step gain G = |G| exp(-i omega dt) encodes, over a step of length dt:
        omega_theta(k) = -arg(G(k)) / dt      (phase advance per unit time)
        gamma_theta(k) =  log|G(k)| / dt      (exponential growth rate per time)
    For a non-dissipative, non-dispersive PDE (e.g. advection) gamma == 0 and
    omega is linear in k with slope = the phase speed c. gamma > 0 anywhere is a
    spurious instability that will blow up an autoregressive rollout.

    Returns (omega_theta, gamma_theta), both real arrays the shape of G.
    """
    G = np.asarray(G)
    omega_theta = -np.angle(G) / dt
    gamma_theta = np.log(np.abs(G) + 1e-30) / dt
    return omega_theta, gamma_theta


def spectral_errors(
    G: np.ndarray,
    true_G: np.ndarray,
    dt: float = 1.0,
    ks: np.ndarray | None = None,
    unstable_tol: float = 0.0,
) -> dict:
    """Compare a learned gain G(k) against the analytic propagator true_G(k).

    Diagnostics (all per-mode unless noted):
      phase_speed_err : |omega_theta - omega_true| / k_phys, the error in the
                        per-mode phase speed c(k) = omega(k)/k_phys (the k=0 mode,
                        which has no phase speed, is set to nan and skipped in the
                        max). This is the frequency-domain analogue of the front-
                        speed error c_hat - c.
      growth_rate_err : gamma_theta - gamma_true (signed): positive => the model
                        is too energetic at that mode (anti-dissipative).
      unstable_bands  : list of (k_lo, k_hi) contiguous wavenumber ranges where
                        gamma_theta > unstable_tol while the true gamma <= 0, i.e.
                        SPURIOUS amplifying modes the operator invented. These are
                        the spectral fingerprint of rollout blow-up.

    Parameters
    ----------
    G, true_G   : (n_k,) complex gains, aligned mode-for-mode.
    dt          : timestep (sets the omega/gamma scale; cancels in band detection).
    ks          : (n_k,) int wavenumber indices (defaults to 0..n_k-1). Used to
                  form the physical wavenumber k_phys = 2*pi*k and to report bands.
    unstable_tol: gamma threshold above which a mode counts as unstable (default 0,
                  i.e. any growth; raise slightly to ignore estimator noise).
    """
    G = np.asarray(G)
    true_G = np.asarray(true_G)
    n_k = len(G)
    if ks is None:
        ks = np.arange(n_k)
    ks = np.asarray(ks)
    k_phys = 2.0 * np.pi * ks.astype(np.float64)

    omega_t, gamma_t = dispersion_and_growth(G, dt)
    omega_true, gamma_true = dispersion_and_growth(true_G, dt)

    # unwrap before differencing so a 2*pi phase wrap is not read as a huge error
    omega_diff = np.angle(np.exp(1j * (omega_t - omega_true) * dt)) / dt

    with np.errstate(divide="ignore", invalid="ignore"):
        phase_speed_true = np.where(k_phys > 0, omega_true / k_phys, np.nan)
        phase_speed_theta = np.where(k_phys > 0, omega_t / k_phys, np.nan)
        phase_speed_err = np.where(k_phys > 0, np.abs(omega_diff) / k_phys, np.nan)

    growth_rate_err = gamma_t - gamma_true

    # spurious unstable bands: model grows where the true operator does not
    spurious = (gamma_t > unstable_tol) & (gamma_true <= unstable_tol)
    bands = []
    in_band = False
    lo = None
    for i in range(n_k):
        if spurious[i] and not in_band:
            in_band = True
            lo = int(ks[i])
        elif not spurious[i] and in_band:
            in_band = False
            bands.append((lo, int(ks[i - 1])))
    if in_band:
        bands.append((lo, int(ks[-1])))

    finite = np.isfinite(phase_speed_err)
    max_phase_speed_err = float(np.nanmax(phase_speed_err)) if finite.any() else 0.0

    return dict(
        ks=ks,
        phase_speed_theta=phase_speed_theta,
        phase_speed_true=phase_speed_true,
        phase_speed_err=phase_speed_err,
        max_phase_speed_err=max_phase_speed_err,
        growth_rate_theta=gamma_t,
        growth_rate_true=gamma_true,
        growth_rate_err=growth_rate_err,
        max_growth_rate_err=float(np.max(np.abs(growth_rate_err))),
        unstable_bands=bands,
        n_unstable_modes=int(spurious.sum()),
    )


def advection_true_G(ks, c: float, dt: float, L: float = 1.0) -> np.ndarray:
    """Analytic advection propagator G(k) = exp(-i c k_phys dt), |G| = 1."""
    ks = np.asarray(list(ks), dtype=np.float64)
    k_phys = 2.0 * np.pi * ks / L
    return np.exp(-1j * c * k_phys * dt)


def diffusion_true_G(ks, nu: float, dt: float, L: float = 1.0) -> np.ndarray:
    """Analytic diffusion propagator G(k) = exp(-nu k_phys^2 dt), real & <= 1."""
    ks = np.asarray(list(ks), dtype=np.float64)
    k_phys = 2.0 * np.pi * ks / L
    return np.exp(-nu * k_phys**2 * dt).astype(np.complex128)
