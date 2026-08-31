"""Superposition defect: how far a one-step operator is from additive.

Measures || F(a + b) - F(a) - F(b) + F(0) || relative to ||F(a)|| + ||F(b)|| for
sampled pairs, optionally by wavenumber band.
"""

from __future__ import annotations

import numpy as np
import torch

from learned_light_cone.pdes.advection import band_limited_ic


def _to_batch(x, device) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=torch.float32, device=device)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t


@torch.no_grad()
def superposition_defect(F, a, b, device=None, reduce: bool = True):
    """Bias-corrected superposition defect of a one-step field map F.

    D_sup = || F(a+b) - F(a) - F(b) + F(0) || / ( ||F(a)|| + ||F(b)|| + eps ),
    with the L2 norm taken over the spatial axis. The +F(0) term cancels any
    affine offset so the defect isolates genuine nonlinearity. Vectorized over a
    batch: a, b are (B, N) (or (N,)); F maps (M, N) -> (M, N).

    Parameters
    ----------
    F      : callable, one step of the operator on a batch of fields.
    a, b   : (B, N) torch/numpy arrays of the two input fields to superpose.
    device : torch device (inferred from a if None).
    reduce : if True return the batch-mean defect (float); else the (B,)
             per-sample defects.
    """
    if device is None:
        device = a.device if torch.is_tensor(a) else torch.device("cpu")
    a = _to_batch(a, device)
    b = _to_batch(b, device)
    B, N = a.shape
    zero = torch.zeros((1, N), dtype=torch.float32, device=device)

    Fab = F(a + b)
    Fa = F(a)
    Fb = F(b)
    F0 = F(zero).expand(B, -1)  # bias term, broadcast over the batch

    eps = 1e-12
    defect = torch.linalg.vector_norm(Fab - Fa - Fb + F0, dim=-1)
    scale = torch.linalg.vector_norm(Fa, dim=-1) + torch.linalg.vector_norm(Fb, dim=-1) + eps
    d = (defect / scale).cpu().numpy()
    return float(d.mean()) if reduce else d


def _band_pass(u: np.ndarray, k_lo: int, k_hi: int) -> np.ndarray:
    """Keep only Fourier modes k_lo <= |k| <= k_hi (inclusive) of each row."""
    uh = np.fft.rfft(u, axis=-1)
    n_modes = uh.shape[-1]
    mask = np.zeros(n_modes, dtype=bool)
    k_hi = min(k_hi, n_modes - 1)
    mask[k_lo : k_hi + 1] = True
    uh = uh * mask
    return np.fft.irfft(uh, n=u.shape[-1], axis=-1)


def sample_superposition_pairs(
    rng, N: int, n_pairs: int, J: int = 12, p: float = 1.0, normalize: bool = True
):
    """Sample two independent band-limited field batches (a, b) for the defect.

    Each of a and b is a fresh band-limited draw (same distribution as the
    training ICs), so D_sup(a, b) measures nonlinearity on in-distribution
    inputs. Returns (a, b), each (n_pairs, N) numpy arrays.
    """
    a = band_limited_ic(rng, N, n_pairs, J=J, p=p, normalize=normalize)
    b = band_limited_ic(rng, N, n_pairs, J=J, p=p, normalize=normalize)
    return a, b


def superposition_defect_by_band(
    F,
    rng,
    N: int,
    n_pairs: int,
    in_band: tuple[int, int] = (1, 8),
    out_band: tuple[int, int] | None = None,
    amp: float = 1.0,
    device=None,
) -> dict:
    """Superposition defect split by spectral regime (in-band vs out-of-band).

    A band-limited operator is expected to superpose cleanly for inputs whose
    energy lives inside its trained band, but to mix modes for high-wavenumber
    (out-of-band) inputs -- the regime that breaks autoregressive rollout. We
    build white-ish random fields, band-pass them into the requested wavenumber
    windows, scale each to RMS = `amp`, and report D_sup per regime.

    Parameters
    ----------
    F        : one-step field map (M, N) -> (M, N).
    rng      : np.random.Generator.
    N        : grid size.
    n_pairs  : number of (a, b) pairs per regime.
    in_band  : (k_lo, k_hi) wavenumber window for the in-distribution regime.
    out_band : (k_lo, k_hi) for the OOD regime; default = modes above in_band up
               to Nyquist, i.e. (in_band[1]+1, N//2).
    amp      : target per-field RMS amplitude (matched across regimes so the
               comparison is amplitude-fair).
    device   : torch device.

    Returns
    -------
    dict mapping regime name -> {"d_mean": float, "d": (n_pairs,) array,
    "band": (k_lo, k_hi)}.
    """
    if out_band is None:
        out_band = (in_band[1] + 1, N // 2)

    def _make(band):
        raw_a = rng.standard_normal((n_pairs, N))
        raw_b = rng.standard_normal((n_pairs, N))
        a = _band_pass(raw_a, band[0], band[1])
        b = _band_pass(raw_b, band[0], band[1])
        a = _rms_scale(a, amp)
        b = _rms_scale(b, amp)
        return a, b

    out = {}
    for name, band in [("in_band", in_band), ("out_band", out_band)]:
        a, b = _make(band)
        d = superposition_defect(F, a, b, device=device, reduce=False)
        out[name] = dict(d_mean=float(d.mean()), d=d, band=band)
    return out


def _rms_scale(u: np.ndarray, amp: float) -> np.ndarray:
    """Scale each row to root-mean-square amplitude `amp` (zero rows untouched)."""
    rms = np.sqrt((u**2).mean(axis=-1, keepdims=True))
    rms = np.where(rms > 0, rms, 1.0)
    return u * (amp / rms)
