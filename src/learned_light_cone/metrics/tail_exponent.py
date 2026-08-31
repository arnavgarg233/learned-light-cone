"""Tail exponent of the real-space response kernel.

Fits a power law and an exponential to the kernel envelope and reports which
fits better. A rectangular spectral cutoff gives an exponent near one; a smooth
taper over the same modes decays faster.
"""

from __future__ import annotations

import numpy as np
import torch

from .cone import signed_periodic_distance, signed_perturbation_probe


def _ols_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """OLS fit y = slope*x + intercept; return (slope, intercept, r2)."""
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    pred = A @ coef
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    return slope, intercept, r2


def kernel_profile(
    step_fn, base: torch.Tensor, bump: torch.Tensor, eps: float, x0: int, k: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Mean |response| vs absolute distance d from the bump center, at step k.

    Returns (d_abs, mag) sorted by increasing distance, where mag[i] is the
    base-averaged |G(d_i)| of the one-step (k=1) signed response. Both the
    bump-side and the opposite side are folded into |d| (the tail is what matters).
    """
    if base.ndim == 1:
        base = base.unsqueeze(0)
    B, N = base.shape
    bump_b = bump.unsqueeze(0).expand(B, -1) if bump.ndim == 1 else bump
    diff = signed_perturbation_probe(step_fn, base, bump_b, eps, max(k, 1))  # (B,K+1,N)
    resp = diff[:, k, :].abs().mean(dim=0).cpu().numpy()  # (N,)
    d_signed = signed_periodic_distance(N, x0)
    d_abs = np.abs(d_signed).astype(np.float64)
    order = np.argsort(d_abs, kind="stable")
    return d_abs[order], resp[order]


def _envelope(d: np.ndarray, g: np.ndarray, bin_w: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Upper envelope of an oscillatory kernel: peak-hold of |G| over distance bins.

    A sinc kernel oscillates and crosses zero, so the RAW |G(d)| is a 1/d envelope
    punctured by deep nulls -- fitting the raw values mixes the true decay rate with
    those nulls and corrupts BOTH the polynomial and exponential fits. The physically
    meaningful decay (the object the theory is about) is the kernel's ENVELOPE. We
    extract it as the max of |G| within each width-`bin_w` distance bin: this is
    monotone-ish, removes the oscillation, and leaves a clean 1/d (rect) vs
    exp(-d/xi) (taper) shape to fit. Two distinct distances can share |d| on a ring,
    so the per-bin max also collapses those to one envelope point.
    """
    if len(d) == 0:
        return d, g
    dmax = int(np.ceil(d.max()))
    edges = np.arange(0, dmax + bin_w + 1, bin_w)
    centers, peaks = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (d >= lo) & (d < hi)
        if sel.any():
            centers.append(0.5 * (lo + hi))
            peaks.append(float(g[sel].max()))
    return np.asarray(centers, dtype=np.float64), np.asarray(peaks, dtype=np.float64)


def tail_exponent(
    step_fn,
    base: torch.Tensor,
    bump: torch.Tensor,
    eps: float,
    x0: int,
    r0: int,
    k: int = 1,
    mag_floor_frac: float = 1e-6,
    max_dist_frac: float = 0.5,
) -> dict:
    """Fit the real-space response-kernel tail: power law vs exponential.

    Parameters
    ----------
    step_fn : (M, N) -> (M, N) one-step map (model or solver).
    base    : (B, N) base states; the kernel is averaged over them.
    bump    : (N,) localized probe (use make_local_bump / a delta).
    eps     : perturbation amplitude (linear-response regime).
    x0      : bump center cell.
    r0      : bump half-support radius; the tail fit starts beyond 2*r0+1 so the
              bump's own support is excluded.
    k       : response step to analyze (1 = the operator's own kernel).
    mag_floor_frac : drop tail points below this fraction of the peak |G| (numerical
              floor; avoids fitting log of estimator noise).
    max_dist_frac  : fit out to this fraction of N (default to the half domain N/2).

    Returns dict:
      alpha          : polynomial-tail exponent (|G| ~ d^-alpha); >0 means decaying.
      poly_r2        : adjusted-R^2 of the log-log (power-law) fit.
      exp_r2         : adjusted-R^2 of the log-linear (exponential) fit.
      xi             : exponential length scale (cells); decay |G| ~ exp(-d/xi).
                       SMALLER xi (and LARGER alpha) = a more suppressed tail. This
                       is the cleanest empirical contrast on a finite periodic ring:
                       the rect sinc has alpha ~ 1 (the theoretical 1/d tail) and a
                       large xi; a smooth taper steepens alpha and shrinks xi.
      is_exponential : True iff the exponential fit beats the power law by more than
                       0.02 adjusted-R^2 AND xi is finite/positive. Note:
                       on a finite ring of N cells with K modes the envelope is fit
                       comparably well by both forms over the resolvable range, so
                       this strict flag is conservative and often False even for
                       tapers; the robust, reportable signal of the taper cure is
                       the steeper `alpha` / smaller `xi`, not this binary.
      n_tail         : number of envelope tail points used.
      d_tail, g_tail : the fitted envelope tail arrays (for plotting / inspection).
    """
    d_abs, mag = kernel_profile(step_fn, base, bump, eps, x0, k=k)
    N = base.shape[-1] if base.ndim > 1 else len(base)

    peak = float(mag.max()) if mag.size else 0.0
    # fit the kernel ENVELOPE (peak-hold), not the raw oscillating |G|: a sinc tail
    # is a 1/d envelope punctured by nulls, and fitting the raw nulls corrupts both
    # the power-law and exponential fits. The theory is about the envelope decay.
    d_env, g_env = _envelope(d_abs, mag, bin_w=2)
    d_start = 2 * r0 + 1
    d_stop = max_dist_frac * N
    sel = (d_env >= d_start) & (d_env <= d_stop) & (g_env > mag_floor_frac * (peak + 1e-30))
    d_tail = d_env[sel]
    g_tail = g_env[sel]

    out = dict(
        alpha=float("nan"),
        poly_r2=float("nan"),
        exp_r2=float("nan"),
        xi=float("nan"),
        is_exponential=False,
        n_tail=int(sel.sum()),
        d_tail=d_tail.tolist(),
        g_tail=g_tail.tolist(),
        peak=peak,
    )
    if sel.sum() < 3:
        return out

    log_g = np.log(g_tail)
    # power law: log|G| = a - alpha * log d  -> slope on log d is -alpha
    slope_p, _, r2_p = _ols_r2(np.log(d_tail), log_g)
    alpha = -slope_p
    # exponential: log|G| = b - d / xi  -> slope on d is -1/xi
    slope_e, _, r2_e = _ols_r2(d_tail, log_g)
    xi = (-1.0 / slope_e) if slope_e < 0 else float("inf")

    is_exp = bool((r2_e - r2_p) > 0.02 and np.isfinite(xi) and xi > 0)
    out.update(
        alpha=float(alpha),
        poly_r2=float(r2_p),
        exp_r2=float(r2_e),
        xi=float(xi),
        is_exponential=is_exp,
    )
    return out
