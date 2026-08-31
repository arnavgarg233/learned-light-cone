"""Response at a distant, causally independent site before the front arrives.

Two separated bumps A and B sit on a periodic advection domain. A perturbation at
A cannot reach B before k_connect steps under the exact dynamics, so response
energy near B at earlier steps measures spurious long-range coupling.
"""

from __future__ import annotations

import numpy as np
import torch

from .cone import signed_periodic_distance, signed_perturbation_probe


def downstream_distance(N: int, x_a: int, x_b: int, m: int) -> int:
    """Cells a +m (rightward) signal travels from A to B on the periodic ring.

    With the convention exact_step = roll(u, +m) (energy moves to larger index),
    a perturbation at A reaches B after traversing the rightward gap. We return
    that rightward distance in cells (0 < d < N); the causal connect TIME is
    d / m steps.
    """
    return int((x_b - x_a) % N)


def cone_connect_step(N: int, x_a: int, x_b: int, m: int, r0: int) -> int:
    """Earliest step k at which the physical cone from A can touch the B window.

    A +m signal covers m cells/step, so it needs d/m steps to span the rightward
    A->B gap d. The bump half-support r0 (and, downstream, the B-window half-width
    handled by the caller) means the leading edge arrives r0 cells early; we
    subtract that so a rigid shift of a finite bump does NOT count as early
    coupling. Steps k < k_connect are the strictly pre-cone (acausal) window.
    """
    d = downstream_distance(N, x_a, x_b, m)
    if m <= 0:
        return 0
    return max(1, int(np.floor((d - r0) / float(m))))


def two_bump_base(N: int, x_a: int, x_b: int, base_field: np.ndarray | None = None) -> np.ndarray:
    """Base state carrying two localized features at A and B (for context).

    The spurious-coupling readout is a JVP (linear-response) quantity, so the
    *value* of the base at B does not enter the perturbation field; we keep the
    two-bump construction only to make the physical picture (two distinct
    teleconnection-prone features) explicit and to seed a non-trivial base. If
    base_field is given it is used directly (e.g. a band-limited IC).
    """
    if base_field is not None:
        return np.asarray(base_field, dtype=np.float64)
    x = np.arange(N)
    da = (x - x_a + N // 2) % N - N // 2
    db = (x - x_b + N // 2) % N - N // 2
    return (np.exp(-0.5 * (da / 3.0) ** 2) + np.exp(-0.5 * (db / 3.0) ** 2)).astype(np.float64)


@torch.no_grad()
def spurious_coupling(
    step_fn,
    base: torch.Tensor,
    bump_a: torch.Tensor,
    x_a: int,
    x_b: int,
    *,
    m: int,
    r0: int,
    eps: float,
    K: int,
    b_halfwidth: int = 5,
    return_field: bool = False,
):
    """Pre-cone cross-talk: energy at B from an A-only perturbation, k < k_connect.

    Parameters
    ----------
    step_fn : callable (M, N) -> (M, N). A trained model or the exact solver.
    base    : (B, N) base states.
    bump_a  : (N,) localized perturbation applied ONLY at site A.
    x_a, x_b: integer cell centers of features A and B.
    m       : exact advection shift (cells/step); sets the physical cone speed.
    r0      : bump half-support (cells) used to dilate the connect time.
    eps     : perturbation amplitude (finite-difference JVP scale).
    K       : rollout steps.
    b_halfwidth : half-width (cells) of the readout window centered on B. The
                  window is also added to r0 when computing k_connect so the
                  physical front must reach the *near edge* of the window before
                  any response there is counted as causal.

    Returns
    -------
    dict with:
      coupling          : scalar, batch-mean pre-cone B-window energy fraction.
      coupling_per_base : (B,) the same per base state.
      k_connect         : the cone-connection step used as the pre-cone cutoff.
      b_response_curve  : (K+1,) batch-mean B-window response energy per step
                          (normalized by total field energy at that step), for
                          the heatmap / time series.
      response_field    : (K+1, N) batch-mean signed-squared response field
                          (only if return_field=True), for the spatiotemporal
                          heatmap.
    """
    bump_b = bump_a.unsqueeze(0).expand(base.shape[0], -1)
    s = signed_perturbation_probe(step_fn, base, bump_b, eps, K)  # (B, K+1, N)
    R = (s**2).cpu().numpy().astype(np.float64)  # response energy

    N = R.shape[-1]
    d_b = np.abs(signed_periodic_distance(N, x_b))
    window = d_b <= b_halfwidth  # (N,) B readout window

    # connect step uses r0 + window halfwidth: the front must reach the near edge
    k_connect = cone_connect_step(N, x_a, x_b, m, r0 + b_halfwidth)
    k_cut = min(k_connect, K + 1)

    B = R.shape[0]
    # per-step B-window energy fraction (normalized by total response energy)
    total_per_step = R.sum(axis=-1)  # (B, K+1)
    bwin_per_step = R[:, :, window].sum(axis=-1)  # (B, K+1)
    frac_per_step = np.divide(
        bwin_per_step,
        total_per_step,
        out=np.zeros_like(bwin_per_step),
        where=total_per_step > 0,
    )  # (B, K+1)

    # pre-cone coupling: B-window energy fraction summed over steps 1..k_cut-1
    pre = np.arange(1, k_cut)
    if pre.size == 0:
        coupling_per_base = np.zeros(B)
    else:
        coupling_per_base = frac_per_step[:, pre].sum(axis=1)

    out = dict(
        coupling=float(coupling_per_base.mean()),
        coupling_per_base=coupling_per_base,
        k_connect=int(k_connect),
        b_response_curve=frac_per_step.mean(axis=0),
        b_halfwidth=int(b_halfwidth),
        x_a=int(x_a),
        x_b=int(x_b),
        downstream_distance=downstream_distance(N, x_a, x_b, m),
    )
    if return_field:
        out["response_field"] = R.mean(axis=0)  # (K+1, N)
    return out


def spurious_coupling_corrected(model_result: dict, solver_result: dict) -> float:
    """Baseline-corrected pre-cone coupling C_corr = C_model - C_solver (clamped >= 0).

    The exact integer-shift solver has C_solver == 0 by construction, but running
    it through the identical probe makes the correction explicit and robust to any
    finite-difference / windowing residue. Clamped at 0: a model below the causal
    floor is reported as no spurious coupling, not negative.
    """
    return float(max(0.0, model_result["coupling"] - solver_result["coupling"]))
