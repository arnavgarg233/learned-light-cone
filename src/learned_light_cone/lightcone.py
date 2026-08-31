"""Perturbation-response cone diagnostics for a one-step operator.

Works with any callable that maps a NumPy state to the next state, in 1D, 2D or
on the sphere (pass the distance array). Reports the out-of-cone energy fraction,
the same fraction minus a causal reference, thresholded reach, antipode response,
and the tail exponent.

    import numpy as np
    from learned_light_cone import cone_report, localized_bump

    N = 128
    base = np.random.randn(8, N).astype("float32")
    dist = np.abs((np.arange(N) - N // 2 + N // 2) % N - N // 2)
    bump = localized_bump(dist, sigma=1.0)
    rep = cone_report(forward, base, bump, dist, cone_radius=m + 3,
                      baseline_forward=exact_forward)
    print(rep["leakage_corrected"], rep["reach_1e-3"], rep["tail"]["is_algebraic"])

`examples()` runs a NumPy-only demo with no model.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "localized_bump",
    "perturbation_response",
    "cone_leakage",
    "tail_exponent",
    "cone_report",
    "torch_forward",
    "examples",
]


# --------------------------------------------------------------------------- #
def localized_bump(distance: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """A localized, zero-mean Gaussian perturbation centred at distance==0.
    `distance` is the per-cell distance-to-source array (any shape/dimension)."""
    b = np.exp(-0.5 * (np.asarray(distance, float) / sigma) ** 2)
    return (b - b.mean()).astype(np.float32)


def perturbation_response(
    forward, base: np.ndarray, bump: np.ndarray, eps: float = 0.5
) -> np.ndarray:
    """Squared linear-response field R = ((F(base+eps*bump) - F(base))/eps)^2.

    forward : callable mapping a batched state array -> next-step array (numpy in/out).
    base    : (B, *grid) base states. bump : (*grid,) perturbation (broadcast over B).
    Returns R summed/aggregated nowhere -- shape (B, *grid). A *difference* of forwards,
    so it isolates the receptive field independent of any base-state fill."""
    base = np.asarray(base, np.float32)
    pert = base + eps * bump[None, ...]
    d = (np.asarray(forward(pert), np.float32) - np.asarray(forward(base), np.float32)) / eps
    return d**2


def cone_leakage(R: np.ndarray, distance: np.ndarray, cone_radius: float) -> float:
    """Mean over the batch of the fraction of response energy OUTSIDE the cone
    (distance > cone_radius). 0 for a perfectly causal operator."""
    out = distance > cone_radius
    fr = []
    for b in range(R.shape[0]):
        e = R[b].astype(np.float64)
        tot = e.sum() + 1e-30
        fr.append(float(e[out].sum() / tot))
    return float(np.mean(fr))


def _reach(R: np.ndarray, distance: np.ndarray, thr_frac: float) -> float:
    """Max distance at which mean response exceeds thr_frac of its peak."""
    e = R.mean(axis=0).astype(np.float64)
    peak = e.max()
    if peak <= 0:
        return 0.0
    mask = e >= thr_frac * peak
    return float(distance[mask].max()) if np.any(mask) else 0.0


def tail_exponent(
    R: np.ndarray, distance: np.ndarray, dmin: float, dmax: float, nbins: int = 24
) -> dict:
    """Fit the response tail vs distance in log space: power-law (algebraic,
    Theorem-T1 hard-cutoff signature) vs exponential (smooth-taper). Returns the
    power-law exponent alpha, both R^2's, and is_algebraic (power-law wins)."""
    e = R.mean(axis=0).astype(np.float64).ravel()
    d = np.asarray(distance, float).ravel()
    edges = np.geomspace(max(dmin, 1e-6), dmax, nbins + 1)
    cen, med = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (d >= a) & (d < b) & (e > 0)
        if m.any():
            cen.append(np.sqrt(a * b))
            med.append(np.median(e[m]))
    cen, med = np.array(cen), np.array(med)
    if len(cen) < 4:
        return dict(
            alpha=float("nan"), r2_power=float("nan"), r2_exp=float("nan"), is_algebraic=False
        )
    ly = np.log(med / med.max())
    Ap = np.polyfit(np.log(cen), ly, 1)
    alpha = -Ap[0]
    r2p = 1 - np.sum((ly - np.polyval(Ap, np.log(cen))) ** 2) / (
        np.sum((ly - ly.mean()) ** 2) + 1e-30
    )
    Ae = np.polyfit(cen, ly, 1)
    r2e = 1 - np.sum((ly - np.polyval(Ae, cen)) ** 2) / (np.sum((ly - ly.mean()) ** 2) + 1e-30)
    return dict(
        alpha=float(alpha), r2_power=float(r2p), r2_exp=float(r2e), is_algebraic=bool(r2p >= r2e)
    )


def cone_report(
    forward,
    base,
    bump,
    distance,
    cone_radius,
    baseline_forward=None,
    eps: float = 0.5,
    antipode: float | None = None,
) -> dict:
    """Cone report for a one-step operator. See module docstring.

    baseline_forward : if given (e.g. the exact solver or a band-limited shift run
        through the IDENTICAL probe), leakage_corrected = model - baseline is the
        *excess* over a same-band-limited causal floor.
    antipode : the farthest distance on the domain; if given, reaches_antipode is set.
    """
    distance = np.asarray(distance, float)
    R = perturbation_response(forward, base, bump, eps)
    leak = cone_leakage(R, distance, cone_radius)
    rep = dict(
        leakage=leak,
        reach_1e_3=_reach(R, distance, 1e-3),
        reach_1e_2=_reach(R, distance, 1e-2),
        cone_radius=float(cone_radius),
    )
    if baseline_forward is not None:
        Rs = perturbation_response(baseline_forward, base, bump, eps)
        rep["baseline_leakage"] = cone_leakage(Rs, distance, cone_radius)
        rep["leakage_corrected"] = leak - rep["baseline_leakage"]
    if antipode is not None:
        rep["antipode"] = float(antipode)
        rep["reaches_antipode_1e_3"] = bool(rep["reach_1e_3"] >= 0.999 * antipode)
    rep["tail"] = tail_exponent(R, distance, dmin=cone_radius, dmax=float(distance.max()))
    rep["verdict"] = (
        "ACAUSAL" if rep.get("leakage_corrected", leak) > 0.02 else "causal (within floor)"
    )
    return rep


# --------------------------------------------------------------------------- #
def torch_forward(model, device="cpu", pre=None, post=None):
    """Wrap a torch model into a numpy forward(x)->x' for the diagnostic.
    pre/post: optional (de)normalization callables applied around the model."""
    import torch

    @torch.no_grad()
    def fwd(x):
        t = torch.as_tensor(np.asarray(x, np.float32), device=device)
        if pre is not None:
            t = pre(t)
        y = model(t)
        if post is not None:
            y = post(y)
        return y.detach().cpu().numpy()

    return fwd


# --------------------------------------------------------------------------- #
def examples():
    """Self-contained numpy demo: a causal roll (zero excess) vs an acausal global
    blur (positive excess, algebraic tail) -- no model/torch needed."""
    N, m = 256, 3
    rng = np.random.default_rng(0)
    base = rng.standard_normal((8, N)).astype(np.float32)
    dist = np.abs((np.arange(N) - N // 2 + N // 2) % N - N // 2).astype(float)
    bump = localized_bump(dist, sigma=1.0)
    cone = m + 3  # physical reach + bump support

    def causal_roll(x):  # exact shift by m -> causal
        return np.roll(x, m, axis=-1)

    def acausal_blur(x):  # global low-pass (FFT truncation) -> leaks
        xf = np.fft.rfft(np.roll(x, m, axis=-1), axis=-1)
        xf[:, 16:] = 0.0  # hard spectral cutoff -> 1/r tail (T1)
        return np.fft.irfft(xf, n=N, axis=-1)

    for name, f in [("causal_roll", causal_roll), ("acausal_blur", acausal_blur)]:
        rep = cone_report(
            f, base, bump, dist, cone_radius=cone, baseline_forward=causal_roll, antipode=N // 2
        )
        print(
            f"{name:14s} leak={rep['leakage']:.4f} corrected={rep['leakage_corrected']:+.4f} "
            f"reach1e-3={rep['reach_1e_3']:.0f} tail_alpha={rep['tail']['alpha']:.2f} "
            f"algebraic={rep['tail']['is_algebraic']} -> {rep['verdict']}"
        )


if __name__ == "__main__":
    examples()
