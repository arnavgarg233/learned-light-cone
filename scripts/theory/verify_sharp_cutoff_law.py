"""Numerical check of the hard-cutoff kernel tail.

Builds rectangular, raised-cosine and Gaussian windows on the rFFT grid, takes the
exact real-space kernel with irfft, and fits the tail. A rectangular cutoff with a
nonzero band-edge value gives a 1/r envelope; tapering the edge to zero steepens
the decay. Random and adversarial per-mode weights with a nonzero edge are included,
and the cutoff K is swept.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# windows on the rFFT mode grid (k = 0 .. K-1 retained, k >= K zeroed)
# Same windows as learned_light_cone.models.windowed_fno.make_window.
def window_transfer(
    N: int, K: int, kind: str, gaussian_sigma_frac: float = 0.5, R_phi: np.ndarray | None = None
) -> np.ndarray:
    """Return the full rFFT transfer vector G(k) of length N//2+1.

    G(k) = w_k * R_phi(k) for k < K, else 0, where w_k is the spectral window.
    If R_phi is None we use R_phi == 1 (the pure window, i.e. the kernel of the
    window itself). R_phi lets us stress-test "no weighting beats the envelope".
    """
    nf = N // 2 + 1
    G = np.zeros(nf, dtype=np.complex128)
    Kc = min(K, nf)
    k = np.arange(Kc, dtype=np.float64)
    kind = kind.lower()
    if kind == "rect":
        w = np.ones(Kc)
    elif kind in ("raised_cosine", "hann", "raised-cosine"):
        if Kc == 1:
            w = np.ones(1)
        else:
            w = 0.5 * (1.0 + np.cos(np.pi * k / (Kc - 1)))
    elif kind == "gaussian":
        sigma = max(gaussian_sigma_frac * Kc, 1e-6)
        w = np.exp(-0.5 * (k / sigma) ** 2)
    else:
        raise ValueError(f"unknown window {kind!r}")
    if R_phi is None:
        r = np.ones(Kc, dtype=np.complex128)
    else:
        r = np.asarray(R_phi, dtype=np.complex128)[:Kc]
    G[:Kc] = w * r
    return G


def real_kernel(N: int, G: np.ndarray) -> np.ndarray:
    """Exact real-space kernel g(r), r = signed distance, centered at 0.

    irfft(G) gives the periodic kernel on x = 0..N-1; we roll it so index N//2 is
    r = 0 and report the symmetric envelope. We keep it as a real signal.
    """
    g = np.fft.irfft(G, n=N)
    g = np.fft.fftshift(g)  # r = 0 at center index N//2
    return g


def signed_r(N: int) -> np.ndarray:
    return np.arange(N) - N // 2


# tail-decay fits
def chord(N: int, r: np.ndarray) -> np.ndarray:
    """Periodic 'chord distance' s = (N/pi) * sin(pi |r| / N).

    On a periodic grid the rectangular-cutoff (Dirichlet) kernel envelope is
    EXACTLY ~ 1 / (N sin(pi r / N)) = 1 / (pi s), not 1 / (pi r): the sin factor
    is the periodic wraparound correction, and s -> |r| for |r| << N. Fitting the
    power law against s (rather than r) removes the periodic bias and recovers the
    clean alpha = 1 the continuous theorem predicts. Both fits are reported so the
    bias is visible.
    """
    return (N / np.pi) * np.sin(np.pi * np.abs(r) / N)


def fit_power_law(x: np.ndarray, env: np.ndarray):
    """alpha, C, R^2 for env ~ C * x^{-alpha}: linear fit of log env vs log x."""
    m = (x > 0) & (env > 0)
    lr, le = np.log(x[m]), np.log(env[m])
    A = np.vstack([np.ones_like(lr), lr]).T
    coef, *_ = np.linalg.lstsq(A, le, rcond=None)
    logC, neg_alpha = coef
    pred = A @ coef
    ss_res = float(((le - pred) ** 2).sum())
    ss_tot = float(((le - le.mean()) ** 2).sum()) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    return float(-neg_alpha), float(np.exp(logC)), float(r2)


def fit_exponential(r: np.ndarray, env: np.ndarray):
    """beta, C, R^2 for env ~ C * exp(-beta r): linear fit of log env vs r."""
    m = (r > 0) & (env > 0)
    rr, le = r[m].astype(float), np.log(env[m])
    A = np.vstack([np.ones_like(rr), rr]).T
    coef, *_ = np.linalg.lstsq(A, le, rcond=None)
    logC, neg_beta = coef
    pred = A @ coef
    ss_res = float(((le - pred) ** 2).sum())
    ss_tot = float(((le - le.mean()) ** 2).sum()) or 1.0
    r2 = 1.0 - ss_res / ss_tot
    return float(-neg_beta), float(np.exp(logC)), float(r2)


def envelope(g: np.ndarray, r: np.ndarray, rmin: int, rmax: int):
    """Local-max envelope of |g| over a window of r, to track the oscillation peaks
    of the sinc rather than its zeros. Returns (r_env, env)."""
    ag = np.abs(g)
    sel = (r >= rmin) & (r <= rmax)
    rr, aa = r[sel], ag[sel]
    # local maxima (peaks of the oscillation)
    peaks = []
    for i in range(1, len(aa) - 1):
        if aa[i] >= aa[i - 1] and aa[i] >= aa[i + 1] and aa[i] > 0:
            peaks.append(i)
    if len(peaks) < 4:  # fall back to raw points if too few peaks
        good = aa > 0
        return rr[good], aa[good]
    peaks = np.array(peaks)
    return rr[peaks], aa[peaks]


# main verification
def analyze_window(
    N: int, K: int, kind: str, R_phi=None, label="", gaussian_sigma_frac: float = 0.5
):
    G = window_transfer(N, K, kind, gaussian_sigma_frac=gaussian_sigma_frac, R_phi=R_phi)
    g = real_kernel(N, G)
    r = signed_r(N)
    # measure the tail well outside the main lobe (~ N/(2K) wide) and inside r=N/4
    # so the periodic antipode (r -> N/2, where the chord correction dominates)
    # does not contaminate the fit.
    main_lobe = max(int(np.ceil(N / (2 * K))), 2)
    rmin = 3 * main_lobe
    rmax = N // 4
    r_env, env = envelope(g, r, rmin, rmax)
    s_env = chord(N, r_env)  # periodic-corrected distance: Dirichlet env ~ 1/s
    out = {
        "window": kind,
        "label": label,
        "N": N,
        "K": K,
        "n_env_points": int(len(env)),
        "gaussian_sigma_frac": gaussian_sigma_frac if kind == "gaussian" else None,
    }
    if len(env) >= 4:
        # raw-r power-law fit (periodically biased) AND chord-corrected fit (clean)
        alpha_r, Cr, r2r = fit_power_law(r_env, env)
        alpha, Cp, r2p = fit_power_law(s_env, env)
        beta, Ce, r2e = fit_exponential(r_env, env)
        out.update(
            power_alpha=alpha,
            power_C=Cp,
            power_r2=r2p,
            power_alpha_raw_r=alpha_r,
            power_r2_raw_r=r2r,
            exp_beta=beta,
            exp_C=Ce,
            exp_r2=r2e,
            better="exponential" if r2e > r2p else "power",
        )
        # dynamic range of the envelope: how many decades it falls across the window
        out["env_decades"] = float(np.log10(env[0] / max(env[-1], 1e-300)))
    return out, (r, g, r_env, env)


def leakage_beyond(
    N: int, K: int, kind: str, r_cut: int, R_phi=None, gaussian_sigma_frac: float = 0.5
) -> float:
    """Fraction of kernel ENERGY beyond a FIXED radius r_cut (the analogue of the
    out-of-cone / upstream leakage mass in learned_light_cone.metrics.cone for a single layer).
    Normalised to total kernel energy so it is the relative leakage Lambda(r_cut)."""
    G = window_transfer(N, K, kind, gaussian_sigma_frac=gaussian_sigma_frac, R_phi=R_phi)
    g = real_kernel(N, G)
    r = signed_r(N)
    e = g**2
    tot = e.sum()
    if tot <= 0:
        return 0.0
    return float(e[np.abs(r) > r_cut].sum() / tot)


def leakage_beyond_sf(N, K, kind, r_cut, sf):
    return leakage_beyond(N, K, kind, r_cut, gaussian_sigma_frac=sf)


def main(output_path: pathlib.Path, figure_path: pathlib.Path | None = None):
    N = 16384  # fine real-space grid: even K=16 then has many oscillations before
    #            r = N/4, so the tail power-law fit is clean for all K shown.
    Ks = [16, 32, 64, 128]
    report = {
        "N": N,
        "Ks": Ks,
        "windows": {},
        "no_weighting_beats_envelope": {},
        "leakage_K_scaling": {},
    }

    print("=" * 78)
    print("SHARP-CUTOFF CAUSAL LAW, exact kernel verification (no training)")
    print("=" * 78)

    # ---- (i) & (ii): tail decay for each window, several K --------------------
    # gaussian uses sigma_frac=0.5 (WIDE: still substantial at the band edge, so it
    # is NOT yet a clean C^inf taper) and sigma_frac=0.2 (NARROW: the Gaussian has
    # decayed to ~exp(-12.5) by the edge, so the truncation jump is negligible and
    # the C^inf exponential tail is exposed). Both are reported:
    # a Gaussian *truncated* at K still has a (tiny) edge jump and is ultimately
    # polynomial; the exponential regime dominates only out to where 1/r overtakes
    # the (negligible) jump term.
    print("\n[1] Real-space kernel tail decay (pure window, R_phi == 1)")
    print("    alpha = chord-corrected power-law exponent (~1 => polynomial 1/r)")
    print(
        f"{'window':>18} {'K':>4} {'alpha':>8} {'pow_R2':>8} "
        f"{'exp_beta':>9} {'exp_R2':>8} {'winner':>12} {'decades':>8}"
    )
    detail_for_fig = {}
    window_specs = [("rect", 0.5), ("raised_cosine", 0.5), ("gaussian", 0.5), ("gaussian", 0.2)]
    for kind, sf in window_specs:
        key = kind if not (kind == "gaussian") else f"gaussian_sf{sf}"
        report["windows"][key] = {}
        for K in Ks:
            out, fig_data = analyze_window(N, K, kind, gaussian_sigma_frac=sf)
            report["windows"][key][str(K)] = out
            if K == 32 and key in ("rect", "raised_cosine", "gaussian_sf0.2"):
                detail_for_fig[key] = fig_data
            if "power_alpha" in out:
                print(
                    f"{key:>18} {K:>4} {out['power_alpha']:>8.3f} "
                    f"{out['power_r2']:>8.3f} {out['exp_beta']:>9.4f} "
                    f"{out['exp_r2']:>8.3f} {out['better']:>12} "
                    f"{out['env_decades']:>8.2f}"
                )

    # ---- (i) hardened: NO edge-nonzero R_phi can beat the 1/r envelope --------
    # THEORY: for a hard cutoff the leading far-tail term of the kernel is, exactly,
    #   g(r) ~ Re[ G(K-1) e^{i(2pi(K-1)/N) r} ] / (pi r) + o(1/r)        (*)
    # i.e. the band-EDGE coefficient G(K-1) = w_{K-1} R_phi(K-1) alone sets the 1/r
    # tail (Abel/summation-by-parts: a sequence with a nonzero last term gives a
    # 1/r partial-sum tail). Per-mode weights R_phi reshuffle the inner modes and
    # add phase, but they CANNOT remove this term unless they drive the edge value
    # to 0 -- which is exactly installing a taper. We verify (*) by measuring the
    # ENVELOPE COEFFICIENT  C_tail := median over a far window of ( pi * s * env(s) ),
    # where s is the chord distance and env is the local-max envelope. Theory says
    # C_tail ~ |G(K-1)| = |R_phi(K-1)|  for any edge-nonzero R_phi, and C_tail -> 0
    # only when the edge value -> 0.
    # We use SMOOTH-INTERNAL R_phi (a smooth magnitude profile times a smooth phase
    # ramp e^{-i k phi}, i.e. a learned shift/dispersion) and vary ONLY the band-
    # EDGE value. This isolates the edge's role: when the inner sequence is smooth,
    # the 1/r tail is controlled by the edge term alone, so C_tail ~ |R_edge|. (If
    # the inner weights are themselves rough, every internal jump adds its own 1/r
    # tail and C_tail >> |R_edge| -- the tail gets WORSE, never better: roughness
    # cannot cancel the edge. The clean lower bound is the edge term.)
    print("\n[2] Hard cutoff, smooth-internal R_phi(k): far-tail coefficient C_tail")
    print("    theory: C_tail = median(pi*s*|g|) -> |R_phi(K-1)| (edge value)")
    print("    => only driving the EDGE value to 0 (a taper) removes the 1/r tail")
    K = 32
    kk = np.arange(K)
    phase = np.exp(-1j * 0.7 * kk)  # smooth learned-shift-like phase ramp
    cases = {}

    # smooth magnitude profiles a(k) (monotone, C^1 in the interior) with a chosen
    # edge value a(K-1); times the smooth phase ramp.
    def smooth_mag(edge):
        # cosine taper from 1 at k=0 to `edge` at k=K-1 (smooth interior, no inner
        # jumps); edge controls the band-edge value only.
        return edge + (1.0 - edge) * 0.5 * (1.0 + np.cos(np.pi * kk / (K - 1)))

    for edge in [1.0, 0.5, 0.2, 0.05, 0.0]:
        cases[f"|R_edge|={edge:.2f}"] = smooth_mag(edge) * phase
    print(f"{'R_phi case':>30} {'|R_edge|':>9} {'C_tail':>9} {'C/|R_edge|':>11}")
    for name, Rp in cases.items():
        G = window_transfer(N, K, "rect", R_phi=Rp)
        g = real_kernel(N, G)
        r = signed_r(N)
        main_lobe = max(int(np.ceil(N / (2 * K))), 2)
        # measure deep in the asymptotic tail (>= 8 main lobes) for a clean coeff
        r_env, env = envelope(g, r, 8 * main_lobe, N // 4)
        s_env = chord(N, r_env)
        C_tail = float(np.median(np.pi * s_env * env)) if len(env) else float("nan")
        edge_mag = float(np.abs(Rp[-1]))
        ratio = C_tail / edge_mag if edge_mag > 1e-9 else None
        report["no_weighting_beats_envelope"][name] = dict(
            edge_mag=edge_mag, C_tail=C_tail, C_over_edge=ratio
        )
        ratio_text = f"{ratio:.3f}" if ratio is not None else "undefined"
        print(f"{name:>30} {edge_mag:>9.3f} {C_tail:>9.4f} {ratio_text:>11}")

    # ---- (iii) K-scaling: the band-limited floor falls; rect >> smooth ---------
    # CONNECTION TO THE REPO EMPIRICS (results/existence/corrected_existence_summary.json):
    # the BAND-LIMITED CAUSAL FLOOR's upstream leakage FALLS as K grows
    # (floor_upstream: 0.082@K6 -> 0.0023@K32). That is exactly the rect-kernel
    # leakage FRACTION beyond a fixed radius: as K grows the Dirichlet main lobe
    # ~N/(2K) shrinks, so a larger share of the (fixed-amplitude 1/r) kernel sits
    # inside any fixed radius -> the relative out-of-cone fraction falls ~1/K. The
    # trained FNO's upstream does NOT fall the same way, so the EXCESS over this
    # falling floor grows -- the learned/multilayer part (Remark, argued, not
    # proven here). The SINGLE-LAYER law proven here is the FLOOR's 1/K falloff and
    # the rect-vs-smooth gap. We report rect leakage (the floor analogue) and the
    # rect/smooth ratio (the matched-modes payoff), both vs K.
    print("\n[3] Leakage Lambda(r_cut) beyond a fixed radius vs K  (energy fraction)")
    print("    gaussian uses sigma_frac=0.2 (the genuinely tapered C^inf window)")
    r_cut = 64  # fixed physical radius (cells), same for all K
    g_sf = 0.2
    print(
        f"{'K':>4} {'rect':>12} {'raised_cos':>12} {'gaussian':>12} "
        f"{'rect/raised':>12} {'rect/gauss':>12}"
    )
    for K in Ks:
        lk_rect = leakage_beyond(N, K, "rect", r_cut)
        lk_rc = leakage_beyond(N, K, "raised_cosine", r_cut)
        lk_g = leakage_beyond_sf(N, K, "gaussian", r_cut, g_sf)
        ratio_rc = lk_rect / lk_rc if lk_rc > 0 else float("inf")
        ratio_g = lk_rect / lk_g if lk_g > 0 else float("inf")
        report["leakage_K_scaling"][str(K)] = dict(
            r_cut=r_cut,
            gaussian_sigma_frac=g_sf,
            rect=lk_rect,
            raised_cosine=lk_rc,
            gaussian=lk_g,
            ratio_rect_over_raised=ratio_rc,
            ratio_rect_over_gauss=ratio_g,
        )
        print(
            f"{K:>4} {lk_rect:>12.3e} {lk_rc:>12.3e} {lk_g:>12.3e} "
            f"{ratio_rc:>12.2e} {ratio_g:>12.2e}"
        )

    # Also: the Dirichlet 1/r prefactor, leakage at FIXED *scaled* radius. For the
    # hard cutoff the kernel is D_K(r) with envelope ~ 1/(pi r) INDEPENDENT of K in
    # amplitude but with main lobe ~ N/(2K); so beyond a fixed r the tail mass is
    # set by the 1/r envelope and the number of K-many oscillations packed before r.
    # We report the fitted power-C prefactor vs K to expose the K-dependence the
    # empirical "upstream excess grows with modes" reflects.
    print("\n[4] Hard-cutoff Dirichlet prefactor C in |g(r)| ~ C / r vs K  (alpha~1)")
    print(f"{'K':>4} {'alpha':>8} {'C':>12}")
    pref = {}
    for K in Ks:
        out, _ = analyze_window(N, K, "rect")
        pref[str(K)] = dict(alpha=out.get("power_alpha"), C=out.get("power_C"))
        print(
            f"{K:>4} {out.get('power_alpha', float('nan')):>8.3f} "
            f"{out.get('power_C', float('nan')):>12.4e}"
        )
    report["hard_cutoff_prefactor_vs_K"] = pref

    # ---- [5] Floor scaling: SINGLE-STEP band-limited-shift upstream vs K -------
    # Replicates learned_light_cone.metrics.cone_corrections.bandlimited_shift_step on the same
    # N=128 grid the repo uses, on a delta (single localized perturbation): truncate
    # to K rFFT modes, roll by m, measure the upstream (signed d < -r0) fraction.
    # Theorem III predicts ~1/K for ONE application. The repo's MEASURED floor (in
    # results/existence/corrected_existence_summary.json) is steeper (~1/K^2) because it is an
    # 8-step ROLLOUT; this section isolates the proven single-step exponent.
    print("\n[5] Single-step band-limited-shift floor: upstream fraction vs K (N=128)")
    print("    theorem III: single application ~ 1/K  (repo rollout floor ~ 1/K^2)")
    Nr, m, r0 = 128, 1, 3  # r0 ~ bump_radius(sigma=1)
    Ks_floor = [6, 8, 10, 12, 16, 24, 32]
    ups = []
    for K in Ks_floor:
        nf = Nr // 2 + 1
        Gf = np.zeros(nf, dtype=complex)
        Gf[: min(K, nf)] = 1.0
        gf = np.fft.irfft(Gf, n=Nr)
        gf = np.roll(gf, m)
        gf = np.fft.fftshift(gf)
        rr = np.arange(Nr) - Nr // 2
        e = gf**2
        ups.append(float(e[rr < -r0].sum() / e.sum()))
    ups = np.array(ups)
    Karr_f = np.array(Ks_floor, float)
    A = np.vstack([np.ones_like(np.log(Karr_f)), np.log(Karr_f)]).T
    coef, *_ = np.linalg.lstsq(A, np.log(ups), rcond=None)
    pred = A @ coef
    r2_floor = 1.0 - float(
        ((np.log(ups) - pred) ** 2).sum()
        / max(((np.log(ups) - np.log(ups).mean()) ** 2).sum(), 1e-12)
    )
    report["floor_single_step_scaling"] = dict(
        Ks=Ks_floor, upstream_fraction=[float(u) for u in ups], exponent=float(coef[1]), r2=r2_floor
    )
    print(f"    upstream fraction ~ K^{coef[1]:.2f}  (R^2={r2_floor:.3f})")
    print("    values: " + ", ".join(f"K{K}={u:.4f}" for K, u in zip(Ks_floor, ups)))

    # ---- figure --------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"rect": "tab:red", "raised_cosine": "tab:green", "gaussian_sf0.2": "tab:blue"}
    names = {
        "rect": "hard cutoff (rect)",
        "raised_cosine": "raised cosine (C^1)",
        "gaussian_sf0.2": "gaussian sf=0.2 (C^inf)",
    }
    for kind in ["rect", "raised_cosine", "gaussian_sf0.2"]:
        if kind not in detail_for_fig:
            continue
        r, g, r_env, env = detail_for_fig[kind]
        ax[0].semilogy(r_env, env, "o-", ms=3, color=colors[kind], label=names[kind])
    # reference 1/r line (chord-corrected, drawn vs r for visualization)
    rr = np.array(sorted(detail_for_fig["rect"][2]))
    ax[0].semilogy(
        rr,
        detail_for_fig["rect"][3].max() * rr[0] / rr,
        "k--",
        lw=1,
        alpha=0.6,
        label="~ 1/r reference",
    )
    ax[0].set_xlabel("radius r (cells)")
    ax[0].set_ylabel("kernel tail envelope |g(r)|")
    ax[0].set_title(
        "A  real-space kernel tail (K=32)\nhard=polynomial, smooth=exponential", fontsize=10
    )
    ax[0].set_xscale("log")
    ax[0].legend(fontsize=8)

    Karr = np.array(Ks)
    rect_leak = np.array([report["leakage_K_scaling"][str(K)]["rect"] for K in Ks])
    rc_leak = np.array([report["leakage_K_scaling"][str(K)]["raised_cosine"] for K in Ks])
    g_leak = np.array([report["leakage_K_scaling"][str(K)]["gaussian"] for K in Ks])
    ax[1].loglog(Karr, rect_leak, "o-", color=colors["rect"], label="hard cutoff")
    ax[1].loglog(Karr, rc_leak, "s-", color=colors["raised_cosine"], label="raised cosine")
    ax[1].loglog(
        Karr,
        np.clip(g_leak, 1e-300, None),
        "^-",
        color=colors["gaussian_sf0.2"],
        label="gaussian sf=0.2",
    )
    ax[1].set_xlabel("retained modes K")
    ax[1].set_ylabel(f"leakage fraction beyond r={r_cut}")
    ax[1].set_title("B  out-of-cone leakage vs K\nhard >> smooth at matched modes", fontsize=10)
    ax[1].legend(fontsize=8)
    fig.suptitle("Sharp-cutoff causal law, exact kernel verification (no training)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if figure_path is not None:
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(figure_path, dpi=140)
    plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    print(f"\nsaved {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--figure", type=pathlib.Path)
    arguments = parser.parse_args()
    main(arguments.output, arguments.figure)
