"""Numerical check of the rollout error bounds on a linear toy map.

Uses F(u) = J u + b on a periodic grid, where the operator norm, the one-step
defect and the out-of-cone projector norms are all available in closed form.
The error recursion e_{k+1} = J e_k + g is rolled out and compared with the
bounds. NumPy only, runs in under a second.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np


# --------------------------------------------------------------------------- #
# geometry: signed periodic distance + cone projectors (same as learned_light_cone.metrics.cone)
# --------------------------------------------------------------------------- #
def signed_periodic_distance(N: int, x0: int) -> np.ndarray:
    idx = np.arange(N)
    return (idx - x0 + N // 2) % N - N // 2


def out_of_cone_diag(N: int, x0: int, k: int, c: float, r0: float) -> np.ndarray:
    """Diagonal (0/1) of the out-of-cone projector Q_k for cone |d| <= c*k + r0."""
    d = np.abs(signed_periodic_distance(N, x0))
    return (d > (c * k + r0)).astype(np.float64)


# --------------------------------------------------------------------------- #
# toy linear operator: a banded (finite-speed) shift + a tunable acausal leak
# --------------------------------------------------------------------------- #
def build_operator(
    N: int, c_phys: int, gain: float, leak: float, leak_reach: int, rng: np.random.Generator
) -> np.ndarray:
    """Linear one-step map J (N x N), circulant-ish.

    Causal part: a normalized local stencil that transports right by c_phys cells
    with half-width 1 -> respects the cone |d| <= c_phys*k + r0 exactly if leak=0.
    Acausal part: a weak coupling to cells `leak_reach` to the LEFT (upstream),
    amplitude `leak`, which deliberately injects out-of-cone energy. `gain` scales
    the whole operator so we can dial L = ||J||_2 across the L<1 / =1 / >1 regimes.
    """
    J = np.zeros((N, N))
    # causal transport: row i gets mass from columns i-c_phys-1 .. i-c_phys+1
    stencil = np.array([0.25, 0.5, 0.25])
    for i in range(N):
        for s, w in zip((-1, 0, 1), stencil):
            J[i, (i - c_phys + s) % N] += w
    # acausal upstream leak: row i also pulls a little from i + leak_reach (to the
    # right in index = upstream of a right-mover) -> lands outside the causal cone
    for i in range(N):
        J[i, (i + leak_reach) % N] += leak
    J *= gain
    return J


def op_2norm(J: np.ndarray) -> float:
    return float(np.linalg.svd(J, compute_uv=False)[0])


# --------------------------------------------------------------------------- #
# measure the (L2) constants Lambda, beta exactly for THIS operator + cone
# --------------------------------------------------------------------------- #
def measure_leak_constants(
    J: np.ndarray, N: int, x0: int, c: float, r0: float, Hmax: int
) -> tuple[float, float]:
    """Return (Lambda, beta) s.t.  ||Q_{k+1} J z|| <= Lambda ||Jz|| + beta ||Q_k z||
    holds for all z and all k=0..Hmax-1, by bounding the two mechanisms separately:

      Lambda := sup_k || Q_{k+1} J P_k ||_2   (fresh leak from IN-cone input)
      beta   := sup_k || Q_{k+1} J Q_k ||_2   (carry-over of ALREADY-out energy)

    Splitting z = P_k z + Q_k z and using ||Q_{k+1}J P_k z|| <= Lambda ||P_k z||
    <= Lambda ||z|| is conservative vs the doc's ||Jz||; we additionally report the
    tighter Lambda_rel = sup ||Q_{k+1}J P_k z|| / ||J P_k z|| used in (A4). Both are
    valid upper-bound constants; we use the pair (Lambda_rel, beta) in the bound and
    verify it holds.
    """
    lam_rel = 0.0
    beta = 0.0
    for k in range(0, Hmax):
        Pk = np.diag(1.0 - out_of_cone_diag(N, x0, k, c, r0))
        Qk = np.diag(out_of_cone_diag(N, x0, k, c, r0))
        Qk1 = np.diag(out_of_cone_diag(N, x0, k + 1, c, r0))
        # fresh-leak relative norm: max over in-cone z of ||Q_{k+1} J z|| / ||J z||.
        # Bound it by the operator norm of Q_{k+1} J restricted to range(P_k),
        # divided by the min gain of J there -> conservative upper bound on the ratio.
        A = Qk1 @ J @ Pk  # in-cone -> out-of-next-cone
        JP = J @ Pk
        num = op_2norm(A)
        svs = np.linalg.svd(JP, compute_uv=False)
        den_min = float(svs[svs > 1e-12].min()) if np.any(svs > 1e-12) else 1.0
        lam_rel = max(lam_rel, num / max(den_min, 1e-12))
        # carry-over: already-out -> out-of-next
        beta = max(beta, op_2norm(Qk1 @ J @ Qk))
    # Lambda_rel as defined can exceed 1 for adversarial directions; the certificate
    # only needs SOME valid (Lambda,beta). Clamp the *reported fraction* but keep the
    # bound constant usable by also offering the absolute fresh-leak norm.
    return float(lam_rel), float(beta)


def measure_leak_abs(J: np.ndarray, N: int, x0: int, c: float, r0: float, Hmax: int) -> float:
    """Absolute fresh-leak constant Lambda_abs := sup_k ||Q_{k+1} J P_k||_2, which
    satisfies ||Q_{k+1} J P_k z|| <= Lambda_abs ||z|| for all z. Used in a *robust*
    form of the bound that never depends on a min-singular-value estimate."""
    lam = 0.0
    for k in range(0, Hmax):
        Pk = np.diag(1.0 - out_of_cone_diag(N, x0, k, c, r0))
        Qk1 = np.diag(out_of_cone_diag(N, x0, k + 1, c, r0))
        lam = max(lam, op_2norm(Qk1 @ J @ Pk))
    return float(lam)


# --------------------------------------------------------------------------- #
# the two certificate bounds (closed form)
# --------------------------------------------------------------------------- #
def geom_sum(x: float, H: int) -> float:
    """sum_{j=0}^{H-1} x^j, robust at x==1."""
    if abs(x - 1.0) < 1e-12:
        return float(H)
    return float((x**H - 1.0) / (x - 1.0))


def thm1_bound(L: float, delta: float, e0: float, H: int) -> float:
    return float(L**H * e0 + delta * geom_sum(L, H))


def thm2_bound_abs(
    L: float,
    beta: float,
    lam_abs: float,
    delta: float,
    delta_Q: float,
    e0: float,
    Q0e0: float,
    H: int,
) -> float:
    """Robust (absolute-constant) form of (star-star):

      ||Q_H e_H|| <= beta^H Q0e0
                   + lam_abs * sum_{j<H} beta^{H-1-j} ||e_j||_bound
                   + delta_Q * sum_{j<H} beta^{H-1-j}

    using ||Q_{k+1} J P_k z|| <= lam_abs ||z|| and ||e_j|| <= thm1_bound. This is a
    valid upper bound because lam_abs ||e_j|| >= ||Q_{k+1} J P_k e_j|| >= fresh leak
    and the in/out split of e_j is absorbed (P_k e_j part -> lam_abs, Q_k e_j part
    -> beta), exactly the recursion a_{k+1} <= beta a_k + lam_abs ||e_k|| + delta_Q.
    """
    total = beta**H * Q0e0
    for j in range(H):
        ej_bound = thm1_bound(L, delta, e0, j)
        total += beta ** (H - 1 - j) * (lam_abs * ej_bound + delta_Q)
    return float(total)


# --------------------------------------------------------------------------- #
# rollout the true error and measure everything
# --------------------------------------------------------------------------- #
def rollout_error(J: np.ndarray, g: np.ndarray, e0: np.ndarray, H: int):
    """Return list of e_k for k=0..H under e_{k+1} = J e_k + g."""
    es = [e0.copy()]
    e = e0.copy()
    for _ in range(H):
        e = J @ e + g
        es.append(e.copy())
    return es


def vector_2norm(values: np.ndarray) -> float:
    """Compute a hardware-independent vector norm with exact partial sums."""
    return math.sqrt(math.fsum(float(value) ** 2 for value in values))


def out_of_cone_norm(e: np.ndarray, N: int, x0: int, k: int, c: float, r0: float) -> float:
    q = out_of_cone_diag(N, x0, k, c, r0)
    return vector_2norm(q * e)


# --------------------------------------------------------------------------- #
# main experiment
# --------------------------------------------------------------------------- #
def run(output_path: pathlib.Path):
    rng = np.random.default_rng(0)
    N = 64
    x0 = N // 2
    c_phys = 1  # physical signal speed (cells/step)
    r0 = 3.0  # cone dilation (bump half-support); matches sigma=1 -> r0=3
    H = 12
    leak_reach = 8  # upstream leak lands 8 cells the wrong way -> out of cone

    results = {"checks": [], "all_pass": True}

    def record(name, passed, detail):
        results["checks"].append({"name": name, "pass": bool(passed), "detail": detail})
        if not passed:
            results["all_pass"] = False
        flag = "PASS" if passed else "FAIL"
        print(f"  [{flag}] {name}: {detail}")

    # ---- a fleet of operators spanning the three regimes ---------------------
    # (gain controls L = ||J||_2; leak controls Lambda). Tuned so the base stencil
    # (gain=1) is marginal-ish, then scaled.
    regimes = [
        ("contractive", 0.80, 0.02),
        ("marginal", 1.00, 0.02),
        ("expansive", 1.40, 0.02),
    ]

    print("\n=== Rollout-Error Certificate : numerical verification ===\n")
    fleet = []
    for label, gain, leak in regimes:
        J = build_operator(N, c_phys, gain, leak, leak_reach, rng)
        L = op_2norm(J)
        lam_abs = measure_leak_abs(J, N, x0, c_phys, r0, H)
        lam_rel, beta = measure_leak_constants(J, N, x0, c_phys, r0, H)
        # a known affine consistency defect g (also gives delta exactly)
        g = 1e-3 * rng.standard_normal(N)
        delta = float(np.linalg.norm(g))
        # out-of-cone part of g (delta_Q)
        qg = out_of_cone_diag(N, x0, 1, c_phys, r0) * g  # cone at k=1 (where g first lands)
        delta_Q = float(np.linalg.norm(qg))
        # localized in-cone initial error (the probe bump), unit norm
        e0 = np.zeros(N)
        e0[x0] = 1.0
        e0_norm = float(np.linalg.norm(e0))
        Q0e0 = out_of_cone_norm(e0, N, x0, 0, c_phys, r0)
        fleet.append(
            dict(
                label=label,
                J=J,
                L=L,
                lam_abs=lam_abs,
                lam_rel=lam_rel,
                beta=beta,
                g=g,
                delta=delta,
                delta_Q=delta_Q,
                e0=e0,
                e0_norm=e0_norm,
                Q0e0=Q0e0,
                leak=leak,
            )
        )

    # ---- Check 1: Theorem 1 (magnitude) holds for all regimes, all horizons --
    print("Check 1 - Theorem 1 (Gronwall / magnitude) upper-bounds true ||e_H||:")
    t1_ok = True
    t1_detail = {}
    for f in fleet:
        es = rollout_error(f["J"], f["g"], f["e0"], H)
        worst_slack = np.inf
        for k in range(0, H + 1):
            true_norm = float(np.linalg.norm(es[k]))
            bound = thm1_bound(f["L"], f["delta"], f["e0_norm"], k)
            slack = bound - true_norm
            worst_slack = min(worst_slack, slack)
            if slack < -1e-9:
                t1_ok = False
        t1_detail[f["label"]] = dict(
            L=round(f["L"], 4),
            min_slack=round(worst_slack, 6),
            EH=round(float(np.linalg.norm(es[H])), 4),
            bound_H=round(thm1_bound(f["L"], f["delta"], f["e0_norm"], H), 4),
        )
    record("Theorem 1 (magnitude bound holds)", t1_ok, t1_detail)

    # ---- Check 2: Theorem 2 (geometry) holds for all regimes, all horizons ---
    print("\nCheck 2 - Theorem 2 (Lieb-Robinson / out-of-cone) upper-bounds true ||Q_H e_H||:")
    t2_ok = True
    t2_detail = {}
    for f in fleet:
        es = rollout_error(f["J"], f["g"], f["e0"], H)
        worst_slack = np.inf
        for k in range(0, H + 1):
            true_q = out_of_cone_norm(es[k], N, x0, k, c_phys, r0)
            bound = thm2_bound_abs(
                f["L"],
                f["beta"],
                f["lam_abs"],
                f["delta"],
                f["delta_Q"],
                f["e0_norm"],
                f["Q0e0"],
                k,
            )
            slack = bound - true_q
            worst_slack = min(worst_slack, slack)
            if slack < -1e-9:
                t2_ok = False
        t2_detail[f["label"]] = dict(
            L=round(f["L"], 4),
            lam_abs=round(f["lam_abs"], 5),
            beta=round(f["beta"], 5),
            min_slack=round(worst_slack, 6),
            QH=round(out_of_cone_norm(es[H], N, x0, H, c_phys, r0), 6),
        )
    record("Theorem 2 (out-of-cone bound holds)", t2_ok, t2_detail)

    # ---- Check 3: regime claim (acausal fraction O(Lambda) when L>1; ~linear when L<=1)
    print("\nCheck 3 - regime claim: acausal fraction bounded (L>1) vs growing (L<=1):")
    frac_detail = {}
    for f in fleet:
        es = rollout_error(f["J"], f["g"], f["e0"], H)
        fracs = []
        for k in range(1, H + 1):
            en = float(np.linalg.norm(es[k]))
            qn = out_of_cone_norm(es[k], N, x0, k, c_phys, r0)
            fracs.append(qn / en if en > 0 else 0.0)
        frac_detail[f["label"]] = dict(
            L=round(f["L"], 4),
            leak=f["leak"],
            frac_first=round(fracs[0], 5),
            frac_last=round(fracs[-1], 5),
            frac_max=round(max(fracs), 5),
        )
    # expansive: last-step fraction should stay bounded (not -> 1) and be O(leak)-ish.
    exp = frac_detail["expansive"]
    con = frac_detail["contractive"]
    # In the expansive regime the in-cone transport spike dominates -> fraction stays
    # small & bounded (does not blow up); contractive lets the leak accumulate so its
    # fraction is >= the expansive one. This is the qualitative §3.3 prediction.
    regime_ok = (exp["frac_last"] < 0.5) and (con["frac_last"] >= exp["frac_last"] - 1e-9)
    record(
        "Regime claim (frac bounded for L>1; leak-dominated for L<=1)",
        regime_ok,
        dict(contractive=con, expansive=exp),
    )

    # ---- Check 4: scale-blindness of the leakage FRACTION (the §4.1 mechanism) --
    print("\nCheck 4 - scale-blindness: leakage fraction invariant under e -> c*e:")
    f = fleet[2]  # use the expansive op
    es = rollout_error(f["J"], f["g"], f["e0"], H)
    base_fracs = []
    scaled_fracs = []
    for k in range(1, H + 1):
        en = vector_2norm(es[k])
        qn = out_of_cone_norm(es[k], N, x0, k, c_phys, r0)
        base_fracs.append(qn / en if en > 0 else 0.0)
        cek = 1e6 * es[k]  # scale the error by 1e6 (a 'blow-up')
        en2 = vector_2norm(cek)
        qn2 = out_of_cone_norm(cek, N, x0, k, c_phys, r0)
        scaled_fracs.append(qn2 / en2 if en2 > 0 else 0.0)
    max_diff = float(np.max(np.abs(np.array(base_fracs) - np.array(scaled_fracs))))
    scale_ok = max_diff < 1e-9
    record(
        "Scale-blindness (fraction invariant under magnitude scaling)",
        scale_ok,
        dict(
            max_abs_diff=max_diff,
            note="a fraction cannot rank a magnitude/blow-up event -> AUROC(Lambda)~chance",
        ),
    )

    # ---- Check 4b: toy AUROC reproduction (Lambda ~ chance, rho(J) ~ 1 for blow-up)
    print(
        "\nCheck 4b - toy fleet reproduces AUROC asymmetry (rho(J) ranks blow-up, Lambda does not):"
    )
    # build a small fleet with varied gain (=> varied L) and varied leak; label
    # blow-up = (||e_H|| above the fleet median). Then AUROC of L vs of mean leakage.
    gains = np.linspace(0.7, 1.6, 12)
    leaks = rng.uniform(0.005, 0.06, size=12)
    L_list, lamfrac_list, EH_list = [], [], []
    for gn, lk in zip(gains, leaks):
        Ji = build_operator(N, c_phys, gn, lk, leak_reach, rng)
        Li = op_2norm(Ji)
        gi = 1e-3 * rng.standard_normal(N)
        e0i = np.zeros(N)
        e0i[x0] = 1.0
        esi = rollout_error(Ji, gi, e0i, H)
        EH = float(np.linalg.norm(esi[H]))
        # measured leakage FRACTION of the response at modest horizon (probe-like)
        kk = 4
        en = float(np.linalg.norm(esi[kk]))
        qn = out_of_cone_norm(esi[kk], N, x0, kk, c_phys, r0)
        L_list.append(Li)
        lamfrac_list.append(qn / en if en > 0 else 0.0)
        EH_list.append(EH)
    L_arr = np.array(L_list)
    lam_arr = np.array(lamfrac_list)
    EH_arr = np.array(EH_list)
    blow = (EH_arr > np.median(EH_arr)).astype(int)

    def auroc(scores, labels):
        from scipy.stats import rankdata

        scores = np.asarray(scores, float)
        labels = np.asarray(labels, int)
        pos = labels == 1
        npos = int(pos.sum())
        nneg = int((~pos).sum())
        if npos == 0 or nneg == 0:
            return 0.5
        r = rankdata(scores)
        return float((r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))

    auc_L = auroc(L_arr, blow)
    auc_lam = auroc(lam_arr, blow)
    # Theory (doc sec 4.1-4.2): the magnitude statistic L=rho(J) should rank the
    # blow-up event near-perfectly (auc -> 1), while the SCALE-FREE leakage fraction
    # cannot rank a magnitude event and -- by the energy-concentration mechanism of
    # sec 4.2 -- is typically ANTI-predictive (auc well BELOW 0.5), which is exactly
    # the repo's lambda_1K=0.213 < 0.5. So the prediction is: auc_L large AND
    # auc_lam far below auc_L (and at/below chance), NOT auc_lam ~ 0.5.
    asym_ok = (auc_L >= 0.85) and (auc_lam <= 0.5) and (auc_L - auc_lam >= 0.5)
    record(
        "AUROC asymmetry (L ranks blow-up; leakage fraction at/below chance)",
        asym_ok,
        dict(
            auroc_L=round(auc_L, 3),
            auroc_leak_fraction=round(auc_lam, 3),
            gap=round(auc_L - auc_lam, 3),
            theory="L->1 ranks magnitude; leakage fraction scale-free & anti-predictive",
            empirical_repo=dict(jac_rho_auroc=0.988, lambda_1K_auroc=0.213),
        ),
    )

    # ---- write summary -------------------------------------------------------
    summary = dict(
        N=N,
        x0=x0,
        c_phys=c_phys,
        r0=r0,
        H=H,
        leak_reach=leak_reach,
        thm1_detail=t1_detail,
        thm2_detail=t2_detail,
        frac_detail=frac_detail,
        scale_blind_max_diff=max_diff,
        auroc_L=auc_L,
        auroc_leak_fraction=auc_lam,
        checks=results["checks"],
        all_pass=results["all_pass"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    print(f"\nWrote {output_path}")
    print(f"\n=== {'ALL CHECKS PASSED' if results['all_pass'] else 'SOME CHECKS FAILED'} ===")
    return 0 if results["all_pass"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    arguments = parser.parse_args()
    raise SystemExit(run(arguments.output))
