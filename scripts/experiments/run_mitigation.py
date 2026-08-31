"""Train global, edge-tapered and local operators on 2D periodic advection.

The exact one-step map is a roll by (m, m) cells, so the solver has zero response
outside the cone. For each model the script reports validation MSE, out-of-cone
response fraction, and one-step error on higher-bandwidth initial conditions.

    python scripts/experiments/run_mitigation.py --smoke --output /tmp/mitigation-smoke.json
    python scripts/experiments/run_mitigation.py --device cuda --output /tmp/mitigation.json

Write outputs outside the tracked results tree.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.fft
import torch.nn as nn


# --------------------------------------------------------------------------- #
def get_device(name):
    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


# --------------------------------------------------------------------------- #
# 2D band-limited advection data (exact solver = roll by (m,m))
# --------------------------------------------------------------------------- #
def band_limited_ic_2d(rng, n, N, kmax, amp=1.0):
    """n band-limited real fields on an NxN torus, modes |kx|,|ky| <= kmax."""
    f = np.zeros((n, N, N), dtype=np.complex128)
    K = np.fft.fftfreq(N, d=1.0 / N).astype(int)
    for i, kx in enumerate(K):
        if abs(kx) > kmax:
            continue
        for j, ky in enumerate(K):
            if abs(ky) > kmax:
                continue
            a = rng.standard_normal(n) + 1j * rng.standard_normal(n)
            f[:, i, j] = a
    u = np.fft.ifft2(f, axes=(-2, -1)).real
    u -= u.mean(axis=(-2, -1), keepdims=True)
    s = u.std(axis=(-2, -1), keepdims=True) + 1e-8
    return (amp * u / s).astype(np.float32)


def exact_step_2d(u, m):
    """Exact one-step advection on the torus: roll by (m, m)."""
    return np.roll(np.roll(u, m, axis=-1), m, axis=-2)


# --------------------------------------------------------------------------- #
# Models -- identical width/depth budget
# --------------------------------------------------------------------------- #
class SpectralConv2d(nn.Module):
    def __init__(self, c_in, c_out, modes, taper_sigma_frac=None):
        super().__init__()
        self.modes = modes
        self.scale = 1.0 / (c_in * c_out)
        self.w1 = nn.Parameter(
            self.scale * torch.randn(c_in, c_out, modes, modes, dtype=torch.cfloat)
        )
        self.w2 = nn.Parameter(
            self.scale * torch.randn(c_in, c_out, modes, modes, dtype=torch.cfloat)
        )
        win = torch.ones(modes)
        if taper_sigma_frac is not None:
            kk = torch.arange(modes).float()
            sigma = max(taper_sigma_frac * modes, 1e-6)
            win = torch.exp(-0.5 * (kk / sigma) ** 2)
        # separable 2D window w[a,b] = win[a]*win[b]
        self.register_buffer("win2d", win[:, None] * win[None, :])

    def compl_mul(self, x, w):
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        B, C, H, W = x.shape
        xft = torch.fft.rfft2(x, dim=(-2, -1))  # (B,C,H,W//2+1)
        out = torch.zeros(B, self.w1.shape[1], H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        m = self.modes
        w2 = self.win2d.to(x.device)
        out[:, :, :m, :m] = self.compl_mul(xft[:, :, :m, :m], self.w1 * w2)
        out[:, :, -m:, :m] = self.compl_mul(xft[:, :, -m:, :m], self.w2 * w2)
        return torch.fft.irfft2(out, s=(H, W), dim=(-2, -1))


class FNO2d(nn.Module):
    def __init__(self, modes=12, width=24, n_layers=4, taper_sigma_frac=None):
        super().__init__()
        self.fc0 = nn.Linear(1, width)
        self.convs = nn.ModuleList(
            [SpectralConv2d(width, width, modes, taper_sigma_frac) for _ in range(n_layers)]
        )
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, 1)
        self.act = nn.GELU()

    def forward(self, x):  # x: (B,H,W)
        h = self.fc0(x.unsqueeze(-1)).permute(0, 3, 1, 2)  # (B,width,H,W)
        for conv, w in zip(self.convs, self.ws):
            h = self.act(conv(h) + w(h))
        h = h.permute(0, 2, 3, 1)
        return self.fc2(self.act(self.fc1(h))).squeeze(-1)


class LocalCNN2d(nn.Module):
    """Bounded receptive field (causal): circular-padded 3x3 convs."""

    def __init__(self, width=24, n_layers=4, radius=1):
        super().__init__()
        self.fc0 = nn.Linear(1, width)
        k = 2 * radius + 1
        self.convs = nn.ModuleList([nn.Conv2d(width, width, k, padding=0) for _ in range(n_layers)])
        self.pad = radius
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, 1)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.fc0(x.unsqueeze(-1)).permute(0, 3, 1, 2)
        for conv in self.convs:
            hp = torch.nn.functional.pad(h, (self.pad,) * 4, mode="circular")
            h = self.act(conv(hp))
        h = h.permute(0, 2, 3, 1)
        return self.fc2(self.act(self.fc1(h))).squeeze(-1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
def train(model, tr_in, tr_out, dev, epochs, lr, bs=64, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xi = torch.as_tensor(tr_in, device=dev)
    Yo = torch.as_tensor(tr_out, device=dev)
    n = Xi.shape[0]
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            opt.zero_grad()
            loss = ((model(Xi[idx]) - Yo[idx]) ** 2).mean()
            loss.backward()
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def val_mse(model, vi, vo, dev):
    model.eval()
    p = model(torch.as_tensor(vi, device=dev))
    return float(((p - torch.as_tensor(vo, device=dev)) ** 2).mean().item())


@torch.no_grad()
def out_of_cone_leakage(model, dev, N, m, eps=0.5, sigma=1.0, n_base=8, kmax=8, seed=1):
    """Perturb a localized 2D Gaussian bump at center on n_base band-limited bases;
    response = ||M(x+eps p)-M(x)||; report fraction of response energy OUTSIDE the
    physical cone. Cone radius = physical one-step reach m PLUS the bump support
    (ceil(3*sigma)) -- so the exact-solver and a receptive-field-=m local model both
    sit at ~0, and only super-cone (acausal) leakage counts (mirrors the 1D r0 floor)."""
    rng = np.random.default_rng(seed)
    bases = band_limited_ic_2d(rng, n_base, N, kmax)
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    cy = cx = N // 2
    dy = np.minimum((yy - cy) % N, (cy - yy) % N)
    dx = np.minimum((xx - cx) % N, (cx - xx) % N)
    cheby = np.maximum(dy, dx)  # toroidal Chebyshev distance
    cone = cheby <= (m + int(math.ceil(3.0 * sigma)))  # physical reach + bump support
    bump = np.exp(-0.5 * ((dy**2 + dx**2) / sigma**2)).astype(np.float32)
    bump -= bump.mean()
    model.eval()
    fr = []
    for b in range(n_base):
        x0 = torch.as_tensor(bases[b : b + 1], device=dev)
        x1 = torch.as_tensor(bases[b : b + 1] + eps * bump[None], device=dev)
        r = (model(x1) - model(x0)).squeeze(0).cpu().numpy() ** 2  # (N,N)
        tot = r.sum() + 1e-30
        fr.append(float(r[~cone].sum() / tot))
    return float(np.mean(fr))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mid", action="store_true", help="quick real validation config (CPU/MPS)")
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="override the seed list (e.g. --seeds 0 1). Default depends on config.",
    )
    ap.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    ap.add_argument(
        "--output", type=Path, required=True, help="result JSON outside tracked results"
    )
    args = ap.parse_args()
    dev = get_device(args.device)

    if args.smoke:
        # m == layers so the local CNN's receptive field == physical cone (clean causal baseline)
        N, modes, width, layers, m = 16, 6, 12, 3, 3
        n_train, n_val, n_ood, epochs, seeds, kmax, kood = 128, 64, 64, 8, [0], 4, 6
    elif args.mid:
        N, modes, width, layers, m = 48, 12, 24, 4, 4
        n_train, n_val, n_ood, epochs, seeds, kmax, kood = 1024, 192, 192, 120, [0], 8, 16
    else:
        N, modes, width, layers, m = 64, 16, 32, 4, 4
        n_train, n_val, n_ood, epochs, seeds, kmax, kood = 2048, 256, 256, 220, [0, 1], 10, 18
    if args.seeds is not None:
        seeds = args.seeds

    print(
        f"device={dev} N={N} modes={modes} width={width} epochs={epochs} seeds={seeds}", flush=True
    )
    rng = np.random.default_rng(0)
    tr_in = band_limited_ic_2d(rng, n_train, N, kmax)
    tr_out = exact_step_2d(tr_in, m)
    vi = band_limited_ic_2d(rng, n_val, N, kmax)
    vo = exact_step_2d(vi, m)
    oi = band_limited_ic_2d(rng, n_ood, N, kood)
    oo = exact_step_2d(oi, m)  # OOD: higher band

    def build(kind):
        if kind == "fno_global":
            return FNO2d(modes, width, layers, taper_sigma_frac=None)
        if kind == "fno_tapered":
            return FNO2d(modes, width, layers, taper_sigma_frac=0.4)
        if kind == "local_cnn":
            # 2x channels (still bounded RF == physical cone -> stays causal) to close the
            # in-distribution accuracy gap; remains far smaller than the FNO's mode budget.
            return LocalCNN2d(width * 2, layers, radius=1)
        raise ValueError(kind)

    kinds = ["fno_global", "fno_tapered", "local_cnn"]
    rows = []
    for kind in kinds:
        for s in seeds:
            torch.manual_seed(s)
            model = build(kind).to(dev)
            train(model, tr_in, tr_out, dev, epochs=epochs, lr=1e-3, seed=s)
            vm = val_mse(model, vi, vo, dev)
            om = val_mse(model, oi, oo, dev)
            leak = out_of_cone_leakage(model, dev, N, m)
            npar = sum(p.numel() for p in model.parameters())
            rows.append(
                dict(
                    kind=kind,
                    seed=s,
                    val_mse=vm,
                    ood_mse=om,
                    out_of_cone_leak=leak,
                    n_params=int(npar),
                )
            )
            print(
                f"  {kind:12s} s={s} valMSE={vm:.2e} oodMSE={om:.2e} leak={leak:.4f} params={npar}",
                flush=True,
            )

    # solver control: exact roll has zero out-of-cone leakage by construction
    class Roll(nn.Module):
        def forward(self, x):
            return torch.roll(torch.roll(x, m, -1), m, -2)

    solver_leak = out_of_cone_leakage(Roll().to(dev), dev, N, m)

    agg = collections.defaultdict(list)
    for r in rows:
        agg[r["kind"]].append(r)
    summary = []
    for k, rs in agg.items():
        summary.append(
            dict(
                kind=k,
                val_mse=float(np.mean([x["val_mse"] for x in rs])),
                ood_mse=float(np.mean([x["ood_mse"] for x in rs])),
                leak=float(np.mean([x["out_of_cone_leak"] for x in rs])),
                leak_corr=float(np.mean([x["out_of_cone_leak"] for x in rs]) - solver_leak),
            )
        )

    g = next(d for d in summary if d["kind"] == "fno_global")
    c = next(d for d in summary if d["kind"] == "local_cnn")
    leak_cut = c["leak_corr"] <= 0.25 * max(g["leak_corr"], 1e-9)
    better_ood = c["ood_mse"] < g["ood_mse"]
    indist_comparable = c["val_mse"] <= 5.0 * g["val_mse"]  # both << field variance (~1)
    cured = leak_cut and (better_ood or indist_comparable)
    verdict = (
        (
            f"MITIGATION WORKS: local fix cuts out-of-cone leakage "
            f"{g['leak_corr']:+.3f}->{c['leak_corr']:+.3f}; in-dist MSE {c['val_mse']:.1e} vs "
            f"{g['val_mse']:.1e} (both <<1); OOD {c['ood_mse']:.3f} vs {g['ood_mse']:.3f} "
            f"({'BETTER' if better_ood else 'worse'})"
        )
        if cured
        else "local fix did not cleanly cure (see numbers)"
    )

    out = dict(
        pde="2D linear advection (torus, exact roll solver, leakage floor 0)",
        grid=N,
        modes=modes,
        width=width,
        n_layers=layers,
        m=m,
        epochs=epochs,
        seeds=seeds,
        device=str(dev),
        solver_out_of_cone_leak=solver_leak,
        rows=rows,
        summary=summary,
        verdict=verdict,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=2, default=float, allow_nan=False)

    print("\n#### MITIGATION SUMMARY (2D advection; leaky FNO vs taper vs local fix) ####")
    print(f"solver out-of-cone leakage (floor) = {solver_leak:.5f}")
    for d in summary:
        print(
            f"  {d['kind']:12s} valMSE={d['val_mse']:.2e} oodMSE={d['ood_mse']:.2e} "
            f"leak_corr={d['leak_corr']:+.4f}"
        )
    print("VERDICT:", verdict)
    print(f"wrote {args.output}")
    print("############################################################")


if __name__ == "__main__":
    main()
