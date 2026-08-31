#!/usr/bin/env python3
"""Validate or externally regenerate the multilayer-tail record.

Default replay validates the frozen strict-JSON record without loading checkpoints.
Optional regeneration requires an explicit checkpoint directory and verifies every
state dictionary against ``data/manifest.json`` before loading it with
``weights_only=True``. Supplying ``--input`` during regeneration additionally
requires byte equality with the committed canonical record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from learned_light_cone.metrics.spectral import learned_transfer
from learned_light_cone.metrics.tail_exponent import kernel_profile, tail_exponent
from learned_light_cone.models.cnn import LocalCNN1d
from learned_light_cone.models.fno import FNO1d
from learned_light_cone.pdes.advection import band_limited_ic, bump_radius, make_local_bump

N = 128
SHIFT = 1
SIGMA = 1.0
EPSILON = 1e-3
CENTER = N // 2
MODES = [6, 8, 10, 12, 16, 24, 32]
EXPECTED_CHECKPOINTS = [
    "cnn_k3_s0.pt",
    "fno_m6_s0.pt",
    "fno_m8_s0.pt",
    "fno_m10_s0.pt",
    "fno_m12_s0.pt",
    "fno_m16_s0.pt",
    "fno_m24_s0.pt",
    "fno_m32_s0.pt",
]


def load_strict(path: Path):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open(encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON value {value!r} in {path}")
            ),
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoints(checkpoint_dir: Path, manifest_path: Path) -> list[Path]:
    manifest = load_strict(manifest_path)
    matches = [
        asset
        for asset in manifest["assets"]
        if asset["asset_class"] == "corrected_existence_state_dicts"
    ]
    if len(matches) != 1:
        raise ValueError("data manifest must declare one corrected-existence checkpoint asset")
    members = matches[0]["members"]
    if [member["path"] for member in members] != EXPECTED_CHECKPOINTS:
        raise ValueError("data manifest has an unexpected checkpoint inventory")
    validated = []
    for member in members:
        path = checkpoint_dir / member["path"]
        if not path.is_file():
            raise FileNotFoundError(f"missing external checkpoint: {path}")
        if path.stat().st_size != member["byte_size"] or sha256(path) != member["sha256"]:
            raise ValueError(f"external checkpoint does not match the manifest: {path.name}")
        validated.append(path)
    return validated


def validate_frozen(input_path: Path) -> int:
    document = load_strict(input_path)
    fno = document["fno"]
    if sorted(map(int, fno)) != MODES:
        raise ValueError("multilayer record has an unexpected retained-mode inventory")
    alphas = [float(row["alpha"]) for row in fno.values()]
    if not all(math.isfinite(value) for value in alphas):
        raise ValueError("multilayer alpha values must be finite")
    verdict = document["verdict"]
    if verdict["all_in_1r_class_below_1p2"] is not True:
        raise ValueError("multilayer record does not retain the banked tail-class result")
    if verdict["cnn_has_no_tail"] is not True:
        raise ValueError("multilayer record does not retain the compact-support control")
    print(f"validated frozen multilayer record: {len(fno)} FNO mode settings")
    return 0


def effective_cutoff(step_fn, bases: torch.Tensor, kmax: int) -> dict:
    ks = np.arange(0, kmax + 1)
    transfer = learned_transfer(step_fn, bases, EPSILON, ks, torch.device("cpu"))["G"]
    absolute = np.abs(transfer)
    low_band_stop = max(2, kmax // 3)
    in_band_median = (
        float(np.median(absolute[1:low_band_stop])) if kmax >= 3 else float(absolute[1])
    )
    threshold = 0.5 * in_band_median
    below = np.where(absolute[1:] < threshold)[0]
    effective = int(below[0] + 1) if below.size else int(kmax)
    return {
        "ks": ks.tolist(),
        "absG": absolute.tolist(),
        "inband_median": in_band_median,
        "k_eff": effective,
        "nominal_K": kmax,
    }


def generate(checkpoint_dir: Path, manifest_path: Path, output_path: Path) -> None:
    validate_checkpoints(checkpoint_dir, manifest_path)
    device = torch.device("cpu")
    radius = bump_radius(SIGMA)
    rng = np.random.default_rng(0)
    bases = torch.as_tensor(band_limited_ic(rng, N, 8, 12, 1.0), dtype=torch.float32, device=device)
    bump = torch.as_tensor(make_local_bump(N, CENTER, SIGMA), dtype=torch.float32, device=device)

    def fno_step(modes: int):
        model = FNO1d(modes=modes, width=32, n_layers=4)
        state = torch.load(
            checkpoint_dir / f"fno_m{modes}_s0.pt",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state)
        model.eval().to(device)
        return lambda value: model(value), model

    document = {
        "meta": {
            "N": N,
            "m": SHIFT,
            "sigma": SIGMA,
            "eps": EPSILON,
            "x0": CENTER,
            "r0": radius,
            "modes": MODES,
            "device": str(device),
            "analytic_single_layer_alpha": 1.0,
            "empirical_anchor": "window_swap_summary.json trained multilayer alpha 0.55-0.70 (advection+burgers)",
        },
        "fno": {},
        "cnn": None,
    }
    alphas = {}
    for modes in MODES:
        step, _ = fno_step(modes)
        exponent = tail_exponent(step, bases, bump, EPSILON, CENTER, radius, k=1)
        cutoff = effective_cutoff(step, bases, kmax=min(modes + 8, N // 2))
        record = {
            "alpha": exponent["alpha"],
            "poly_r2": exponent["poly_r2"],
            "exp_r2": exponent["exp_r2"],
            "xi": exponent["xi"],
            "is_exponential": bool(exponent["is_exponential"]),
            "n_tail": exponent["n_tail"],
            "peak": exponent["peak"],
            "d_tail": exponent["d_tail"],
            "g_tail": exponent["g_tail"],
            "k_eff": cutoff["k_eff"],
            "inband_median_absG": cutoff["inband_median"],
            "absG": cutoff["absG"],
            "ks": cutoff["ks"],
            "nominal_K": modes,
        }
        document["fno"][str(modes)] = record
        alphas[modes] = exponent["alpha"]

    cnn = LocalCNN1d(kernel=3, width=32, n_layers=4)
    cnn.load_state_dict(
        torch.load(
            checkpoint_dir / "cnn_k3_s0.pt",
            map_location=device,
            weights_only=True,
        )
    )
    cnn.eval().to(device)

    def cnn_step(value):
        return cnn(value)

    cnn_exponent = tail_exponent(cnn_step, bases, bump, EPSILON, CENTER, radius, k=1)
    distance, magnitude = kernel_profile(cnn_step, bases, bump, EPSILON, CENTER, k=1)
    peak = float(magnitude.max()) if magnitude.size else 0.0
    signal = distance[magnitude > 1e-2 * (peak + 1e-30)]
    floor = distance[magnitude > 1e-5 * (peak + 1e-30)]
    signal_radius = float(signal.max()) if signal.size else 0.0
    floor_radius = float(floor.max()) if floor.size else 0.0
    fno_distance, fno_magnitude = kernel_profile(fno_step(16)[0], bases, bump, EPSILON, CENTER, k=1)
    fno_peak = float(fno_magnitude.max()) if fno_magnitude.size else 1.0
    far = fno_magnitude[fno_distance >= (N // 2 - 4)]
    half_domain_tail = float(far.max() / (fno_peak + 1e-30)) if far.size else float("nan")
    document["cnn"] = {
        "alpha": cnn_exponent["alpha"],
        "poly_r2": cnn_exponent["poly_r2"],
        "exp_r2": cnn_exponent["exp_r2"],
        "xi": cnn_exponent["xi"],
        "is_exponential": bool(cnn_exponent["is_exponential"]),
        "n_tail": cnn_exponent["n_tail"],
        "peak": cnn_exponent["peak"],
        "signal_support_radius_cells": signal_radius,
        "noise_floor_support_radius_cells": floor_radius,
        "arch_receptive_radius": int(cnn.receptive_radius),
        "fno_K16_tail_frac_at_halfdomain": half_domain_tail,
        "note": "CNN alpha fits the ~1e-5 float32 probe noise floor and is meaningless; the fact is COMPACT SUPPORT: signal dies by ~r_sig cells, beyond which response is at the single-precision floor (~1e-5 of peak, no systematic decay). Contrast the FNO, ~6% of peak at N/2 -- a tail.",
    }
    alpha_values = np.array([alphas[modes] for modes in MODES], dtype=float)
    finite = np.isfinite(alpha_values)
    in_class = bool(np.all(alpha_values[finite] < 1.2)) and bool(
        np.median(alpha_values[finite]) < 1.0
    )
    document["verdict"] = {
        "alpha_min": float(np.nanmin(alpha_values)),
        "alpha_max": float(np.nanmax(alpha_values)),
        "alpha_median": float(np.nanmedian(alpha_values)),
        "all_in_1r_class_below_1p2": bool(np.all(alpha_values[finite] < 1.2)),
        "median_below_1": in_class,
        "below_analytic_single_layer_alpha1": bool(np.median(alpha_values[finite]) < 0.95),
        "cnn_has_no_tail": bool(signal_radius < N // 4),
        "statement": "Trained multilayer FNO retains a polynomial (1/r-class) tail alpha<~1 across K, set by the trained effective cutoff (alpha below the analytic single-layer 1); CNN has compact support, no tail. T1's mechanism survives training + nonlinearity.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, default=float, allow_nan=False)
    print(f"regenerated multilayer record: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.checkpoint_dir is None:
        if (
            arguments.input is None
            or arguments.manifest is not None
            or arguments.output is not None
        ):
            parser.error("validation requires only --input")
        return validate_frozen(arguments.input)
    if arguments.manifest is None or arguments.output is None:
        parser.error("regeneration requires --checkpoint-dir, --manifest, and --output")
    generate(arguments.checkpoint_dir, arguments.manifest, arguments.output)
    if (
        arguments.input is not None
        and arguments.output.read_bytes() != arguments.input.read_bytes()
    ):
        raise ValueError("regenerated multilayer record differs from the canonical bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
