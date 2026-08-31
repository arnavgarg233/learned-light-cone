#!/usr/bin/env python3
"""Optionally rerun the corrected-existence probe with externally supplied state dictionaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from learned_light_cone.metrics.cone import perturbation_probe
from learned_light_cone.metrics.cone_corrections import (
    bandlimited_shift_step,
    directional_leakage,
)
from learned_light_cone.models.cnn import LocalCNN1d
from learned_light_cone.models.fno import FNO1d
from learned_light_cone.pdes.advection import (
    band_limited_ic,
    bump_radius,
    make_local_bump,
)

N = 128
SHIFT = 1
SIGMA = 1.0
EPSILON = 1e-3
HORIZON = 16
SUMMARY_HORIZON = 8
MODES = [6, 8, 10, 12, 16, 24, 32]
CENTERS = [N // 4, N // 2, 3 * N // 4, N // 2 + 7]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoints(checkpoint_dir: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    asset = next(
        item
        for item in manifest["assets"]
        if item["asset_class"] == "corrected_existence_state_dicts"
    )
    for member in asset["members"]:
        path = checkpoint_dir / member["path"]
        if not path.is_file():
            raise FileNotFoundError(f"missing external checkpoint: {path}")
        if path.stat().st_size != member["byte_size"] or sha256(path) != member["sha256"]:
            raise ValueError(f"external checkpoint does not match the manifest: {path.name}")


def directional_curves(step, bases, bumps, radius):
    upstream, ahead, total = [], [], []
    for center in CENTERS:
        bump_batch = bumps[center].unsqueeze(0).expand(bases.shape[0], -1)
        response = perturbation_probe(step, bases, bump_batch, EPSILON, HORIZON).numpy()
        for row in response:
            leakage = directional_leakage(row, center, SHIFT, radius)
            upstream.append(leakage["upstream"])
            ahead.append(leakage["downstream_ahead"])
            total.append(leakage["total_out"])
    return np.array(upstream), np.array(ahead), np.array(total)


def mean_and_se(curves):
    values = curves[:, :SUMMARY_HORIZON].mean(axis=1)
    return float(values.mean()), float(values.std() / np.sqrt(len(values)))


def run(checkpoint_dir: Path, manifest_path: Path, output_path: Path) -> None:
    validate_checkpoints(checkpoint_dir, manifest_path)
    device = torch.device("cpu")
    radius = bump_radius(SIGMA)
    rng = np.random.default_rng(0)
    bases = torch.as_tensor(band_limited_ic(rng, N, 8, 12, 1.0), dtype=torch.float32, device=device)
    bumps = {
        center: torch.as_tensor(
            make_local_bump(N, center, SIGMA), dtype=torch.float32, device=device
        )
        for center in CENTERS
    }

    def fno_step(modes):
        model = FNO1d(modes=modes, width=32, n_layers=4)
        state = torch.load(
            checkpoint_dir / f"fno_m{modes}_s0.pt",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state)
        model.eval()
        return lambda value: model(value)

    solver_upstream, _, solver_total = directional_curves(
        lambda value: torch.roll(value, SHIFT, dims=-1), bases, bumps, radius
    )
    cnn = LocalCNN1d(kernel=3, width=32, n_layers=4)
    cnn.load_state_dict(
        torch.load(
            checkpoint_dir / "cnn_k3_s0.pt",
            map_location=device,
            weights_only=True,
        )
    )
    cnn.eval()
    cnn_upstream, _, cnn_total = directional_curves(lambda value: cnn(value), bases, bumps, radius)
    torch.manual_seed(100)
    untrained = FNO1d(modes=16, width=32, n_layers=4).eval()
    untrained_upstream, _, _ = directional_curves(
        lambda value: untrained(value), bases, bumps, radius
    )

    table = {}
    for modes in MODES:
        fno_upstream, _, fno_total = directional_curves(fno_step(modes), bases, bumps, radius)
        floor_upstream, _, floor_total = directional_curves(
            bandlimited_shift_step(SHIFT, modes, N), bases, bumps, radius
        )
        upstream_mean, upstream_se = mean_and_se(fno_upstream)
        floor_mean, _ = mean_and_se(floor_upstream)
        total_mean, _ = mean_and_se(fno_total)
        table[modes] = {
            "sym_total": total_mean,
            "upstream": upstream_mean,
            "upstream_se": upstream_se,
            "floor_upstream": floor_mean,
            "upstream_excess": upstream_mean - floor_mean,
            "floor_total": float(floor_total[:, :SUMMARY_HORIZON].mean()),
        }

    summary = {
        "modes": table,
        "cnn_upstream": mean_and_se(cnn_upstream)[0],
        "solver_upstream": mean_and_se(solver_upstream)[0],
        "untrained_upstream": mean_and_se(untrained_upstream)[0],
        "note": "upstream_excess = FNO upstream leakage - identically-band-limited causal shift; >0 = genuine learned backward acausality",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.checkpoint_dir, arguments.manifest, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
