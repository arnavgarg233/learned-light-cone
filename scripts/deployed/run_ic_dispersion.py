#!/usr/bin/env python3
"""Run the local-operator IC-dispersion probe with explicitly supplied external arrays."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from learned_light_cone.deployed.afno_model import VARIABLES
from learned_light_cone.deployed.cone_metrics import (
    ANTIPODE,
    CONES,
    cone_radius_km,
    cos_lat_weights,
    leakage_metrics,
)
from learned_light_cone.deployed.local_conv_operator import LocalConvNet
from learned_light_cone.geometry_2d import haversine_distance

DATES = [
    "2020-01-01T00",
    "2020-02-15T12",
    "2020-04-01T00",
    "2020-05-15T12",
    "2020-07-15T12",
    "2020-08-20T00",
    "2020-10-01T12",
    "2020-11-15T00",
    "2021-01-10T00",
    "2021-06-21T12",
]
REGIONS = {
    "N.Pacific(30N,200E)": (30.0, 200.0),
    "Equator(0,0E)": (0.0, 0.0),
    "S.Ocean(50S,90E)": (-50.0, 90.0),
}
CONE_RADIUS_KM = cone_radius_km(CONES["gravity_sound_300ms"])


def require_array(root: Path, name: str) -> np.ndarray:
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(f"missing declared external array: {path}")
    return np.load(path).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=os.environ.get("LIGHT_CONE_DATA_ROOT"),
        help="external data root, or set LIGHT_CONE_DATA_ROOT",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.data_root is None:
        parser.error("--data-root or LIGHT_CONE_DATA_ROOT is required")
    root = arguments.data_root.resolve()
    output_dir = arguments.output_dir.resolve()
    if output_dir == root or root in output_dir.parents:
        raise ValueError("output directory must be separate from the external data root")

    lat = require_array(root, "lat.npy")
    lon = require_array(root, "lon.npy")
    means = require_array(root, "global_means.npy").reshape(26, 1, 1)
    standard_deviations = require_array(root, "global_stds.npy").reshape(26, 1, 1)
    weights = cos_lat_weights(lat, len(lon))
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    torch.manual_seed(arguments.seed)
    model = LocalConvNet(in_channels=26, hidden=48, n_layers=6, dilation=1).eval()

    def step(state):
        normalized = ((state - means) / standard_deviations)[None]
        with torch.no_grad():
            prediction = model(torch.from_numpy(normalized).float()).numpy()[0]
        return prediction * standard_deviations + means

    rows = []
    for date in DATES:
        filename = "era5_ic_" + date.replace("-", "").replace("T", "") + ".npy"
        state = require_array(root, filename)
        if state.shape[0] != len(VARIABLES):
            raise ValueError(f"{filename} must contain 26 channels")
        for region, (center_lat, center_lon) in REGIONS.items():
            distance = haversine_distance(center_lat, center_lon, lat_grid, lon_grid)
            bump = np.exp(-0.5 * (distance / 500.0) ** 2)
            bump -= (bump * weights).sum() / weights.sum()
            perturbation = np.zeros_like(state)
            channel = VARIABLES.index("t2m")
            perturbation[channel] = 0.5 * float(np.std(state[channel])) * bump
            baseline = step(state)
            perturbed = step(state + perturbation)
            response = np.sqrt((((perturbed - baseline) / standard_deviations) ** 2).sum(axis=0))
            panel = leakage_metrics(response, distance, weights)
            outside = distance > CONE_RADIUS_KM
            energy = weights * response**2
            raw_fraction = float(energy[outside].sum() / energy.sum())
            bump_energy = weights * perturbation[channel] ** 2
            floor = float(bump_energy[outside].sum() / bump_energy.sum())
            rows.append(
                {
                    "Lambda": max(raw_fraction - floor, 0.0),
                    "raw_outside_fraction": raw_fraction,
                    "causal_floor_fraction": floor,
                    "peak": float(response.max()),
                    "max_reach_km_thr1e-2": panel["max_reach_km_thr0.01"],
                    "reaches_antipode_thr1e-2": panel["reaches_antipode_thr0.01"],
                    "ic": date,
                    "region": region,
                }
            )

    values = np.array([row["Lambda"] for row in rows])
    document = {
        "model": "Local-conv operator (finite 3x3-kernel stack)",
        "globality_class": "local (finite convolution receptive field)",
        "device": "cpu",
        "status": "OPERATOR RAN",
        "cone": {
            "name": "gravity_sound_300ms",
            "speed_ms": 300.0,
            "radius_6h_km": CONE_RADIUS_KM,
            "antipode_km": ANTIPODE,
        },
        "n_ics": len(DATES),
        "ic_dates": DATES,
        "n_regions": len(REGIONS),
        "regions": REGIONS,
        "Lambda_mean": float(values.mean()),
        "Lambda_std": float(values.std()),
        "n_samples": len(rows),
        "per_ic": rows,
        "seed": arguments.seed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "local_operator_ic_region_dispersion.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
