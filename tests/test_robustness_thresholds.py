"""Banked robustness threshold tests, adapted to canonical result paths."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS = ROOT / "results" / "tables" / "robustness"
OOD_RATIO_MAX = 0.10
ID_MATCH_FACTOR = 5.0
ID_TRAINED_CEIL = 1e-2
LEAK_CORR_MAX = 0.01
SEPARATION_SIGMA = 3.0


def load(name: str):
    path = ROBUSTNESS / name

    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(f"duplicate JSON key {key!r} in {name}")
            result[key] = value
        return result

    with path.open(encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AssertionError(f"nonfinite JSON value {value!r} in {name}")
            ),
        )


def seed_rows(document):
    grouped = {}
    for row in document["rows"]:
        grouped.setdefault(row["seed"], {})[row["kind"]] = row
    return grouped


def ratios(document, seed):
    rows = seed_rows(document)[seed]
    global_fno = rows["fno_global"]
    local_cnn = rows["local_cnn"]
    return {
        "ood_ratio": local_cnn["ood_mse"] / global_fno["ood_mse"],
        "id_ratio": local_cnn["val_mse"] / global_fno["val_mse"],
        "trained_minimum": min(local_cnn["val_mse"], global_fno["val_mse"]),
        "leak_corr": local_cnn["out_of_cone_leak"] - document["solver_out_of_cone_leak"],
    }


def test_mid_scale_gate_is_preserved_as_failed():
    document = load("mitigation_mid_scale_multiseed.json")
    assert document["seeds"] == [0, 1, 2, 3, 4]
    observed = [ratios(document, seed) for seed in document["seeds"]]
    assert all(row["ood_ratio"] < 1.0 for row in observed)
    assert max(row["ood_ratio"] for row in observed) > OOD_RATIO_MAX
    assert all(row["id_ratio"] <= ID_MATCH_FACTOR for row in observed)
    assert all(row["trained_minimum"] < ID_TRAINED_CEIL for row in observed)
    assert all(row["leak_corr"] <= LEAK_CORR_MAX for row in observed)


def test_full_scale_mps_per_seed_gates_pass():
    document = load("mitigation_full_scale_mps_seeds_2_3.json")
    assert document["seeds"] == [2, 3]
    observed = [ratios(document, seed) for seed in document["seeds"]]
    assert all(row["ood_ratio"] <= OOD_RATIO_MAX for row in observed)
    assert all(row["id_ratio"] <= ID_MATCH_FACTOR for row in observed)
    assert all(row["trained_minimum"] < ID_TRAINED_CEIL for row in observed)
    assert all(row["leak_corr"] <= LEAK_CORR_MAX for row in observed)


def test_prespecified_three_sigma_gate_is_preserved_as_failed():
    fourcastnet = load("fourcastnet_ic_region_dispersion.json")
    local = load("local_operator_ic_region_dispersion.json")
    f_mean, f_std = fourcastnet["Lambda_mean"], fourcastnet["Lambda_std"]
    l_mean, l_std = local["Lambda_mean"], local["Lambda_std"]
    assert f_mean - f_std > l_mean + l_std
    separation = (f_mean - l_mean) / math.sqrt(f_std**2 + l_std**2)
    assert round(separation, 2) == 2.55
    assert separation < SEPARATION_SIGMA
