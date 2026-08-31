"""Banked robustness sample and ingest characterization tests."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS = ROOT / "results" / "tables" / "robustness"


def load(name: str):
    path = ROBUSTNESS / name
    with path.open(encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AssertionError(f"nonfinite JSON value {value!r} in {name}")
            ),
        )


def test_dispersion_records_cover_ten_dates_three_regions_and_thirty_samples():
    for name in (
        "fourcastnet_ic_region_dispersion.json",
        "local_operator_ic_region_dispersion.json",
    ):
        document = load(name)
        assert document["n_ics"] == len(document["ic_dates"]) == 10
        assert document["n_regions"] == len(document["regions"]) == 3
        assert document["n_samples"] == len(document["per_ic"]) == 30
        assert len({row["ic"] for row in document["per_ic"]}) == 10
        assert len({row["region"] for row in document["per_ic"]}) == 3


def test_local_stays_at_floor_and_fourcastnet_reaches_antipode():
    local = load("local_operator_ic_region_dispersion.json")
    fourcastnet = load("fourcastnet_ic_region_dispersion.json")
    assert local["Lambda_mean"] <= 0.01
    assert not any(row["reaches_antipode_thr1e-2"] for row in local["per_ic"])
    assert fourcastnet["Lambda_mean"] >= 0.01
    assert any(row["reaches_antipode_thr1e-2"] for row in fourcastnet["per_ic"])
    assert local["numerical_control"]["max"] <= 1e-6
    assert fourcastnet["numerical_control"]["max"] <= 1e-6


def test_validated_ingest_covers_all_channels_exactly():
    validation = load("era5_ingest_validation.json")
    assert validation["channels_checked"] == 26
    assert len(validation["per_channel_max_abs_diff"]) == 26
    assert validation["max_abs_diff"] == 0.0
    assert validation["tol"] == 0.001


def test_no_robustness_record_claims_unrun_models():
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in ROBUSTNESS.glob("*.json"))
    assert "stormer" not in text
    assert "aurora" not in text
