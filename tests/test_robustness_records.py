"""Derived robustness status and scope-record checks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"


def test_derived_gate_table_keeps_negative_and_not_run_statuses():
    with (TABLES / "robustness_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = {row["gate"]: row for row in csv.DictReader(handle)}
    assert rows["mid_scale_ood_ratio"]["status"] == "fail"
    assert rows["full_scale_mps_ood_ratio"]["status"] == "pass"
    assert rows["ic_three_sigma_separation"]["status"] == "fail"
    assert float(rows["ic_three_sigma_separation"]["observed"]) < 3.0
    assert rows["ic_standard_deviation_bands"]["status"] == "pass"
    assert rows["stormer_round_two"]["status"] == "not_run"
    assert rows["aurora_round_two"]["status"] == "not_run"


def test_claim_scope_records_population_and_replay_boundaries():
    scope = json.loads((TABLES / "claim_scope.json").read_text(encoding="utf-8"))
    populations = scope["population_boundaries"]
    assert populations == {
        "controlled_trained_operators": 26,
        "local_references": 1,
        "populations_are_distinct": True,
        "released_checkpoints": 6,
    }
    assert scope["headline_evidence_boundary"] == {
        "headline_bullets": 5,
        "rely_on_frozen_external_records_without_tracked_generator": 4,
    }
    replay = scope["default_replay_scope"]
    assert replay["downloads_era5"] is False
    assert replay["reruns_deployed_model_inference"] is False
    assert replay["retrains_mitigation_models"] is False
    assert replay["loads_checkpoints"] is False
    limitations = "\n".join(scope["limitations"]).lower()
    assert "2.55 sigma" in limitations
    assert "stormer and aurora were not run" in limitations
