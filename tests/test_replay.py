"""Deterministic regeneration and canonical-output equality checks."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE_OUTPUTS = ["summary.csv", "robustness_summary.csv", "claim_scope.json"]
FIGURE_OUTPUTS = [
    "fig_attention_window_sweep.pdf",
    "fig_consequence_and_mitigation.pdf",
    "fig_theory_checks.pdf",
]


def run(*arguments: str) -> None:
    subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def assert_json_close(observed, expected) -> None:
    if isinstance(expected, dict):
        assert observed.keys() == expected.keys()
        for key in expected:
            assert_json_close(observed[key], expected[key])
    elif isinstance(expected, list):
        assert len(observed) == len(expected)
        for observed_item, expected_item in zip(observed, expected, strict=True):
            assert_json_close(observed_item, expected_item)
    elif isinstance(expected, float):
        assert math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12)
    else:
        assert observed == expected


def test_derived_tables_and_figures_match_canonical_bytes(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    for output in (first, second):
        run(
            "scripts/replay/build_tables.py",
            "--input-dir",
            "results/tables",
            "--output-dir",
            str(output / "tables"),
        )
        run(
            "scripts/replay/build_figures.py",
            "--input-dir",
            "results/tables",
            "--output-dir",
            str(output / "figures"),
        )
    for name in TABLE_OUTPUTS:
        first_bytes = (first / "tables" / name).read_bytes()
        assert first_bytes == (second / "tables" / name).read_bytes()
        assert first_bytes == (ROOT / "results" / "tables" / name).read_bytes()
    for name in FIGURE_OUTPUTS:
        first_bytes = (first / "figures" / name).read_bytes()
        assert first_bytes == (second / "figures" / name).read_bytes()
        assert first_bytes == (ROOT / "results" / "figures" / name).read_bytes()


def test_closed_form_records_match_canonical_values(tmp_path):
    sharp = tmp_path / "sharp.json"
    rollout = tmp_path / "rollout.json"
    run("scripts/theory/verify_sharp_cutoff_law.py", "--output", str(sharp))
    run("scripts/theory/verify_rollout_certificate.py", "--output", str(rollout))
    for observed_path, canonical_path in (
        (
            sharp,
            ROOT / "results" / "tables" / "theory" / "sharp_cutoff_law_verification.json",
        ),
        (
            rollout,
            ROOT / "results" / "tables" / "theory" / "rollout_certificate_verification.json",
        ),
    ):
        assert_json_close(
            json.loads(observed_path.read_text(encoding="utf-8")),
            json.loads(canonical_path.read_text(encoding="utf-8")),
        )


def test_manifest_and_frozen_multilayer_verifiers_pass():
    run("scripts/replay/verify_evidence.py")
    run(
        "scripts/theory/verify_multilayer_tail.py",
        "--input",
        "results/tables/theory/multilayer_tail_verification.json",
    )
