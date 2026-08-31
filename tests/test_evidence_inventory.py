"""Canonical result inventory, strict JSON, and provenance checks."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_strict(path: Path):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open(encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AssertionError(f"nonfinite JSON value {value!r} in {path}")
            ),
        )


def test_all_json_is_strict_and_checksums_match():
    for path in ROOT.rglob("*.json"):
        if ".git" not in path.parts and ".venv" not in path.parts:
            load_strict(path)
    result = subprocess.run(
        ["shasum", "-a", "256", "-c", "results/CHECKSUMS.txt"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_result_seeds_are_declared_by_seed_authority():
    seed_document = load_strict(ROOT / "configs/seeds.json")
    allowed = {seed for values in seed_document["studies"].values() for seed in values}
    assert {0, 1, 2, 3, 4} <= allowed


def test_manifest_verifier_rejects_an_unexpected_result(tmp_path):
    shutil.copytree(ROOT / "results", tmp_path / "results")
    shutil.copy2(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    shutil.copy2(ROOT / "uv.lock", tmp_path / "uv.lock")
    unexpected = tmp_path / "results" / "tables" / "unexpected.json"
    unexpected.write_text("{}\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "replay" / "verify_evidence.py"),
            "--root",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "result declaration mismatch" in result.stderr
