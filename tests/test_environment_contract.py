"""Single-authority environment and seed contracts."""

from __future__ import annotations

import json
import math
import subprocess
import tomllib
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RUNTIME = {
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "matplotlib": "3.11.0",
    "torch": "2.12.0",
    "pandas": "3.0.3",
    "seaborn": "0.13.2",
    "PyYAML": "6.0.3",
}
FORBIDDEN_AUTHORITIES = {
    "requirements.txt",
    "requirements.lock",
    "requirements.lock.txt",
    "environment.yml",
    "environment.yaml",
    "Pipfile",
    "Pipfile.lock",
    "poetry.lock",
    "setup.py",
    "setup.cfg",
}


def load_strict_json(path: Path):
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


def test_pyproject_is_the_only_dependency_declaration():
    assert (ROOT / "pyproject.toml").is_file()
    assert (ROOT / "uv.lock").is_file()
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8")
    tracked = [PurePosixPath(path) for path in output.rstrip("\0").split("\0") if path]
    assert not ({path.name for path in tracked} & FORBIDDEN_AUTHORITIES)


def test_python_and_load_bearing_versions_are_exact():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["requires-python"] == ">=3.11,<3.12"
    dependencies = project["project"]["dependencies"]
    pins = dict(item.split("==", 1) for item in dependencies)
    assert pins == EXPECTED_RUNTIME
    assert project["dependency-groups"]["dev"]


def test_seed_inventory_covers_all_retained_studies():
    seeds = load_strict_json(ROOT / "configs/seeds.json")
    assert seeds["schema_version"] == 1
    studies = seeds["studies"]
    assert studies["theory_checks"] == [0]
    assert studies["corrected_existence_probe"] == [0]
    assert studies["headline_mitigation"] == [0, 1]
    assert studies["mitigation_full_scale_mps"] == [2, 3]
    assert studies["mitigation_mid_scale"] == [0, 1, 2, 3, 4]
    assert all(
        isinstance(seed, int) and math.isfinite(seed)
        for inventory in studies.values()
        for seed in inventory
    )
