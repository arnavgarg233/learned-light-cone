"""Public script CLI and import-discipline checks."""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    "scripts/replay/verify_claims.py",
    "scripts/replay/verify_evidence.py",
    "scripts/replay/build_tables.py",
    "scripts/replay/build_figures.py",
    "scripts/theory/verify_sharp_cutoff_law.py",
    "scripts/theory/verify_rollout_certificate.py",
    "scripts/theory/verify_multilayer_tail.py",
    "scripts/experiments/run_corrected_existence.py",
    "scripts/experiments/run_mitigation.py",
    "scripts/experiments/run_predictive_battery.py",
    "scripts/deployed/run_ic_dispersion.py",
]


def test_each_retained_script_exposes_help_from_outside_repository(tmp_path):
    for relative in SCRIPTS:
        result = subprocess.run(
            [sys.executable, str(ROOT / relative), "--help"],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 0, (relative, result.stdout, result.stderr)
        assert "usage:" in result.stdout.lower()


def test_external_checkpoint_consumers_require_explicit_directories():
    for relative in (
        "scripts/experiments/run_corrected_existence.py",
        "scripts/experiments/run_predictive_battery.py",
        "scripts/theory/verify_multilayer_tail.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert 'parser.add_argument("--checkpoint-dir", type=Path' in source
        assert "weights_only=True" in source
    assert not (ROOT / "results" / "checkpoints").exists()


def test_predictive_battery_preserves_source_protocol_and_explicit_paths():
    path = ROOT / "scripts" / "experiments" / "run_predictive_battery.py"
    spec = importlib.util.spec_from_file_location("predictive_battery", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = json.loads(
        (ROOT / "results" / "tables" / "prediction" / "predictive_summary.json").read_text(
            encoding="utf-8"
        )
    )["cfg"]
    assert module.make_cfg(False) == expected
    source = path.read_text(encoding="utf-8")
    assert 'parser.add_argument("--output", type=Path, required=True)' in source
    assert 'parser.add_argument("--checkpoint-dir", type=Path, required=True)' in source
    assert "weights_only=True" in source


def test_mitigation_docstring_matches_current_cli():
    source = (ROOT / "scripts" / "experiments" / "run_mitigation.py").read_text(encoding="utf-8")
    docstring = ast.get_docstring(ast.parse(source))
    assert docstring is not None
    assert "scripts/experiments/run_mitigation.py" in docstring
    assert "--output" in docstring
    assert "run_mitigation_gpu.py" not in docstring
    assert "Writes mitigation_results.json in the CWD" not in docstring


def test_no_python_file_mutates_sys_path():
    for path in [*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("scripts").rglob("*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                target = ast.unparse(node.func)
                assert target not in {"sys.path.append", "sys.path.insert", "sys.path.extend"}, path
