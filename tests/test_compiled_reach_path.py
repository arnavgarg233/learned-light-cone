"""Contracts for the optional compiled-reach utility.

The utility needs a 1.18 GB released checkpoint that this repository does not
redistribute, so these tests exercise the vendored interpreter, driver, and retained record.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "tools" / "compiled_reach"
RECORD = ROOT / "results" / "tables" / "deployed" / "compiled_reach_pangu.json"

# Vendored unmodified from the research tree that produced the retained record.
VENDORED = {
    "reach.py": "977cab35940c9f9fe4ac77d315e5ce88795f3c21f539b8449ac4f4a5ad4f5c5e",
    "onnx_skim.py": "b25d0138f618ec63a9b606ef53170e02d8358876ad615eb21a5caad8df6bcb0f",
}


def load_record():
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_vendored_interpreter_is_unmodified():
    for name, expected in VENDORED.items():
        data = (KIT / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == expected, name
        assert expected in (KIT / "FETCH.md").read_text(encoding="utf-8"), name


def test_driver_exposes_help_and_imports_the_interpreter_without_path_surgery():
    result = subprocess.run(
        [sys.executable, str(KIT / "cone_reach.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "--checkpoint" in result.stdout
    imported = subprocess.run(
        [sys.executable, "-c", "import onnx_skim, reach; print(reach.Interp.__name__)"],
        cwd=KIT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert imported.returncode == 0, imported.stderr
    assert "sys.path" not in (KIT / "cone_reach.py").read_text(encoding="utf-8")


def test_driver_and_record_agree_on_the_sites_and_the_cone():
    source = (KIT / "cone_reach.py").read_text(encoding="utf-8")
    record = load_record()
    for site in record["sites"]:
        declared = f'"{site["site"]}": ({site["site_lat"]:g}.0, {site["site_lon"]:g}.0)'
        assert declared.replace(".0.0", ".0") in source, declared
    cone = record["physical_cones"]["300_ms_24h"]
    assert f"default={cone['speed_ms']:g}.0" in source
    assert f"default={cone['step_hours']:g}.0" in source
    assert f"EARTH_R_KM = {record['earth_radius_km']:g}" in source


def test_documented_expected_output_matches_the_retained_record():
    fetch = (KIT / "FETCH.md").read_text(encoding="utf-8")
    record = load_record()
    headline = {site["site"]: site for site in record["sites"]}[record["headline_site"]]
    percent = 100.0 * headline["compiled_area_fraction"]
    cone_percent = 100.0 * record["physical_cones"]["300_ms_24h"]["area_fraction"]
    assert f"{percent:.2f}% of the globe" in fetch
    assert f"{cone_percent:.2f}% of the globe" in fetch
    assert f"{headline['compiled_reach_km']:,.6f} km" in fetch
    assert (
        f"{headline['n_cells_reachable']:,} of {record['grid']['n_cells_total']:,} cells" in fetch
    )
    lowest = 100.0 * record["certified_zero_area_fraction_min"]
    highest = 100.0 * record["certified_zero_area_fraction_max"]
    assert f"{lowest:.2f} percent to {highest:.2f} percent" in fetch
