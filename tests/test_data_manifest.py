"""External-data boundary and checkpoint exclusion contracts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ASSET_CLASSES = {
    "weatherbench2_arco_era5",
    "fourcastnet_v1_checkpoint",
    "fourcastnet_v1_normalization_and_grid",
    "pangu_weather_24_checkpoint",
    "corrected_existence_state_dicts",
}
REQUIRED_FIELDS = {
    "asset_id",
    "asset_class",
    "stable_identifier",
    "immutable_url_or_doi",
    "version",
    "license",
    "access_terms",
    "role",
    "byte_size",
    "sha256",
    "retrieval_command",
    "availability",
}


def load_strict_json(path: Path):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    with path.open(encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AssertionError(f"nonfinite JSON value {value!r}")
            ),
        )


def test_required_external_asset_boundaries_are_explicit():
    document = load_strict_json(ROOT / "data" / "manifest.json")
    assert document["schema_version"] == 1
    assets = document["assets"]
    assert {asset["asset_class"] for asset in assets} == REQUIRED_ASSET_CLASSES
    for asset in assets:
        assert set(asset) >= REQUIRED_FIELDS
        assert asset["availability"] in {"available", "unavailable"}
        if asset["availability"] == "available":
            assert isinstance(asset["byte_size"], int) and asset["byte_size"] > 0
            assert len(asset["sha256"]) == 64
            assert asset["retrieval_command"]
        else:
            assert asset["unavailable_reason"]
            assert asset["retrieval_command"] is None


def test_dates_variables_and_checkpoint_hashes_are_complete():
    document = load_strict_json(ROOT / "data" / "manifest.json")
    assets = {asset["asset_class"]: asset for asset in document["assets"]}
    era5 = assets["weatherbench2_arco_era5"]
    assert len(era5["selected_dates_utc"]) == 10
    assert len(era5["selected_variables"]) == 26
    checkpoint_asset = assets["corrected_existence_state_dicts"]
    assert checkpoint_asset["consumers"] == [
        "scripts/experiments/run_corrected_existence.py",
        "scripts/theory/verify_multilayer_tail.py",
    ]
    checkpoints = checkpoint_asset["members"]
    assert len(checkpoints) == 8
    assert len({member["path"] for member in checkpoints}) == 8
    for member in checkpoints:
        assert len(member["sha256"]) == 64
        assert member["byte_size"] > 0
        path = PurePosixPath(member["path"])
        assert not path.is_absolute() and ".." not in path.parts


def test_no_checkpoint_or_raw_data_is_tracked():
    forbidden_suffixes = {".pt", ".pth", ".ckpt", ".mdlus", ".onnx", ".nc", ".npy"}
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8")
    tracked = [PurePosixPath(path) for path in output.rstrip("\0").split("\0") if path]
    assert not [path for path in tracked if path.suffix in forbidden_suffixes]
