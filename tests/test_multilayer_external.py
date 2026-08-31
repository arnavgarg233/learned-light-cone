"""External checkpoint validation for optional multilayer regeneration."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "theory" / "verify_multilayer_tail.py"
NAMES = [
    "cnn_k3_s0.pt",
    "fno_m6_s0.pt",
    "fno_m8_s0.pt",
    "fno_m10_s0.pt",
    "fno_m12_s0.pt",
    "fno_m16_s0.pt",
    "fno_m24_s0.pt",
    "fno_m32_s0.pt",
]


def load_script():
    spec = importlib.util.spec_from_file_location("multilayer_verifier", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_external_assets(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    members = []
    for index, name in enumerate(NAMES):
        payload = f"external-checkpoint-{index}".encode()
        path = checkpoint_dir / name
        path.write_bytes(payload)
        members.append(
            {
                "path": name,
                "byte_size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest = {
        "assets": [
            {
                "asset_class": "corrected_existence_state_dicts",
                "members": members,
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return checkpoint_dir, manifest_path


def test_external_checkpoint_inventory_is_hash_validated(tmp_path):
    module = load_script()
    checkpoint_dir, manifest = make_external_assets(tmp_path)
    validated = module.validate_checkpoints(checkpoint_dir, manifest)
    assert [path.name for path in validated] == NAMES


def test_external_checkpoint_validation_rejects_missing_or_changed_bytes(tmp_path):
    module = load_script()
    checkpoint_dir, manifest = make_external_assets(tmp_path)
    (checkpoint_dir / NAMES[0]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="does not match the manifest"):
        module.validate_checkpoints(checkpoint_dir, manifest)
    (checkpoint_dir / NAMES[0]).unlink()
    with pytest.raises(FileNotFoundError, match="missing external checkpoint"):
        module.validate_checkpoints(checkpoint_dir, manifest)
