"""Code-only tree, portability, sensitive-pattern, and file-policy contracts."""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_TOP_LEVEL = {
    ".github",
    ".gitignore",
    "LICENSE",
    "README.md",
    "data",
    "configs",
    "pyproject.toml",
    "reproduce.sh",
    "results",
    "scripts",
    "src",
    "tests",
    "tools",
    "uv.lock",
}
FORBIDDEN_PARTS = {
    "artifact",
    "paper",
    "submission",
    "evidence",
    "prespecification",
    "logs",
    "cache",
    "checkpoints",
    "stage",
    "dist",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
}
FORBIDDEN_SUFFIXES = {
    ".aux",
    ".ckpt",
    ".ipynb",
    ".log",
    ".mdlus",
    ".nc",
    ".npy",
    ".onnx",
    ".pt",
    ".pth",
    ".pyc",
    ".tex",
    ".zip",
}


def tracked_paths() -> list[PurePosixPath]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode("utf-8")
    return [PurePosixPath(item) for item in output.rstrip("\0").split("\0") if item]


def test_top_level_and_single_package_layout_are_exact():
    assert {path.parts[0] for path in tracked_paths()} == ALLOWED_TOP_LEVEL
    packages = [
        path.name
        for path in (ROOT / "src").iterdir()
        if path.is_dir() and not path.name.endswith(".egg-info") and path.name != "__pycache__"
    ]
    assert packages == ["learned_light_cone"]


def test_tracked_paths_are_safe_regular_and_allowlisted():
    paths = tracked_paths()
    assert paths
    for relative in paths:
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        assert not (set(part.lower() for part in relative.parts) & FORBIDDEN_PARTS)
        assert relative.suffix.lower() not in FORBIDDEN_SUFFIXES
        path = ROOT / relative
        mode = path.lstat().st_mode
        assert stat.S_ISREG(mode), relative
        assert path.stat().st_size > 0, relative
        assert path.stat().st_size < 50 * 1024 * 1024, relative


def test_ignore_file_does_not_hide_scientific_extensions():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("*.npy", "*.npz", "*.pt", "*.h5", "*.zip"):
        assert pattern not in text


def test_all_tracked_json_is_strict():
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            assert key not in result
            result[key] = value
        return result

    for relative in tracked_paths():
        if relative.suffix == ".json":
            with (ROOT / relative).open(encoding="utf-8") as handle:
                json.load(
                    handle,
                    object_pairs_hook=object_pairs,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        AssertionError(f"nonfinite JSON value {value!r} in {relative}")
                    ),
                )


def test_repository_scan_finds_no_sensitive_pattern():
    result = subprocess.run(
        [sys.executable, "scripts/replay/verify_evidence.py", "--scan"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "repository scan passed" in result.stdout


def test_license_is_unchanged_from_the_audited_parent():
    # SHA-256 of the LICENSE blob at audited parent 4239de8c84453c00ef46160cf678bb8d19949ace.
    expected = "9de98cec21b21672650edee0e715c4a4677223fc0abb0ee3c285988e9595e258"
    data = (ROOT / "LICENSE").read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected
    assert 'license = { file = "LICENSE" }' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


def test_tracked_text_has_no_em_or_en_dashes():
    for relative in tracked_paths():
        try:
            text = (ROOT / relative).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        assert "\u2014" not in text and "\u2013" not in text, relative
