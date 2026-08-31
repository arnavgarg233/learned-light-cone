"""Installed-package import contract."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path


def test_stale_spectral_docstring_path_is_removed():
    from learned_light_cone.metrics import jacobian

    source = Path(jacobian.__file__).read_text(encoding="utf-8")
    assert "src.metrics.spectral" not in source
    assert "learned_light_cone.metrics.spectral" in source


def test_package_and_all_modules_import_without_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    package = importlib.import_module("learned_light_cone")
    modules = sorted(
        module.name for module in pkgutil.walk_packages(package.__path__, package.__name__ + ".")
    )
    assert modules
    for name in modules:
        importlib.import_module(name)
    assert list(Path(tmp_path).iterdir()) == []
