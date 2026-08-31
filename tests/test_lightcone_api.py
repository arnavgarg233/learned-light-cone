"""Focused contracts for the model-agnostic NumPy benchmark."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import learned_light_cone
from learned_light_cone.lightcone import __all__ as module_api

ROOT = Path(__file__).resolve().parents[1]


def test_public_api_and_top_level_dependencies_are_preserved():
    expected = [
        "localized_bump",
        "perturbation_response",
        "cone_leakage",
        "tail_exponent",
        "cone_report",
        "torch_forward",
        "examples",
    ]
    assert module_api == expected
    assert learned_light_cone.__all__ == ["__version__", *expected]
    assert all(getattr(learned_light_cone, name) for name in expected)
    path = ROOT / "src" / "learned_light_cone" / "lightcone.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".", 1)[0]
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
        if alias.name != "annotations"
    }
    assert imported == {"numpy"}
    source = path.read_text(encoding="utf-8")
    assert "from learned_light_cone import cone_report, localized_bump" in source
    assert 'rep["tail"]["is_algebraic"]' in source
    assert 'rep["is_algebraic"]' not in source
    assert "from lightcone import" not in source


def test_model_agnostic_one_dimensional_cone_report():
    size = 32
    center = size // 2
    base = np.zeros((3, size), dtype=np.float32)
    bump = np.zeros(size, dtype=np.float32)
    bump[center] = 1.0
    distance = np.abs((np.arange(size) - center + size // 2) % size - size // 2)

    def causal_roll(state):
        return np.roll(state, 1, axis=-1)

    def callable_with_remote_coupling(state):
        output = causal_roll(state)
        output[..., 0] += state[..., center]
        return output

    report = learned_light_cone.cone_report(
        callable_with_remote_coupling,
        base,
        bump,
        distance,
        cone_radius=2,
        baseline_forward=causal_roll,
        eps=0.25,
        antipode=size // 2,
    )
    assert report["baseline_leakage"] == pytest.approx(0.0)
    assert report["leakage_corrected"] > 0.4
    assert report["reach_1e_3"] == pytest.approx(size // 2)
    assert report["reaches_antipode_1e_3"] is True
    assert report["verdict"] == "ACAUSAL"


def test_custom_distance_array_controls_multidimensional_reporting():
    base = np.zeros((2, 2, 3), dtype=np.float32)
    bump = np.zeros((2, 3), dtype=np.float32)
    bump[0, 0] = 1.0
    custom_distance = np.array([[0.0, 1.0, 2.0], [7.0, 9.0, 12.0]])

    def arbitrary_forward(state):
        output = np.array(state, copy=True)
        output[..., 1, 2] += state[..., 0, 0]
        return output

    report = learned_light_cone.cone_report(
        arbitrary_forward,
        base,
        bump,
        custom_distance,
        cone_radius=2.0,
        eps=0.5,
        antipode=12.0,
    )
    assert report["leakage"] == pytest.approx(0.5)
    assert report["reach_1e_2"] == pytest.approx(12.0)
    assert report["reach_1e_3"] == pytest.approx(12.0)
    assert report["reaches_antipode_1e_3"] is True
