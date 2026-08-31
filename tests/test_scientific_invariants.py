"""Small deterministic checks for retained scientific operations."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from learned_light_cone.deployed.era5_ingest import (
    PRESSURE_CHANNELS,
    SURFACE_CHANNELS,
    assemble_fourcastnet_channels,
)
from learned_light_cone.metrics.cone import leakage_mass, signed_periodic_distance
from learned_light_cone.models.windowed_fno import make_window
from learned_light_cone.pdes.advection import band_limited_ic, exact_rollout


def test_signed_periodic_distance_has_expected_wrap():
    np.testing.assert_array_equal(
        signed_periodic_distance(8, 4), np.array([-4, -3, -2, -1, 0, 1, 2, 3])
    )


def test_exact_roll_respects_dilated_causal_cone():
    n = 32
    center = n // 2
    base = np.zeros(n)
    base[center] = 1.0
    trajectory = exact_rollout(base, m=1, K=5)
    response_energy = trajectory**2
    np.testing.assert_array_equal(leakage_mass(response_energy, x0=center, m=1, r0=0), np.zeros(5))


def test_spectral_windows_preserve_control_and_taper_edge():
    torch.testing.assert_close(make_window(8, "rect"), torch.ones(8))
    raised = make_window(8, "raised_cosine")
    assert raised[0].item() == pytest.approx(1.0)
    assert raised[-1].item() == pytest.approx(0.0, abs=1e-7)
    assert torch.all(raised[:-1] >= raised[1:])


def test_seeded_initial_conditions_are_reproducible():
    first = band_limited_ic(np.random.default_rng(0), 64, 4)
    second = band_limited_ic(np.random.default_rng(0), 64, 4)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first.mean(axis=1), 0.0, atol=1e-14)
    np.testing.assert_allclose(first.std(axis=1), 1.0, atol=1e-14)


def test_era5_transform_has_validated_channel_order_and_shape():
    grid = np.full((3, 4), 280.0, dtype=np.float32)
    surface = {name: grid + index for index, name in SURFACE_CHANNELS.items()}
    pressure = {key: grid + index for index, key in PRESSURE_CHANNELS.items()}
    pressure[("specific_humidity", 500)] = np.full_like(grid, 0.001)
    pressure[("specific_humidity", 850)] = np.full_like(grid, 0.002)
    output = assemble_fourcastnet_channels(surface, pressure, target_latitudes=2)
    assert output.shape == (26, 2, 4)
    np.testing.assert_array_equal(output[20], output[6])
    np.testing.assert_array_equal(output[21], output[7])
    assert np.all((0.0 <= output[17]) & (output[17] <= 100.0))
    assert np.all((0.0 <= output[18]) & (output[18] <= 100.0))
