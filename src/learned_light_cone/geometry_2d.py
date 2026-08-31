"""2D geometry helpers for lat-lon real-emulator diagnostics."""

from __future__ import annotations

import numpy as np


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: np.ndarray,
    lon2: np.ndarray,
    earth_radius_km: float = 6371.0,
) -> np.ndarray:
    """Return great-circle distance in kilometers.

    lat/lon should be in degrees. lat2 and lon2 can be 2D broadcastable grids.
    """
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    lambda1, lambda2 = np.deg2rad(lon1), np.deg2rad(lon2)

    dphi = phi2 - phi1
    dlambda = lambda2 - lambda1

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return earth_radius_km * c


def cone_masks_2d(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    center_lat: float,
    center_lon: float,
    speed_km_per_step: float,
    radius0_km: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return inside/outside physical-cone masks for steps ``0..steps``.

    The masks have shape ``(steps + 1, height, width)``.
    """
    distance = haversine_distance(center_lat, center_lon, lat_grid, lon_grid)
    inside = np.zeros((steps + 1,) + distance.shape, dtype=bool)
    for k in range(steps + 1):
        inside[k] = distance <= (radius0_km + speed_km_per_step * k)
    return inside, ~inside
