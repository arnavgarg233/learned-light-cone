"""Validated ERA5-to-FourCastNet channel transform without I/O side effects."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .afno_model import VARIABLES

ARCO_ERA5_URI = (
    "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
)

SURFACE_CHANNELS = {
    0: "10m_u_component_of_wind",
    1: "10m_v_component_of_wind",
    2: "2m_temperature",
    3: "surface_pressure",
    4: "mean_sea_level_pressure",
    19: "total_column_water_vapour",
}

PRESSURE_CHANNELS = {
    5: ("temperature", 850),
    6: ("u_component_of_wind", 1000),
    7: ("v_component_of_wind", 1000),
    8: ("geopotential", 1000),
    9: ("u_component_of_wind", 850),
    10: ("v_component_of_wind", 850),
    11: ("geopotential", 850),
    12: ("u_component_of_wind", 500),
    13: ("v_component_of_wind", 500),
    14: ("geopotential", 500),
    15: ("temperature", 500),
    16: ("geopotential", 50),
    22: ("u_component_of_wind", 250),
    23: ("v_component_of_wind", 250),
    24: ("geopotential", 250),
    25: ("temperature", 250),
}


def bolton_relative_humidity(
    specific_humidity: np.ndarray,
    temperature_kelvin: np.ndarray,
    pressure_pa: float,
) -> np.ndarray:
    """Return relative humidity in percent using the validated Bolton transform."""
    q = np.asarray(specific_humidity, dtype=np.float32)
    temperature = np.asarray(temperature_kelvin, dtype=np.float32)
    vapour_pressure = q * pressure_pa / (0.622 + 0.378 * q)
    saturation = 611.2 * np.exp(17.67 * (temperature - 273.15) / (temperature - 29.65))
    return np.clip(100.0 * vapour_pressure / saturation, 0.0, 100.0).astype(np.float32)


def assemble_fourcastnet_channels(
    surface: Mapping[str, np.ndarray],
    pressure: Mapping[tuple[str, int], np.ndarray],
    *,
    target_latitudes: int = 720,
) -> np.ndarray:
    """Assemble the 26-channel state and drop the final pole row.

    ``surface`` maps ERA5 two-dimensional variable names to latitude-longitude
    arrays. ``pressure`` maps ``(variable, pressure_hpa)`` to arrays. The transform
    follows the banked validation: retain the first 720 latitude rows, derive
    relative humidity from specific humidity, temperature, and pressure, and use
    1000 hPa winds for the two 100 m wind channels.
    """
    sample = np.asarray(next(iter(surface.values())))
    if sample.ndim != 2 or sample.shape[0] < target_latitudes:
        raise ValueError("surface fields must be latitude-longitude arrays")
    nlon = sample.shape[1]
    output = np.zeros((len(VARIABLES), target_latitudes, nlon), dtype=np.float32)

    def trimmed(value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != sample.shape:
            raise ValueError("all ERA5 fields must share one grid")
        return array[:target_latitudes]

    for index, name in SURFACE_CHANNELS.items():
        output[index] = trimmed(surface[name])
    for index, key in PRESSURE_CHANNELS.items():
        output[index] = trimmed(pressure[key])

    output[20] = output[6]
    output[21] = output[7]
    q500 = trimmed(pressure[("specific_humidity", 500)])
    q850 = trimmed(pressure[("specific_humidity", 850)])
    output[17] = bolton_relative_humidity(q500, output[15], 50_000.0)
    output[18] = bolton_relative_humidity(q850, output[5], 85_000.0)
    return output
