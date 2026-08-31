"""Cone response metrics shared by every weather-model probe.

Response magnitude |M(x + eps p) - M(x)| per normalized variable, great-circle
distance from the perturbation centre, cones at 100 and 300 m/s for a 6 h step,
and the cos-latitude-weighted energy fraction beyond the cone.
"""

from __future__ import annotations

import numpy as np

EARTH_R = 6371.0
ANTIPODE = np.pi * EARTH_R
CONES = {"advective_100ms": 0.100, "gravity_sound_300ms": 0.300}
STEP_HOURS = 6.0


def cone_radius_km(speed_kms):
    return speed_kms * 3600.0 * STEP_HOURS


def leakage_metrics(resp, dist, warea):
    """resp,(H,W) response magnitude; dist,(H,W) gc-km; warea,(H,W) cos-lat wts."""
    peak = float(resp.max())
    we = warea * resp**2
    tot_e = float(we.sum())
    out = {"peak": peak, "antipode_km": ANTIPODE}
    # reach at two thresholds: 1e-2 (genuine signal) vs 1e-3 (sensitive)
    for thr in (1e-2, 1e-3):
        mask = resp > thr * peak
        key = f"thr{thr:g}"
        out[f"max_reach_km_{key}"] = float(dist[mask].max()) if mask.any() else 0.0
        out[f"area_fraction_{key}"] = float((warea * mask).sum() / warea.sum())
        out[f"reaches_antipode_{key}"] = bool(mask.any() and dist[mask].max() > 0.95 * ANTIPODE)
    mask3 = resp > 1e-3 * peak
    for name, spd in CONES.items():
        cr = cone_radius_km(spd)
        outside = dist > cr
        out[f"cone_radius_{name}_km"] = float(cr)
        out[f"energy_frac_beyond_{name}"] = float(we[outside].sum() / tot_e) if tot_e > 0 else 0.0
        out[f"area_frac_signal_beyond_{name}"] = float(
            (warea * (mask3 & outside)).sum() / warea.sum()
        )
        # reach at a STRICT (1e-2) threshold restricted to outside the cone
        m2 = (resp > 1e-2 * peak) & outside
        out[f"max_reach_beyond_{name}_km_thr1e-2"] = float(dist[m2].max()) if m2.any() else 0.0
    return out


def decay_profile(resp, dist, peak=None):
    pk = peak or float(resp.max())
    bins = [0, 200, 500, 1000, 2160, 6480, 12000, 20015]
    prof = []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (dist >= a) & (dist < b)
        if m.any():
            prof.append(
                {
                    "dist_km": [a, b],
                    "median_over_peak": float(np.median(resp[m]) / pk),
                    "max_over_peak": float(resp[m].max() / pk),
                }
            )
    return prof


def cos_lat_weights(lat, nlon=1440):
    w = np.maximum(np.cos(np.deg2rad(lat)), 0.0)
    return np.repeat(w[:, None], nlon, axis=1)
