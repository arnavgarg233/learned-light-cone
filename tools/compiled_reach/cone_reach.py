#!/usr/bin/env python3
"""Compile one released checkpoint's causal reach and print it against the physical cone.

Static reachability over the released ONNX operator sequence. Zero forward passes: every
weight entry is treated as structurally nonzero, so the compiled dependency set is an upper
bound on the one-step domain of dependence and is read from the graph rather than from the
trained values.

Usage:
    python tools/compiled_reach/cone_reach.py --checkpoint /path/to/pangu_weather_24.onnx \
        --step-hours 24

The deterministic fields are compared against the record retained at
results/tables/deployed/compiled_reach_pangu.json, and a mismatch exits nonzero.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import onnx_skim as osk
import reach

EARTH_R_KM = 6371.0
ANTIPODE_KM = float(math.pi * EARTH_R_KM)
REPOSITORY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RECORD = os.path.join(
    REPOSITORY_ROOT, "results", "tables", "deployed", "compiled_reach_pangu.json"
)

SITES = {
    "pacific": (0.0, 200.0),
    "north_atlantic": (45.0, 330.0),
    "southern_ocean": (-55.0, 90.0),
    "siberia": (65.0, 100.0),
}

LAT = np.linspace(90.0, -90.0, 721)
LON = np.arange(1440) * 0.25

COMPARED = (
    "compiled_reach_km",
    "compiled_reach_deg",
    "n_cells_reachable",
    "compiled_area_fraction",
    "ratio_compiled_over_cone",
)


def haversine_km(lat0, lon0, lat_grid, lon_grid):
    p0, l0 = np.deg2rad(lat0), np.deg2rad(lon0)
    p1, l1 = np.deg2rad(lat_grid), np.deg2rad(lon_grid)
    a = (
        np.sin((p1 - p0) / 2.0) ** 2
        + np.cos(p0) * np.cos(p1) * np.sin((l1 - l0) / 2.0) ** 2
    )
    return 2.0 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def cone_area_fraction(radius_km: float) -> float:
    """Exact spherical-cap area fraction for a cone of the given radius."""
    if radius_km >= ANTIPODE_KM:
        return 1.0
    return (1.0 - math.cos(radius_km / EARTH_R_KM)) / 2.0


def compare_with_record(record_path: str, site: str, observed: dict) -> int:
    """Return 0 when every compared field equals the retained record for this site."""
    if not os.path.isfile(record_path):
        print(f"  no retained record at {record_path}, comparison skipped")
        print("")
        return 0
    with open(record_path, encoding="utf-8") as handle:
        record = json.load(handle)
    rows = {row["site"]: row for row in record["sites"]}
    if site not in rows:
        print(f"  site {site} is not in the retained record, comparison skipped")
        print("")
        return 0
    expected = rows[site]
    differences = [
        (field, observed[field], expected[field])
        for field in COMPARED
        if not (
            observed[field] == expected[field]
            or (
                isinstance(observed[field], float)
                and isinstance(expected[field], float)
                and math.isclose(observed[field], expected[field], rel_tol=1e-12, abs_tol=1e-12)
            )
        )
    ]
    if differences:
        print("  DIFFERS from the retained record:")
        for field, got, want in differences:
            print(f"    {field}: computed {got!r}, retained {want!r}")
        print("")
        return 1
    print(f"  retained record         matches all {len(COMPARED)} compared fields")
    print(f"                          {os.path.relpath(record_path, REPOSITORY_ROOT)}")
    print("")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="path to a released ONNX checkpoint")
    parser.add_argument("--site", default="pacific", choices=sorted(SITES))
    parser.add_argument("--cone-speed-ms", type=float, default=300.0)
    parser.add_argument("--step-hours", type=float, default=24.0)
    parser.add_argument("--record", default=DEFAULT_RECORD)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    clat, clon = SITES[args.site]
    i = int(np.argmin(np.abs(LAT - clat)))
    j = int(np.argmin(np.abs(LON - (clon % 360.0))))

    wall0 = time.time()
    graph = osk.load(args.checkpoint)
    parse_s = time.time() - wall0
    graph_nodes = len(graph.nodes)

    interp = reach.Interp(graph, (i, j))
    t0 = time.time()
    outs = interp.run()
    compile_s = time.time() - t0

    mask = np.zeros((721, 1440), dtype=bool)
    for value in outs.values():
        dense = reach.dep_full(value.dep, value.shape)
        while dense.ndim > 2:
            dense = dense.any(axis=0)
        mask |= dense
    graph.close()

    lat2d, lon2d = np.meshgrid(LAT, LON, indexing="ij")
    dist = haversine_km(float(LAT[i]), float(LON[j]), lat2d, lon2d)
    weight = np.maximum(np.cos(np.deg2rad(lat2d)), 0.0)

    n_cells = int(mask.sum())
    area_frac = float((weight * mask).sum() / weight.sum())
    reach_km = float(dist[mask].max()) if n_cells else 0.0
    reach_deg = reach_km / EARTH_R_KM * 180.0 / math.pi

    cone_radius_km = args.cone_speed_ms * args.step_hours * 3600.0 / 1000.0
    cone_frac = cone_area_fraction(cone_radius_km)
    total_s = time.time() - wall0

    print("")
    print("  compiled causal reach of a released checkpoint, zero forward passes")
    print("  " + "-" * 68)
    print(f"  checkpoint              {os.path.basename(args.checkpoint)}")
    print(f"  graph nodes             {graph_nodes:,}")
    print(f"  source site             {args.site} ({clat:g} lat, {clon:g} lon)")
    print("")
    print(f"  compiled reach          {reach_km:,.6f} km  ({reach_deg:.4f} deg)")
    print(f"  antipodal distance      {ANTIPODE_KM:,.6f} km")
    print(
        f"  compiled support        {area_frac * 100:.2f}% of the globe by area"
        f"  ({n_cells:,} of {mask.size:,} cells)"
    )
    print(f"  certified-zero area     {(1.0 - area_frac) * 100:.2f}% of the globe by area")
    print(
        f"  physical cone           {cone_frac * 100:.2f}% of the globe by area"
        f"  ({cone_radius_km:,.0f} km at {args.cone_speed_ms:g} m/s over"
        f" {args.step_hours:g} h)"
    )
    print(f"  ratio                   {area_frac / cone_frac:.2f}x the physical cone")
    print("")
    print("  model forward passes    0")
    print(
        f"  wall clock              {total_s:.2f} s"
        f"  (parse {parse_s:.2f} s, compile {compile_s:.2f} s)"
    )
    print("")

    record = {
        "checkpoint": os.path.basename(args.checkpoint),
        "graph_nodes": graph_nodes,
        "site": args.site,
        "site_lat": clat,
        "site_lon": clon,
        "compiled_reach_km": reach_km,
        "compiled_reach_deg": reach_deg,
        "antipode_km": ANTIPODE_KM,
        "n_cells_reachable": n_cells,
        "n_cells_total": int(mask.size),
        "compiled_area_fraction": area_frac,
        "cone_radius_km": cone_radius_km,
        "cone_area_fraction": cone_frac,
        "ratio_compiled_over_cone": area_frac / cone_frac,
        "forward_passes": 0,
        "seconds_total": total_s,
        "seconds_parse": parse_s,
        "seconds_compile": compile_s,
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
    }
    status = compare_with_record(args.record, args.site, record) if args.record else 0
    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, sort_keys=True)
        print(f"  wrote {args.json_out}")
        print("")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
