#!/usr/bin/env python3
"""Build compact claim, robustness, and scope records from canonical evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def load_strict(path: Path):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open(encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON value {value!r} in {path}")
            ),
        )


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def value(number: float | int) -> str:
    return str(number) if isinstance(number, int) else repr(float(number))


def mitigation_ratios(document):
    grouped = {}
    for row in document["rows"]:
        grouped.setdefault(row["seed"], {})[row["kind"]] = row
    return [
        grouped[seed]["local_cnn"]["ood_mse"] / grouped[seed]["fno_global"]["ood_mse"]
        for seed in sorted(grouped)
    ]


def headline_ood(documents, kind):
    """Pool the four headline initializations, two per device record, for one operator kind."""
    values = [
        row["ood_mse"] for document in documents for row in document["rows"] if row["kind"] == kind
    ]
    return statistics.fmean(values), statistics.stdev(values)


def real_parameters(document, kind):
    """Stored n_params counts each complex spectral entry once, so restore its second
    real component for the spectral arm; the convolutional arm is already real."""
    count = {row["kind"]: row["n_params"] for row in document["rows"]}[kind]
    if kind != "fno_global":
        return count
    complex_entries = document["n_layers"] * 2 * document["width"] ** 2 * document["modes"] ** 2
    return count + complex_entries


def equal_width_wins(document, arm):
    analysis = document["experiment_a_analysis"]
    systems = sorted(key for key in analysis if key != "conclusions")
    return sum(
        analysis[system]["paired_vs_global"][arm]["wins_ties_losses"][0] for system in systems
    )


def build(input_dir: Path, output_dir: Path) -> None:
    day1 = load_strict(input_dir / "existence" / "day1_summary.json")
    reliance = load_strict(input_dir / "consequence" / "causal_reliance.json")
    mitigation = load_strict(input_dir / "mitigation" / "mitigation_2d_full.json")
    full = load_strict(input_dir / "robustness" / "mitigation_full_scale_mps_seeds_2_3.json")
    locality_1d = load_strict(input_dir / "controls" / "locality_control_1d.json")
    era5 = load_strict(input_dir / "era5" / "six_hour_z500.json")
    prediction = load_strict(input_dir / "prediction" / "predictive_summary.json")
    with (input_dir / "deployed" / "deployed_fleet.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        fleet = list(csv.DictReader(handle))
    released_checkpoints = [row for row in fleet if row["record_role"] == "released_checkpoint"]
    local_references = [row for row in fleet if row["record_role"] == "local_reference"]
    if len(released_checkpoints) + len(local_references) != len(fleet):
        raise ValueError("deployed fleet contains an unknown record_role")

    headline_records = [mitigation, full]
    fno_ood_mean, fno_ood_sd = headline_ood(headline_records, "fno_global")
    local_ood_mean, local_ood_sd = headline_ood(headline_records, "local_cnn")
    fno_real_params = real_parameters(mitigation, "fno_global")
    local_real_params = real_parameters(mitigation, "local_cnn")
    headline_source = (
        "mitigation/mitigation_2d_full.json;robustness/mitigation_full_scale_mps_seeds_2_3.json"
    )
    reliance_rows = reliance["synthetic_control"]["in_distribution_J12"]
    summary_rows = [
        {
            "metric": "trained_fno_leakage",
            "value": value(day1["lam_1K"]["trained_fno"]),
            "unit": "fraction",
            "source": "existence/day1_summary.json",
        },
        {
            "metric": "local_cnn_leakage",
            "value": value(day1["lam_1K"]["cnn"]),
            "unit": "fraction",
            "source": "existence/day1_summary.json",
        },
        {
            "metric": "fno_masking_degradation",
            "value": value(reliance_rows["FNO_leaky"]["degradation_x"]),
            "unit": "ratio",
            "source": "consequence/causal_reliance.json",
        },
        {
            "metric": "local_masking_degradation",
            "value": value(reliance_rows["CNN_causal"]["degradation_x"]),
            "unit": "ratio",
            "source": "consequence/causal_reliance.json",
        },
        {
            "metric": "headline_fno_ood_mse",
            "value": value(fno_ood_mean),
            "unit": "mse",
            "source": headline_source,
        },
        {
            "metric": "headline_fno_ood_mse_sd",
            "value": value(fno_ood_sd),
            "unit": "mse",
            "source": headline_source,
        },
        {
            "metric": "headline_local_ood_mse",
            "value": value(local_ood_mean),
            "unit": "mse",
            "source": headline_source,
        },
        {
            "metric": "headline_local_ood_mse_sd",
            "value": value(local_ood_sd),
            "unit": "mse",
            "source": headline_source,
        },
        {
            "metric": "headline_fno_real_parameters",
            "value": value(fno_real_params),
            "unit": "parameters",
            "source": "mitigation/mitigation_2d_full.json",
        },
        {
            "metric": "headline_local_real_parameters",
            "value": value(local_real_params),
            "unit": "parameters",
            "source": "mitigation/mitigation_2d_full.json",
        },
        {
            "metric": "real_parameter_ratio_fno_to_local",
            "value": value(fno_real_params / local_real_params),
            "unit": "ratio",
            "source": "mitigation/mitigation_2d_full.json",
        },
        {
            "metric": "equal_width_1d_draws_won",
            "value": value(equal_width_wins(locality_1d, "local_true_width")),
            "unit": "draws",
            "source": "controls/locality_control_1d.json",
        },
        {
            "metric": "era5_six_hour_improvement_width720",
            "value": value(
                era5["contrasts_vs_reference"]["local_cone_r10_w720"]["relative_improvement_mean"]
            ),
            "unit": "fraction",
            "source": "era5/six_hour_z500.json",
        },
        {
            "metric": "era5_six_hour_improvement_width96",
            "value": value(
                era5["contrasts_vs_reference"]["local_cone_r10_w96"]["relative_improvement_mean"]
            ),
            "unit": "fraction",
            "source": "era5/six_hour_z500.json",
        },
        {
            "metric": "trained_operator_count",
            "value": value(len(prediction["rows"])),
            "unit": "operators",
            "source": "prediction/predictive_summary.json",
        },
        {
            "metric": "released_checkpoint_count",
            "value": value(len(released_checkpoints)),
            "unit": "checkpoints",
            "source": "deployed/deployed_fleet.csv",
        },
        {
            "metric": "local_reference_count",
            "value": value(len(local_references)),
            "unit": "references",
            "source": "deployed/deployed_fleet.csv",
        },
        {
            "metric": "jacobian_spectral_radius_auroc",
            "value": value(prediction["auroc"]["jac_rho"]),
            "unit": "auroc",
            "source": "prediction/predictive_summary.json",
        },
        {
            "metric": "leakage_fraction_auroc",
            "value": value(prediction["auroc"]["lambda_1K"]),
            "unit": "auroc",
            "source": "prediction/predictive_summary.json",
        },
        {
            "metric": "released_acausal_group_minimum",
            "value": value(
                min(
                    float(row["leakage_mean_pct"])
                    for row in released_checkpoints
                    if row["verdict"] == "acausal"
                )
            ),
            "unit": "percent",
            "source": "deployed/deployed_fleet.csv",
        },
        {
            "metric": "released_near_floor_group_maximum",
            "value": value(
                max(
                    float(row["leakage_mean_pct"])
                    for row in released_checkpoints
                    if row["verdict"] == "near-floor"
                )
            ),
            "unit": "percent",
            "source": "deployed/deployed_fleet.csv",
        },
    ]
    write_csv(
        output_dir / "summary.csv",
        ["metric", "value", "unit", "source"],
        summary_rows,
    )

    mid = load_strict(input_dir / "robustness" / "mitigation_mid_scale_multiseed.json")
    fourcastnet = load_strict(input_dir / "robustness" / "fourcastnet_ic_region_dispersion.json")
    local = load_strict(input_dir / "robustness" / "local_operator_ic_region_dispersion.json")
    ingest = load_strict(input_dir / "robustness" / "era5_ingest_validation.json")
    mid_worst = max(mitigation_ratios(mid))
    full_worst = max(mitigation_ratios(full))
    separation = (fourcastnet["Lambda_mean"] - local["Lambda_mean"]) / math.sqrt(
        fourcastnet["Lambda_std"] ** 2 + local["Lambda_std"] ** 2
    )
    bands_separate = (
        fourcastnet["Lambda_mean"] - fourcastnet["Lambda_std"]
        > local["Lambda_mean"] + local["Lambda_std"]
    )
    robustness_rows = [
        {
            "gate": "mid_scale_ood_ratio",
            "status": "fail",
            "threshold": "<=0.10 for every seed",
            "observed": value(mid_worst),
            "scope": "five seeds at the lower-cost configuration",
        },
        {
            "gate": "full_scale_mps_ood_ratio",
            "status": "pass",
            "threshold": "<=0.10 for every seed",
            "observed": value(full_worst),
            "scope": "device-specific seeds 2 and 3",
        },
        {
            "gate": "ic_three_sigma_separation",
            "status": "fail",
            "threshold": ">=3.0 sigma",
            "observed": value(separation),
            "scope": "10 dates by 3 regions",
        },
        {
            "gate": "ic_standard_deviation_bands",
            "status": "pass" if bands_separate else "fail",
            "threshold": "non-overlap",
            "observed": str(bands_separate).lower(),
            "scope": "10 dates by 3 regions",
        },
        {
            "gate": "era5_ingest_exactness",
            "status": "pass",
            "threshold": "max_abs_diff<=0.001",
            "observed": value(ingest["max_abs_diff"]),
            "scope": "26 channels",
        },
        {
            "gate": "stormer_round_two",
            "status": "not_run",
            "threshold": "not applicable",
            "observed": "not_run",
            "scope": "no round-two evidence",
        },
        {
            "gate": "aurora_round_two",
            "status": "not_run",
            "threshold": "not applicable",
            "observed": "not_run",
            "scope": "no round-two evidence",
        },
    ]
    write_csv(
        output_dir / "robustness_summary.csv",
        ["gate", "status", "threshold", "observed", "scope"],
        robustness_rows,
    )

    claim_scope = {
        "schema_version": 1,
        "population_boundaries": {
            "controlled_trained_operators": 26,
            "released_checkpoints": len(released_checkpoints),
            "local_references": len(local_references),
            "populations_are_distinct": True,
        },
        "headline_evidence_boundary": {
            "headline_bullets": 5,
            "rely_on_frozen_external_records_without_tracked_generator": 4,
        },
        "default_replay_scope": {
            "regenerates": [
                "closed-form sharp-cutoff record",
                "closed-form rollout-certificate record",
                "compact derived tables",
                "three canonical figures",
            ],
            "compares_frozen_records": True,
            "downloads_era5": False,
            "reruns_deployed_model_inference": False,
            "retrains_mitigation_models": False,
            "loads_checkpoints": False,
        },
        "limitations": [
            "The released-checkpoint values and local-reference value are verified from frozen compact records, not regenerated by default replay.",
            "The roughly 20-fold out-of-distribution comparison is scoped to the headline configuration; the five-seed lower-cost configuration misses the frozen 0.10 ratio gate.",
            "The FourCastNet versus local comparison has non-overlapping one-standard-deviation bands but reaches 2.55 sigma, below the frozen 3-sigma threshold.",
            "Stormer and Aurora were not run in the retained robustness round.",
            "Dependence under masking does not by itself identify a causal mechanism in a deployed emulator.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "claim_scope.json").open("w", encoding="utf-8") as handle:
        json.dump(claim_scope, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    build(arguments.input_dir, arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
