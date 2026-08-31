"""Trace retained quantitative claims to compact canonical records."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"


def load_json(relative: str):
    path = TABLES / relative

    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise AssertionError(f"duplicate JSON key {key!r} in {relative}")
            result[key] = value
        return result

    with path.open(encoding="utf-8") as handle:
        return json.load(
            handle,
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AssertionError(f"nonfinite JSON value {value!r} in {relative}")
            ),
        )


def test_existence_reliance_and_mitigation_claims():
    day1 = load_json("existence/day1_summary.json")
    assert day1["lam_1K"]["trained_fno"] == pytest.approx(0.116, abs=0.002)
    assert day1["lam_1K"]["cnn"] == pytest.approx(0.0, abs=0.001)

    reliance = load_json("consequence/causal_reliance.json")
    models = reliance["synthetic_control"]["in_distribution_J12"]
    assert models["FNO_leaky"]["degradation_x"] == pytest.approx(415.0, rel=0.02)
    assert models["CNN_causal"]["degradation_x"] == pytest.approx(1.0, rel=1e-6)

    mitigation = load_json("mitigation/mitigation_2d_full.json")
    summary = {row["kind"]: row for row in mitigation["summary"]}
    assert summary["fno_global"]["leak_corr"] == pytest.approx(0.050, abs=0.002)
    assert summary["local_cnn"]["leak_corr"] == pytest.approx(0.0, abs=0.001)


def headline_ood(kind):
    """The headline comparison is four initializations on one fixed data draw, two per device."""
    records = [
        load_json("mitigation/mitigation_2d_full.json"),
        load_json("robustness/mitigation_full_scale_mps_seeds_2_3.json"),
    ]
    values = [row["ood_mse"] for record in records for row in record["rows"] if row["kind"] == kind]
    assert len(values) == 4
    return statistics.fmean(values), statistics.stdev(values)


def test_headline_two_dimensional_comparison_uses_four_initializations():
    fno_mean, fno_sd = headline_ood("fno_global")
    local_mean, local_sd = headline_ood("local_cnn")
    assert fno_mean == pytest.approx(0.663, abs=0.0005)
    assert fno_sd == pytest.approx(0.008, abs=0.0005)
    assert local_mean == pytest.approx(0.032, abs=0.0005)
    assert local_sd == pytest.approx(0.004, abs=0.0005)
    assert fno_mean / local_mean == pytest.approx(20.6, abs=0.05)

    # Stored n_params counts each complex spectral entry once, so the spectral arm needs its
    # second real component restored before the two arms can be compared.
    mitigation = load_json("mitigation/mitigation_2d_full.json")
    rows = {row["kind"]: row for row in mitigation["rows"]}
    complex_entries = (
        mitigation["n_layers"] * 2 * mitigation["width"] ** 2 * mitigation["modes"] ** 2
    )
    fno_real = rows["fno_global"]["n_params"] + complex_entries
    local_real = rows["local_cnn"]["n_params"]
    assert fno_real == 4200769
    assert local_real == 152065
    assert fno_real / local_real == pytest.approx(27.6, abs=0.05)

    with (TABLES / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = {row["metric"]: float(row["value"]) for row in csv.DictReader(handle)}
    assert summary["headline_fno_ood_mse"] == pytest.approx(fno_mean, rel=1e-12)
    assert summary["headline_fno_ood_mse_sd"] == pytest.approx(fno_sd, rel=1e-12)
    assert summary["headline_local_ood_mse"] == pytest.approx(local_mean, rel=1e-12)
    assert summary["headline_local_ood_mse_sd"] == pytest.approx(local_sd, rel=1e-12)
    assert summary["headline_fno_real_parameters"] == fno_real
    assert summary["headline_local_real_parameters"] == local_real
    assert summary["real_parameter_ratio_fno_to_local"] == pytest.approx(
        fno_real / local_real, rel=1e-12
    )


def test_operator_released_checkpoint_and_local_reference_populations_stay_distinct():
    prediction = load_json("prediction/predictive_summary.json")
    assert len(prediction["rows"]) == 26
    assert prediction["full"]["lambda_1K"]["vs_rmse"]["n"] == 26
    assert prediction["auroc"]["jac_rho"] == pytest.approx(0.988, abs=0.002)
    assert prediction["auroc"]["lambda_1K"] == pytest.approx(0.213, abs=0.002)

    with (TABLES / "deployed" / "deployed_fleet.csv").open(encoding="utf-8", newline="") as handle:
        fleet = list(csv.DictReader(handle))
    released = [row for row in fleet if row["record_role"] == "released_checkpoint"]
    local_references = [row for row in fleet if row["record_role"] == "local_reference"]
    assert [row["model"] for row in released] == [
        "Stormer",
        "FourCastNet v1",
        "ACE2-SOM",
        "Pangu-Weather",
        "GenCast",
        "Aurora",
    ]
    assert [row["model"] for row in local_references] == ["local CNN"]
    assert len(released) == 6
    assert len(local_references) == 1
    assert len(fleet) == len(released) + len(local_references)
    by_model = {row["model"]: row for row in fleet}
    assert by_model["Aurora"]["n_cells"] == "2"
    assert by_model["ACE2-SOM"]["n_cells"] == "1"

    # The released checkpoints are reported as a two-group ordering. The local reference
    # is a separate control, and the withdrawn single-ratio span must not be reconstructed.
    acausal = [row for row in released if row["verdict"] == "acausal"]
    near_floor = [row for row in released if row["verdict"] == "near-floor"]
    assert len(acausal) == 2
    assert len(near_floor) == 4
    assert min(float(row["leakage_mean_pct"]) for row in acausal) > max(
        float(row["leakage_mean_pct"]) for row in near_floor
    )
    assert local_references[0]["verdict"] == "near-floor"

    with (TABLES / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = {row["metric"]: row for row in csv.DictReader(handle)}
    assert "deployed_leakage_span" not in summary
    assert "deployed_checkpoint_count" not in summary
    assert int(summary["released_checkpoint_count"]["value"]) == len(released)
    assert int(summary["local_reference_count"]["value"]) == len(local_references)
    assert float(summary["released_acausal_group_minimum"]["value"]) == min(
        float(row["leakage_mean_pct"]) for row in acausal
    )
    assert float(summary["released_near_floor_group_maximum"]["value"]) == max(
        float(row["leakage_mean_pct"]) for row in near_floor
    )


def test_local_reference_row_traces_to_its_measured_cells():
    with (TABLES / "deployed" / "deployed_fleet.csv").open(encoding="utf-8", newline="") as handle:
        row = {item["model"]: item for item in csv.DictReader(handle)}["local CNN"]
    reference = load_json("deployed/local_conv_reference.json")

    assert row["record_role"] == "local_reference"
    assert reference["n_cells"] == len(reference["cells"]) == 4
    cells = [cell["energy_frac_beyond_gravity_sound_300ms"] for cell in reference["cells"]]
    assert reference["energy_frac_beyond_300ms_mean"] == pytest.approx(
        sum(cells) / len(cells), rel=1e-12
    )
    assert reference["leakage_mean_pct"] == pytest.approx(
        100.0 * reference["energy_frac_beyond_300ms_mean"], rel=1e-12
    )

    # The shipped row is the measurement, rounded, and not the withdrawn constant.
    assert row["n_cells"] == "30"
    assert float(row["leakage_mean_pct"]) == pytest.approx(0.4621560638770461, abs=0.005)
    assert float(row["corrected_excess_mean_pct"]) == pytest.approx(0.00486952718347311, abs=0.0005)
    assert float(row["leakage_mean_pct"]) != reference["retired_definitional_value_pct"]

    # The withdrawn constant is unattainable: it sits below the probe's own t = 0
    # out-of-cone footprint, and the measurement sits above it.
    floor_pct = 100.0 * reference["protocol_causal_floor_fraction"]
    assert reference["retired_definitional_value_pct"] < floor_pct
    assert reference["leakage_mean_pct"] > floor_pct


def test_compiled_reach_record_is_internally_consistent():
    record = load_json("deployed/compiled_reach_pangu.json")
    cone = record["physical_cones"]["300_ms_24h"]
    assert cone["radius_km"] == pytest.approx(
        cone["speed_ms"] * cone["step_hours"] * 3600.0 / 1000.0
    )
    assert cone["radius_km"] > record["antipode_km"]
    assert cone["area_fraction"] == 1.0
    assert record["compiler"]["forward_passes"] == 0
    assert record["checkpoint"]["redistributed_here"] is False
    assert record["antipode_km"] == pytest.approx(math.pi * record["earth_radius_km"], rel=1e-12)

    sites = {site["site"]: site for site in record["sites"]}
    assert len(sites) == 4
    for site in sites.values():
        assert 0.0 < site["compiled_area_fraction"] <= 1.0
        assert site["n_cells_reachable"] <= record["grid"]["n_cells_total"]
        assert site["compiled_reach_km"] <= record["antipode_km"]
        assert site["certified_zero_area_fraction"] == pytest.approx(
            1.0 - site["compiled_area_fraction"], abs=1e-15
        )
    fractions = [site["certified_zero_area_fraction"] for site in sites.values()]
    assert record["certified_zero_area_fraction_min"] == pytest.approx(min(fractions), rel=1e-12)
    assert record["certified_zero_area_fraction_max"] == pytest.approx(max(fractions), rel=1e-12)


def test_deployed_response_census_rederives_current_rows():
    census = load_json("deployed/deployed_response_census.json")
    records = {record["model"]: record for record in census["records"]}
    assert set(records) == {"Stormer", "Pangu-Weather", "GenCast", "Aurora"}
    with (TABLES / "deployed" / "deployed_fleet.csv").open(encoding="utf-8", newline="") as handle:
        fleet = {row["model"]: row for row in csv.DictReader(handle)}
    for model, record in records.items():
        cells = record["energy_fraction_cells"]
        assert record["n_cells"] == len(cells)
        assert record["mean"] == pytest.approx(sum(cells) / len(cells), rel=1e-15)
        assert float(fleet[model]["leakage_mean_pct"]) == pytest.approx(
            100.0 * record["mean"], abs=0.05
        )
        assert fleet[model]["n_cells"] == str(record["n_cells"])
    assert records["GenCast"]["sampler_scope"].startswith("released full diffusion sampler")
    assert records["GenCast"]["numerical_control_status"] == "failed"


def test_equal_width_controls_win_every_draw_at_lower_mixing_cost():
    one_d = load_json("controls/locality_control_1d.json")
    analysis = one_d["experiment_a_analysis"]
    systems = sorted(key for key in analysis if key != "conclusions")
    assert systems == ["advection", "burgers", "heterogeneous_wave"]
    arms = ("local_true_width", "local_rule_width")
    for arm in arms:
        paired = [analysis[system]["paired_vs_global"][arm] for system in systems]
        assert [item["wins_ties_losses"] for item in paired] == [[12, 0, 0]] * len(systems)
        assert sum(item["n_paired_draws"] for item in paired) == 36

    # Width is matched across arms, so the spatial-mixing cost follows the mixing extent alone.
    architectures = one_d["architectures"]["experiment_a"]
    extents = {
        (system, arm): architectures[system][arm]["spatial_weight_shape"][-1]
        for system in systems
        for arm in ("global_width",) + arms
    }
    assert {architectures[system][arm]["width"] for system, arm in extents} == {5}
    ratios = [
        extents[(system, "global_width")] / extents[(system, arm)]
        for system in systems
        for arm in arms
    ]
    assert round(min(ratios), 1) == 3.6
    assert round(max(ratios), 1) == 10.7

    two_d = load_json("controls/locality_control_2d.json")
    summaries = two_d["arm_summaries"]
    assert [round(item, 3) for item in summaries["global_width32"]["ood_mse_t95"]] == [
        0.655,
        0.654,
        0.655,
    ]
    assert [round(item, 4) for item in summaries["local_true_width32"]["ood_mse_t95"]] == [
        0.0747,
        0.0730,
        0.0764,
    ]
    assert [round(item, 3) for item in summaries["local_nonoracle_width32"]["ood_mse_t95"]] == [
        0.246,
        0.244,
        0.247,
    ]
    assert [item["wins_ties_losses"] for item in two_d["comparisons"]] == [[12, 0, 0]] * 2
    global_real = summaries["global_width32"]["n_params"] + 4 * 2 * 32**2 * 16**2
    assert round(global_real / summaries["local_true_width32"]["n_params"]) == 107


def test_static_support_census_and_reanalysis_records():
    census = load_json("deployed/static_support_census.json")
    rows = census["rows"]
    assert len(rows) == 12
    assert sum(row["global_at_native_lead"] for row in rows) == census["native_global_count"] == 10
    assert (
        sum(row["global_at_harmonized_24h"] for row in rows)
        == census["harmonized_24h_global_count"]
        == 11
    )
    assert [row["model"] for row in rows if not row["global_at_harmonized_24h"]] == [
        census["harmonized_24h_only_blind_model"]
    ]

    resources = load_json("era5/resource_accounting.json")
    assert resources["global"]["real_parameters"] == 37757126
    widths = {row["width"]: row for row in resources["local"]}
    assert widths[720]["real_parameters"] == 18727958
    assert widths[96]["real_parameters"] == 340550
    assert round(widths[720]["forward_flops"] / 1e9, 1) == 76.7
    assert round(resources["global"]["forward_flops"] / 1e9, 3) == 0.224

    era5 = load_json("era5/six_hour_z500.json")
    assert era5["lead_hours"] == 6
    arms = era5["arms"]
    assert round(arms["global_spectral_w96"]["mean_rmse_m"], 3) == 8.738
    assert round(arms["local_cone_r10_w720"]["mean_rmse_m"], 3) == 5.541
    assert round(arms["local_cone_r10_w96"]["mean_rmse_m"], 3) == 8.488
    for arm, quoted, interval, decimals in (
        ("local_cone_r10_w720", 36.6, [35.7, 37.6], 1),
        ("local_cone_r10_w96", 2.85, [2.56, 3.16], 2),
    ):
        contrast = era5["contrasts_vs_reference"][arm]
        assert round(100.0 * contrast["relative_improvement_mean"], decimals) == quoted
        assert [
            round(100.0 * bound, decimals) for bound in contrast["relative_improvement_interval95"]
        ] == interval
        assert contrast["wins_ties_losses"] == [8, 0, 0]
        assert len(contrast["relative_improvement_by_block"]) == era5["n_blocks"] == 8


def test_theory_records_are_finite_and_retain_checks():
    sharp = load_json("theory/sharp_cutoff_law_verification.json")
    rect = sharp["windows"]["rect"]["128"]
    assert rect["power_alpha"] == pytest.approx(1.0, abs=0.01)
    assert rect["power_C"] == pytest.approx(1.0 / math.pi, abs=0.01)
    assert sharp["no_weighting_beats_envelope"]["|R_edge|=0.00"]["C_over_edge"] is None

    rollout = load_json("theory/rollout_certificate_verification.json")
    assert rollout["all_pass"] is True
    assert rollout["auroc_L"] == pytest.approx(1.0)
    assert rollout["auroc_leak_fraction"] == pytest.approx(0.056, abs=0.002)

    tail = load_json("theory/higher_dimensional_tail.json")
    assert tail["all_pass"] is True
    assert tail["max_abs_amplitude_ratio_error"] < 1e-4
    assert tail["max_abs_exponent_error"] < 1e-3
    assert tail["max_quadrature_rel_error_vs_closed_form"] < 1e-12


def test_horizon_by_capacity_and_matched_skill_records():
    ladder = load_json("era5/horizon_by_capacity.json")["contrasts_vs_reference"]
    quoted = {
        "local_cone_r10_w96": [2.85, 2.62, -28.7, -103, -195],
        "local_cone_r10_w264": [27.1, 30.3, 25.7, 16.7, -19.8],
        "local_cone_r10_w720": [36.6, 41.1, 36.3, 29.5, 21.3],
    }
    for arm, values in quoted.items():
        for lead, value in zip(("6", "24", "72", "120", "240"), values, strict=True):
            observed = 100.0 * ladder[arm][lead]["relative_improvement_mean"]
            assert observed == pytest.approx(value, abs=0.6 if abs(value) >= 100 else 0.06)
    assert all(
        ladder["local_cone_r10_w720"][lead]["wins_ties_losses"] == [8, 0, 0]
        for lead in ladder["local_cone_r10_w720"]
    )
    assert ladder["local_cone_r10_w264"]["240"]["wins_ties_losses"] == [1, 0, 7]
    assert ladder["local_cone_r10_w96"]["72"]["wins_ties_losses"] == [0, 0, 8]

    ratios = load_json("consequence/matched_skill_shifted_band.json")[
        "ratios_fno_full_over_local_causal"
    ]
    assert ratios["indist_one_step_nrmse"]["geomean"] == pytest.approx(0.99, abs=0.005)
    band = ratios["shift_band_one_step_nrmse"]
    assert round(band["geomean"], 1) == 18.3
    assert [round(band["ci_lo"], 1), round(band["ci_hi"], 1)] == [17.5, 19.1]
    assert band["units_local_better"] == band["n"] == 10
    assert (
        ratios["rollout_shift_slope"]["8"]["ci_lo"]
        > 1.0
        > ratios["rollout_shift_slope"]["16"]["ci_hi"]
    )


def test_horizon_record_block_errors_reproduce_stored_improvements():
    record = load_json("era5/horizon_by_capacity.json")
    errors = record["block_rmse_m"]
    reference = errors[record["reference_arm"]]
    assert len(record["block_order"]) == record["n_blocks"] == 8
    for arm, by_lead in record["contrasts_vs_reference"].items():
        for lead, contrast in by_lead.items():
            per_block = [
                1.0 - local / glob
                for local, glob in zip(errors[arm][lead], reference[lead], strict=True)
            ]
            assert per_block == pytest.approx(contrast["relative_improvement_by_block"], abs=1e-12)
            assert sum(per_block) / len(per_block) == pytest.approx(
                contrast["relative_improvement_mean"], abs=1e-12
            )
    six_hour = {arm: sum(errors[arm]["6"]) / 8 for arm in errors}
    assert round(six_hour["global_spectral_w96"], 3) == 8.738
    assert round(six_hour["local_cone_r10_w720"], 3) == 5.541
