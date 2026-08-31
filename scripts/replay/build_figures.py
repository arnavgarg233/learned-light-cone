#!/usr/bin/env python3
"""Render the three canonical figures from compact result records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from learned_light_cone.viz.style import COLORS, FIGURE_METADATA, apply_style


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON value {value!r} in {path}")
            ),
        )


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata=FIGURE_METADATA)
    plt.close(fig)


def attention_figure(input_dir: Path, output_dir: Path) -> None:
    with (input_dir / "attention_seeds.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = ["global" if row["window"] == "inf" else row["window"] for row in rows]
    x = np.arange(len(rows))
    leakage = np.array([float(row["leak_mean"]) for row in rows])
    leakage_std = np.array([float(row["leak_std"]) for row in rows])
    reach = np.array([float(row["reach_mean"]) for row in rows])
    reach_std = np.array([float(row["reach_std"]) for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.45))
    axes[0].errorbar(
        x,
        leakage,
        yerr=leakage_std,
        marker="o",
        capsize=2.5,
        color=COLORS["global"],
    )
    axes[0].set_ylabel("out-of-cone leakage")
    axes[0].set_title("a  Leakage")
    axes[1].errorbar(
        x,
        reach,
        yerr=reach_std,
        marker="o",
        capsize=2.5,
        color=COLORS["accent"],
    )
    axes[1].set_ylabel("response reach (cells)")
    axes[1].set_title("b  Reach")
    for axis in axes:
        axis.set_xticks(x, labels)
        axis.set_xlabel("attention window")
        axis.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    save(fig, output_dir / "fig_attention_window_sweep.pdf")


def consequence_figure(input_dir: Path, output_dir: Path) -> None:
    reliance = load_json(input_dir / "consequence" / "causal_reliance.json")
    mitigation = load_json(input_dir / "mitigation" / "mitigation_2d_full.json")
    prediction = load_json(input_dir / "prediction" / "predictive_summary.json")
    dependence = reliance["synthetic_control"]["in_distribution_J12"]
    summary = {row["kind"]: row for row in mitigation["summary"]}

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45))
    axes[0].bar(
        ["FNO", "local"],
        [
            dependence["FNO_leaky"]["degradation_x"],
            dependence["CNN_causal"]["degradation_x"],
        ],
        color=[COLORS["global"], COLORS["local"]],
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("masked / full RMSE")
    axes[0].set_title("a  Masking dependence")

    axes[1].bar(
        ["FNO", "local"],
        [summary["fno_global"]["ood_mse"], summary["local_cnn"]["ood_mse"]],
        color=[COLORS["global"], COLORS["local"]],
    )
    axes[1].set_ylabel("OOD MSE")
    axes[1].set_title("b  Mitigation")

    axes[2].bar(
        ["Jacobian", "leakage"],
        [prediction["auroc"]["jac_rho"], prediction["auroc"]["lambda_1K"]],
        color=[COLORS["accent"], COLORS["control"]],
    )
    axes[2].axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    axes[2].set_ylim(0, 1.05)
    axes[2].set_ylabel("rollout blow-up AUROC")
    axes[2].set_title("c  Prediction")
    axes[2].tick_params(axis="x", rotation=20)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    save(fig, output_dir / "fig_consequence_and_mitigation.pdf")


def theory_figure(input_dir: Path, output_dir: Path) -> None:
    sharp = load_json(input_dir / "theory" / "sharp_cutoff_law_verification.json")
    rollout = load_json(input_dir / "theory" / "rollout_certificate_verification.json")
    multilayer = load_json(input_dir / "theory" / "multilayer_tail_verification.json")

    modes = np.array([16, 32, 64, 128])
    rect = sharp["windows"]["rect"]
    alpha = np.array([rect[str(mode)]["power_alpha"] for mode in modes])
    trained_modes = np.array(sorted(map(int, multilayer["fno"])))
    trained_alpha = np.array([multilayer["fno"][str(mode)]["alpha"] for mode in trained_modes])

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45))
    axes[0].plot(modes, alpha, "o-", color=COLORS["global"])
    axes[0].axhline(1.0, color="black", linestyle=":", linewidth=0.8)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(modes, [str(mode) for mode in modes])
    axes[0].set_xlabel("hard-cutoff modes")
    axes[0].set_ylabel("tail exponent")
    axes[0].set_title("a  Sharp cutoff")

    axes[1].bar(
        ["magnitude", "fraction"],
        [rollout["auroc_L"], rollout["auroc_leak_fraction"]],
        color=[COLORS["accent"], COLORS["control"]],
    )
    axes[1].axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("toy blow-up AUROC")
    axes[1].set_title("b  Rollout certificate")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].plot(trained_modes, trained_alpha, "o-", color=COLORS["local"])
    axes[2].axhline(1.0, color="black", linestyle=":", linewidth=0.8)
    axes[2].set_xlabel("trained FNO modes")
    axes[2].set_ylabel("tail exponent")
    axes[2].set_title("c  Multilayer check")
    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    save(fig, output_dir / "fig_theory_checks.pdf")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    apply_style(matplotlib)
    attention_figure(arguments.input_dir, arguments.output_dir)
    consequence_figure(arguments.input_dir, arguments.output_dir)
    theory_figure(arguments.input_dir, arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
