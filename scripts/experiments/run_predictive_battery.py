#!/usr/bin/env python3
"""Run the retained predictive battery as an explicit secondary reproduction."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from learned_light_cone.metrics.cone import (
    fit_front_speed,
    leakage_mass,
    perturbation_probe,
)
from learned_light_cone.metrics.cone_corrections import directional_leakage
from learned_light_cone.metrics.jacobian import (
    jacobian_frobenius,
    jacobian_spectral_radius,
    spectral_gain_from_G,
)
from learned_light_cone.metrics.predictive import (
    auroc,
    dominance,
    nested_regression,
    partial_spearman,
)
from learned_light_cone.metrics.rollout import model_rollout, rollout_rmse_curve
from learned_light_cone.metrics.spectral import learned_transfer
from learned_light_cone.models.fno import FNO1d
from learned_light_cone.pdes.advection import (
    band_limited_ic,
    bump_radius,
    exact_rollout,
    make_local_bump,
)
from learned_light_cone.train.causality_reg import train_with_causality
from learned_light_cone.train.train_one_step import evaluate_one_step, train_one_step


def get_device(name: str) -> torch.device:
    device = torch.device(name)
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def make_cfg(smoke: bool) -> dict:
    if smoke:
        return {
            "N": 64,
            "J_train": 10,
            "J_ood": 18,
            "p": 1.0,
            "n_train": 256,
            "n_val": 128,
            "n_ood": 128,
            "width": 24,
            "n_layers": 4,
            "modes_sweep": [6, 10, 16],
            "doses": [0.0, 0.1],
            "seeds": [0],
            "epochs": 20,
            "lr": 1e-3,
            "eps": 1e-3,
            "K": 6,
            "q": 0.95,
            "n_base": 4,
            "H_ood": 8,
            "m": 1,
            "c": 1.0,
            "sigma": 1.0,
            "x0": 32,
            "tau": 1.0,
        }
    return {
        "N": 128,
        "J_train": 12,
        "J_ood": 24,
        "p": 1.0,
        "n_train": 2048,
        "n_val": 256,
        "n_ood": 256,
        "width": 32,
        "n_layers": 4,
        "modes_sweep": [6, 8, 10, 12, 16, 24, 32],
        "doses": [0.0, 0.01, 0.05, 0.1, 0.3, 1.0],
        "seeds": [0, 1],
        "epochs": 400,
        "lr": 1e-3,
        "eps": 1e-3,
        "K": 8,
        "q": 0.95,
        "n_base": 8,
        "H_ood": 16,
        "m": 1,
        "c": 1.0,
        "sigma": 1.0,
        "x0": 64,
        "tau": 1.0,
    }


def solver_step(m):
    return lambda x: torch.roll(x, m, dims=-1)


def make_step(model):
    model.eval()

    @torch.no_grad()
    def step(x):
        return model(x)

    return step


def measure_features(model, cfg, device, bases_t, bump_t, lam_solver_curve):
    """Return the source protocol's scalar feature battery for one model."""
    m, K, eps, q, x0 = cfg["m"], cfg["K"], cfg["eps"], cfg["q"], cfg["x0"]
    r0 = bump_radius(cfg["sigma"])
    step = make_step(model)
    bump_b = bump_t.unsqueeze(0).expand(bases_t.shape[0], -1)
    response = perturbation_probe(step, bases_t, bump_b, eps, K).cpu().numpy()
    leakage = np.array(
        [leakage_mass(response[index], x0, m, r0) for index in range(response.shape[0])]
    )
    leakage_corrected = leakage.mean(0) - lam_solver_curve
    upstream = np.array(
        [
            directional_leakage(response[index], x0, m, r0)["upstream"]
            for index in range(response.shape[0])
        ]
    )
    upstream_1k = float(upstream.mean(0)[: cfg["K"]].mean())
    chat = np.mean(
        [fit_front_speed(response[index], x0, q)[0] for index in range(response.shape[0])]
    )
    jac_fro = float(np.mean(jacobian_frobenius(step, bases_t, eps, n_probe=16)))
    jac_rho = float(np.mean(jacobian_spectral_radius(step, bases_t, eps, n_iter=25)))
    ks = np.arange(1, min(cfg["N"] // 2, 33))
    gain = spectral_gain_from_G(learned_transfer(step, bases_t, eps, ks, device)["G"])
    return {
        "lambda_1K": float(leakage_corrected.mean()),
        "lambda_max": float(leakage_corrected.max()),
        "lambda_up_1K": upstream_1k,
        "chat": float(chat),
        "jac_fro": jac_fro,
        "jac_rho": jac_rho,
        "spectral_max_gain": gain["max_gain"],
        "spectral_hf_gain": gain["hf_mean_gain"],
        "n_params": float(model.n_params()),
    }


def run(*, smoke: bool, device_name: str, checkpoint_dir: Path, output: Path) -> dict:
    cfg = make_cfg(smoke)
    device = get_device(device_name)
    started = time.time()
    rng = np.random.default_rng(0)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    size = cfg["N"]

    train_input = band_limited_ic(rng, size, cfg["n_train"], cfg["J_train"], cfg["p"])
    train_output = np.roll(train_input, cfg["m"], axis=-1)
    val_input = band_limited_ic(rng, size, cfg["n_val"], cfg["J_train"], cfg["p"])
    val_output = np.roll(val_input, cfg["m"], axis=-1)
    ood_input = band_limited_ic(rng, size, cfg["n_ood"], cfg["J_ood"], cfg["p"])
    ood_true = exact_rollout(ood_input, cfg["m"], cfg["H_ood"])
    base_states = band_limited_ic(rng, size, cfg["n_base"], cfg["J_train"], cfg["p"])
    bases_t = torch.as_tensor(base_states, dtype=torch.float32, device=device)
    bump_t = torch.as_tensor(
        make_local_bump(size, cfg["x0"], cfg["sigma"]), dtype=torch.float32, device=device
    )
    radius = bump_radius(cfg["sigma"])

    bump_batch = bump_t.unsqueeze(0).expand(bases_t.shape[0], -1)
    solver_response = (
        perturbation_probe(solver_step(cfg["m"]), bases_t, bump_batch, cfg["eps"], cfg["K"])
        .cpu()
        .numpy()
    )
    solver_leakage = np.array(
        [
            leakage_mass(solver_response[index], cfg["x0"], cfg["m"], radius)
            for index in range(solver_response.shape[0])
        ]
    ).mean(0)

    def new_fno(modes, seed=0):
        torch.manual_seed(seed)
        return FNO1d(modes=modes, width=cfg["width"], n_layers=cfg["n_layers"]).to(device)

    rows = []

    def evaluate_model(model, name, modes):
        features = measure_features(model, cfg, device, bases_t, bump_t, solver_leakage)
        val_mse = evaluate_one_step(model, val_input, val_output, device)
        prediction = model_rollout(model, ood_input, cfg["H_ood"], device)
        horizon_error = float(rollout_rmse_curve(prediction, ood_true)[cfg["H_ood"]])
        features.update(
            name=name,
            modes=modes,
            one_step_rmse=float(np.sqrt(val_mse)),
            val_mse=float(val_mse),
            E_H=horizon_error,
        )
        rows.append(features)

    for modes in cfg["modes_sweep"]:
        for seed in cfg["seeds"]:
            model = new_fno(modes, seed)
            path = checkpoint_dir / f"sweep_m{modes}_s{seed}.pt"
            if path.exists():
                model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            else:
                train_one_step(
                    model,
                    train_input,
                    train_output,
                    device,
                    epochs=cfg["epochs"],
                    lr=cfg["lr"],
                    seed=seed,
                )
                torch.save(model.state_dict(), path)
            evaluate_model(model, f"modes{modes}_s{seed}", modes)

    for regularization in cfg["doses"]:
        for seed in cfg["seeds"]:
            model = new_fno(16, seed)
            path = checkpoint_dir / f"dose_{regularization}_s{seed}.pt"
            if path.exists():
                model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            else:
                train_with_causality(
                    model,
                    train_input,
                    train_output,
                    device,
                    lam_reg=regularization,
                    base_states=base_states,
                    bump=bump_t.cpu().numpy(),
                    m=cfg["m"],
                    r0=radius,
                    eps=cfg["eps"],
                    K=cfg["K"],
                    tau=cfg["tau"],
                    epochs=cfg["epochs"],
                    lr=cfg["lr"],
                    x0=cfg["x0"],
                    seed=seed,
                )
                torch.save(model.state_dict(), path)
            evaluate_model(model, f"dose{regularization}_s{seed}", 16)

    def column(key):
        return np.array([row[key] for row in rows], dtype=float)

    features = {
        key: column(key)
        for key in [
            "one_step_rmse",
            "lambda_1K",
            "lambda_up_1K",
            "lambda_max",
            "chat",
            "jac_fro",
            "jac_rho",
            "spectral_max_gain",
            "n_params",
        ]
    }
    features["chat_err"] = np.abs(column("chat") - cfg["c"])
    horizon_error = column("E_H")
    blowup = (horizon_error > np.median(horizon_error)).astype(int)

    def report(selected_features, target):
        base_keys = ["one_step_rmse", "jac_fro", "jac_rho", "spectral_max_gain"]
        report_rows = {}
        for cone_key in ["lambda_1K", "lambda_up_1K"]:
            versus_rmse = nested_regression(
                selected_features, target, ["one_step_rmse"], [cone_key]
            )
            versus_all = nested_regression(selected_features, target, base_keys, [cone_key])
            rho_rmse, p_rmse = partial_spearman(
                selected_features[cone_key], target, [selected_features["one_step_rmse"]]
            )
            controls = [
                selected_features["one_step_rmse"],
                selected_features["jac_fro"],
                selected_features["jac_rho"],
                selected_features["spectral_max_gain"],
                selected_features["n_params"],
            ]
            rho_all, p_all = partial_spearman(selected_features[cone_key], target, controls)
            report_rows[cone_key] = {
                "vs_rmse": versus_rmse,
                "vs_rmse_jac_spectral": versus_all,
                "partial_spearman_vs_rmse": [float(rho_rmse), float(p_rmse)],
                "partial_spearman_vs_all": [float(rho_all), float(p_all)],
            }
        return report_rows

    full = report(features, horizon_error)
    roc = {
        key: auroc(features[key], blowup)
        for key in [
            "one_step_rmse",
            "lambda_1K",
            "jac_fro",
            "jac_rho",
            "spectral_max_gain",
        ]
    }
    shares = dominance(
        {
            key: features[key]
            for key in ["one_step_rmse", "lambda_1K", "jac_fro", "spectral_max_gain"]
        },
        horizon_error,
    )
    dose_indices = [index for index, row in enumerate(rows) if row["name"].startswith("dose")]
    matched = None
    if len(dose_indices) >= 4:
        matched_features = {key: values[dose_indices] for key, values in features.items()}
        matched = report(matched_features, horizon_error[dose_indices])
    best_delta = max(
        full["lambda_1K"]["vs_rmse_jac_spectral"]["delta_r2_cv"],
        full["lambda_up_1K"]["vs_rmse_jac_spectral"]["delta_r2_cv"],
    )
    verdict = (
        "CONE BEATS RMSE+Jacobian+spectral (held-out)"
        if best_delta > 0
        else "cone does NOT add held-out signal beyond magnitude baselines"
    )
    summary = {
        "cfg": cfg,
        "smoke": smoke,
        "rows": rows,
        "blowup_rate": float(blowup.mean()),
        "full": full,
        "auroc": roc,
        "dominance": shares,
        "matched": matched,
        "verdict": verdict,
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"saved {output}; elapsed {time.time() - started:.1f}s")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    run(
        smoke=arguments.smoke,
        device_name=arguments.device,
        checkpoint_dir=arguments.checkpoint_dir,
        output=arguments.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
