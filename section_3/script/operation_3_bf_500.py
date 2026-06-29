#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "section_3" / "results" / "part_3_bf_500"


def _pytensor_flags_with_defaults(raw: str | None, *, cache_dir: Path) -> str:
    flags = "" if raw is None else raw.strip()
    defaults = {
        "optimizer_excluding": "fusion",
        "base_compiledir": str(cache_dir),
        "compile__timeout": "600",
        "compile__wait": "10",
    }

    for key, value in defaults.items():
        if f"{key}=" not in flags:
            flags = f"{flags},{key}={value}" if flags else f"{key}={value}"

    return flags


def _prepare_runtime_environment(out_dir: Path) -> None:
    cache_root = Path(os.environ.get("TMPDIR", "/tmp"))
    pytensor_cache = Path(
        os.environ.get("PYTENSOR_BASE_COMPILEDIR", str(out_dir / "pytensor_cache"))
    )
    pytensor_cache.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTENSOR_BASE_COMPILEDIR", str(pytensor_cache))
    os.environ["PYTENSOR_FLAGS"] = _pytensor_flags_with_defaults(
        os.environ.get("PYTENSOR_FLAGS"),
        cache_dir=pytensor_cache,
    )
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / ".cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def _import_bf_helpers():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from section_3.script import operation_2_bf_trajectory as bfop

    return bfop


def _simulation_seed(bfop, scenario: dict[str, Any], *, replicate: int, base_seed: int) -> int:
    return int(base_seed + scenario["seed_offset"] + replicate * 10_000)


def _fit_seed(
    *,
    base_seed: int,
    scenario_idx: int,
    replicate: int,
    n_cells: int,
    model_idx: int,
) -> int:
    return int(
        base_seed
        + scenario_idx * 1_000_000
        + replicate * 100_000
        + n_cells * 100
        + model_idx
    )


def _append_summary_rows(
    rows: list[dict[str, Any]],
    *,
    bfop,
    scenario: dict[str, Any],
    replicate: int,
    n_cells: int,
    base_seed: int,
    logml_by_model: dict[str, float],
) -> None:
    true_model = str(scenario["true_model"])
    true_logml = float(logml_by_model[true_model])
    best_model = max(logml_by_model, key=logml_by_model.get)
    best_logml = float(logml_by_model[best_model])

    for model_name in bfop.MODEL_ORDER:
        logml = float(logml_by_model[model_name])
        log_bf_model_vs_true = logml - true_logml
        log_bf_model_vs_best = logml - best_logml
        rows.append(
            {
                "scenario": scenario["scenario"],
                "scenario_label": scenario["label"],
                "replicate": int(replicate),
                "simulation_seed": _simulation_seed(
                    bfop,
                    scenario,
                    replicate=replicate,
                    base_seed=base_seed,
                ),
                "gt_mean_lambda": float(scenario["gt_mean_lambda"]),
                "gt_sigma_lambda": float(scenario["gt_sigma_lambda"]),
                "gt_p0": float(scenario["gt_p0"]),
                "gt_sigma_eta": float(scenario["gt_sigma_eta"]),
                "gt_beta_x": float(scenario["gt_beta_x"]),
                "gt_beta_y": float(scenario["gt_beta_y"]),
                "true_model": true_model,
                "best_model": best_model,
                "n_cells": int(n_cells),
                "model": model_name,
                "logml": logml,
                "log_bf_model_vs_true": log_bf_model_vs_true,
                "log10_bf_model_vs_true": log_bf_model_vs_true / np.log(10.0),
                "log_bf_model_vs_best": log_bf_model_vs_best,
                "log10_bf_model_vs_best": log_bf_model_vs_best / np.log(10.0),
            }
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed-n section-3 contact-kill decision scenarios, fit the "
            "four candidate models, and save Bayes-factor summaries."
        )
    )
    parser.add_argument("--n_cells", type=int, default=500, help="Synthetic NK cells per dataset.")
    parser.add_argument("--T", type=float, default=1.0, help="Observation duration.")
    parser.add_argument("--base_seed", type=int, default=309, help="Base simulation seed.")
    parser.add_argument("--replicates", type=int, default=1, help="Independent synthetic datasets per scenario.")
    parser.add_argument("--scenarios", type=str, default="all", help="all, or comma list such as No1,No3.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Directory for saved simulations, posteriors, and logml summary.",
    )
    parser.add_argument("--chains", type=int, default=6, help="Number of SMC chains.")
    parser.add_argument("--smc_particles", type=int, default=10000, help="SMC particles per chain.")
    parser.add_argument("--smc_cores", type=int, default=6, help="SMC CPU processes; 0 means all available cores.")
    parser.add_argument("--prior_draws", type=int, default=0, help="Prior predictive draws to store with each posterior.")
    parser.add_argument("--n_quad", type=int, default=60, help="Quadrature nodes for logit-normal decision likelihoods.")
    parser.add_argument("--lambda_prior_lower", type=float, default=-1.0, help="Lower bound for eta where lambda scale is 10**eta.")
    parser.add_argument("--lambda_prior_upper", type=float, default=1.5, help="Upper bound for eta where lambda scale is 10**eta.")
    parser.add_argument("--sigma_lambda_prior", type=float, default=2.0, help="HalfNormal prior scale for contact-rate sigma_lambda.")
    parser.add_argument("--p0_prior_alpha", type=float, default=1.0, help="Beta prior alpha for p0_centre.")
    parser.add_argument("--p0_prior_beta", type=float, default=1.0, help="Beta prior beta for p0_centre.")
    parser.add_argument("--sigma_eta_prior", type=float, default=1.0, help="HalfNormal prior scale for sigma_eta.")
    parser.add_argument("--beta_prior_sd", type=float, default=1.0, help="Normal prior SD for history effects.")
    parser.add_argument("--threshold", type=float, default=0.5, help="PyMC SMC threshold.")
    parser.add_argument("--correlation_threshold", type=float, default=0.01, help="PyMC SMC correlation threshold.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing simulations/posteriors.")
    parser.add_argument("--no_progressbar", action="store_true")
    parser.add_argument("--no_retry_sequential", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if int(args.n_cells) <= 0:
        raise ValueError("--n_cells must be a positive integer")
    if int(args.replicates) <= 0:
        raise ValueError("--replicates must be a positive integer")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    _prepare_runtime_environment(out_dir)
    bfop = _import_bf_helpers()

    scenarios = bfop._selected_scenarios(args.scenarios)
    run_config = {
        "n_cells": int(args.n_cells),
        "T": float(args.T),
        "base_seed": int(args.base_seed),
        "replicates": int(args.replicates),
        "scenarios": [scenario["scenario"] for scenario in scenarios],
        "models": list(bfop.MODEL_ORDER),
        "chains": int(args.chains),
        "smc_particles": int(args.smc_particles),
        "smc_cores": int(args.smc_cores),
        "prior_draws": int(args.prior_draws),
        "n_quad": int(args.n_quad),
        "lambda_prior_bounds": [float(args.lambda_prior_lower), float(args.lambda_prior_upper)],
        "sigma_lambda_prior": float(args.sigma_lambda_prior),
        "p0_prior": [float(args.p0_prior_alpha), float(args.p0_prior_beta)],
        "sigma_eta_prior": float(args.sigma_eta_prior),
        "beta_prior_sd": float(args.beta_prior_sd),
        "threshold": float(args.threshold),
        "correlation_threshold": float(args.correlation_threshold),
    }
    bfop._save_json(out_dir / f"run_config_n{int(args.n_cells)}.json", run_config)
    pd.DataFrame(scenarios).to_csv(out_dir / "scenarios.csv", index=False)

    rows: list[dict[str, Any]] = []
    summary_path = out_dir / f"logml_summary_n{int(args.n_cells)}.csv"

    for scenario_idx, scenario in enumerate(scenarios):
        print(f"Scenario {scenario['scenario']}: {scenario['label']}")
        for replicate in range(1, int(args.replicates) + 1):
            print(f"  Replicate {replicate}/{args.replicates}")
            full_sim = bfop._simulate_or_load_full_dataset(
                scenario,
                replicate=replicate,
                max_cells=int(args.n_cells),
                obs_time=float(args.T),
                base_seed=int(args.base_seed),
                out_dir=out_dir,
                force=bool(args.force),
            )

            subset_dir = out_dir / str(scenario["scenario"]) / f"rep_{replicate:03d}" / f"n_{int(args.n_cells)}"
            subset_sim = bfop._slice_simulation(full_sim, int(args.n_cells))
            bfop._write_subset_metadata(subset_dir, subset_sim, scenario, float(args.T))

            logml_by_model: dict[str, float] = {}
            for model_idx, model_name in enumerate(bfop.MODEL_ORDER):
                seed = _fit_seed(
                    base_seed=int(args.base_seed),
                    scenario_idx=scenario_idx,
                    replicate=replicate,
                    n_cells=int(args.n_cells),
                    model_idx=model_idx,
                )
                _idata, logml = bfop._fit_or_load_model(
                    model_name=model_name,
                    df=subset_sim,
                    obs_time=float(args.T),
                    out_dir=subset_dir,
                    args=args,
                    random_seed=seed,
                )
                logml_by_model[model_name] = float(logml)

            _append_summary_rows(
                rows,
                bfop=bfop,
                scenario=scenario,
                replicate=replicate,
                n_cells=int(args.n_cells),
                base_seed=int(args.base_seed),
                logml_by_model=logml_by_model,
            )
            pd.DataFrame(rows).to_csv(summary_path, index=False)
            print("Updated:", summary_path)

    print("Done:", summary_path)


if __name__ == "__main__":
    main()
