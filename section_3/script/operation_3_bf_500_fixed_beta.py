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
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "section_3"
    / "results"
    / "part_3_bf_500_fixed_beta"
)

MODEL_ORDER = (
    "homogeneous_fixed_beta",
    "heterogeneous_fixed_beta",
)
MODEL_IS_HETEROGENEOUS = {
    "homogeneous_fixed_beta": False,
    "heterogeneous_fixed_beta": True,
}

# Stage 1 deliberately holds the history effects constant and changes only
# decision-level heterogeneity. beta_x and beta_y are displayed as beta_f and
# beta_s in the validation notebook.
SCENARIOS: list[dict[str, Any]] = [
    {
        "scenario": "No1",
        "label": "Heterogeneous truth: sigma_eta=0.75, beta_f=0.8, beta_s=-0.8",
        "gt_mean_lambda": 4.0,
        "gt_sigma_lambda": 2.0,
        "gt_p0": 0.25,
        "gt_sigma_eta": 0.75,
        "gt_beta_x": 0.8,
        "gt_beta_y": -0.8,
        "true_model": "heterogeneous_fixed_beta",
        "seed_offset": 1,
    },
    {
        "scenario": "No3",
        "label": "Homogeneous truth: sigma_eta=0, beta_f=0.8, beta_s=-0.8",
        "gt_mean_lambda": 4.0,
        "gt_sigma_lambda": 2.0,
        "gt_p0": 0.25,
        "gt_sigma_eta": 0.0,
        "gt_beta_x": 0.8,
        "gt_beta_y": -0.8,
        "true_model": "homogeneous_fixed_beta",
        "seed_offset": 3,
    },
]


def _selected_scenarios(raw: str) -> list[dict[str, Any]]:
    if str(raw).strip().lower() == "all":
        return list(SCENARIOS)

    wanted = {item.strip() for item in str(raw).split(",") if item.strip()}
    selected = [scenario for scenario in SCENARIOS if scenario["scenario"] in wanted]
    missing = wanted.difference({scenario["scenario"] for scenario in selected})
    if missing:
        raise ValueError(f"Unknown scenario(s): {', '.join(sorted(missing))}")
    if not selected:
        raise ValueError("--scenarios must select at least one scenario")
    return selected


def _fit_seed(
    *,
    base_seed: int,
    scenario_index: int,
    replicate: int,
    n_cells: int,
    model_index: int,
) -> int:
    return int(
        base_seed
        + scenario_index * 1_000_000
        + replicate * 100_000
        + n_cells * 100
        + model_index
    )


def _fit_config(
    *,
    model_name: str,
    scenario: dict[str, Any],
    n_cells: int,
    obs_time: float,
    args: argparse.Namespace,
    random_seed: int,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "heterogeneous": bool(MODEL_IS_HETEROGENEOUS[model_name]),
        "history_effects": "fixed_to_ground_truth",
        "fixed_beta_x": float(scenario["gt_beta_x"]),
        "fixed_beta_y": float(scenario["gt_beta_y"]),
        "n_cells": int(n_cells),
        "T": float(obs_time),
        "random_seed": int(random_seed),
        "chains": int(args.chains),
        "smc_particles": int(args.smc_particles),
        "n_quad": int(args.n_quad),
        "lambda_prior_bounds": [
            float(args.lambda_prior_lower),
            float(args.lambda_prior_upper),
        ],
        "sigma_lambda_prior": float(args.sigma_lambda_prior),
        "p0_prior": [float(args.p0_prior_alpha), float(args.p0_prior_beta)],
        "sigma_eta_prior": float(args.sigma_eta_prior),
        "threshold": float(args.threshold),
        "correlation_threshold": float(args.correlation_threshold),
    }


def _infer_or_load_fixed_model(
    *,
    bfop,
    inf,
    model_name: str,
    scenario: dict[str, Any],
    data: pd.DataFrame,
    obs_time: float,
    out_dir: Path,
    args: argparse.Namespace,
    random_seed: int,
):
    import arviz as az

    model_dir = out_dir / model_name
    posterior_path = model_dir / "posterior.nc"
    summary_path = model_dir / "posterior_summary.csv"
    logml_path = model_dir / "log_marginal_likelihood.csv"
    config_path = model_dir / "fit_config.json"
    current_config = _fit_config(
        model_name=model_name,
        scenario=scenario,
        n_cells=len(data),
        obs_time=obs_time,
        args=args,
        random_seed=random_seed,
    )

    if posterior_path.exists() and not args.force:
        if not config_path.exists():
            raise RuntimeError(
                f"{posterior_path} exists without fit_config.json. "
                "Use a fresh output directory or rerun with --force."
            )
        previous_config = json.loads(config_path.read_text())
        if previous_config != current_config:
            raise RuntimeError(
                f"Saved fit configuration differs for {model_dir}. "
                "Use a fresh output directory or rerun with --force."
            )
        idata = az.from_netcdf(posterior_path)
        logml = float(inf.log_evidence(idata))
        print("Loaded:", posterior_path)
        return idata, logml

    model_dir.mkdir(parents=True, exist_ok=True)
    bfop._save_json(config_path, current_config)

    history_data = inf.prepare_data(data)
    spec = inf.ModelSpec(
        name=model_name,
        heterogeneous=bool(MODEL_IS_HETEROGENEOUS[model_name]),
        history_dependent=True,
    )
    model = inf.build_model(
        history_data,
        spec,
        duration=float(obs_time),
        lambda_prior_bounds=(
            float(args.lambda_prior_lower),
            float(args.lambda_prior_upper),
        ),
        sigma_lambda_prior=float(args.sigma_lambda_prior),
        p0_prior=(float(args.p0_prior_alpha), float(args.p0_prior_beta)),
        sigma_eta_prior=float(args.sigma_eta_prior),
        fixed_beta_x=float(scenario["gt_beta_x"]),
        fixed_beta_y=float(scenario["gt_beta_y"]),
        n_quad=int(args.n_quad),
    )

    print(
        f"Running {model_name}: n={history_data.n_cells}, "
        f"events={history_data.z.size}, beta_f={scenario['gt_beta_x']:g}, "
        f"beta_s={scenario['gt_beta_y']:g}, particles={args.smc_particles}, "
        f"chains={args.chains}, n_quad={args.n_quad}"
    )
    idata = inf.sample_smc(
        model,
        draws=int(args.smc_particles),
        chains=int(args.chains),
        cores=int(args.smc_cores),
        random_seed=int(random_seed),
        prior_draws=int(args.prior_draws),
        threshold=float(args.threshold),
        correlation_threshold=float(args.correlation_threshold),
        progressbar=not bool(args.no_progressbar),
        retry_sequential=not bool(args.no_retry_sequential),
    )

    bfop._save_idata_netcdf(idata, posterior_path)
    az.summary(idata, hdi_prob=0.95).to_csv(summary_path)
    logml = float(inf.log_evidence(idata))
    pd.DataFrame(
        [
            {
                "model": model_name,
                "sampler": "smc",
                "n_cells": int(history_data.n_cells),
                "fixed_beta_x": float(scenario["gt_beta_x"]),
                "fixed_beta_y": float(scenario["gt_beta_y"]),
                "logml": logml,
            }
        ]
    ).to_csv(logml_path, index=False)
    bfop._save_json(config_path, current_config)
    return idata, logml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "At fixed N, compare homogeneous and heterogeneous decision models "
            "while beta_f and beta_s are fixed at their generating values."
        )
    )
    parser.add_argument("--n_cells", type=int, default=500)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--base_seed", type=int, default=309)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--scenarios", type=str, default="No1,No3")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--chains", type=int, default=6)
    parser.add_argument("--smc_particles", type=int, default=10000)
    parser.add_argument("--smc_cores", type=int, default=6)
    parser.add_argument("--prior_draws", type=int, default=0)
    parser.add_argument("--n_quad", type=int, default=60)
    parser.add_argument("--lambda_prior_lower", type=float, default=-1.0)
    parser.add_argument("--lambda_prior_upper", type=float, default=1.5)
    parser.add_argument("--sigma_lambda_prior", type=float, default=2.0)
    parser.add_argument("--p0_prior_alpha", type=float, default=1.0)
    parser.add_argument("--p0_prior_beta", type=float, default=1.0)
    parser.add_argument("--sigma_eta_prior", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--correlation_threshold", type=float, default=0.01)
    parser.add_argument(
        "--force",
        "--force_rerun",
        dest="force",
        action="store_true",
        help="Regenerate simulations and refit every requested model.",
    )
    parser.add_argument("--no_progressbar", action="store_true")
    parser.add_argument("--no_retry_sequential", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if int(args.n_cells) <= 0:
        raise ValueError("--n_cells must be positive")
    if int(args.replicates) <= 0:
        raise ValueError("--replicates must be positive")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from section_3.script.operation_3_bf_500 import _prepare_runtime_environment

    _prepare_runtime_environment(out_dir)

    import arviz as az  # noqa: F401

    from section_3.script import operation_2_bf_trajectory as bfop
    from section_3.src import inference as inf

    scenarios = _selected_scenarios(args.scenarios)
    run_config = {
        "analysis": "fixed_beta_homogeneous_vs_heterogeneous",
        "n_cells": int(args.n_cells),
        "T": float(args.T),
        "base_seed": int(args.base_seed),
        "replicates": int(args.replicates),
        "scenarios": [scenario["scenario"] for scenario in scenarios],
        "models": list(MODEL_ORDER),
        "beta_mode": "fixed_to_ground_truth",
        "beta_display_names": {"beta_x": "beta_f", "beta_y": "beta_s"},
        "chains": int(args.chains),
        "smc_particles": int(args.smc_particles),
        "smc_cores": int(args.smc_cores),
        "prior_draws": int(args.prior_draws),
        "n_quad": int(args.n_quad),
        "lambda_prior_bounds": [
            float(args.lambda_prior_lower),
            float(args.lambda_prior_upper),
        ],
        "sigma_lambda_prior": float(args.sigma_lambda_prior),
        "p0_prior": [float(args.p0_prior_alpha), float(args.p0_prior_beta)],
        "sigma_eta_prior": float(args.sigma_eta_prior),
        "threshold": float(args.threshold),
        "correlation_threshold": float(args.correlation_threshold),
    }
    config_path = out_dir / f"run_config_n{int(args.n_cells)}.json"
    if config_path.exists() and not args.force:
        previous_config = json.loads(config_path.read_text())
        if previous_config != run_config:
            raise RuntimeError(
                f"{config_path} contains different settings. Use a fresh "
                "output directory or rerun with --force."
            )
    bfop._save_json(config_path, run_config)
    pd.DataFrame(scenarios).to_csv(out_dir / "scenarios.csv", index=False)

    rows: list[dict[str, Any]] = []
    summary_path = out_dir / f"logml_summary_n{int(args.n_cells)}.csv"

    for scenario_index, scenario in enumerate(scenarios):
        print(f"Scenario {scenario['scenario']}: {scenario['label']}")
        for replicate in range(1, int(args.replicates) + 1):
            print(f"  Replicate {replicate}/{args.replicates}")
            full_simulation = bfop._simulate_or_load_full_dataset(
                scenario,
                replicate=replicate,
                max_cells=int(args.n_cells),
                obs_time=float(args.T),
                base_seed=int(args.base_seed),
                out_dir=out_dir,
                force=bool(args.force),
            )
            subset_dir = (
                out_dir
                / str(scenario["scenario"])
                / f"rep_{replicate:03d}"
                / f"n_{int(args.n_cells)}"
            )
            subset = bfop._slice_simulation(full_simulation, int(args.n_cells))
            bfop._write_subset_metadata(subset_dir, subset, scenario, float(args.T))

            logml_by_model: dict[str, float] = {}
            for model_index, model_name in enumerate(MODEL_ORDER):
                seed = _fit_seed(
                    base_seed=int(args.base_seed),
                    scenario_index=scenario_index,
                    replicate=replicate,
                    n_cells=int(args.n_cells),
                    model_index=model_index,
                )
                _idata, logml = _infer_or_load_fixed_model(
                    bfop=bfop,
                    inf=inf,
                    model_name=model_name,
                    scenario=scenario,
                    data=subset,
                    obs_time=float(args.T),
                    out_dir=subset_dir,
                    args=args,
                    random_seed=seed,
                )
                logml_by_model[model_name] = float(logml)

            log_bf_het_vs_hom = (
                logml_by_model["heterogeneous_fixed_beta"]
                - logml_by_model["homogeneous_fixed_beta"]
            )
            best_model = max(logml_by_model, key=logml_by_model.get)
            rows.append(
                {
                    "scenario": scenario["scenario"],
                    "scenario_label": scenario["label"],
                    "replicate": int(replicate),
                    "simulation_seed": int(
                        args.base_seed
                        + scenario["seed_offset"]
                        + replicate * 10_000
                    ),
                    "n_cells": int(args.n_cells),
                    "gt_mean_lambda": float(scenario["gt_mean_lambda"]),
                    "gt_sigma_lambda": float(scenario["gt_sigma_lambda"]),
                    "gt_p0": float(scenario["gt_p0"]),
                    "gt_sigma_eta": float(scenario["gt_sigma_eta"]),
                    "gt_beta_x": float(scenario["gt_beta_x"]),
                    "gt_beta_y": float(scenario["gt_beta_y"]),
                    "gt_beta_f": float(scenario["gt_beta_x"]),
                    "gt_beta_s": float(scenario["gt_beta_y"]),
                    "true_model": scenario["true_model"],
                    "best_model": best_model,
                    "logml_homogeneous_fixed_beta": float(
                        logml_by_model["homogeneous_fixed_beta"]
                    ),
                    "logml_heterogeneous_fixed_beta": float(
                        logml_by_model["heterogeneous_fixed_beta"]
                    ),
                    "log_bf_heterogeneous_vs_homogeneous": float(log_bf_het_vs_hom),
                    "log10_bf_heterogeneous_vs_homogeneous": float(
                        log_bf_het_vs_hom / np.log(10.0)
                    ),
                }
            )
            pd.DataFrame(rows).to_csv(summary_path, index=False)
            print("Updated:", summary_path)

    print("Done:", summary_path)


if __name__ == "__main__":
    main()
