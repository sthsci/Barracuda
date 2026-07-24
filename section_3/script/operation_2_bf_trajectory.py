#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

CACHE_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
PYTENSOR_CACHE_ROOT = Path(
    os.environ.get(
        "PYTENSOR_BASE_COMPILEDIR",
        str(CACHE_ROOT / "pytensor_orca_section3_bf"),
    )
)


def _pytensor_flags_with_defaults(raw: str | None) -> str:
    flags = "" if raw is None else raw.strip()
    defaults = {
        "optimizer_excluding": "fusion",
        "base_compiledir": str(PYTENSOR_CACHE_ROOT),
        "compile__timeout": "600",
        "compile__wait": "10",
    }

    for key, value in defaults.items():
        if f"{key}=" not in flags:
            flags = f"{flags},{key}={value}" if flags else f"{key}={value}"

    return flags


os.environ["PYTENSOR_FLAGS"] = _pytensor_flags_with_defaults(
    os.environ.get("PYTENSOR_FLAGS")
)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / ".cache"))
PYTENSOR_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import arviz as az
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from section_3.src import inference as inf  # noqa: E402
from section_3.src import simulator as sim  # noqa: E402


SCENARIOS: list[dict[str, Any]] = [
    {
        "scenario": "No1",
        "label": "No1: sigma_eta=0.75, beta=(0.8,-0.8)",
        "gt_mean_lambda": 4.0,
        "gt_sigma_lambda": 2.0,
        "gt_p0": 0.25,
        "gt_sigma_eta": 0.75,
        "gt_beta_x": 0.8,
        "gt_beta_y": -0.8,
        "true_model": "heterogeneous_history_dependent",
        "seed_offset": 1,
    },
    {
        "scenario": "No2",
        "label": "No2: sigma_eta=0.75, beta=(0,0)",
        "gt_mean_lambda": 4.0,
        "gt_sigma_lambda": 2.0,
        "gt_p0": 0.25,
        "gt_sigma_eta": 0.75,
        "gt_beta_x": 0.0,
        "gt_beta_y": 0.0,
        "true_model": "heterogeneous_history_independent",
        "seed_offset": 2,
    },
    {
        "scenario": "No3",
        "label": "No3: sigma_eta=0, beta=(0.8,-0.8)",
        "gt_mean_lambda": 4.0,
        "gt_sigma_lambda": 2.0,
        "gt_p0": 0.25,
        "gt_sigma_eta": 0.0,
        "gt_beta_x": 0.8,
        "gt_beta_y": -0.8,
        "true_model": "homogeneous_history_dependent",
        "seed_offset": 3,
    },
    {
        "scenario": "No4",
        "label": "No4: sigma_eta=0, beta=(0,0)",
        "gt_mean_lambda": 4.0,
        "gt_sigma_lambda": 2.0,
        "gt_p0": 0.25,
        "gt_sigma_eta": 0.0,
        "gt_beta_x": 0.0,
        "gt_beta_y": 0.0,
        "true_model": "homogeneous_history_independent",
        "seed_offset": 4,
    },
]

MODEL_SPECS = tuple(inf.default_model_specs())
MODEL_ORDER = tuple(spec.name for spec in MODEL_SPECS)
MODEL_BY_NAME = {spec.name: spec for spec in MODEL_SPECS}


def _parse_sample_sizes(raw: str) -> list[int]:
    pieces = raw.replace(",", " ").split()
    sizes = sorted({int(piece) for piece in pieces})
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("--sample_sizes must contain positive integers")
    return sizes


def _selected_scenarios(raw: str) -> list[dict[str, Any]]:
    if str(raw).strip().lower() == "all":
        return SCENARIOS

    wanted = {item.strip() for item in str(raw).split(",") if item.strip()}
    selected = [scenario for scenario in SCENARIOS if scenario["scenario"] in wanted]
    missing = wanted.difference({scenario["scenario"] for scenario in selected})
    if missing:
        raise ValueError(f"Unknown scenario(s): {', '.join(sorted(missing))}")
    return selected


def _selected_models(raw: str) -> list[str]:
    if str(raw).strip().lower() == "all":
        return list(MODEL_ORDER)

    wanted = {
        item.strip()
        for item in str(raw).replace(",", " ").split()
        if item.strip()
    }
    missing = wanted.difference(MODEL_BY_NAME)
    if missing:
        raise ValueError(f"Unknown model(s): {', '.join(sorted(missing))}")

    selected = [model_name for model_name in MODEL_ORDER if model_name in wanted]
    if not selected:
        raise ValueError("--models must select at least one model")
    return selected


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _save_idata_netcdf(idata: az.InferenceData, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.stem}.{os.getpid()}.tmp{out_path.suffix}")
    try:
        if tmp_path.exists():
            tmp_path.unlink()
        az.to_netcdf(idata, str(tmp_path))
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _contact_kill_params(
    scenario: dict[str, Any],
    *,
    n_cells: int,
    obs_time: float,
) -> sim.ContactKillParams:
    return sim.ContactKillParams(
        n_cells=int(n_cells),
        mean_lambda=float(scenario["gt_mean_lambda"]),
        sigma_lambda=float(scenario["gt_sigma_lambda"]),
        p0=float(scenario["gt_p0"]),
        sigma_eta=float(scenario["gt_sigma_eta"]),
        beta_x=float(scenario["gt_beta_x"]),
        beta_y=float(scenario["gt_beta_y"]),
        duration=float(obs_time),
    )


def _simulation_config(
    scenario: dict[str, Any],
    *,
    replicate: int,
    n_cells: int,
    obs_time: float,
    seed: int,
) -> dict[str, Any]:
    return {
        "scenario": scenario["scenario"],
        "label": scenario["label"],
        "replicate": int(replicate),
        "n_cells": int(n_cells),
        "T": float(obs_time),
        "gt_mean_lambda": float(scenario["gt_mean_lambda"]),
        "gt_sigma_lambda": float(scenario["gt_sigma_lambda"]),
        "gt_p0": float(scenario["gt_p0"]),
        "gt_sigma_eta": float(scenario["gt_sigma_eta"]),
        "gt_beta_x": float(scenario["gt_beta_x"]),
        "gt_beta_y": float(scenario["gt_beta_y"]),
        "true_model": scenario["true_model"],
        "seed": int(seed),
    }


def _simulate_or_load_full_dataset(
    scenario: dict[str, Any],
    *,
    replicate: int,
    max_cells: int,
    obs_time: float,
    base_seed: int,
    out_dir: Path,
    force: bool,
) -> pd.DataFrame:
    simulation_seed = int(base_seed + scenario["seed_offset"] + replicate * 10_000)
    scenario_dir = out_dir / str(scenario["scenario"]) / f"rep_{replicate:03d}"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    sim_path = scenario_dir / "simulation_full.csv"
    config_path = scenario_dir / "simulation_config.json"

    config = _simulation_config(
        scenario,
        replicate=replicate,
        n_cells=max_cells,
        obs_time=obs_time,
        seed=simulation_seed,
    )

    if sim_path.exists() and config_path.exists() and not force:
        previous = json.loads(config_path.read_text())
        if previous == config:
            return pd.read_csv(sim_path)

    params = _contact_kill_params(
        scenario,
        n_cells=int(max_cells),
        obs_time=float(obs_time),
    )
    df = sim.simulate_contact_kill(params, seed=simulation_seed, return_latent=True)
    df.to_csv(sim_path, index=False)
    _save_json(config_path, config)
    print("Saved:", sim_path)
    return df


def _slice_simulation(sim_data: pd.DataFrame, n_cells: int) -> pd.DataFrame:
    return sim_data.head(int(n_cells)).copy()


def _write_subset_metadata(path: Path, df: pd.DataFrame, scenario: dict[str, Any], obs_time: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path / "simulation_data.csv", index=False)

    params = _contact_kill_params(
        scenario,
        n_cells=int(len(df)),
        obs_time=float(obs_time),
    )
    true_parameters = sim.true_parameter_dict(params)
    _save_json(path / "true_parameters.json", true_parameters)
    pd.DataFrame([true_parameters]).to_csv(path / "true_parameters.csv", index=False)


def _infer_or_load_model(
    *,
    model_name: str,
    df: pd.DataFrame,
    obs_time: float,
    out_dir: Path,
    args: argparse.Namespace,
    random_seed: int,
) -> tuple[az.InferenceData, float]:
    model_dir = out_dir / model_name
    posterior_path = model_dir / "posterior.nc"
    summary_path = model_dir / "posterior_summary.csv"
    logml_path = model_dir / "log_marginal_likelihood.csv"

    if posterior_path.exists() and not args.force:
        idata = az.from_netcdf(posterior_path)
        return idata, inf.log_evidence(idata)

    history_data = inf.prepare_data(df)
    spec = MODEL_BY_NAME[model_name]
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
        beta_prior_sd=float(args.beta_prior_sd),
        n_quad=int(args.n_quad),
    )

    print(
        f"Running {model_name}: n={history_data.n_cells}, "
        f"events={history_data.z.size}, particles={args.smc_particles}, "
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

    _save_idata_netcdf(idata, posterior_path)
    az.summary(idata, hdi_prob=0.95).to_csv(summary_path)
    logml = inf.log_evidence(idata)
    pd.DataFrame(
        [
            {
                "model": model_name,
                "sampler": "smc",
                "n_cells": int(history_data.n_cells),
                "logml": float(logml),
            }
        ]
    ).to_csv(logml_path, index=False)
    return idata, logml


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate section-3 contact-kill decision scenarios and run selected "
            "SMC model fits across cumulative sample sizes for Bayes-factor "
            "trajectories."
        )
    )
    parser.add_argument(
        "--sample_sizes",
        type=str,
        default="10,20,30,50,100,200,300,400,500,1000",
        help="Comma- or space-separated cumulative cell counts.",
    )
    parser.add_argument("--T", type=float, default=1.0, help="Observation duration.")
    parser.add_argument("--base_seed", type=int, default=2026, help="Base simulation seed.")
    parser.add_argument("--replicates", type=int, default=3, help="Independent synthetic datasets per scenario.")
    parser.add_argument("--scenarios", type=str, default="all", help="all, or comma list such as No1,No3.")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=(
            "all, or a comma/space-separated list of canonical model names "
            "(homogeneous_history_independent, homogeneous_history_dependent, "
            "heterogeneous_history_independent, heterogeneous_history_dependent). "
            "Each selected scenario's declared true model must be included."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="../results/part_2_bf_trajectory",
        help="Directory for saved simulations, posteriors, and logml summary.",
    )
    parser.add_argument("--chains", type=int, default=4, help="Number of SMC chains.")
    parser.add_argument("--smc_particles", type=int, default=5000, help="SMC particles per chain.")
    parser.add_argument("--smc_cores", type=int, default=0, help="SMC CPU processes; 0 means all available cores.")
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
    args = parser.parse_args(argv)

    sample_sizes = _parse_sample_sizes(args.sample_sizes)
    if int(args.replicates) <= 0:
        raise ValueError("--replicates must be a positive integer")
    scenarios = _selected_scenarios(args.scenarios)
    model_order = _selected_models(args.models)

    missing_true_models = [
        f"{scenario['scenario']} ({scenario['true_model']})"
        for scenario in scenarios
        if str(scenario["true_model"]) not in model_order
    ]
    if missing_true_models:
        raise ValueError(
            "--models must include the declared true model for every selected "
            f"scenario; missing: {', '.join(missing_true_models)}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "sample_sizes": sample_sizes,
        "T": float(args.T),
        "base_seed": int(args.base_seed),
        "replicates": int(args.replicates),
        "scenarios": [scenario["scenario"] for scenario in scenarios],
        "models": list(model_order),
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
    _save_json(out_dir / "run_config.json", run_config)
    pd.DataFrame(scenarios).to_csv(out_dir / "scenarios.csv", index=False)

    rows: list[dict[str, Any]] = []
    summary_path = out_dir / "logml_summary.csv"
    max_cells = max(sample_sizes)

    for scenario_idx, scenario in enumerate(scenarios):
        print(f"Scenario {scenario['scenario']}: {scenario['label']}")
        for replicate in range(1, int(args.replicates) + 1):
            print(f"  Replicate {replicate}/{args.replicates}")
            full_sim = _simulate_or_load_full_dataset(
                scenario,
                replicate=replicate,
                max_cells=max_cells,
                obs_time=float(args.T),
                base_seed=int(args.base_seed),
                out_dir=out_dir,
                force=bool(args.force),
            )

            for n_cells in sample_sizes:
                subset_dir = out_dir / str(scenario["scenario"]) / f"rep_{replicate:03d}" / f"n_{n_cells}"
                subset_sim = _slice_simulation(full_sim, int(n_cells))
                _write_subset_metadata(subset_dir, subset_sim, scenario, float(args.T))

                logml_by_model: dict[str, float] = {}
                for model_name in model_order:
                    model_idx = MODEL_ORDER.index(model_name)
                    seed = int(
                        args.base_seed
                        + scenario_idx * 1_000_000
                        + replicate * 100_000
                        + n_cells * 100
                        + model_idx
                    )
                    _idata, logml = _infer_or_load_model(
                        model_name=model_name,
                        df=subset_sim,
                        obs_time=float(args.T),
                        out_dir=subset_dir,
                        args=args,
                        random_seed=seed,
                    )
                    logml_by_model[model_name] = float(logml)

                true_logml = logml_by_model[str(scenario["true_model"])]
                best_model = max(logml_by_model, key=logml_by_model.get)
                best_logml = logml_by_model[best_model]

                for model_name in model_order:
                    row = {
                        "scenario": scenario["scenario"],
                        "scenario_label": scenario["label"],
                        "replicate": int(replicate),
                        "simulation_seed": int(
                            args.base_seed
                            + scenario["seed_offset"]
                            + replicate * 10_000
                        ),
                        "gt_mean_lambda": float(scenario["gt_mean_lambda"]),
                        "gt_sigma_lambda": float(scenario["gt_sigma_lambda"]),
                        "gt_p0": float(scenario["gt_p0"]),
                        "gt_sigma_eta": float(scenario["gt_sigma_eta"]),
                        "gt_beta_x": float(scenario["gt_beta_x"]),
                        "gt_beta_y": float(scenario["gt_beta_y"]),
                        "true_model": scenario["true_model"],
                        "best_model": best_model,
                        "n_cells": int(n_cells),
                        "model": model_name,
                        "logml": float(logml_by_model[model_name]),
                        "log10_bf_model_vs_true": float(
                            (logml_by_model[model_name] - true_logml) / np.log(10.0)
                        ),
                        "log10_bf_model_vs_best": float(
                            (logml_by_model[model_name] - best_logml) / np.log(10.0)
                        ),
                    }
                    rows.append(row)

                pd.DataFrame(rows).to_csv(summary_path, index=False)
                print("Updated:", summary_path)

    print("Done:", summary_path)


if __name__ == "__main__":
    main()
