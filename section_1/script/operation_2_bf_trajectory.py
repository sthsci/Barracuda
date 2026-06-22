#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

os.environ.setdefault("PYTENSOR_FLAGS", "optimizer_excluding=fusion")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
CACHE_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import arviz as az
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from section_1.src import inference as cih  # noqa: E402
from section_1.src import simulator as sim  # noqa: E402


SCENARIOS: list[dict[str, Any]] = [
    {
        "scenario": "No1",
        "label": "No1: sigma=3, p0=0.2",
        "mu_lambda": 4.0,
        "sigma_lambda": 3.0,
        "p_zero": 0.2,
        "true_model": "hetero3",
        "seed_offset": 1,
    },
    {
        "scenario": "No2",
        "label": "No2: sigma=0, p0=0.2",
        "mu_lambda": 4.0,
        "sigma_lambda": 0.0,
        "p_zero": 0.2,
        "true_model": "Z2P",
        "seed_offset": 2,
    },
    {
        "scenario": "No3",
        "label": "No3: sigma=3, p0=0",
        "mu_lambda": 4.0,
        "sigma_lambda": 3.0,
        "p_zero": 0.0,
        "true_model": "Dis2P",
        "seed_offset": 3,
    },
    {
        "scenario": "No4",
        "label": "No4: sigma=0, p0=0",
        "mu_lambda": 4.0,
        "sigma_lambda": 0.0,
        "p_zero": 0.0,
        "true_model": "homo",
        "seed_offset": 4,
    },
]

MODEL_ORDER = ("homo", "Z2P", "Dis2P", "hetero3")


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


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


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


def _simulate_or_load_full_dataset(
    scenario: dict[str, Any],
    *,
    replicate: int,
    max_cells: int,
    obs_time: float,
    base_seed: int,
    out_dir: Path,
    force: bool,
) -> dict[str, Any]:
    simulation_seed = int(base_seed + scenario["seed_offset"] + replicate * 10_000)
    scenario_dir = out_dir / str(scenario["scenario"]) / f"rep_{replicate:03d}"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    sim_path = scenario_dir / "simulation_full.npz"
    config_path = scenario_dir / "simulation_config.json"

    config = {
        "scenario": scenario["scenario"],
        "label": scenario["label"],
        "replicate": int(replicate),
        "n_cells": int(max_cells),
        "T": float(obs_time),
        "mu_lambda": float(scenario["mu_lambda"]),
        "sigma_lambda": float(scenario["sigma_lambda"]),
        "p_zero": float(scenario["p_zero"]),
        "true_model": scenario["true_model"],
        "seed": simulation_seed,
    }

    if sim_path.exists() and config_path.exists() and not force:
        previous = json.loads(config_path.read_text())
        if previous == config:
            return _load_npz(sim_path)

    sim_data = sim.simulate_Population(
        n_cells=int(max_cells),
        T=float(obs_time),
        mode="heterogeneous",
        mu_lambda=float(scenario["mu_lambda"]),
        sd_lambda=float(scenario["sigma_lambda"]),
        p0_lambda=float(scenario["p_zero"]),
        Dist_mode="gamma",
        seed=simulation_seed,
    )
    sim_data.update(config)
    np.savez_compressed(sim_path, **sim_data)
    _save_json(config_path, config)
    print("Saved:", sim_path)
    return sim_data


def _slice_simulation(sim_data: dict[str, Any], n_cells: int) -> dict[str, Any]:
    out = dict(sim_data)
    out["n_cells"] = int(n_cells)
    out["rates"] = np.asarray(sim_data["rates"], dtype=float)[:n_cells]
    out["n_events"] = np.asarray(sim_data["n_events"], dtype=int)[:n_cells]
    return out


def _model_runner(model_name: str) -> Callable[..., dict[str, Any]]:
    runners = {
        "homo": cih.inference_homo,
        "Z2P": cih.inference_Z2P,
        "Dis2P": cih.inference_Dis2P,
        "hetero3": cih.inference_hetero3,
    }
    return runners[model_name]


def _fit_or_load_model(
    *,
    model_name: str,
    counts: np.ndarray,
    obs_time: float,
    out_dir: Path,
    args: argparse.Namespace,
    random_seed: int,
) -> tuple[az.InferenceData, float]:
    model_dir = out_dir / model_name
    posterior_path = model_dir / "posterior.nc"
    summary_path = model_dir / "posterior_summary.csv"

    if posterior_path.exists() and not args.force:
        idata = az.from_netcdf(posterior_path)
        return idata, cih.smc_log_evidence(idata)

    kwargs: dict[str, Any] = {
        "draws": int(args.smc_particles),
        "chains": int(args.chains),
        "cores": int(args.smc_cores),
        "lambda_prior_bounds": (float(args.lambda_prior_lower), float(args.lambda_prior_upper)),
        "random_seed": int(random_seed),
        "threshold": float(args.threshold),
        "correlation_threshold": float(args.correlation_threshold),
    }

    if model_name in {"Z2P", "hetero3"}:
        kwargs["p_prior_bounds"] = (float(args.p_prior_alpha), float(args.p_prior_beta))

    if model_name in {"Dis2P", "hetero3"}:
        kwargs["std_prior_factor"] = float(args.std_prior_factor)

    print(f"Running {model_name}: n={counts.size}, particles={args.smc_particles}, chains={args.chains}")
    pack = _model_runner(model_name)(counts, float(obs_time), **kwargs)
    idata = pack["idata"]

    _save_idata_netcdf(idata, posterior_path)
    az.summary(idata, hdi_prob=0.95).to_csv(summary_path)
    logml = cih.smc_log_evidence(idata)
    return idata, logml


def _write_subset_metadata(path: Path, sim_data: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "simulation_data.npz", **sim_data)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate four synthetic NK-cell count scenarios and run four SMC model "
            "fits across cumulative sample sizes for Bayes-factor trajectories."
        )
    )
    parser.add_argument(
        "--sample_sizes",
        type=str,
        default="10,20,30,50,75,100,150,200,300,400,500",
        help="Comma- or space-separated cumulative cell counts.",
    )
    parser.add_argument("--T", type=float, default=1.0, help="Observation time.")
    parser.add_argument("--base_seed", type=int, default=2026, help="Base simulation seed.")
    parser.add_argument("--replicates", type=int, default=3, help="Independent synthetic datasets per scenario.")
    parser.add_argument("--scenarios", type=str, default="all", help="all, or comma list such as No1,No3.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="../results/part_2_bf_trajectory",
        help="Directory for saved simulations, posteriors, and logml summary.",
    )
    parser.add_argument("--chains", type=int, default=4, help="Number of SMC chains.")
    parser.add_argument("--smc_particles", type=int, default=5000, help="SMC particles per chain.")
    parser.add_argument("--smc_cores", type=int, default=0, help="SMC CPU processes; 0 means all available cores.")
    parser.add_argument("--std_prior_factor", type=float, default=5.0, help="HalfNormal prior scale for sigma_lambda.")
    parser.add_argument("--lambda_prior_lower", type=float, default=-1.0, help="Lower bound for eta where lambda scale is 10**eta.")
    parser.add_argument("--lambda_prior_upper", type=float, default=1.5, help="Upper bound for eta where lambda scale is 10**eta.")
    parser.add_argument("--p_prior_alpha", type=float, default=1.0, help="Beta prior alpha for p_zero.")
    parser.add_argument("--p_prior_beta", type=float, default=1.0, help="Beta prior beta for p_zero.")
    parser.add_argument("--threshold", type=float, default=0.6, help="PyMC SMC threshold.")
    parser.add_argument("--correlation_threshold", type=float, default=0.01, help="PyMC SMC correlation threshold.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing simulations/posteriors.")
    args = parser.parse_args(argv)

    sample_sizes = _parse_sample_sizes(args.sample_sizes)
    if int(args.replicates) <= 0:
        raise ValueError("--replicates must be a positive integer")
    scenarios = _selected_scenarios(args.scenarios)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "sample_sizes": sample_sizes,
        "T": float(args.T),
        "base_seed": int(args.base_seed),
        "replicates": int(args.replicates),
        "scenarios": [scenario["scenario"] for scenario in scenarios],
        "models": list(MODEL_ORDER),
        "chains": int(args.chains),
        "smc_particles": int(args.smc_particles),
        "smc_cores": int(args.smc_cores),
        "std_prior_factor": float(args.std_prior_factor),
        "lambda_prior_bounds": [float(args.lambda_prior_lower), float(args.lambda_prior_upper)],
        "p_prior_bounds": [float(args.p_prior_alpha), float(args.p_prior_beta)],
        "threshold": float(args.threshold),
        "correlation_threshold": float(args.correlation_threshold),
    }
    _save_json(out_dir / "run_config.json", run_config)
    pd.DataFrame(scenarios).to_csv(out_dir / "scenarios.csv", index=False)

    rows: list[dict[str, Any]] = []
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
                _write_subset_metadata(subset_dir, subset_sim)
                counts = np.asarray(subset_sim["n_events"], dtype=int)

                logml_by_model: dict[str, float] = {}
                for model_idx, model_name in enumerate(MODEL_ORDER):
                    seed = int(
                        args.base_seed
                        + scenario_idx * 1_000_000
                        + replicate * 100_000
                        + n_cells * 100
                        + model_idx
                    )
                    _idata, logml = _fit_or_load_model(
                        model_name=model_name,
                        counts=counts,
                        obs_time=float(args.T),
                        out_dir=subset_dir,
                        args=args,
                        random_seed=seed,
                    )
                    logml_by_model[model_name] = float(logml)

                true_logml = logml_by_model[str(scenario["true_model"])]
                best_model = max(logml_by_model, key=logml_by_model.get)
                best_logml = logml_by_model[best_model]

                for model_name in MODEL_ORDER:
                    row = {
                        "scenario": scenario["scenario"],
                        "scenario_label": scenario["label"],
                        "replicate": int(replicate),
                        "simulation_seed": int(np.asarray(full_sim["seed"]).item()),
                        "mu_lambda": float(scenario["mu_lambda"]),
                        "sigma_lambda": float(scenario["sigma_lambda"]),
                        "p_zero": float(scenario["p_zero"]),
                        "true_model": scenario["true_model"],
                        "best_model": best_model,
                        "n_cells": int(n_cells),
                        "model": model_name,
                        "logml": float(logml_by_model[model_name]),
                        "log10_bf_model_vs_true": float((logml_by_model[model_name] - true_logml) / np.log(10.0)),
                        "log10_bf_model_vs_best": float((logml_by_model[model_name] - best_logml) / np.log(10.0)),
                    }
                    rows.append(row)

                pd.DataFrame(rows).to_csv(out_dir / "logml_summary.csv", index=False)
                print("Updated:", out_dir / "logml_summary.csv")

    print("Done:", out_dir / "logml_summary.csv")


if __name__ == "__main__":
    main()
