#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("PYTENSOR_FLAGS", "optimizer_excluding=fusion")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import arviz as az
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from section_1.src import inference as cih  # noqa: E402
from section_1.src import simulator as sim  # noqa: E402


SCENARIO_NAME = "ZI-gamma"
SIM_METADATA_KEYS = (
    "scenario",
    "gt_mu_lambda",
    "gt_sigma_lambda",
    "gt_p0_lambda",
    "sim_seed",
    "sim_mode",
    "sim_dist_mode",
)


def _safe_slug(s: str) -> str:
    out = []
    for ch in str(s).strip():
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch.isspace() or ch in ("/", "\\", ":", "."):
            out.append("_")
        else:
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "scenario"


def _estimate_logml(idata: az.InferenceData) -> float:
    sample_stats = getattr(idata, "sample_stats", None)
    if sample_stats is not None and "log_marginal_likelihood" in getattr(sample_stats, "data_vars", {}):
        raw = sample_stats["log_marginal_likelihood"].values
    else:
        attrs = getattr(idata, "attrs", {})
        if "log_marginal_likelihood" not in attrs:
            raise RuntimeError("SMC log_marginal_likelihood is missing from the posterior object.")
        raw = attrs["log_marginal_likelihood"]

    vals = np.asarray(raw, dtype=object).ravel()
    parsed: list[float] = []
    for val in vals:
        if isinstance(val, (list, tuple, np.ndarray)):
            arr = np.asarray(val, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                parsed.append(float(arr[-1]))
            continue
        try:
            fval = float(val)
        except Exception:
            continue
        if np.isfinite(fval):
            parsed.append(fval)

    if not parsed:
        raise RuntimeError("Could not parse finite SMC log marginal likelihood values.")
    return float(np.mean(parsed))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print("Saved:", str(path))


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
    print("Saved:", str(out_path))


def _load_sim_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _simulation_n_cells(sim_data: dict[str, Any]) -> int:
    return int(np.asarray(sim_data["n_events"]).size)


def _slice_sim_data(sim_data: dict[str, Any], n_cells: int) -> dict[str, Any]:
    sliced = {
        "n_cells": int(n_cells),
        "max_time": float(np.asarray(sim_data["max_time"])),
        "rates": np.asarray(sim_data["rates"], dtype=float)[:n_cells],
        "n_events": np.asarray(sim_data["n_events"], dtype=int)[:n_cells],
    }
    for key in SIM_METADATA_KEYS:
        if key in sim_data:
            sliced[key] = sim_data[key]
    return sliced


def _append_sim_data(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    max_time = float(np.asarray(base["max_time"]))
    extra_max_time = float(np.asarray(extra["max_time"]))
    if not np.isclose(max_time, extra_max_time):
        raise ValueError(f"Cannot append simulations with different T values: {max_time} vs {extra_max_time}")

    rates = np.concatenate(
        [np.asarray(base["rates"], dtype=float), np.asarray(extra["rates"], dtype=float)]
    )
    n_events = np.concatenate(
        [np.asarray(base["n_events"], dtype=int), np.asarray(extra["n_events"], dtype=int)]
    )
    appended = {
        "n_cells": int(n_events.size),
        "max_time": max_time,
        "rates": rates,
        "n_events": n_events,
    }
    for key in SIM_METADATA_KEYS:
        if key in base:
            appended[key] = base[key]
    return appended


def _simulate_population(args: argparse.Namespace, n_cells: int, seed: Optional[int]) -> dict[str, Any]:
    return sim.simulate_Population(
        n_cells=int(n_cells),
        T=float(args.T),
        mode="heterogeneous",
        mu_lambda=float(args.gt_mu_lambda),
        sd_lambda=float(args.gt_sigma_lambda),
        p0_lambda=float(args.gt_p0_lambda),
        Dist_mode="gamma",
        seed=seed,
    )


def _sim_seed_value(seed: Optional[int]) -> int:
    return -1 if seed is None else int(seed)


def _with_sim_metadata(sim_data: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(sim_data)
    out.update(
        {
            "scenario": SCENARIO_NAME,
            "gt_mu_lambda": float(args.gt_mu_lambda),
            "gt_sigma_lambda": float(args.gt_sigma_lambda),
            "gt_p0_lambda": float(args.gt_p0_lambda),
            "sim_seed": _sim_seed_value(args.seed),
            "sim_mode": "heterogeneous",
            "sim_dist_mode": "gamma",
        }
    )
    return out


def _scalar_value(sim_data: dict[str, Any], key: str) -> Any:
    val = np.asarray(sim_data[key])
    return val.item() if val.shape == () else val


def _validate_cumulative_sim_data(sim_data: dict[str, Any], args: argparse.Namespace, path: Path) -> None:
    missing = [key for key in SIM_METADATA_KEYS if key not in sim_data]
    if missing:
        raise ValueError(
            f"{path} is missing simulation metadata ({', '.join(missing)}). "
            "It may be an older cumulative file. Use a new --cumulative_sim_path or delete/regenerate it."
        )

    expected = {
        "scenario": SCENARIO_NAME,
        "gt_mu_lambda": float(args.gt_mu_lambda),
        "gt_sigma_lambda": float(args.gt_sigma_lambda),
        "gt_p0_lambda": float(args.gt_p0_lambda),
        "sim_seed": _sim_seed_value(args.seed),
        "sim_mode": "heterogeneous",
        "sim_dist_mode": "gamma",
    }
    for key, expected_value in expected.items():
        actual_value = _scalar_value(sim_data, key)
        if isinstance(expected_value, float):
            ok = np.isclose(float(actual_value), expected_value)
        else:
            ok = str(actual_value) == str(expected_value)
        if not ok:
            raise ValueError(
                f"{path} metadata mismatch for {key}: found {actual_value!r}, "
                f"but this run requested {expected_value!r}. Use a new cumulative path or regenerate it."
            )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Simulate one ZI-gamma count dataset, run hetero3 SMC inference, and save results."
    )
    parser.add_argument("--n_cell", type=int, default=500, help="Number of cells.")
    parser.add_argument("--T", type=float, default=1.0, help="Observation time.")
    parser.add_argument("--gt_mu_lambda", type=float, default=2.0, help="Ground-truth Gamma mean.")
    parser.add_argument("--gt_sigma_lambda", type=float, default=1.0, help="Ground-truth Gamma standard deviation.")
    parser.add_argument("--gt_p0_lambda", type=float, default=0.1, help="Ground-truth zero-inflation probability.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for simulation and SMC.")
    parser.add_argument("--out_dir", type=str, default="../results/operation_1", help="Directory for saved outputs.")
    parser.add_argument(
        "--cumulative_sim_path",
        type=str,
        default=None,
        help="Reusable NPZ simulation file. If set, extend it up to --n_cell and use its first --n_cell cells.",
    )
    parser.add_argument("--chains", type=int, default=4, help="Number of SMC chains.")
    parser.add_argument("--smc_particles", type=int, default=5000, help="SMC particles per chain.")
    parser.add_argument("--smc_cores", type=int, default=0, help="SMC CPU processes; 0 means all available cores.")
    parser.add_argument("--std_prior_factor", type=float, default=5.0, help="HalfNormal prior scale for sigma_lambda.")
    parser.add_argument("--lambda_prior_lower", type=float, default=-1.0, help="Lower bound for eta where lambda scale is 10**eta.")
    parser.add_argument("--lambda_prior_upper", type=float, default=1.5, help="Upper bound for eta where lambda scale is 10**eta.")
    parser.add_argument("--threshold", type=float, default=0.6, help="PyMC SMC threshold.")
    parser.add_argument("--correlation_threshold", type=float, default=0.01, help="PyMC SMC correlation threshold.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _safe_slug(SCENARIO_NAME)

    config = {
        "scenario": SCENARIO_NAME,
        "n_cell": int(args.n_cell),
        "T": float(args.T),
        "gt_mu_lambda": float(args.gt_mu_lambda),
        "gt_sigma_lambda": float(args.gt_sigma_lambda),
        "gt_p0_lambda": float(args.gt_p0_lambda),
        "seed": args.seed,
        "chains": int(args.chains),
        "smc_particles": int(args.smc_particles),
        "smc_cores": int(args.smc_cores),
        "std_prior_factor": float(args.std_prior_factor),
        "lambda_prior_bounds": [float(args.lambda_prior_lower), float(args.lambda_prior_upper)],
        "threshold": float(args.threshold),
        "correlation_threshold": float(args.correlation_threshold),
    }
    _save_json(out_dir / f"config_{slug}.json", config)

    print(f"Preparing simulation for {SCENARIO_NAME}: n_cell={args.n_cell}, T={args.T}")
    if args.cumulative_sim_path:
        cumulative_sim_path = Path(args.cumulative_sim_path)
        cumulative_sim_path.parent.mkdir(parents=True, exist_ok=True)

        if cumulative_sim_path.exists():
            cumulative_data = _load_sim_npz(cumulative_sim_path)
            existing_n = _simulation_n_cells(cumulative_data)
            existing_t = float(np.asarray(cumulative_data["max_time"]))
            if not np.isclose(existing_t, float(args.T)):
                raise ValueError(
                    f"{cumulative_sim_path} has T={existing_t}, but this run requested T={args.T}."
                )
            _validate_cumulative_sim_data(cumulative_data, args, cumulative_sim_path)

            if existing_n < int(args.n_cell):
                missing_n = int(args.n_cell) - existing_n
                print(f"Extending cumulative simulation: {existing_n} -> {args.n_cell} cells")
                append_seed = None if args.seed is None else int(args.seed) + existing_n
                extra_data = _simulate_population(args, missing_n, append_seed)
                cumulative_data = _append_sim_data(cumulative_data, extra_data)
                np.savez_compressed(cumulative_sim_path, **cumulative_data)
                print("Saved:", str(cumulative_sim_path))
            else:
                print(f"Using existing cumulative simulation with {existing_n} cells")
        else:
            print(f"Creating cumulative simulation with {args.n_cell} cells")
            cumulative_data = _with_sim_metadata(
                _simulate_population(args, int(args.n_cell), args.seed),
                args,
            )
            np.savez_compressed(cumulative_sim_path, **cumulative_data)
            print("Saved:", str(cumulative_sim_path))

        sim_data = _slice_sim_data(cumulative_data, int(args.n_cell))
    else:
        print(f"Simulating fresh dataset with {args.n_cell} cells")
        sim_data = _with_sim_metadata(_simulate_population(args, int(args.n_cell), args.seed), args)

    sim_path = out_dir / f"simulation_data_{slug}.npz"
    np.savez_compressed(sim_path, **sim_data)
    print("Saved:", str(sim_path))

    counts = np.asarray(sim_data["n_events"], dtype=int)
    print(f"Running hetero3 SMC inference on {counts.size} count observations.")
    pack = cih.inference_hetero3(
        counts,
        float(sim_data["max_time"]),
        draws=int(args.smc_particles),
        chains=int(args.chains),
        cores=int(args.smc_cores),
        lambda_prior_bounds=(float(args.lambda_prior_lower), float(args.lambda_prior_upper)),
        p_prior_bounds=(1.0, 1.0),
        std_prior_factor=float(args.std_prior_factor),
        random_seed=args.seed,
        threshold=float(args.threshold),
        correlation_threshold=float(args.correlation_threshold),
    )

    idata = pack["idata"]
    posterior_path = out_dir / f"posterior_{slug}_hetero3_smc.nc"
    _save_idata_netcdf(idata, posterior_path)

    summary_path = out_dir / f"posterior_summary_{slug}_hetero3_smc.csv"
    az.summary(idata, hdi_prob=0.95).to_csv(summary_path)
    print("Saved:", str(summary_path))

    logml = _estimate_logml(idata)
    logml_path = out_dir / f"log_marginal_likelihood_{slug}_hetero3_smc.csv"
    pd.DataFrame(
        [
            {
                "scenario": SCENARIO_NAME,
                "model": "hetero3",
                "sampler": "smc",
                "logml": logml,
            }
        ]
    ).to_csv(logml_path, index=False)
    print("Saved:", str(logml_path))

    print("Done.")


if __name__ == "__main__":
    main()
