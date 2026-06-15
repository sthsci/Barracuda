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


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Simulate one ZI-gamma count dataset, run hetero3 SMC inference, and save results."
    )
    parser.add_argument("--n_cell", type=int, default=500, help="Number of cells.")
    parser.add_argument("--T", type=float, default=1.0, help="Observation time.")
    parser.add_argument("--gt_mu_lambda", type=float, default=4.0, help="Ground-truth Gamma mean.")
    parser.add_argument("--gt_sigma_lambda", type=float, default=3.0, help="Ground-truth Gamma standard deviation.")
    parser.add_argument("--gt_p0_lambda", type=float, default=0.2, help="Ground-truth zero-inflation probability.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for simulation and SMC.")
    parser.add_argument("--out_dir", type=str, default="../results/operation_1", help="Directory for saved outputs.")
    parser.add_argument("--chains", type=int, default=4, help="Number of SMC chains.")
    parser.add_argument("--smc_particles", type=int, default=5000, help="SMC particles per chain.")
    parser.add_argument("--smc_cores", type=int, default=0, help="SMC CPU processes; 0 means all available cores.")
    parser.add_argument("--std_prior_factor", type=float, default=5.0, help="HalfNormal prior scale for sigma_lambda.")
    parser.add_argument("--lambda_prior_lower", type=float, default=0.0, help="Lower bound for eta where lambda scale is 10**eta.")
    parser.add_argument("--lambda_prior_upper", type=float, default=2.0, help="Upper bound for eta where lambda scale is 10**eta.")
    parser.add_argument("--threshold", type=float, default=0.5, help="PyMC SMC threshold.")
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

    print(f"Simulating {SCENARIO_NAME}: n_cell={args.n_cell}, T={args.T}")
    sim_data = sim.simulate_Population(
        n_cells=int(args.n_cell),
        T=float(args.T),
        mode="heterogeneous",
        mu_lambda=float(args.gt_mu_lambda),
        sd_lambda=float(args.gt_sigma_lambda),
        p0_lambda=float(args.gt_p0_lambda),
        Dist_mode="gamma",
        seed=args.seed,
    )
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
    az.summary(idata).to_csv(summary_path)
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
