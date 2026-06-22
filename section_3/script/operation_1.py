#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

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

from section_3.src import inference as inf  # noqa: E402
from section_3.src import simulator as sim  # noqa: E402


SCENARIO_NAME = "complex_history_dependent"
MODEL_NAME = "heterogeneous_history_dependent"


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print("Saved:", path)


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
    print("Saved:", out_path)


def _simulation_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "scenario": SCENARIO_NAME,
        "duration": float(args.T),
        "gt_mean_lambda": float(args.gt_mean_lambda),
        "gt_sigma_lambda": float(args.gt_sigma_lambda),
        "gt_p0": float(args.gt_p0),
        "gt_sigma_eta": float(args.gt_sigma_eta),
        "gt_beta_x": float(args.gt_beta_x),
        "gt_beta_y": float(args.gt_beta_y),
        "seed": None if args.seed is None else int(args.seed),
    }


def _operation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        **_simulation_metadata(args),
        "n_cell": int(args.n_cell),
        "model": MODEL_NAME,
        "chains": int(args.chains),
        "smc_particles": int(args.smc_particles),
        "smc_cores": int(args.smc_cores),
        "lambda_prior_bounds": [
            float(args.lambda_prior_lower),
            float(args.lambda_prior_upper),
        ],
        "sigma_lambda_prior": float(args.sigma_lambda_prior),
        "p0_prior": [float(args.p0_prior_alpha), float(args.p0_prior_beta)],
        "sigma_eta_prior": float(args.sigma_eta_prior),
        "beta_prior_sd": float(args.beta_prior_sd),
        "n_quad": int(args.n_quad),
        "prior_draws": int(args.prior_draws),
        "threshold": float(args.threshold),
        "correlation_threshold": float(args.correlation_threshold),
    }


def _simulate_cells(
    args: argparse.Namespace,
    *,
    n_cells: int,
    seed: Optional[int],
    cell_id_offset: int = 0,
) -> pd.DataFrame:
    params = sim.ContactKillParams(
        n_cells=int(n_cells),
        mean_lambda=float(args.gt_mean_lambda),
        sigma_lambda=float(args.gt_sigma_lambda),
        p0=float(args.gt_p0),
        sigma_eta=float(args.gt_sigma_eta),
        beta_x=float(args.gt_beta_x),
        beta_y=float(args.gt_beta_y),
        duration=float(args.T),
    )
    df = sim.simulate_contact_kill(params, seed=seed, return_latent=True)
    df["cell_id"] = df["cell_id"].astype(int) + int(cell_id_offset)
    return df


def _load_cumulative(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _write_cumulative(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print("Saved:", path)


def _validate_cumulative_metadata(path: Path, args: argparse.Namespace) -> None:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        raise ValueError(
            f"{metadata_path} is missing. Use a new --cumulative_sim_path or regenerate the cumulative simulation."
        )

    previous = json.loads(metadata_path.read_text())
    current = _simulation_metadata(args)
    if previous != current:
        raise ValueError(
            f"{path} metadata do not match this run. Use a new cumulative path or pass --force."
        )


def _prepare_simulation(args: argparse.Namespace) -> pd.DataFrame:
    if not args.cumulative_sim_path:
        print(f"Simulating fresh dataset with {args.n_cell} cells.")
        return _simulate_cells(args, n_cells=int(args.n_cell), seed=args.seed)

    cumulative_path = Path(args.cumulative_sim_path)
    metadata_path = cumulative_path.with_suffix(".json")

    if cumulative_path.exists() and not args.force:
        _validate_cumulative_metadata(cumulative_path, args)
        cumulative_df = _load_cumulative(cumulative_path)
        existing_n = len(cumulative_df)
        print(f"Using cumulative simulation with {existing_n} existing cells.")
    else:
        cumulative_df = pd.DataFrame()
        existing_n = 0
        _save_json(metadata_path, _simulation_metadata(args))

    if existing_n < int(args.n_cell):
        missing_n = int(args.n_cell) - existing_n
        append_seed = None if args.seed is None else int(args.seed) + existing_n
        print(f"Extending cumulative simulation: {existing_n} -> {args.n_cell} cells.")
        extra_df = _simulate_cells(
            args,
            n_cells=missing_n,
            seed=append_seed,
            cell_id_offset=existing_n,
        )
        cumulative_df = pd.concat([cumulative_df, extra_df], ignore_index=True)
        _write_cumulative(cumulative_path, cumulative_df)
    elif cumulative_path.exists():
        print(f"Reusing first {args.n_cell} cells from cumulative simulation.")

    return cumulative_df.head(int(args.n_cell)).copy()


def _expanded_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    data = inf.prepare_data(df)
    cell_ids = (
        np.asarray(df["cell_id"].to_numpy(), dtype=int)
        if "cell_id" in df
        else np.arange(data.n_cells, dtype=int)
    )

    if data.z.size == 0:
        return pd.DataFrame(
            columns=["cell_id", "event_id", "x_before", "y_before", "z"]
        )

    event_id = (data.x_before + data.y_before).astype(int)
    return pd.DataFrame(
        {
            "cell_id": cell_ids[data.cell_idx],
            "event_id": event_id,
            "x_before": data.x_before.astype(int),
            "y_before": data.y_before.astype(int),
            "z": data.z.astype(int),
        }
    )


def _true_parameters(args: argparse.Namespace) -> dict[str, float]:
    params = sim.ContactKillParams(
        n_cells=int(args.n_cell),
        mean_lambda=float(args.gt_mean_lambda),
        sigma_lambda=float(args.gt_sigma_lambda),
        p0=float(args.gt_p0),
        sigma_eta=float(args.gt_sigma_eta),
        beta_x=float(args.gt_beta_x),
        beta_y=float(args.gt_beta_y),
        duration=float(args.T),
    )
    return sim.true_parameter_dict(params)


def _fit_or_load(
    args: argparse.Namespace,
    df: pd.DataFrame,
    posterior_path: Path,
) -> az.InferenceData:
    if posterior_path.exists() and not args.force:
        print("Loading existing posterior:", posterior_path)
        return az.from_netcdf(posterior_path)

    history_data = inf.prepare_data(df)
    spec = inf.ModelSpec(
        name=MODEL_NAME,
        heterogeneous=True,
        history_dependent=True,
    )
    model = inf.build_model(
        history_data,
        spec,
        duration=float(args.T),
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
        f"Running {MODEL_NAME}: n={history_data.n_cells}, "
        f"events={history_data.z.size}, particles={args.smc_particles}, "
        f"chains={args.chains}, n_quad={args.n_quad}"
    )
    idata = inf.sample_smc(
        model,
        draws=int(args.smc_particles),
        chains=int(args.chains),
        cores=int(args.smc_cores),
        random_seed=args.seed,
        prior_draws=int(args.prior_draws),
        threshold=float(args.threshold),
        correlation_threshold=float(args.correlation_threshold),
        progressbar=not bool(args.no_progressbar),
        retry_sequential=not bool(args.no_retry_sequential),
    )
    _save_idata_netcdf(idata, posterior_path)
    return idata


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate section 3 contact-kill histories and fit the most complex "
            "heterogeneous history-dependent model with PyMC SMC."
        )
    )
    parser.add_argument("--n_cell", type=int, default=100, help="Number of cells.")
    parser.add_argument("--T", type=float, default=1.0, help="Observation duration.")
    parser.add_argument("--gt_mean_lambda", type=float, default=4.0)
    parser.add_argument("--gt_sigma_lambda", type=float, default=2.0)
    parser.add_argument("--gt_p0", type=float, default=0.25)
    parser.add_argument("--gt_sigma_eta", type=float, default=0.75)
    parser.add_argument("--gt_beta_x", type=float, default=0.4)
    parser.add_argument("--gt_beta_y", type=float, default=-0.4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="../results/part_1_complex")
    parser.add_argument(
        "--cumulative_sim_path",
        type=str,
        default=None,
        help="Reusable cumulative simulation CSV. The first --n_cell rows are used.",
    )
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--smc_particles", type=int, default=2000)
    parser.add_argument("--smc_cores", type=int, default=0)
    parser.add_argument("--lambda_prior_lower", type=float, default=-5.0)
    parser.add_argument("--lambda_prior_upper", type=float, default=2.0)
    parser.add_argument("--sigma_lambda_prior", type=float, default=2.0)
    parser.add_argument("--p0_prior_alpha", type=float, default=1.0)
    parser.add_argument("--p0_prior_beta", type=float, default=1.0)
    parser.add_argument("--sigma_eta_prior", type=float, default=1.0)
    parser.add_argument("--beta_prior_sd", type=float, default=1.0)
    parser.add_argument("--n_quad", type=int, default=30)
    parser.add_argument("--prior_draws", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--correlation_threshold", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_progressbar", action="store_true")
    parser.add_argument("--no_retry_sequential", action="store_true")
    args = parser.parse_args(argv)

    if int(args.n_cell) <= 0:
        raise ValueError("--n_cell must be positive.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = _operation_config(args)
    _save_json(out_dir / "simulation_config.json", config)

    df = _prepare_simulation(args)
    simulation_path = out_dir / "simulation_data.csv"
    df.to_csv(simulation_path, index=False)
    print("Saved:", simulation_path)

    expanded_path = out_dir / "expanded_history.csv"
    _expanded_history_frame(df).to_csv(expanded_path, index=False)
    print("Saved:", expanded_path)

    true_parameters = _true_parameters(args)
    _save_json(out_dir / "true_parameters.json", true_parameters)
    pd.DataFrame([true_parameters]).to_csv(out_dir / "true_parameters.csv", index=False)
    print("Saved:", out_dir / "true_parameters.csv")

    posterior_path = out_dir / "posterior_complex_smc.nc"
    idata = _fit_or_load(args, df, posterior_path)

    summary_path = out_dir / "posterior_summary_complex_smc.csv"
    az.summary(idata, hdi_prob=0.95).to_csv(summary_path)
    print("Saved:", summary_path)

    logml = inf.log_evidence(idata)
    logml_path = out_dir / "log_marginal_likelihood_complex_smc.csv"
    pd.DataFrame(
        [
            {
                "scenario": SCENARIO_NAME,
                "model": MODEL_NAME,
                "sampler": "smc",
                "n_cell": int(args.n_cell),
                "logml": logml,
            }
        ]
    ).to_csv(logml_path, index=False)
    print("Saved:", logml_path)

    p0_summary_path = out_dir / "p0_population_summary_complex_smc.csv"
    inf.summarise_p0_population(idata, seed=args.seed).to_frame().T.to_csv(
        p0_summary_path,
        index=False,
    )
    print("Saved:", p0_summary_path)

    if int(args.prior_draws) > 0:
        try:
            bf = inf.history_effect_bayes_factors(idata)
        except Exception as exc:
            warnings.warn(
                f"Could not compute history-effect Bayes factors: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            if len(bf):
                bf_path = out_dir / "history_effect_bayes_factors.csv"
                bf.to_csv(bf_path, index=False)
                print("Saved:", bf_path)

    print("Done.")


if __name__ == "__main__":
    main()
