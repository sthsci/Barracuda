#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("PYTENSOR_FLAGS", "optimizer_excluding=fusion")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
CACHE_ROOT = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / ".cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import arviz as az

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "section_1" / "results" / "part_2_bf_groundtruth"
SIMULATION_SCHEMA_VERSION = 1
infer_SCHEMA_VERSION = 1
SUMMARY_SCHEMA_VERSION = 1
sys.path.insert(0, str(PROJECT_ROOT))

from section_1.script import operation_2_bf_trajectory as bfop  # noqa: E402
from section_1.src import simulator as sim  # noqa: E402


def _atomic_save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _atomic_to_csv(data: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        data.to_csv(tmp_path, index=index)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _atomic_save_npz(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _config_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _parse_float_values(raw: str, *, option: str) -> list[float]:
    pieces = str(raw).replace(",", " ").split()
    if not pieces:
        raise ValueError(f"{option} must contain at least one number")

    try:
        values = [float(piece) for piece in pieces]
    except ValueError as exc:
        raise ValueError(f"{option} must be a comma- or space-separated list of numbers") from exc

    if not all(np.isfinite(value) for value in values):
        raise ValueError(f"{option} values must all be finite")
    return sorted(set(values))


def _number_slug(value: float) -> str:
    text = np.format_float_positional(float(value), unique=True, trim="-")
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def _minimal_true_model(sigma_lambda: float, p_zero: float) -> str:
    has_heterogeneity = float(sigma_lambda) > 0.0
    has_zero_inflation = float(p_zero) > 0.0
    if has_heterogeneity and has_zero_inflation:
        return "hetero3"
    if has_zero_inflation:
        return "Z2P"
    if has_heterogeneity:
        return "Dis2P"
    return "homo"


def _build_sweep_points(
    *,
    mu_lambda: float,
    baseline_sigma_lambda: float,
    baseline_p_zero: float,
    sigma_lambda_values: list[float],
    p_zero_values: list[float],
    reference_model: str,
) -> list[dict[str, Any]]:
    memberships: dict[tuple[float, float], set[str]] = {}

    for sigma_lambda in sigma_lambda_values:
        memberships.setdefault(
            (float(sigma_lambda), float(baseline_p_zero)), set()
        ).add("sigma_lambda")

    for p_zero in p_zero_values:
        memberships.setdefault(
            (float(baseline_sigma_lambda), float(p_zero)), set()
        ).add("p_zero")

    points: list[dict[str, Any]] = []
    for point_index, (sigma_lambda, p_zero) in enumerate(sorted(memberships), start=1):
        point_id = (
            f"mu_{_number_slug(mu_lambda)}__"
            f"sigma_{_number_slug(sigma_lambda)}__"
            f"pzero_{_number_slug(p_zero)}"
        )
        is_baseline = bool(
            sigma_lambda == float(baseline_sigma_lambda)
            and p_zero == float(baseline_p_zero)
        )
        points.append(
            {
                "point_index": int(point_index),
                "point_id": point_id,
                "scenario": point_id,
                "scenario_label": (
                    f"mu={float(mu_lambda):g}, sigma={sigma_lambda:g}, p0={p_zero:g}"
                ),
                "sweep_membership": ",".join(sorted(memberships[(sigma_lambda, p_zero)])),
                "mu_lambda": float(mu_lambda),
                "sigma_lambda": float(sigma_lambda),
                "p_zero": float(p_zero),
                "minimal_true_model": _minimal_true_model(sigma_lambda, p_zero),
                "reference_model": str(reference_model),
                "is_no1_baseline": is_baseline,
            }
        )

    point_ids = [point["point_id"] for point in points]
    if len(point_ids) != len(set(point_ids)):
        raise RuntimeError("Ground-truth values produced duplicate point directory names")
    return points


def _stable_seed(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=4).digest()
    # PyMC and NumPy both accept this positive uint32-range seed.
    return 1 + int.from_bytes(digest, "little") % (2**32 - 2)


def _load_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _simulate_or_load_point(
    point: dict[str, Any],
    *,
    replicate: int,
    n_cells: int,
    obs_time: float,
    base_seed: int,
    point_dir: Path,
    force: bool,
) -> tuple[dict[str, Any], int]:
    simulation_seed = _stable_seed(
        "simulation",
        int(base_seed),
        f"{float(point['mu_lambda']):.17g}",
        f"{float(point['sigma_lambda']):.17g}",
        f"{float(point['p_zero']):.17g}",
        int(replicate),
    )
    sim_path = point_dir / "simulation_data.npz"
    config_path = point_dir / "simulation_config.json"
    config = {
        "simulation_schema_version": SIMULATION_SCHEMA_VERSION,
        "run_fingerprint": point["run_fingerprint"],
        "point_id": point["point_id"],
        "replicate": int(replicate),
        "n_cells": int(n_cells),
        "T": float(obs_time),
        "mu_lambda": float(point["mu_lambda"]),
        "sigma_lambda": float(point["sigma_lambda"]),
        "p_zero": float(point["p_zero"]),
        "minimal_true_model": point["minimal_true_model"],
        "simulation_seed": int(simulation_seed),
    }

    if sim_path.exists() or config_path.exists():
        cache_is_complete = sim_path.exists() and config_path.exists()
        previous = json.loads(config_path.read_text()) if config_path.exists() else None
        if cache_is_complete and previous == config and not force:
            return _load_npz(sim_path), simulation_seed
        if not force:
            raise RuntimeError(
                f"Cached simulation at {point_dir} is incomplete or uses different settings. "
                "Use a different --out_dir or pass --force."
            )

    point_dir.mkdir(parents=True, exist_ok=True)
    sim_data = sim.simulate_Population(
        n_cells=int(n_cells),
        T=float(obs_time),
        mode="heterogeneous",
        mu_lambda=float(point["mu_lambda"]),
        sd_lambda=float(point["sigma_lambda"]),
        p0_lambda=float(point["p_zero"]),
        Dist_mode="gamma",
        seed=int(simulation_seed),
    )
    sim_data.update(config)
    _atomic_save_npz(sim_path, sim_data)
    _atomic_save_json(config_path, config)
    print("Saved:", sim_path)
    return sim_data, simulation_seed


def _counts_checksum(counts: np.ndarray) -> str:
    values = np.ascontiguousarray(counts, dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _infer_config(
    *,
    point: dict[str, Any],
    replicate: int,
    simulation_seed: int,
    counts: np.ndarray,
    model_name: str,
    infer_seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    config = {
        "infer_schema_version": infer_SCHEMA_VERSION,
        "run_fingerprint": point["run_fingerprint"],
        "point_id": point["point_id"],
        "replicate": int(replicate),
        "simulation_seed": int(simulation_seed),
        "counts_sha256": _counts_checksum(counts),
        "model": model_name,
        "infer_seed": int(infer_seed),
        "n_cells": int(args.n_cells),
        "T": float(args.T),
        "chains": int(args.chains),
        "smc_particles": int(args.smc_particles),
        "lambda_prior_bounds": [
            float(args.lambda_prior_lower),
            float(args.lambda_prior_upper),
        ],
        "threshold": float(args.threshold),
        "correlation_threshold": float(args.correlation_threshold),
    }
    if model_name in {"Dis2P", "hetero3"}:
        config["std_prior_factor"] = float(args.std_prior_factor)
    if model_name in {"Z2P", "hetero3"}:
        config["p_prior_bounds"] = [float(args.p_prior_alpha), float(args.p_prior_beta)]
    return config


def _infer_or_load_checked(
    *,
    point: dict[str, Any],
    replicate: int,
    simulation_seed: int,
    counts: np.ndarray,
    model_name: str,
    infer_seed: int,
    point_dir: Path,
    args: argparse.Namespace,
) -> float:
    model_dir = point_dir / model_name
    posterior_path = model_dir / "posterior.nc"
    summary_path = model_dir / "posterior_summary.csv"
    config_path = model_dir / "infer_config.json"
    expected = _infer_config(
        point=point,
        replicate=replicate,
        simulation_seed=simulation_seed,
        counts=counts,
        model_name=model_name,
        infer_seed=infer_seed,
        args=args,
    )

    previous = json.loads(config_path.read_text()) if config_path.exists() else None
    if posterior_path.exists() and previous is None and not args.force:
        raise RuntimeError(
            f"Cached infer at {model_dir} has no infer_config.json. Pass --force to regenerate it."
        )
    if (posterior_path.exists() or config_path.exists()) and previous != expected:
        if not args.force:
            raise RuntimeError(
                f"Cached infer at {model_dir} uses different data or inference settings. "
                "Use a different --out_dir or pass --force."
            )

    idata, logml = bfop._fit_or_load_model(
        model_name=model_name,
        counts=counts,
        obs_time=float(args.T),
        out_dir=point_dir,
        args=args,
        random_seed=int(infer_seed),
    )
    if not summary_path.exists():
        _atomic_to_csv(az.summary(idata, hdi_prob=0.95), summary_path, index=True)
    _atomic_save_json(config_path, expected)
    return float(logml)


def _validate_args(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    if int(args.n_cells) <= 0:
        raise ValueError("--n_cells must be a positive integer")
    if float(args.T) <= 0.0 or not np.isfinite(float(args.T)):
        raise ValueError("--T must be finite and > 0")
    if int(args.replicates) <= 0:
        raise ValueError("--replicates must be a positive integer")
    if int(args.chains) <= 0:
        raise ValueError("--chains must be a positive integer")
    if int(args.smc_particles) <= 0:
        raise ValueError("--smc_particles must be a positive integer")
    if int(args.smc_cores) < 0:
        raise ValueError("--smc_cores must be >= 0")
    if float(args.mu_lambda) <= 0.0 or not np.isfinite(float(args.mu_lambda)):
        raise ValueError("--mu_lambda must be finite and > 0")
    if float(args.baseline_sigma_lambda) < 0.0 or not np.isfinite(
        float(args.baseline_sigma_lambda)
    ):
        raise ValueError("--baseline_sigma_lambda must be finite and >= 0")
    if not 0.0 <= float(args.baseline_p_zero) < 1.0:
        raise ValueError("--baseline_p_zero must satisfy 0 <= value < 1")
    if float(args.std_prior_factor) <= 0.0 or not np.isfinite(
        float(args.std_prior_factor)
    ):
        raise ValueError("--std_prior_factor must be finite and > 0")
    lambda_lower = float(args.lambda_prior_lower)
    lambda_upper = float(args.lambda_prior_upper)
    if not np.isfinite(lambda_lower) or not np.isfinite(lambda_upper) or lambda_lower >= lambda_upper:
        raise ValueError("lambda prior bounds must be finite with lower < upper")
    if float(args.p_prior_alpha) <= 0.0 or not np.isfinite(float(args.p_prior_alpha)):
        raise ValueError("--p_prior_alpha must be finite and > 0")
    if float(args.p_prior_beta) <= 0.0 or not np.isfinite(float(args.p_prior_beta)):
        raise ValueError("--p_prior_beta must be finite and > 0")
    if not 0.0 < float(args.threshold) < 1.0:
        raise ValueError("--threshold must satisfy 0 < value < 1")
    if not 0.0 < float(args.correlation_threshold) < 1.0:
        raise ValueError("--correlation_threshold must satisfy 0 < value < 1")

    sigma_values = _parse_float_values(
        args.sigma_lambda_values, option="--sigma_lambda_values"
    )
    p_zero_values = _parse_float_values(args.p_zero_values, option="--p_zero_values")
    if any(value < 0.0 for value in sigma_values):
        raise ValueError("--sigma_lambda_values must all be >= 0")
    if any(not 0.0 <= value < 1.0 for value in p_zero_values):
        raise ValueError("--p_zero_values must all satisfy 0 <= value < 1")

    # Always include the No1 anchor in both slices, even if a custom list omits it.
    sigma_values = sorted(set(sigma_values + [float(args.baseline_sigma_lambda)]))
    p_zero_values = sorted(set(p_zero_values + [float(args.baseline_p_zero)]))
    return sigma_values, p_zero_values


def _guard_output_root(out_dir: Path, *, run_fingerprint: str, force: bool) -> None:
    config_path = out_dir / "run_config.json"
    summary_path = out_dir / "logml_summary.csv"
    cache_patterns = (
        "posterior.nc",
        "posterior_summary.csv",
        "infer_config.json",
        "simulation_data.npz",
        "simulation_config.json",
    )
    has_cached_artifacts = any(
        next(out_dir.rglob(pattern), None) is not None for pattern in cache_patterns
    )
    has_results = summary_path.exists() or has_cached_artifacts

    if not config_path.exists():
        if has_results and not force:
            raise RuntimeError(
                f"{out_dir} contains results but no run_config.json. "
                "Use a different --out_dir or pass --force."
            )
        return

    try:
        previous = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        if not force:
            raise RuntimeError(
                f"Could not validate existing {config_path}. Pass --force to replace it."
            ) from exc
        return

    previous_fingerprint = previous.get("run_fingerprint")
    if previous_fingerprint != run_fingerprint and not force:
        raise RuntimeError(
            f"{out_dir} already contains a different ground-truth analysis. "
            "Use a new --out_dir or pass --force; canonical metadata was not changed."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "At fixed cell number, vary the No1 ground-truth sigma_lambda and p_zero "
            "one at a time, infer all four count models, and save Bayes-factor summaries."
        )
    )
    parser.add_argument("--n_cells", type=int, default=500, help="Synthetic NK cells per dataset.")
    parser.add_argument("--T", type=float, default=1.0, help="Observation time.")
    parser.add_argument("--mu_lambda", type=float, default=4.0, help="Fixed active-cell Gamma mean.")
    parser.add_argument(
        "--baseline_sigma_lambda",
        type=float,
        default=3.0,
        help="No1 sigma_lambda; held fixed while p_zero is varied.",
    )
    parser.add_argument(
        "--baseline_p_zero",
        type=float,
        default=0.2,
        help="No1 p_zero; held fixed while sigma_lambda is varied.",
    )
    parser.add_argument(
        "--sigma_lambda_values",
        type=str,
        default="0,0.5,1,2,3,4,5,6",
        help="Comma- or space-separated sigma_lambda ground truths.",
    )
    parser.add_argument(
        "--p_zero_values",
        type=str,
        default="0,0.05,0.1,0.2,0.3,0.4,0.5",
        help="Comma- or space-separated p_zero ground truths.",
    )
    parser.add_argument("--base_seed", type=int, default=2027, help="Base seed for stable simulation/infer seeds.")
    parser.add_argument("--replicates", type=int, default=5, help="Independent datasets per ground-truth point.")
    parser.add_argument(
        "--reference_model",
        choices=bfop.MODEL_ORDER,
        default="hetero3",
        help="Fixed Bayes-factor denominator across the No1 sensitivity sweep.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Directory for simulations, posteriors, and logml_summary.csv.",
    )
    parser.add_argument("--chains", type=int, default=4, help="Number of SMC chains.")
    parser.add_argument("--smc_particles", type=int, default=5000, help="SMC particles per chain.")
    parser.add_argument("--smc_cores", type=int, default=0, help="SMC CPU processes; 0 means all available cores.")
    parser.add_argument("--std_prior_factor", type=float, default=5.0, help="HalfNormal scale for sigma_lambda.")
    parser.add_argument("--lambda_prior_lower", type=float, default=-1.0, help="Lower Uniform bound for eta=log10(lambda).")
    parser.add_argument("--lambda_prior_upper", type=float, default=1.5, help="Upper Uniform bound for eta=log10(lambda).")
    parser.add_argument("--p_prior_alpha", type=float, default=1.0, help="Beta prior alpha for p_zero.")
    parser.add_argument("--p_prior_beta", type=float, default=1.0, help="Beta prior beta for p_zero.")
    parser.add_argument("--threshold", type=float, default=0.6, help="PyMC SMC threshold.")
    parser.add_argument("--correlation_threshold", type=float, default=0.01, help="PyMC SMC correlation threshold.")
    parser.add_argument("--force", action="store_true", help="Overwrite cached simulations and posteriors.")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Print the sweep and write planned_run_config.json/planned_sweep_points.csv "
            "without changing canonical results."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        sigma_values, p_zero_values = _validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    points = _build_sweep_points(
        mu_lambda=float(args.mu_lambda),
        baseline_sigma_lambda=float(args.baseline_sigma_lambda),
        baseline_p_zero=float(args.baseline_p_zero),
        sigma_lambda_values=sigma_values,
        p_zero_values=p_zero_values,
        reference_model=str(args.reference_model),
    )
    run_config = {
        "analysis": "No1 fixed-n one-at-a-time ground-truth sweep",
        "simulation_schema_version": SIMULATION_SCHEMA_VERSION,
        "infer_schema_version": infer_SCHEMA_VERSION,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "n_cells": int(args.n_cells),
        "T": float(args.T),
        "mu_lambda": float(args.mu_lambda),
        "baseline_sigma_lambda": float(args.baseline_sigma_lambda),
        "baseline_p_zero": float(args.baseline_p_zero),
        "sigma_lambda_values": sigma_values,
        "p_zero_values": p_zero_values,
        "base_seed": int(args.base_seed),
        "replicates": int(args.replicates),
        "reference_model": str(args.reference_model),
        "models": list(bfop.MODEL_ORDER),
        "chains": int(args.chains),
        "smc_particles": int(args.smc_particles),
        "smc_cores": int(args.smc_cores),
        "std_prior_factor": float(args.std_prior_factor),
        "lambda_prior_bounds": [
            float(args.lambda_prior_lower),
            float(args.lambda_prior_upper),
        ],
        "p_prior_bounds": [float(args.p_prior_alpha), float(args.p_prior_beta)],
        "threshold": float(args.threshold),
        "correlation_threshold": float(args.correlation_threshold),
        "n_unique_points": len(points),
    }
    fingerprint_payload = {
        key: value for key, value in run_config.items() if key != "smc_cores"
    }
    run_fingerprint = _config_fingerprint(fingerprint_payload)
    run_config["run_fingerprint"] = run_fingerprint
    for point in points:
        point["run_fingerprint"] = run_fingerprint

    point_columns = [
        "run_fingerprint",
        "point_index",
        "point_id",
        "scenario",
        "scenario_label",
        "sweep_membership",
        "mu_lambda",
        "sigma_lambda",
        "p_zero",
        "minimal_true_model",
        "reference_model",
        "is_no1_baseline",
    ]
    point_df = pd.DataFrame(points, columns=point_columns)

    n_infers = len(points) * int(args.replicates) * len(bfop.MODEL_ORDER)
    print(
        f"Prepared {len(points)} unique points, {args.replicates} replicate(s), "
        f"and {n_infers} model infers ({n_infers * int(args.chains)} SMC chain runs, "
        f"{int(args.smc_particles)} particles per chain)."
    )
    print(pd.DataFrame(points)[["point_id", "sweep_membership", "minimal_true_model"]].to_string(index=False))
    if args.dry_run:
        planned_config_path = out_dir / "planned_run_config.json"
        planned_points_path = out_dir / "planned_sweep_points.csv"
        _atomic_save_json(planned_config_path, run_config)
        _atomic_to_csv(point_df, planned_points_path)
        print("Dry run complete; canonical results were not changed:", planned_points_path)
        return

    try:
        _guard_output_root(
            out_dir,
            run_fingerprint=run_fingerprint,
            force=bool(args.force),
        )
    except RuntimeError as exc:
        parser.error(str(exc))
    _atomic_save_json(out_dir / "run_config.json", run_config)
    _atomic_to_csv(point_df, out_dir / "sweep_points.csv")

    rows: list[dict[str, Any]] = []
    summary_path = out_dir / "logml_summary.csv"

    for point in points:
        print(f"Point {point['point_index']}/{len(points)}: {point['scenario_label']}")
        for replicate in range(1, int(args.replicates) + 1):
            print(f"  Replicate {replicate}/{args.replicates}")
            point_dir = (
                out_dir
                / str(point["point_id"])
                / f"rep_{replicate:03d}"
                / f"n_{int(args.n_cells)}"
            )
            sim_data, simulation_seed = _simulate_or_load_point(
                point,
                replicate=replicate,
                n_cells=int(args.n_cells),
                obs_time=float(args.T),
                base_seed=int(args.base_seed),
                point_dir=point_dir,
                force=bool(args.force),
            )
            counts = np.asarray(sim_data["n_events"], dtype=int)

            logml_by_model: dict[str, float] = {}
            for model_idx, model_name in enumerate(bfop.MODEL_ORDER):
                infer_seed = _stable_seed(
                    "infer", int(simulation_seed), model_name, int(model_idx)
                )
                logml_by_model[model_name] = _infer_or_load_checked(
                    point=point,
                    replicate=replicate,
                    simulation_seed=simulation_seed,
                    counts=counts,
                    model_name=model_name,
                    infer_seed=infer_seed,
                    point_dir=point_dir,
                    args=args,
                )

            reference_model = str(point["reference_model"])
            true_model = str(point["minimal_true_model"])
            reference_logml = logml_by_model[reference_model]
            true_logml = logml_by_model[true_model]
            best_model = max(logml_by_model, key=logml_by_model.get)
            best_logml = logml_by_model[best_model]

            for model_name in bfop.MODEL_ORDER:
                logml = float(logml_by_model[model_name])
                log_bf_vs_reference = logml - reference_logml
                log_bf_vs_true = logml - true_logml
                log_bf_vs_best = logml - best_logml
                rows.append(
                    {
                        **point,
                        "replicate": int(replicate),
                        "simulation_seed": int(simulation_seed),
                        "n_cells": int(args.n_cells),
                        "T": float(args.T),
                        "true_model": true_model,
                        "best_model": best_model,
                        "model": model_name,
                        "logml": logml,
                        "log_bf_model_vs_reference": float(log_bf_vs_reference),
                        "log10_bf_model_vs_reference": float(log_bf_vs_reference / np.log(10.0)),
                        "log_bf_model_vs_true": float(log_bf_vs_true),
                        "log10_bf_model_vs_true": float(log_bf_vs_true / np.log(10.0)),
                        "log_bf_model_vs_best": float(log_bf_vs_best),
                        "log10_bf_model_vs_best": float(log_bf_vs_best / np.log(10.0)),
                    }
                )

            _atomic_to_csv(pd.DataFrame(rows), summary_path)
            print("Updated:", summary_path)

    print("Done:", summary_path)


if __name__ == "__main__":
    main()
