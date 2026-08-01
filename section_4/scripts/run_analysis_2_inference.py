#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations
from pathlib import Path
from typing import Sequence

os.environ.setdefault("PYTENSOR_FLAGS", "optimizer_excluding=fusion")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "orca_matplotlib"),
)

import arviz as az
import numpy as np
import pandas as pd


def find_project_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)

    for candidate in [start, *start.parents]:
        if (candidate / "section_4" / "src" / "inference.py").exists() and (
            candidate / "data"
        ).exists():
            return candidate

    raise RuntimeError("Could not find project root containing section_4/src/inference.py and data/.")


PROJECT_ROOT = find_project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from section_4.src import inference as h4  # noqa: E402


DATA_ROOT = PROJECT_ROOT / "data"
RESULT_ROOT = PROJECT_ROOT / "section_4" / "results" / "analysis_1"

CONDITIONS = {
    "no_treatment": {
        "label": "No treatment",
        "path": DATA_ROOT / "NT-No Treatment.csv",
    },
    "rituximab": {
        "label": "Rituximab",
        "path": DATA_ROOT / "RTX-Rituximab.csv",
    },
    "bispecific_ab": {
        "label": "Bispecific Ab",
        "path": DATA_ROOT / "Bispecific-Bispecific.csv",
    },
}

CONDITION_ORDER = ["No treatment", "Rituximab", "Bispecific Ab"]
MODEL_SPECS = h4.default_model_specs()
MODEL_ORDER = [spec.name for spec in MODEL_SPECS]
MODEL_BY_NAME = {spec.name: spec for spec in MODEL_SPECS}


def slug(text: str) -> str:
    return str(text).lower().replace(" ", "_").replace("-", "_")


def _normalise_history_row(row: pd.Series) -> tuple[int, ...]:
    values = []

    for value in row.dropna().to_numpy():
        numeric = float(value)

        if not numeric.is_integer():
            raise ValueError("Each contact-history value must be 0 or 1.")

        z = int(numeric)

        if z not in (0, 1):
            raise ValueError("Each contact-history value must be 0 or 1.")

        values.append(z)

    return tuple(values)


def load_history_table(path: Path, condition: str, label: str) -> pd.DataFrame:
    raw = pd.read_csv(path)

    history = raw.drop(columns="Cell", errors="ignore").apply(pd.to_numeric, errors="coerce")
    history = history.loc[:, history.notna().any()]
    histories = [_normalise_history_row(row) for _, row in history.iterrows()]

    if "Cell" in raw:
        cell = raw["Cell"]
    else:
        cell = np.arange(len(raw))

    return pd.DataFrame(
        {
            "Cell": cell,
            "condition": condition,
            "condition_label": label,
            "history": histories,
            "history_string": ["".join(map(str, h)) for h in histories],
            "contact_count": [len(h) for h in histories],
            "kill_count": [sum(h) for h in histories],
        }
    )


def load_all_conditions(conditions: dict = CONDITIONS) -> pd.DataFrame:
    return pd.concat(
        [
            load_history_table(
                path=Path(meta["path"]),
                condition=condition,
                label=str(meta["label"]),
            )
            for condition, meta in conditions.items()
        ],
        ignore_index=True,
    )


def save_cell_histories(cells: pd.DataFrame, result_root: Path = RESULT_ROOT) -> Path:
    result_root.mkdir(parents=True, exist_ok=True)
    out = result_root / "cell_histories.csv"
    cells.drop(columns="history").to_csv(out, index=False)
    return out


def save_idata(idata: az.InferenceData, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.stem}.{os.getpid()}.tmp{out_path.suffix}")

    if tmp_path.exists():
        tmp_path.unlink()

    az.to_netcdf(idata, str(tmp_path))
    tmp_path.replace(out_path)


def run_or_load_model(
    condition_label: str,
    condition_df: pd.DataFrame,
    model_name: str,
    *,
    result_root: Path = RESULT_ROOT,
    force_rerun: bool = False,
    draws: int = 10000,
    chains: int = 8,
    cores: int | None = 6,
    random_seed: int | None = None,
    duration: float = 1.0,
    lambda_prior_bounds=(-5.0, 2.0),
    sigma_lambda_prior: float = 5.0,
    p0_prior=(1.0, 1.0),
    sigma_eta_prior: float = 1.0,
    beta_prior_sd: float = 1.0,
    n_quad: int = 30,
    prior_draws: int = 2000,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
    progressbar: bool = True,
) -> az.InferenceData:
    if model_name not in MODEL_BY_NAME:
        raise ValueError(f"Unknown model {model_name!r}.")

    out_dir = result_root / slug(condition_label)
    out_dir.mkdir(parents=True, exist_ok=True)

    posterior_path = out_dir / f"posterior_{model_name}_smc.nc"
    summary_path = out_dir / f"posterior_summary_{model_name}_smc.csv"
    log_evidence_path = out_dir / f"log_evidence_{model_name}_smc.csv"
    history_bf_path = out_dir / f"history_effect_bayes_factors_{model_name}_smc.csv"

    if posterior_path.exists() and log_evidence_path.exists() and not force_rerun:
        print(f"Loading cached {condition_label} / {model_name}")
        idata = az.from_netcdf(str(posterior_path))
    else:
        print(f"Running {condition_label} / {model_name}")
        data = h4.prepare_data(condition_df[["history"]])
        model = h4.build_model(
            data,
            MODEL_BY_NAME[model_name],
            duration=duration,
            lambda_prior_bounds=lambda_prior_bounds,
            sigma_lambda_prior=sigma_lambda_prior,
            p0_prior=p0_prior,
            sigma_eta_prior=sigma_eta_prior,
            beta_prior_sd=beta_prior_sd,
            n_quad=n_quad,
        )
        idata = h4.sample_smc(
            model,
            draws=draws,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            prior_draws=prior_draws,
            threshold=threshold,
            correlation_threshold=correlation_threshold,
            progressbar=progressbar,
        )
        save_idata(idata, posterior_path)

    h4.posterior_summary(idata).to_csv(summary_path)
    log_evidence = h4.log_evidence(idata)

    pd.DataFrame(
        [
            {
                "condition": condition_label,
                "model": model_name,
                "log_evidence": log_evidence,
                "sampler": "smc",
                "chains": chains,
                "smc_particles": draws,
                "prior_draws": prior_draws,
                "n_quad": n_quad,
            }
        ]
    ).to_csv(log_evidence_path, index=False)

    try:
        h4.history_effect_bayes_factors(idata).to_csv(history_bf_path, index=False)
    except Exception as exc:
        print(f"Skipping history-effect Bayes factors for {condition_label} / {model_name}: {exc}")

    return idata


def bayes_factor_pairs(
    log_evidence_by_model: dict[str, float],
    model_order: Sequence[str] = MODEL_ORDER,
) -> pd.DataFrame:
    columns = [
        "model_1",
        "model_2",
        "log_evidence_1",
        "log_evidence_2",
        "delta_log_evidence",
        "log10_BF",
        "BF",
    ]
    rows = []
    available = [model for model in model_order if model in log_evidence_by_model]

    for m1, m2 in combinations(available, 2):
        delta = float(log_evidence_by_model[m1] - log_evidence_by_model[m2])
        rows.append(
            {
                "model_1": m1,
                "model_2": m2,
                "log_evidence_1": float(log_evidence_by_model[m1]),
                "log_evidence_2": float(log_evidence_by_model[m2]),
                "delta_log_evidence": delta,
                "log10_BF": delta / np.log(10.0),
                "BF": float(np.exp(delta)) if delta < 709 else np.inf,
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows, columns=columns).sort_values("delta_log_evidence", ascending=False).reset_index(drop=True)


def run_all_inference(
    cells: pd.DataFrame,
    *,
    result_root: Path = RESULT_ROOT,
    condition_labels: Sequence[str] = CONDITION_ORDER,
    model_order: Sequence[str] = MODEL_ORDER,
    force_rerun: bool = False,
    draws: int = 10000,
    chains: int = 8,
    cores: int | None = 6,
    random_seed: int | None = None,
    progressbar: bool = True,
    **model_kwargs,
) -> pd.DataFrame:
    log_rows = []

    for condition_label in condition_labels:
        condition_df = cells[cells["condition_label"] == condition_label].copy()

        if condition_df.empty:
            print(f"Skipping missing condition {condition_label!r}.")
            continue

        log_evidence_by_model = {}

        for model_index, model_name in enumerate(model_order):
            seed = None if random_seed is None else int(random_seed) + model_index
            idata = run_or_load_model(
                condition_label,
                condition_df,
                model_name,
                result_root=result_root,
                force_rerun=force_rerun,
                draws=draws,
                chains=chains,
                cores=cores,
                random_seed=seed,
                progressbar=progressbar,
                **model_kwargs,
            )
            log_evidence = h4.log_evidence(idata)
            log_evidence_by_model[model_name] = log_evidence
            log_rows.append(
                {
                    "condition": condition_label,
                    "model": model_name,
                    "log_evidence": log_evidence,
                }
            )

        bf = bayes_factor_pairs(log_evidence_by_model, model_order=model_order)
        out_dir = result_root / slug(condition_label)
        out_dir.mkdir(parents=True, exist_ok=True)
        bf.to_csv(out_dir / "bayes_factors_pairs_smc.csv", index=False)

    log_df = pd.DataFrame(log_rows)
    write_aggregate_outputs(log_df, result_root=result_root)
    return log_df


def collect_cached_log_evidence(
    *,
    result_root: Path = RESULT_ROOT,
    condition_labels: Sequence[str] = CONDITION_ORDER,
    model_order: Sequence[str] = MODEL_ORDER,
) -> pd.DataFrame:
    rows = []

    for condition_label in condition_labels:
        condition_dir = result_root / slug(condition_label)

        for model_name in model_order:
            path = condition_dir / f"log_evidence_{model_name}_smc.csv"

            if not path.exists():
                continue

            df = pd.read_csv(path)

            if len(df):
                rows.append(df.iloc[0].to_dict())

    return pd.DataFrame(rows)


def add_bayes_factor_vs_best(log_df: pd.DataFrame) -> pd.DataFrame:
    df = log_df.copy()

    if df.empty:
        return df

    df["best_log_evidence"] = df.groupby("condition")["log_evidence"].transform("max")
    df["delta_log_evidence_vs_best"] = df["log_evidence"] - df["best_log_evidence"]
    df["log10_BF_model_vs_best"] = df["delta_log_evidence_vs_best"] / np.log(10.0)
    df["log10_BF_best_vs_model"] = -df["log10_BF_model_vs_best"]
    df["BF_best_vs_model"] = np.exp(np.clip(-df["delta_log_evidence_vs_best"], -745, 709))

    best_models = (
        df.sort_values("log_evidence", ascending=False)
        .groupby("condition")["model"]
        .first()
        .rename("best_model")
        .reset_index()
    )

    return df.merge(best_models, on="condition", how="left")


def overall_model_comparison(log_df: pd.DataFrame) -> pd.DataFrame:
    if log_df.empty:
        return log_df.copy()

    df = (
        log_df.groupby("model", as_index=False)
        .agg(total_log_evidence=("log_evidence", "sum"), n_conditions=("condition", "nunique"))
        .sort_values("total_log_evidence", ascending=False)
        .reset_index(drop=True)
    )

    best = float(df["total_log_evidence"].max())
    df["delta_log_evidence_vs_best"] = df["total_log_evidence"] - best
    df["log10_BF_model_vs_best"] = df["delta_log_evidence_vs_best"] / np.log(10.0)
    df["log10_BF_best_vs_model"] = -df["log10_BF_model_vs_best"]
    df["BF_best_vs_model"] = np.exp(np.clip(-df["delta_log_evidence_vs_best"], -745, 709))

    return df


def write_aggregate_outputs(log_df: pd.DataFrame, *, result_root: Path = RESULT_ROOT) -> None:
    result_root.mkdir(parents=True, exist_ok=True)

    if log_df.empty:
        return

    log_df.to_csv(result_root / "log_evidence_all_conditions_smc.csv", index=False)
    add_bayes_factor_vs_best(log_df).to_csv(
        result_root / "bayes_factor_vs_best_by_condition_smc.csv",
        index=False,
    )
    overall_model_comparison(log_df).to_csv(
        result_root / "overall_model_comparison_smc.csv",
        index=False,
    )


def _resolve_requested_conditions(values: Sequence[str] | None) -> list[str]:
    if not values:
        return CONDITION_ORDER

    by_key = {key: str(meta["label"]) for key, meta in CONDITIONS.items()}
    by_slug = {slug(label): label for label in CONDITION_ORDER}
    out = []

    for value in values:
        if value in CONDITION_ORDER:
            out.append(value)
        elif value in by_key:
            out.append(by_key[value])
        elif value in by_slug:
            out.append(by_slug[value])
        else:
            raise ValueError(f"Unknown condition {value!r}.")

    return out


def _resolve_requested_models(values: Sequence[str] | None) -> list[str]:
    if not values:
        return MODEL_ORDER

    unknown = [value for value in values if value not in MODEL_BY_NAME]

    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")

    return list(values)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Section 4 SMC inference and save cached results.")
    parser.add_argument("--conditions", nargs="+", help="Condition labels or keys to run. Default: all.")
    parser.add_argument("--models", nargs="+", help="Model names to run. Default: all.")
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT, help="Directory for cached inference outputs.")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore cached posterior/evidence files.")
    parser.add_argument("--draws", type=int, default=int(os.environ.get("SECTION4_SMC_PARTICLES", 10000)))
    parser.add_argument("--chains", type=int, default=int(os.environ.get("SECTION4_CHAINS", 6)))
    parser.add_argument("--cores", type=int, default=int(os.environ.get("SECTION4_SMC_CORES", 0)))
    parser.add_argument("--prior-draws", type=int, default=int(os.environ.get("SECTION4_PRIOR_DRAWS", 2000)))
    parser.add_argument("--n-quad", type=int, default=int(os.environ.get("SECTION4_N_QUAD", 60)))
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--no-progressbar", action="store_true")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Do not run models; rebuild aggregate CSVs from cached per-model log-evidence files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    condition_labels = _resolve_requested_conditions(args.conditions)
    model_order = _resolve_requested_models(args.models)
    result_root = Path(args.result_root)

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DATA_ROOT:", DATA_ROOT)
    print("RESULT_ROOT:", result_root)
    print("Conditions:", ", ".join(condition_labels))
    print("Models:", ", ".join(model_order))

    cells = load_all_conditions()
    histories_path = save_cell_histories(cells, result_root=result_root)
    print("Saved:", histories_path)

    if args.collect_only:
        log_df = collect_cached_log_evidence(
            result_root=result_root,
            condition_labels=condition_labels,
            model_order=model_order,
        )
        write_aggregate_outputs(log_df, result_root=result_root)
    else:
        log_df = run_all_inference(
            cells,
            result_root=result_root,
            condition_labels=condition_labels,
            model_order=model_order,
            force_rerun=args.force_rerun,
            draws=args.draws,
            chains=args.chains,
            cores=args.cores,   
            random_seed=args.random_seed,
            prior_draws=args.prior_draws,
            n_quad=args.n_quad,
            progressbar=not args.no_progressbar,
        )

    if len(log_df):
        print(log_df.sort_values(["condition", "log_evidence"], ascending=[True, False]))
    else:
        print("No log-evidence rows available yet.")


if __name__ == "__main__":
    main()
