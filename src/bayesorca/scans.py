"""Cumulative sample-size Bayes-factor scans and sensitivity planning.

Each scan simulates one maximum-size dataset per scenario and replicate.  A
requested size ``N`` then fits ``frame.iloc[:N]`` from that same dataset.
Consequently, sample sizes within a scenario/replicate are nested cumulative
prefixes, not independently simulated datasets.  Independent replicates are
obtained with the ``replicates`` argument.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Final

import numpy as np
import pandas as pd

from . import event_counts, trajectories
from .validation import (
    COUNT_MODEL_KEYS,
    COUNT_SCENARIOS,
    TRAJECTORY_MODEL_KEYS,
    TRAJECTORY_SCENARIOS,
    EventCountScenario,
    TrajectoryScenario,
    stable_seed,
)


ScanProgressCallback = Callable[[int, int, str], None]
LOG_10: Final[float] = float(np.log(10.0))


def _finite(value: Any, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be a positive integer")
    converted = int(value)
    if converted < 1:
        raise ValueError(f"{name} must be a positive integer")
    return converted


def _base_seed(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("base_seed must be a non-negative integer")
    converted = int(value)
    if converted < 0 or converted > np.iinfo(np.uint32).max:
        raise ValueError("base_seed must be between 0 and 2**32 - 1")
    return converted


def _sample_sizes(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("sample_sizes must be a sequence of integers")
    converted = sorted({_positive_int(value, "sample size") for value in values})
    if not converted:
        raise ValueError("sample_sizes must contain at least one value")
    return tuple(converted)


def _models(
    values: Sequence[str] | str | None,
    available: Sequence[str],
) -> tuple[str, ...]:
    if values is None:
        return tuple(available)
    requested = [values] if isinstance(values, str) else list(values)
    if not requested:
        raise ValueError("model_keys must contain at least one model")
    converted = tuple(str(value).strip().lower() for value in requested)
    if any(not value for value in converted):
        raise ValueError("model_keys must not contain empty names")
    if len(converted) != len(set(converted)):
        raise ValueError("model_keys must not contain duplicates")
    unknown = sorted(set(converted).difference(available))
    if unknown:
        raise ValueError(f"unknown model_keys: {', '.join(unknown)}")
    return converted


def _scenarios(values: Sequence[Any], expected_type: type) -> tuple[Any, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("scenarios must be a sequence of scenario objects")
    selected = tuple(values)
    if not selected:
        raise ValueError("scenarios must contain at least one scenario")
    if any(not isinstance(value, expected_type) for value in selected):
        raise TypeError(f"scenarios must contain only {expected_type.__name__} values")
    names = [value.scenario for value in selected]
    if len(names) != len(set(names)):
        raise ValueError("scenario names must be unique")
    return selected


def _log_evidence(
    results: Mapping[str, Any],
    model_keys: Sequence[str],
) -> dict[str, float]:
    if not isinstance(results, Mapping):
        raise TypeError("fit function must return a mapping")
    values: dict[str, float] = {}
    for requested_key, result in results.items():
        key = str(getattr(result, "model_key", requested_key)).strip().lower()
        if isinstance(result, Mapping):
            raw = result.get("log_evidence")
        else:
            raw = getattr(result, "log_evidence", None)
        values[key] = _finite(raw, f"log_evidence[{key!r}]")
    expected = set(model_keys)
    missing = expected.difference(values)
    extra = set(values).difference(expected)
    if missing or extra:
        raise ValueError(
            "fit result models do not match model_keys "
            f"(missing={sorted(missing)}, extra={sorted(extra)})"
        )
    return {key: values[key] for key in model_keys}


_SCAN_COLUMNS: Final[list[str]] = [
    "workflow",
    "scenario",
    "scenario_label",
    "replicate",
    "simulation_seed",
    "inference_seed",
    "n_cells",
    "model_key",
    "true_model",
    "best_model",
    "log_evidence",
    "log_bf_model_vs_true",
    "log10_bf_model_vs_true",
    "log_bf_model_vs_best",
    "log10_bf_model_vs_best",
    "is_best",
]


def _comparison_rows(
    values: Mapping[str, float],
    *,
    true_model: str,
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if true_model not in values:
        raise ValueError("model_keys must include each scenario's true model")
    true_log_evidence = values[true_model]
    best_model = max(values, key=values.get)
    best_log_evidence = values[best_model]
    rows: list[dict[str, Any]] = []
    for model_key, log_evidence in values.items():
        log_bf_true = float(log_evidence - true_log_evidence)
        log_bf_best = float(log_evidence - best_log_evidence)
        rows.append(
            {
                **metadata,
                "model_key": model_key,
                "true_model": true_model,
                "best_model": best_model,
                "log_evidence": float(log_evidence),
                # Positive values support model_key over the named denominator.
                "log_bf_model_vs_true": log_bf_true,
                "log10_bf_model_vs_true": log_bf_true / LOG_10,
                "log_bf_model_vs_best": log_bf_best,
                "log10_bf_model_vs_best": log_bf_best / LOG_10,
                "is_best": bool(np.isclose(log_evidence, best_log_evidence)),
            }
        )
    return rows


# Module-level adapters are intentional.  Tests and workflow managers can
# monkeypatch the expensive fit operations while retaining real scan logic.
simulate_event_count_data = event_counts.simulate_event_counts
fit_event_count_models = event_counts.run_count_models
simulate_trajectory_data = trajectories.simulate_trajectory_frame
fit_trajectory_models = trajectories.run_trajectory_conditions


def run_count_bf_scan(
    sample_sizes: Sequence[int],
    *,
    scenarios: Sequence[EventCountScenario] = COUNT_SCENARIOS,
    replicates: int = 1,
    observation_time: float = 1.0,
    base_seed: int = 2026,
    settings: event_counts.InferenceSettings | None = None,
    model_keys: Sequence[str] | str | None = COUNT_MODEL_KEYS,
    progress_callback: ScanProgressCallback | None = None,
) -> pd.DataFrame:
    """Run the event-count models over nested cumulative sample-size prefixes.

    One dataset with ``max(sample_sizes)`` cells is simulated for each scenario
    and replicate.  Every smaller fit receives ``full_frame.iloc[:N]``.  The
    output is long form with one row per fitted model.  In columns named
    ``log_bf_model_vs_*``, positive values favour ``model_key`` over the model
    named after ``vs``.
    """

    sizes = _sample_sizes(sample_sizes)
    selected_scenarios = _scenarios(scenarios, EventCountScenario)
    replicate_count = _positive_int(replicates, "replicates")
    seed_base = _base_seed(base_seed)
    duration = _finite(observation_time, "observation_time")
    if duration <= 0:
        raise ValueError("observation_time must be greater than zero")
    selected_models = _models(model_keys, COUNT_MODEL_KEYS)
    missing_truths = [
        scenario.true_model
        for scenario in selected_scenarios
        if scenario.true_model not in selected_models
    ]
    if missing_truths:
        raise ValueError("model_keys must include every selected scenario's true model")
    if settings is not None and not isinstance(settings, event_counts.InferenceSettings):
        raise TypeError("settings must be an InferenceSettings instance or None")

    total = len(selected_scenarios) * replicate_count * len(sizes)
    completed = 0
    rows: list[dict[str, Any]] = []
    maximum = max(sizes)
    for scenario in selected_scenarios:
        for replicate_number in range(1, replicate_count + 1):
            simulation_seed = stable_seed(
                "count_bf_scan",
                "simulation",
                seed_base,
                scenario.scenario,
                scenario.seed_offset,
                replicate_number,
            )
            full_frame, _truth = simulate_event_count_data(
                model_key=scenario.true_model,
                n_cells=maximum,
                obs_time=duration,
                mu_lambda=scenario.mu_lambda,
                sigma_lambda=scenario.sigma_lambda,
                p_zero=scenario.p_zero,
                seed=simulation_seed,
            )
            if len(full_frame) < maximum:
                raise RuntimeError("event-count simulator returned too few cells")
            for n_cells in sizes:
                completed += 1
                if progress_callback is not None:
                    progress_callback(
                        completed,
                        total,
                        f"{scenario.scenario} replicate {replicate_number}, N={n_cells}",
                    )
                inference_seed = stable_seed(
                    "count_bf_scan",
                    "inference",
                    simulation_seed,
                    n_cells,
                    selected_models,
                )
                controls = (
                    event_counts.InferenceSettings(seed=inference_seed)
                    if settings is None
                    else replace(settings, seed=inference_seed)
                )
                prefix = full_frame.iloc[:n_cells].copy()
                fitted = fit_event_count_models(
                    prefix,
                    duration,
                    settings=controls,
                    model_keys=selected_models,
                )
                values = _log_evidence(fitted, selected_models)
                metadata = {
                    "workflow": "event_count",
                    "scenario": scenario.scenario,
                    "scenario_label": scenario.label,
                    "replicate": replicate_number,
                    "simulation_seed": simulation_seed,
                    "inference_seed": inference_seed,
                    "n_cells": n_cells,
                    "mu_lambda": scenario.mu_lambda,
                    "sigma_lambda": scenario.sigma_lambda,
                    "p_zero": scenario.p_zero,
                }
                rows.extend(
                    _comparison_rows(
                        values,
                        true_model=scenario.true_model,
                        metadata=metadata,
                    )
                )
    result = pd.DataFrame(rows)
    ordered = [*_SCAN_COLUMNS, "mu_lambda", "sigma_lambda", "p_zero"]
    result = result.reindex(columns=ordered)
    return validate_bf_scan_schema(result, workflow="event_count")


def run_trajectory_bf_scan(
    sample_sizes: Sequence[int],
    *,
    scenarios: Sequence[TrajectoryScenario] = TRAJECTORY_SCENARIOS,
    replicates: int = 1,
    observation_time: float = 1.0,
    base_seed: int = 2026,
    settings: trajectories.TrajectorySettings | None = None,
    model_keys: Sequence[str] | str | None = TRAJECTORY_MODEL_KEYS,
    progress_callback: ScanProgressCallback | None = None,
) -> pd.DataFrame:
    """Run trajectory models over nested cumulative sample-size prefixes.

    One maximum-size trajectory frame is generated for each scenario and
    replicate.  Fits at smaller sample sizes use its first ``N`` cells, so the
    evidence trajectory reflects accumulating data rather than resimulation.
    Positive ``log_bf_model_vs_*`` values favour ``model_key``.
    """

    sizes = _sample_sizes(sample_sizes)
    selected_scenarios = _scenarios(scenarios, TrajectoryScenario)
    replicate_count = _positive_int(replicates, "replicates")
    seed_base = _base_seed(base_seed)
    duration = _finite(observation_time, "observation_time")
    if duration <= 0:
        raise ValueError("observation_time must be greater than zero")
    selected_models = _models(model_keys, TRAJECTORY_MODEL_KEYS)
    missing_truths = [
        scenario.true_model
        for scenario in selected_scenarios
        if scenario.true_model not in selected_models
    ]
    if missing_truths:
        raise ValueError("model_keys must include every selected scenario's true model")
    if settings is not None and not isinstance(settings, trajectories.TrajectorySettings):
        raise TypeError("settings must be a TrajectorySettings instance or None")

    total = len(selected_scenarios) * replicate_count * len(sizes)
    completed = 0
    rows: list[dict[str, Any]] = []
    maximum = max(sizes)
    for scenario in selected_scenarios:
        for replicate_number in range(1, replicate_count + 1):
            simulation_seed = stable_seed(
                "trajectory_bf_scan",
                "simulation",
                seed_base,
                scenario.scenario,
                scenario.seed_offset,
                replicate_number,
            )
            full_frame, _truths = simulate_trajectory_data(
                condition=scenario.scenario,
                n_cells=maximum,
                mu_lambda=scenario.mu_lambda,
                sigma_lambda=scenario.sigma_lambda,
                p0=scenario.p0,
                sigma_eta=scenario.sigma_eta,
                beta_f=scenario.beta_f,
                beta_s=scenario.beta_s,
                observation_time=duration,
                seed=simulation_seed,
            )
            if len(full_frame) < maximum:
                raise RuntimeError("trajectory simulator returned too few cells")
            for n_cells in sizes:
                completed += 1
                if progress_callback is not None:
                    progress_callback(
                        completed,
                        total,
                        f"{scenario.scenario} replicate {replicate_number}, N={n_cells}",
                    )
                inference_seed = stable_seed(
                    "trajectory_bf_scan",
                    "inference",
                    simulation_seed,
                    n_cells,
                    selected_models,
                )
                controls = (
                    trajectories.TrajectorySettings(seed=inference_seed)
                    if settings is None
                    else replace(settings, seed=inference_seed)
                )
                prefix = full_frame.iloc[:n_cells].copy()
                nested = fit_trajectory_models(
                    prefix,
                    observation_time=duration,
                    settings=controls,
                    model_keys=selected_models,
                )
                if scenario.scenario not in nested:
                    raise RuntimeError(
                        f"trajectory fit returned no condition {scenario.scenario!r}"
                    )
                values = _log_evidence(nested[scenario.scenario], selected_models)
                metadata = {
                    "workflow": "trajectory",
                    "scenario": scenario.scenario,
                    "scenario_label": scenario.label,
                    "replicate": replicate_number,
                    "simulation_seed": simulation_seed,
                    "inference_seed": inference_seed,
                    "n_cells": n_cells,
                    "mu_lambda": scenario.mu_lambda,
                    "sigma_lambda": scenario.sigma_lambda,
                    "p0": scenario.p0,
                    "mu_eta": scenario.mu_eta,
                    "sigma_eta": scenario.sigma_eta,
                    "beta_f": scenario.beta_f,
                    "beta_s": scenario.beta_s,
                }
                rows.extend(
                    _comparison_rows(
                        values,
                        true_model=scenario.true_model,
                        metadata=metadata,
                    )
                )
    result = pd.DataFrame(rows)
    ordered = [
        *_SCAN_COLUMNS,
        "mu_lambda",
        "sigma_lambda",
        "p0",
        "mu_eta",
        "sigma_eta",
        "beta_f",
        "beta_s",
    ]
    result = result.reindex(columns=ordered)
    return validate_bf_scan_schema(result, workflow="trajectory")


def _number_slug(value: float) -> str:
    text = np.format_float_positional(float(value), unique=True, trim="-")
    return text.replace("-", "m").replace("+", "").replace(".", "p")


def _float_values(values: Sequence[float], name: str) -> list[float]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of numbers")
    converted = sorted({_finite(value, name) for value in values})
    return converted


def plan_count_ground_truth_grid(
    *,
    mu_lambda: float = 4.0,
    baseline_sigma_lambda: float = 3.0,
    baseline_p_zero: float = 0.2,
    sigma_lambda_values: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    p_zero_values: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5),
    reference_model: str = "hetero3",
) -> pd.DataFrame:
    """Plan the canonical one-at-a-time count ground-truth sensitivity grid.

    ``sigma_lambda`` varies while ``p_zero`` stays at its baseline, and
    ``p_zero`` varies while ``sigma_lambda`` stays at its baseline.  The shared
    baseline is emitted once with membership in both slices; this is not a
    Cartesian product.  Model identifiers are lowercase canonical keys.
    """

    mean_rate = _finite(mu_lambda, "mu_lambda")
    baseline_sigma = _finite(baseline_sigma_lambda, "baseline_sigma_lambda")
    baseline_zero = _finite(baseline_p_zero, "baseline_p_zero")
    if mean_rate <= 0:
        raise ValueError("mu_lambda must be greater than zero")
    if baseline_sigma < 0:
        raise ValueError("baseline_sigma_lambda must be non-negative")
    if not 0 <= baseline_zero < 1:
        raise ValueError("baseline_p_zero must satisfy 0 <= value < 1")
    sigma_values = sorted(
        set(_float_values(sigma_lambda_values, "sigma_lambda_values") + [baseline_sigma])
    )
    zero_values = sorted(
        set(_float_values(p_zero_values, "p_zero_values") + [baseline_zero])
    )
    if any(value < 0 for value in sigma_values):
        raise ValueError("sigma_lambda_values must be non-negative")
    if any(not 0 <= value < 1 for value in zero_values):
        raise ValueError("p_zero_values must satisfy 0 <= value < 1")
    reference = str(reference_model).strip().lower()
    if reference not in COUNT_MODEL_KEYS:
        raise ValueError(f"unknown reference_model {reference_model!r}")

    memberships: dict[tuple[float, float], set[str]] = {}
    for sigma_lambda in sigma_values:
        memberships.setdefault((sigma_lambda, baseline_zero), set()).add(
            "sigma_lambda"
        )
    for p_zero in zero_values:
        memberships.setdefault((baseline_sigma, p_zero), set()).add("p_zero")

    rows: list[dict[str, Any]] = []
    for point_index, ((sigma_lambda, p_zero), membership) in enumerate(
        sorted(memberships.items()),
        start=1,
    ):
        point_id = (
            f"mu_{_number_slug(mean_rate)}__"
            f"sigma_{_number_slug(sigma_lambda)}__"
            f"pzero_{_number_slug(p_zero)}"
        )
        rows.append(
            {
                "point_index": point_index,
                "point_id": point_id,
                "scenario": point_id,
                "scenario_label": (
                    f"mu_lambda={mean_rate:g}, sigma_lambda={sigma_lambda:g}, "
                    f"p_zero={p_zero:g}"
                ),
                "sweep_membership": ",".join(sorted(membership)),
                "mu_lambda": mean_rate,
                "sigma_lambda": sigma_lambda,
                "p_zero": p_zero,
                "true_model": (
                    "hetero3"
                    if sigma_lambda > 0 and p_zero > 0
                    else "z2p"
                    if p_zero > 0
                    else "dis2p"
                    if sigma_lambda > 0
                    else "homo"
                ),
                "reference_model": reference,
                "is_baseline": bool(
                    sigma_lambda == baseline_sigma and p_zero == baseline_zero
                ),
            }
        )
    return pd.DataFrame(rows)


def validate_bf_scan_schema(
    frame: pd.DataFrame,
    *,
    workflow: str | None = None,
    require_true_model: bool = True,
    require_consistent_models: bool = True,
) -> pd.DataFrame:
    """Validate and return a copy of a standardized long-form BF scan.

    Direction is checked numerically: ``log_bf_model_vs_true`` must equal the
    row model's log evidence minus the true model's log evidence, and the
    analogous best-model columns must use the same numerator convention.
    By default, every replicate and sample-size fit for a scenario must contain
    the same candidate-model set.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = [column for column in _SCAN_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"scan is missing columns: {', '.join(missing)}")
    validated = frame.copy()
    requested_workflow = None if workflow is None else str(workflow).strip().lower()
    if requested_workflow not in {None, "event_count", "trajectory"}:
        raise ValueError("workflow must be 'event_count', 'trajectory', or None")
    for column in ("workflow", "scenario", "scenario_label", "model_key", "true_model", "best_model"):
        validated[column] = validated[column].astype(str)
        if validated[column].str.strip().eq("").any():
            raise ValueError(f"{column} must not contain empty values")
    if requested_workflow is not None and set(validated["workflow"]) != {
        requested_workflow
    }:
        raise ValueError(f"scan does not contain only workflow {requested_workflow!r}")
    if not set(validated["workflow"]).issubset({"event_count", "trajectory"}):
        raise ValueError("scan contains an unknown workflow")
    for column in ("model_key", "true_model", "best_model"):
        if not validated[column].eq(validated[column].str.lower()).all():
            raise ValueError(f"{column} must use lowercase canonical model keys")

    numeric_columns = [
        "log_evidence",
        "log_bf_model_vs_true",
        "log10_bf_model_vs_true",
        "log_bf_model_vs_best",
        "log10_bf_model_vs_best",
    ]
    for column in numeric_columns:
        validated[column] = pd.to_numeric(validated[column], errors="raise")
        if not np.isfinite(validated[column].to_numpy(float)).all():
            raise ValueError(f"{column} must contain only finite values")
    for column in ("replicate", "simulation_seed", "inference_seed", "n_cells"):
        numeric = pd.to_numeric(validated[column], errors="raise")
        if not np.equal(numeric, np.floor(numeric)).all():
            raise ValueError(f"{column} must contain integers")
        validated[column] = numeric.astype(np.int64)
    if (validated["replicate"] < 1).any() or (validated["n_cells"] < 1).any():
        raise ValueError("replicate and n_cells must be positive")
    if (validated["simulation_seed"] < 0).any() or (
        validated["inference_seed"] < 0
    ).any():
        raise ValueError("simulation_seed and inference_seed must be non-negative")
    if not validated["is_best"].map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError("is_best must contain booleans")

    key_columns = ["workflow", "scenario", "replicate", "n_cells", "model_key"]
    if validated.duplicated(key_columns).any():
        raise ValueError("scan contains duplicate workflow/scenario/replicate/N/model rows")
    group_columns = ["workflow", "scenario", "replicate", "n_cells"]
    for keys, group in validated.groupby(group_columns, sort=False):
        workflow_name = str(keys[0])
        allowed = set(
            COUNT_MODEL_KEYS if workflow_name == "event_count" else TRAJECTORY_MODEL_KEYS
        )
        for column in ("model_key", "true_model", "best_model"):
            unknown = set(group[column]).difference(allowed)
            if unknown:
                raise ValueError(
                    f"scan contains unknown {workflow_name} {column}: {sorted(unknown)}"
                )
        if group["true_model"].nunique() != 1 or group["best_model"].nunique() != 1:
            raise ValueError("true_model and best_model must be constant within a fit")
        if group["scenario_label"].nunique() != 1:
            raise ValueError("scenario_label must be constant within a fit")
        if group["simulation_seed"].nunique() != 1 or group["inference_seed"].nunique() != 1:
            raise ValueError(
                "simulation_seed and inference_seed must be constant within a fit"
            )
        true_model = str(group["true_model"].iloc[0])
        best_model = str(group["best_model"].iloc[0])
        indexed = group.set_index("model_key")
        if require_true_model and true_model not in indexed.index:
            raise ValueError("each scan fit must contain its true model")
        best_log_evidence = float(group["log_evidence"].max())
        if best_model not in indexed.index or not np.isclose(
            float(indexed.loc[best_model, "log_evidence"]),
            best_log_evidence,
        ):
            raise ValueError("best_model does not identify a maximum-evidence model")
        expected_best = group["log_evidence"].to_numpy(float) - best_log_evidence
        if not np.allclose(group["log_bf_model_vs_best"], expected_best):
            raise ValueError("log_bf_model_vs_best has the wrong direction or values")
        if not np.allclose(
            group["log10_bf_model_vs_best"], expected_best / LOG_10
        ):
            raise ValueError("log10_bf_model_vs_best is inconsistent with natural logs")
        expected_is_best = np.isclose(
            group["log_evidence"].to_numpy(float), best_log_evidence
        )
        if not np.array_equal(group["is_best"].astype(bool), expected_is_best):
            raise ValueError("is_best is inconsistent with log_evidence")
        if true_model in indexed.index:
            true_log_evidence = float(indexed.loc[true_model, "log_evidence"])
            expected_true = group["log_evidence"].to_numpy(float) - true_log_evidence
            if not np.allclose(group["log_bf_model_vs_true"], expected_true):
                raise ValueError("log_bf_model_vs_true has the wrong direction or values")
            if not np.allclose(
                group["log10_bf_model_vs_true"], expected_true / LOG_10
            ):
                raise ValueError(
                    "log10_bf_model_vs_true is inconsistent with natural logs"
                )

    scenario_columns = ["workflow", "scenario"]
    generating_columns = [
        column
        for column in (
            "scenario_label",
            "true_model",
            "mu_lambda",
            "sigma_lambda",
            "p_zero",
            "p0",
            "mu_eta",
            "sigma_eta",
            "beta_f",
            "beta_s",
        )
        if column in validated.columns
    ]
    for _keys, scenario_group in validated.groupby(scenario_columns, sort=False):
        for column in generating_columns:
            if scenario_group[column].nunique(dropna=False) != 1:
                raise ValueError(
                    f"{column} must be constant within a workflow/scenario"
                )
        for _replicate, replicate_group in scenario_group.groupby(
            "replicate", sort=False
        ):
            if replicate_group["simulation_seed"].nunique() != 1:
                raise ValueError(
                    "simulation_seed must be constant across sample sizes within "
                    "a scenario replicate"
                )
        if require_consistent_models:
            model_sets = {
                tuple(sorted(group["model_key"].astype(str)))
                for _fit_keys, group in scenario_group.groupby(
                    ["replicate", "n_cells"], sort=False
                )
            }
            if len(model_sets) > 1:
                raise ValueError(
                    "candidate model coverage must be identical for every "
                    "replicate and sample size within a scenario"
                )
    return validated


def summarize_bf_scan(
    frame: pd.DataFrame,
    *,
    interval: tuple[float, float] = (0.025, 0.975),
) -> pd.DataFrame:
    """Summarize replicate-wise evidence trajectories by scenario, N, and model."""

    if len(interval) != 2:
        raise ValueError("interval must contain lower and upper probabilities")
    lower = _finite(interval[0], "interval lower")
    upper = _finite(interval[1], "interval upper")
    if not 0 <= lower < upper <= 1:
        raise ValueError("interval must satisfy 0 <= lower < upper <= 1")
    validated = validate_bf_scan_schema(frame)
    groups = [
        "workflow",
        "scenario",
        "scenario_label",
        "n_cells",
        "model_key",
        "true_model",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in validated.groupby(groups, sort=False):
        values = group["log_bf_model_vs_true"].to_numpy(float)
        log10_values = group["log10_bf_model_vs_true"].to_numpy(float)
        rows.append(
            {
                **dict(zip(groups, keys, strict=True)),
                "n_replicates": int(group["replicate"].nunique()),
                "mean_log_evidence": float(group["log_evidence"].mean()),
                "sd_log_evidence": float(group["log_evidence"].std(ddof=1))
                if len(group) > 1
                else 0.0,
                "mean_log_bf_model_vs_true": float(np.mean(values)),
                "median_log_bf_model_vs_true": float(np.median(values)),
                "lower_log_bf_model_vs_true": float(np.quantile(values, lower)),
                "upper_log_bf_model_vs_true": float(np.quantile(values, upper)),
                "mean_log10_bf_model_vs_true": float(np.mean(log10_values)),
                "selection_rate": float(group["is_best"].astype(bool).mean()),
                "interval_lower_probability": lower,
                "interval_upper_probability": upper,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "ScanProgressCallback",
    "fit_event_count_models",
    "fit_trajectory_models",
    "plan_count_ground_truth_grid",
    "run_count_bf_scan",
    "run_trajectory_bf_scan",
    "simulate_event_count_data",
    "simulate_trajectory_data",
    "summarize_bf_scan",
    "validate_bf_scan_schema",
]
