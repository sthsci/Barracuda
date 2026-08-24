"""Reusable validation and posterior-recovery utilities.

The functions in this module are deliberately independent of the web UI and
filesystem layout used by the research scripts.  Validation runs use the same
public simulators and inference entry points as the rest of :mod:`barracuda`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any, Final

import arviz as az
import numpy as np
import pandas as pd

from . import event_counts, trajectories


COUNT_MODEL_KEYS: Final[tuple[str, ...]] = (
    "homo",
    "z2p",
    "dis2p",
    "hetero3",
)
TRAJECTORY_MODEL_KEYS: Final[tuple[str, ...]] = tuple(
    trajectories.TRAJECTORY_MODEL_SPECS
)


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


def _label(value: Any, name: str) -> str:
    converted = str(value).strip()
    if not converted:
        raise ValueError(f"{name} must not be empty")
    return converted


def _count_truth_model(sigma_lambda: float, p_zero: float) -> str:
    heterogeneous = float(sigma_lambda) > 0.0
    zero_inflated = float(p_zero) > 0.0
    if heterogeneous and zero_inflated:
        return "hetero3"
    if zero_inflated:
        return "z2p"
    if heterogeneous:
        return "dis2p"
    return "homo"


def _trajectory_truth_model(
    sigma_eta: float,
    beta_f: float,
    beta_s: float,
) -> str:
    heterogeneous = float(sigma_eta) > 0.0
    history_dependent = float(beta_f) != 0.0 or float(beta_s) != 0.0
    if heterogeneous and history_dependent:
        return "heterogeneous_history_dependent"
    if heterogeneous:
        return "heterogeneous_history_independent"
    if history_dependent:
        return "homogeneous_history_dependent"
    return "homogeneous_history_independent"


@dataclass(frozen=True)
class EventCountScenario:
    """Typed ground truth for one canonical event-count experiment."""

    scenario: str
    label: str
    mu_lambda: float
    sigma_lambda: float
    p_zero: float
    true_model: str
    seed_offset: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario", _label(self.scenario, "scenario"))
        object.__setattr__(self, "label", _label(self.label, "label"))
        mu_lambda = _finite(self.mu_lambda, "mu_lambda")
        sigma_lambda = _finite(self.sigma_lambda, "sigma_lambda")
        p_zero = _finite(self.p_zero, "p_zero")
        if mu_lambda <= 0:
            raise ValueError("mu_lambda must be greater than zero")
        if sigma_lambda < 0:
            raise ValueError("sigma_lambda must be non-negative")
        if not 0 <= p_zero < 1:
            raise ValueError("p_zero must satisfy 0 <= p_zero < 1")
        model = _label(self.true_model, "true_model").lower()
        if model not in COUNT_MODEL_KEYS:
            raise ValueError(f"unknown event-count true_model {self.true_model!r}")
        expected = _count_truth_model(sigma_lambda, p_zero)
        if model != expected:
            raise ValueError(
                f"true_model must be {expected!r} for sigma_lambda={sigma_lambda:g} "
                f"and p_zero={p_zero:g}"
            )
        if isinstance(self.seed_offset, (bool, np.bool_)) or not isinstance(
            self.seed_offset, (int, np.integer)
        ):
            raise ValueError("seed_offset must be a non-negative integer")
        if int(self.seed_offset) < 0:
            raise ValueError("seed_offset must be a non-negative integer")
        object.__setattr__(self, "mu_lambda", mu_lambda)
        object.__setattr__(self, "sigma_lambda", sigma_lambda)
        object.__setattr__(self, "p_zero", p_zero)
        object.__setattr__(self, "true_model", model)
        object.__setattr__(self, "seed_offset", int(self.seed_offset))


@dataclass(frozen=True)
class TrajectoryScenario:
    """Typed ground truth for one canonical contact-trajectory experiment."""

    scenario: str
    label: str
    mu_lambda: float
    sigma_lambda: float
    p0: float
    sigma_eta: float
    beta_f: float
    beta_s: float
    true_model: str
    seed_offset: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario", _label(self.scenario, "scenario"))
        object.__setattr__(self, "label", _label(self.label, "label"))
        mu_lambda = _finite(self.mu_lambda, "mu_lambda")
        sigma_lambda = _finite(self.sigma_lambda, "sigma_lambda")
        p0 = _finite(self.p0, "p0")
        sigma_eta = _finite(self.sigma_eta, "sigma_eta")
        beta_f = _finite(self.beta_f, "beta_f")
        beta_s = _finite(self.beta_s, "beta_s")
        if mu_lambda <= 0:
            raise ValueError("mu_lambda must be greater than zero")
        if sigma_lambda < 0:
            raise ValueError("sigma_lambda must be non-negative")
        if not 0 < p0 < 1:
            raise ValueError("p0 must satisfy 0 < p0 < 1")
        if sigma_eta < 0:
            raise ValueError("sigma_eta must be non-negative")
        model = _label(self.true_model, "true_model").lower()
        if model not in TRAJECTORY_MODEL_KEYS:
            raise ValueError(f"unknown trajectory true_model {self.true_model!r}")
        expected = _trajectory_truth_model(sigma_eta, beta_f, beta_s)
        if model != expected:
            raise ValueError(
                f"true_model must be {expected!r} for the supplied decision "
                "heterogeneity and history effects"
            )
        if isinstance(self.seed_offset, (bool, np.bool_)) or not isinstance(
            self.seed_offset, (int, np.integer)
        ):
            raise ValueError("seed_offset must be a non-negative integer")
        if int(self.seed_offset) < 0:
            raise ValueError("seed_offset must be a non-negative integer")
        object.__setattr__(self, "mu_lambda", mu_lambda)
        object.__setattr__(self, "sigma_lambda", sigma_lambda)
        object.__setattr__(self, "p0", p0)
        object.__setattr__(self, "sigma_eta", sigma_eta)
        object.__setattr__(self, "beta_f", beta_f)
        object.__setattr__(self, "beta_s", beta_s)
        object.__setattr__(self, "true_model", model)
        object.__setattr__(self, "seed_offset", int(self.seed_offset))

    @property
    def mu_eta(self) -> float:
        """Logit-scale population killing propensity implied by ``p0``."""

        return float(np.log(self.p0 / (1.0 - self.p0)))


COUNT_SCENARIOS: Final[tuple[EventCountScenario, ...]] = (
    EventCountScenario(
        "No1",
        "No1: sigma_lambda=3, p_zero=0.2",
        4.0,
        3.0,
        0.2,
        "hetero3",
        1,
    ),
    EventCountScenario(
        "No2",
        "No2: sigma_lambda=0, p_zero=0.2",
        4.0,
        0.0,
        0.2,
        "z2p",
        2,
    ),
    EventCountScenario(
        "No3",
        "No3: sigma_lambda=3, p_zero=0",
        4.0,
        3.0,
        0.0,
        "dis2p",
        3,
    ),
    EventCountScenario(
        "No4",
        "No4: sigma_lambda=0, p_zero=0",
        4.0,
        0.0,
        0.0,
        "homo",
        4,
    ),
)

TRAJECTORY_SCENARIOS: Final[tuple[TrajectoryScenario, ...]] = (
    TrajectoryScenario(
        "No1",
        "No1: sigma_eta=1, beta=(0.8,-0.8)",
        4.0,
        2.0,
        0.25,
        1.0,
        0.8,
        -0.8,
        "heterogeneous_history_dependent",
        1,
    ),
    TrajectoryScenario(
        "No2",
        "No2: sigma_eta=1, beta=(0,0)",
        4.0,
        2.0,
        0.25,
        1.0,
        0.0,
        0.0,
        "heterogeneous_history_independent",
        2,
    ),
    TrajectoryScenario(
        "No3",
        "No3: sigma_eta=0, beta=(0.8,-0.8)",
        4.0,
        2.0,
        0.25,
        0.0,
        0.8,
        -0.8,
        "homogeneous_history_dependent",
        3,
    ),
    TrajectoryScenario(
        "No4",
        "No4: sigma_eta=0, beta=(0,0)",
        4.0,
        2.0,
        0.25,
        0.0,
        0.0,
        0.0,
        "homogeneous_history_independent",
        4,
    ),
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "__fspath__"):
        return str(value)
    raise TypeError(f"cannot derive a stable seed from {type(value).__name__}")


def stable_seed(*parts: Any, namespace: str = "barracuda") -> int:
    """Derive a reproducible non-zero uint32 seed from structured values.

    Unlike Python's built-in ``hash``, this value is stable across processes.
    Mappings are JSON encoded with sorted keys, so their insertion order does
    not affect the seed.
    """

    scope = _label(namespace, "namespace")
    payload = json.dumps(
        parts,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")
    digest = hashlib.blake2s(
        scope.encode("utf-8") + b"\0" + payload,
        digest_size=4,
    ).digest()
    return 1 + int.from_bytes(digest, "little") % (2**32 - 2)


@dataclass(frozen=True)
class EventCountValidationResult:
    """Complete in-memory result for one simulated event-count validation."""

    scenario: EventCountScenario
    replicate: int
    simulation_seed: int
    inference_seed: int
    frame: pd.DataFrame
    truth: Mapping[str, Any]
    fits: Mapping[str, Any]
    evidence: pd.DataFrame
    recovery: pd.DataFrame


@dataclass(frozen=True)
class TrajectoryValidationResult:
    """Complete in-memory result for one simulated trajectory validation."""

    scenario: TrajectoryScenario
    replicate: int
    simulation_seed: int
    inference_seed: int
    frame: pd.DataFrame
    truth: Mapping[str, Any]
    fits: Mapping[str, Any]
    evidence: pd.DataFrame
    recovery: pd.DataFrame


@dataclass(frozen=True)
class PosteriorProbabilityResult:
    """Exact cross-draw probabilities for ``first - second`` and a ROPE."""

    rope_lower: float
    rope_upper: float
    probability_below: float
    probability_in_rope: float
    probability_above: float
    n_first: int
    n_second: int
    n_pairs: int

    @property
    def probability_first_superior(self) -> float:
        """Probability that ``first - second`` is above the ROPE."""

        return self.probability_above

    @property
    def probability_second_superior(self) -> float:
        """Probability that ``first - second`` is below the ROPE."""

        return self.probability_below

    def as_dict(self) -> dict[str, int | float]:
        """Return a serialization-friendly representation."""

        return asdict(self)


_RECOVERY_COLUMNS: Final[list[str]] = [
    "condition",
    "model_key",
    "parameter",
    "posterior_variable",
    "truth",
    "mean",
    "median",
    "sd",
    "hdi_lower",
    "hdi_upper",
    "hdi_probability",
    "error",
    "absolute_error",
    "relative_error",
    "covered",
    "n_draws",
]


def _posterior_values(idata: Any, variable: str) -> np.ndarray:
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        raise ValueError("idata must contain a posterior group")
    if variable not in posterior:
        raise ValueError(f"posterior variable {variable!r} is not present")
    values = posterior[variable]
    extra_dims = [dim for dim in values.dims if dim not in {"chain", "draw"}]
    if any(int(values.sizes[dim]) != 1 for dim in extra_dims):
        raise ValueError(f"posterior variable {variable!r} is not scalar")
    vector = np.asarray(values, dtype=float).reshape(-1)
    vector = vector[np.isfinite(vector)]
    if vector.size == 0:
        raise ValueError(f"posterior variable {variable!r} has no finite draws")
    return vector


def posterior_recovery_table(
    idata: Any,
    truth: Mapping[str, Any],
    *,
    parameters: Sequence[str] | None = None,
    parameter_map: Mapping[str, str] | None = None,
    model_key: str | None = None,
    condition: str | None = None,
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    """Compare scalar posterior parameters with their generating truths.

    ``parameter_map`` maps public/truth names to posterior variable names.  The
    returned error is ``posterior mean - truth``; ``covered`` states whether
    the closed HDI contains the truth.
    """

    if not isinstance(truth, Mapping):
        raise TypeError("truth must be a mapping")
    probability = _finite(hdi_prob, "hdi_prob")
    if not 0 < probability < 1:
        raise ValueError("hdi_prob must be between zero and one")
    mapping = {str(key): str(value) for key, value in (parameter_map or {}).items()}
    if parameters is None:
        requested = list(mapping) if mapping else [str(key) for key in truth]
    else:
        requested = [str(parameter) for parameter in parameters]
        if not requested:
            raise ValueError("parameters must contain at least one parameter")
    if len(requested) != len(set(requested)):
        raise ValueError("parameters must not contain duplicates")

    rows: list[dict[str, Any]] = []
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        raise ValueError("idata must contain a posterior group")
    for parameter in requested:
        if parameter not in truth:
            raise ValueError(f"truth does not contain parameter {parameter!r}")
        posterior_variable = mapping.get(parameter, parameter)
        if posterior_variable not in posterior:
            if parameters is None and not mapping:
                continue
            raise ValueError(
                f"posterior variable {posterior_variable!r} for {parameter!r} is not present"
            )
        true_value = _finite(truth[parameter], f"truth[{parameter!r}]")
        values = _posterior_values(idata, posterior_variable)
        hdi = np.asarray(az.hdi(values, hdi_prob=probability), dtype=float).reshape(-1)
        if hdi.size != 2 or not np.isfinite(hdi).all():
            raise RuntimeError(f"could not calculate a finite HDI for {parameter!r}")
        mean = float(np.mean(values))
        error = mean - true_value
        rows.append(
            {
                "condition": None if condition is None else str(condition),
                "model_key": None if model_key is None else str(model_key).lower(),
                "parameter": parameter,
                "posterior_variable": posterior_variable,
                "truth": true_value,
                "mean": mean,
                "median": float(np.median(values)),
                "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "hdi_lower": float(hdi[0]),
                "hdi_upper": float(hdi[1]),
                "hdi_probability": probability,
                "error": error,
                "absolute_error": abs(error),
                "relative_error": (
                    error / abs(true_value) if true_value != 0.0 else np.nan
                ),
                "covered": bool(hdi[0] <= true_value <= hdi[1]),
                "n_draws": int(values.size),
            }
        )
    return pd.DataFrame(rows, columns=_RECOVERY_COLUMNS)


_COUNT_PARAMETER_MAPS: Final[dict[str, dict[str, str]]] = {
    "homo": {"mu_lambda": "lambda"},
    "z2p": {"mu_lambda": "lambda", "p_zero": "p_zero"},
    "dis2p": {"mu_lambda": "mu_lambda", "sigma_lambda": "sigma_lambda"},
    "hetero3": {
        "mu_lambda": "mu_lambda",
        "sigma_lambda": "sigma_lambda",
        "p_zero": "p_zero",
    },
}


def event_count_recovery_table(
    results: Mapping[str, Any],
    truth: Mapping[str, Any],
    *,
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    """Build recovery rows for every fitted event-count model."""

    if not isinstance(results, Mapping):
        raise TypeError("results must be a mapping")
    tables: list[pd.DataFrame] = []
    for requested_key, result in results.items():
        key = str(getattr(result, "model_key", requested_key)).lower()
        if key not in _COUNT_PARAMETER_MAPS:
            raise ValueError(f"unknown event-count model key {key!r}")
        mapping = _COUNT_PARAMETER_MAPS[key]
        tables.append(
            posterior_recovery_table(
                result.idata,
                truth,
                parameters=tuple(mapping),
                parameter_map=mapping,
                model_key=key,
                hdi_prob=hdi_prob,
            )
        )
    if not tables:
        return pd.DataFrame(columns=_RECOVERY_COLUMNS)
    return pd.concat(tables, ignore_index=True).reindex(columns=_RECOVERY_COLUMNS)


def trajectory_recovery_table(
    results: Mapping[str, Any],
    truth: Mapping[str, Any],
    *,
    condition: str | None = None,
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    """Build recovery rows for every fitted trajectory decision model."""

    if not isinstance(results, Mapping):
        raise TypeError("results must be a mapping")
    tables: list[pd.DataFrame] = []
    for requested_key, result in results.items():
        key = str(getattr(result, "model_key", requested_key)).lower()
        if key not in trajectories.TRAJECTORY_MODEL_SPECS:
            raise ValueError(f"unknown trajectory model key {key!r}")
        parameters = trajectories.TRAJECTORY_MODEL_SPECS[key].parameters
        mapping = {
            parameter: trajectories.PUBLIC_TO_BACKEND_PARAMETER[parameter]
            for parameter in parameters
        }
        tables.append(
            posterior_recovery_table(
                result.idata,
                truth,
                parameters=parameters,
                parameter_map=mapping,
                model_key=key,
                condition=condition,
                hdi_prob=hdi_prob,
            )
        )
    if not tables:
        return pd.DataFrame(columns=_RECOVERY_COLUMNS)
    return pd.concat(tables, ignore_index=True).reindex(columns=_RECOVERY_COLUMNS)


def _validated_group_columns(
    frame: pd.DataFrame,
    group_by: Sequence[str],
) -> list[str]:
    columns = [str(column) for column in group_by]
    if not columns:
        raise ValueError("group_by must contain at least one column")
    if len(columns) != len(set(columns)):
        raise ValueError("group_by must not contain duplicates")
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"recovery is missing group columns: {', '.join(missing)}")
    return columns


def coverage_summary(
    recovery: pd.DataFrame,
    *,
    group_by: Sequence[str] = ("parameter",),
) -> pd.DataFrame:
    """Aggregate HDI coverage, bias, RMSE, and interval width."""

    if not isinstance(recovery, pd.DataFrame):
        raise TypeError("recovery must be a pandas DataFrame")
    required = {
        "covered",
        "error",
        "absolute_error",
        "hdi_lower",
        "hdi_upper",
    }
    missing = sorted(required.difference(recovery.columns))
    if missing:
        raise ValueError(f"recovery is missing columns: {', '.join(missing)}")
    groups = _validated_group_columns(recovery, group_by)
    columns = [
        *groups,
        "n_runs",
        "coverage_rate",
        "mean_error",
        "mean_absolute_error",
        "rmse",
        "mean_hdi_width",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in recovery.groupby(groups, sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        error = pd.to_numeric(group["error"], errors="raise").to_numpy(float)
        absolute = pd.to_numeric(
            group["absolute_error"], errors="raise"
        ).to_numpy(float)
        widths = (
            pd.to_numeric(group["hdi_upper"], errors="raise").to_numpy(float)
            - pd.to_numeric(group["hdi_lower"], errors="raise").to_numpy(float)
        )
        if (
            not np.isfinite(error).all()
            or not np.isfinite(absolute).all()
            or not np.isfinite(widths).all()
        ):
            raise ValueError("recovery errors and HDI widths must be finite")
        rows.append(
            {
                **dict(zip(groups, key_values, strict=True)),
                "n_runs": int(len(group)),
                "coverage_rate": float(group["covered"].astype(bool).mean()),
                "mean_error": float(np.mean(error)),
                "mean_absolute_error": float(np.mean(absolute)),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "mean_hdi_width": float(np.mean(widths)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


_DEFAULT_BOUNDARIES: Final[dict[str, float]] = {
    "sigma_lambda": 0.0,
    "p_zero": 0.0,
    "sigma_eta": 0.0,
    "beta_f": 0.0,
    "beta_s": 0.0,
}


def boundary_recovery_summary(
    recovery: pd.DataFrame,
    *,
    boundaries: Mapping[str, float] | None = None,
    truth_tolerance: float = 1e-12,
    estimate_tolerance: float = 0.1,
    group_by: Sequence[str] = ("model_key", "parameter"),
) -> pd.DataFrame:
    """Summarise recovery rows whose generating truth lies on a boundary.

    ``boundary_coverage_rate`` records whether the posterior HDI includes the
    boundary.  ``estimate_within_tolerance_rate`` uses the posterior mean and
    the absolute ``estimate_tolerance`` supplied by the caller.
    """

    if not isinstance(recovery, pd.DataFrame):
        raise TypeError("recovery must be a pandas DataFrame")
    required = {"parameter", "truth", "mean", "covered", "absolute_error"}
    missing = sorted(required.difference(recovery.columns))
    if missing:
        raise ValueError(f"recovery is missing columns: {', '.join(missing)}")
    truth_tol = _finite(truth_tolerance, "truth_tolerance")
    estimate_tol = _finite(estimate_tolerance, "estimate_tolerance")
    if truth_tol < 0 or estimate_tol < 0:
        raise ValueError("boundary tolerances must be non-negative")
    boundary_values = {
        str(parameter): _finite(value, f"boundaries[{parameter!r}]")
        for parameter, value in (boundaries or _DEFAULT_BOUNDARIES).items()
    }
    groups = _validated_group_columns(recovery, group_by)
    selected = recovery.copy()
    selected["_boundary"] = selected["parameter"].map(boundary_values)
    selected = selected.loc[selected["_boundary"].notna()].copy()
    if not selected.empty:
        selected = selected.loc[
            np.isclose(
                pd.to_numeric(selected["truth"], errors="raise").to_numpy(float),
                selected["_boundary"].to_numpy(float),
                atol=truth_tol,
                rtol=0.0,
            )
        ]
    columns = [
        *groups,
        "boundary",
        "n_runs",
        "mean_estimate",
        "mean_absolute_error",
        "boundary_coverage_rate",
        "boundary_exclusion_rate",
        "estimate_within_tolerance_rate",
        "estimate_tolerance",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in selected.groupby(groups, sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        boundary = float(group["_boundary"].iloc[0])
        means = pd.to_numeric(group["mean"], errors="raise").to_numpy(float)
        if not np.isfinite(means).all():
            raise ValueError("boundary recovery estimates must be finite")
        covered = group["covered"].astype(bool).to_numpy()
        rows.append(
            {
                **dict(zip(groups, key_values, strict=True)),
                "boundary": boundary,
                "n_runs": int(len(group)),
                "mean_estimate": float(np.mean(means)),
                "mean_absolute_error": float(
                    pd.to_numeric(
                        group["absolute_error"], errors="raise"
                    ).mean()
                ),
                "boundary_coverage_rate": float(np.mean(covered)),
                "boundary_exclusion_rate": float(1.0 - np.mean(covered)),
                "estimate_within_tolerance_rate": float(
                    np.mean(np.abs(means - boundary) <= estimate_tol)
                ),
                "estimate_tolerance": estimate_tol,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _finite_draws(values: Any, name: str) -> np.ndarray:
    try:
        draws = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric posterior draws") from exc
    if draws.size == 0:
        raise ValueError(f"{name} must contain at least one draw")
    if not np.isfinite(draws).all():
        raise ValueError(f"{name} must contain only finite draws")
    return draws


def posterior_superiority_probability(
    first: Any,
    second: Any,
    *,
    margin: float = 0.0,
) -> float:
    """Return exact ``P(first - second > margin)`` over all cross-draw pairs.

    The samples are treated as independent posterior populations.  Sorting and
    binary search avoid constructing the potentially huge Cartesian product.
    """

    first_draws = _finite_draws(first, "first")
    second_draws = np.sort(_finite_draws(second, "second"))
    threshold = _finite(margin, "margin")
    favourable = np.searchsorted(
        second_draws,
        first_draws - threshold,
        side="left",
    ).sum(dtype=np.int64)
    return float(favourable / (first_draws.size * second_draws.size))


def posterior_rope_probabilities(
    first: Any,
    second: Any,
    *,
    rope: tuple[float, float] = (-0.1, 0.1),
) -> PosteriorProbabilityResult:
    """Return exact probabilities below, within, and above a difference ROPE.

    The ROPE is closed: ``lower <= first - second <= upper``.  The below and
    above events are strict, so the three returned probabilities partition all
    cross-draw pairs exactly, including ties.
    """

    if len(rope) != 2:
        raise ValueError("rope must contain lower and upper bounds")
    lower = _finite(rope[0], "rope lower")
    upper = _finite(rope[1], "rope upper")
    if lower > upper:
        raise ValueError("rope lower bound must not exceed the upper bound")
    first_draws = _finite_draws(first, "first")
    second_draws = np.sort(_finite_draws(second, "second"))
    n_second = second_draws.size
    below = (
        n_second
        - np.searchsorted(second_draws, first_draws - lower, side="right")
    ).sum(dtype=np.int64)
    above = np.searchsorted(
        second_draws,
        first_draws - upper,
        side="left",
    ).sum(dtype=np.int64)
    pairs = int(first_draws.size * n_second)
    within = pairs - int(below) - int(above)
    return PosteriorProbabilityResult(
        rope_lower=lower,
        rope_upper=upper,
        probability_below=float(below / pairs),
        probability_in_rope=float(within / pairs),
        probability_above=float(above / pairs),
        n_first=int(first_draws.size),
        n_second=int(n_second),
        n_pairs=pairs,
    )


def _selected_models(
    requested: Sequence[str] | str | None,
    available: Sequence[str],
) -> tuple[str, ...]:
    if requested is None:
        return tuple(available)
    values = [requested] if isinstance(requested, str) else list(requested)
    if not values:
        raise ValueError("model_keys must contain at least one model")
    selected = tuple(str(value).strip().lower() for value in values)
    if len(selected) != len(set(selected)):
        raise ValueError("model_keys must not contain duplicates")
    unknown = sorted(set(selected).difference(available))
    if unknown:
        raise ValueError(f"unknown model_keys: {', '.join(unknown)}")
    return selected


def _single_evidence_table(
    fits: Mapping[str, Any],
    *,
    true_model: str,
) -> pd.DataFrame:
    values: dict[str, float] = {}
    for requested_key, result in fits.items():
        key = str(getattr(result, "model_key", requested_key)).lower()
        value = _finite(getattr(result, "log_evidence"), f"log_evidence[{key!r}]")
        if key in values:
            raise ValueError(f"duplicate fitted model key {key!r}")
        values[key] = value
    if true_model not in values:
        raise ValueError("the fitted model set must include the true model")
    true_log_evidence = values[true_model]
    best_model = max(values, key=values.get)
    best_log_evidence = values[best_model]
    rows = []
    for model_key, log_evidence in values.items():
        log_bf_true = log_evidence - true_log_evidence
        log_bf_best = log_evidence - best_log_evidence
        rows.append(
            {
                "model_key": model_key,
                "true_model": true_model,
                "best_model": best_model,
                "log_evidence": log_evidence,
                "log_bf_model_vs_true": log_bf_true,
                "log10_bf_model_vs_true": log_bf_true / np.log(10.0),
                "log_bf_model_vs_best": log_bf_best,
                "log10_bf_model_vs_best": log_bf_best / np.log(10.0),
                "is_best": bool(np.isclose(log_evidence, best_log_evidence)),
            }
        )
    return pd.DataFrame(rows)


# Module-level adapters are intentional: tests and downstream orchestration can
# replace the expensive fit functions without importing or running PyMC.
simulate_event_count_data = event_counts.simulate_event_counts
fit_event_count_models = event_counts.run_count_models
simulate_trajectory_data = trajectories.simulate_trajectory_frame
fit_trajectory_models = trajectories.run_trajectory_conditions


def run_event_count_validation(
    scenario: EventCountScenario,
    n_cells: int,
    *,
    observation_time: float = 1.0,
    replicate: int = 1,
    base_seed: int = 2026,
    settings: event_counts.InferenceSettings | None = None,
    model_keys: Sequence[str] | str | None = None,
    hdi_prob: float = 0.95,
    progress_callback: Any = None,
) -> EventCountValidationResult:
    """Simulate, fit, and assess one event-count validation dataset."""

    if not isinstance(scenario, EventCountScenario):
        raise TypeError("scenario must be an EventCountScenario")
    replicate_number = _positive_int(replicate, "replicate")
    seed_base = _base_seed(base_seed)
    if settings is not None and not isinstance(settings, event_counts.InferenceSettings):
        raise TypeError("settings must be an InferenceSettings instance or None")
    selected = _selected_models(model_keys, COUNT_MODEL_KEYS)
    if scenario.true_model not in selected:
        raise ValueError("model_keys must include scenario.true_model")
    simulation_seed = stable_seed(
        "event_count",
        "simulation",
        seed_base,
        scenario.scenario,
        scenario.seed_offset,
        replicate_number,
    )
    inference_seed = stable_seed(
        "event_count",
        "inference",
        simulation_seed,
        selected,
    )
    frame, truth = simulate_event_count_data(
        model_key=scenario.true_model,
        n_cells=n_cells,
        obs_time=observation_time,
        mu_lambda=scenario.mu_lambda,
        sigma_lambda=scenario.sigma_lambda,
        p_zero=scenario.p_zero,
        seed=simulation_seed,
    )
    controls = (
        event_counts.InferenceSettings(seed=inference_seed)
        if settings is None
        else replace(settings, seed=inference_seed)
    )
    fits = fit_event_count_models(
        frame,
        observation_time,
        settings=controls,
        model_keys=selected,
        progress_callback=progress_callback,
    )
    recovery = event_count_recovery_table(fits, truth, hdi_prob=hdi_prob)
    evidence = _single_evidence_table(fits, true_model=scenario.true_model)
    return EventCountValidationResult(
        scenario=scenario,
        replicate=replicate_number,
        simulation_seed=simulation_seed,
        inference_seed=inference_seed,
        frame=frame.copy(),
        truth=dict(truth),
        fits=dict(fits),
        evidence=evidence,
        recovery=recovery,
    )


def run_trajectory_validation(
    scenario: TrajectoryScenario,
    n_cells: int,
    *,
    observation_time: float = 1.0,
    replicate: int = 1,
    base_seed: int = 2026,
    settings: trajectories.TrajectorySettings | None = None,
    model_keys: Sequence[str] | str | None = None,
    hdi_prob: float = 0.95,
    progress_callback: Any = None,
) -> TrajectoryValidationResult:
    """Simulate, fit, and assess one trajectory validation dataset."""

    if not isinstance(scenario, TrajectoryScenario):
        raise TypeError("scenario must be a TrajectoryScenario")
    replicate_number = _positive_int(replicate, "replicate")
    seed_base = _base_seed(base_seed)
    if settings is not None and not isinstance(settings, trajectories.TrajectorySettings):
        raise TypeError("settings must be a TrajectorySettings instance or None")
    selected = _selected_models(model_keys, TRAJECTORY_MODEL_KEYS)
    if scenario.true_model not in selected:
        raise ValueError("model_keys must include scenario.true_model")
    simulation_seed = stable_seed(
        "trajectory",
        "simulation",
        seed_base,
        scenario.scenario,
        scenario.seed_offset,
        replicate_number,
    )
    inference_seed = stable_seed(
        "trajectory",
        "inference",
        simulation_seed,
        selected,
    )
    frame, truths = simulate_trajectory_data(
        condition=scenario.scenario,
        n_cells=n_cells,
        mu_lambda=scenario.mu_lambda,
        sigma_lambda=scenario.sigma_lambda,
        p0=scenario.p0,
        sigma_eta=scenario.sigma_eta,
        beta_f=scenario.beta_f,
        beta_s=scenario.beta_s,
        observation_time=observation_time,
        seed=simulation_seed,
    )
    controls = (
        trajectories.TrajectorySettings(seed=inference_seed)
        if settings is None
        else replace(settings, seed=inference_seed)
    )
    nested_fits = fit_trajectory_models(
        frame,
        observation_time=observation_time,
        settings=controls,
        model_keys=selected,
        progress_callback=progress_callback,
    )
    if scenario.scenario not in nested_fits:
        raise RuntimeError(
            f"trajectory fit returned no condition {scenario.scenario!r}"
        )
    fits = nested_fits[scenario.scenario]
    truth = dict(truths[scenario.scenario])
    recovery = trajectory_recovery_table(
        fits,
        truth,
        condition=scenario.scenario,
        hdi_prob=hdi_prob,
    )
    evidence = _single_evidence_table(fits, true_model=scenario.true_model)
    return TrajectoryValidationResult(
        scenario=scenario,
        replicate=replicate_number,
        simulation_seed=simulation_seed,
        inference_seed=inference_seed,
        frame=frame.copy(),
        truth=truth,
        fits=dict(fits),
        evidence=evidence,
        recovery=recovery,
    )


__all__ = [
    "COUNT_MODEL_KEYS",
    "COUNT_SCENARIOS",
    "TRAJECTORY_MODEL_KEYS",
    "TRAJECTORY_SCENARIOS",
    "EventCountScenario",
    "EventCountValidationResult",
    "PosteriorProbabilityResult",
    "TrajectoryScenario",
    "TrajectoryValidationResult",
    "boundary_recovery_summary",
    "coverage_summary",
    "event_count_recovery_table",
    "fit_event_count_models",
    "fit_trajectory_models",
    "posterior_recovery_table",
    "posterior_rope_probabilities",
    "posterior_superiority_probability",
    "run_event_count_validation",
    "run_trajectory_validation",
    "simulate_event_count_data",
    "simulate_trajectory_data",
    "stable_seed",
    "trajectory_recovery_table",
]
