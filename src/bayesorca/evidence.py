"""Model-evidence and Bayes-factor utilities.

The inference workflows return log marginal likelihoods because their raw
Bayes factors can overflow even for moderately decisive comparisons.  This
module keeps calculations in log space for as long as possible and provides
tables that can be used with event-count and trajectory results alike.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import combinations
import math
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import gaussian_kde


LOG_10: Final[float] = float(np.log(10.0))
_MAX_EXPONENT: Final[float] = float(np.log(np.finfo(float).max))
_MIN_EXPONENT: Final[float] = float(np.log(np.nextafter(0.0, 1.0)))


@dataclass(frozen=True)
class SavageDickeyResult:
    """Density-ratio evidence for a point null nested in a larger model.

    ``bf_01`` supports the point null and ``bf_10`` supports the alternative.
    The calculation assumes that the nuisance-parameter priors under both
    models are compatible, as required by the Savage--Dickey identity.
    """

    parameter: str
    reference: float
    prior_density: float
    posterior_density: float
    bf_01: float
    bf_10: float
    log_bf_10: float

    def as_dict(self) -> dict[str, float | str]:
        """Return a serialization-friendly representation."""

        return asdict(self)


def _finite_number(value: Any, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _validated_log_evidence(
    log_evidence: Mapping[str, float],
) -> dict[str, float]:
    if not isinstance(log_evidence, Mapping):
        raise TypeError("log_evidence must be a mapping of model names to values")
    if not log_evidence:
        raise ValueError("log_evidence must contain at least one model")
    values: dict[str, float] = {}
    for model, value in log_evidence.items():
        label = str(model).strip()
        if not label:
            raise ValueError("model names must not be empty")
        if label in values:
            raise ValueError(f"duplicate model name after normalization: {label!r}")
        values[label] = _finite_number(value, f"log_evidence[{label!r}]")
    return values


def _safe_exp(value: float) -> float:
    if value > _MAX_EXPONENT:
        return math.inf
    if value < _MIN_EXPONENT:
        return 0.0
    return float(np.exp(value))


def log_bayes_factor(log_evidence_1: float, log_evidence_2: float) -> float:
    """Return ``log p(data|model_1) - log p(data|model_2)``.

    Positive values favour model 1, negative values favour model 2, and zero
    means equal evidence under the fitted priors.
    """

    first = _finite_number(log_evidence_1, "log_evidence_1")
    second = _finite_number(log_evidence_2, "log_evidence_2")
    return first - second


def bayes_factor(log_evidence_1: float, log_evidence_2: float) -> float:
    """Return the Bayes factor for model 1 against model 2.

    Very large values are returned as ``inf`` and underflowing values as
    ``0.0``.  Use :func:`log_bayes_factor` for lossless downstream work.
    """

    return _safe_exp(log_bayes_factor(log_evidence_1, log_evidence_2))


def classify_bayes_factor(log_bf: float) -> str:
    """Describe evidence strength using Kass--Raftery log-BF thresholds.

    The magnitude thresholds are 1, 3, and 5 natural-log units.  The returned
    label describes strength only; inspect the sign to determine which model
    is favoured.
    """

    magnitude = abs(_finite_number(log_bf, "log_bf"))
    if magnitude < 1.0:
        return "negligible"
    if magnitude < 3.0:
        return "positive"
    if magnitude < 5.0:
        return "strong"
    return "very strong"


def pairwise_bayes_factors(
    log_evidence: Mapping[str, float],
    *,
    model_order: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compare every pair of models from a log-evidence mapping.

    Parameters
    ----------
    log_evidence:
        Mapping from model identifiers to finite log marginal likelihoods.
    model_order:
        Optional comparison order.  It must name every supplied model exactly
        once.  When omitted, insertion order is preserved.

    Returns
    -------
    pandas.DataFrame
        One row per unordered pair.  Positive ``log_BF_1_vs_2`` values favour
        ``model_1``.  Both natural-log and base-10 representations are kept so
        plots never need to reconstruct them from rounded raw factors.
    """

    values = _validated_log_evidence(log_evidence)
    if model_order is None:
        models = list(values)
    else:
        models = [str(model).strip() for model in model_order]
        if len(models) != len(set(models)):
            raise ValueError("model_order must not contain duplicates")
        missing = set(values).difference(models)
        extra = set(models).difference(values)
        if missing or extra:
            raise ValueError(
                "model_order must contain every model exactly once "
                f"(missing={sorted(missing)}, extra={sorted(extra)})"
            )

    columns = [
        "model_1",
        "model_2",
        "log_evidence_1",
        "log_evidence_2",
        "log_BF_1_vs_2",
        "log10_BF_1_vs_2",
        "BF_1_vs_2",
        "favoured_model",
        "evidence_strength",
    ]
    rows: list[dict[str, Any]] = []
    for model_1, model_2 in combinations(models, 2):
        delta = log_bayes_factor(values[model_1], values[model_2])
        rows.append(
            {
                "model_1": model_1,
                "model_2": model_2,
                "log_evidence_1": values[model_1],
                "log_evidence_2": values[model_2],
                "log_BF_1_vs_2": delta,
                "log10_BF_1_vs_2": delta / LOG_10,
                "BF_1_vs_2": _safe_exp(delta),
                "favoured_model": (
                    model_1 if delta > 0 else model_2 if delta < 0 else "tie"
                ),
                "evidence_strength": classify_bayes_factor(delta),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def posterior_model_probabilities(
    log_evidence: Mapping[str, float],
    *,
    prior_probabilities: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Convert model evidence and model priors into posterior probabilities.

    Equal model priors are used by default.  User-supplied priors are
    normalized after validation, so they may be probabilities or positive
    relative weights.
    """

    values = _validated_log_evidence(log_evidence)
    models = list(values)
    if prior_probabilities is None:
        prior = np.full(len(models), 1.0 / len(models), dtype=float)
    else:
        normalized_priors = {
            str(model).strip(): value
            for model, value in prior_probabilities.items()
        }
        if set(normalized_priors) != set(models):
            raise ValueError("prior_probabilities must name exactly the evidence models")
        prior = np.asarray(
            [normalized_priors[model] for model in models],
            dtype=float,
        )
        if not np.all(np.isfinite(prior)) or np.any(prior <= 0):
            raise ValueError("prior probabilities must be finite and greater than zero")
        prior = prior / prior.sum()

    evidence_array = np.asarray([values[model] for model in models], dtype=float)
    log_joint = evidence_array + np.log(prior)
    posterior = np.exp(log_joint - logsumexp(log_joint))
    best_index = int(np.argmax(posterior))
    best_probability = float(posterior[best_index])
    table = pd.DataFrame(
        {
            "model_key": models,
            "log_evidence": evidence_array,
            "prior_probability": prior,
            "posterior_probability": posterior,
            "posterior_odds_best_vs_model": [
                math.inf if probability == 0 else best_probability / float(probability)
                for probability in posterior
            ],
            "is_best": np.arange(len(models)) == best_index,
        }
    )
    return table.sort_values(
        ["posterior_probability", "model_key"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def combine_independent_evidence(
    evidence: pd.DataFrame,
    *,
    model_column: str = "model_key",
    log_evidence_column: str = "log_evidence",
    dataset_columns: Sequence[str] | str | None = None,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Sum log evidence across independent datasets for each model.

    The caller is responsible for the scientific independence assumption.  A
    common use is to combine condition-wise model comparisons after fitting the
    same candidate set separately to each condition.  When ``condition`` is
    present it is used as the dataset identifier automatically.  Complete and
    duplicate-free model coverage is required by default, preventing a model
    from winning merely because a difficult dataset was omitted.
    """

    if not isinstance(evidence, pd.DataFrame):
        raise TypeError("evidence must be a pandas DataFrame")
    if dataset_columns is None:
        datasets = ["condition"] if "condition" in evidence.columns else []
    elif isinstance(dataset_columns, str):
        datasets = [dataset_columns]
    else:
        datasets = [str(column) for column in dataset_columns]
    missing = [
        column
        for column in (model_column, log_evidence_column, *datasets)
        if column not in evidence.columns
    ]
    if missing:
        raise ValueError(f"evidence is missing columns: {', '.join(missing)}")
    if evidence.empty:
        return pd.DataFrame(
            columns=[
                "model_key",
                "total_log_evidence",
                "n_datasets",
                "delta_log_evidence_vs_best",
                "log10_BF_best_vs_model",
                "is_best",
            ]
        )
    frame = evidence.loc[:, [*datasets, model_column, log_evidence_column]].copy()
    frame[model_column] = frame[model_column].astype(str)
    frame[log_evidence_column] = pd.to_numeric(
        frame[log_evidence_column], errors="raise"
    )
    if not np.isfinite(frame[log_evidence_column].to_numpy(float)).all():
        raise ValueError("log evidence values must all be finite")
    if datasets:
        keys = [*datasets, model_column]
        if frame.duplicated(keys).any():
            raise ValueError(
                "evidence contains duplicate dataset-model rows for keys: "
                + ", ".join(keys)
            )
        if require_complete:
            expected = {
                tuple(values)
                for values in frame.loc[:, datasets].itertuples(index=False, name=None)
            }
            for model, subset in frame.groupby(model_column, sort=False):
                observed = {
                    tuple(values)
                    for values in subset.loc[:, datasets].itertuples(index=False, name=None)
                }
                if observed != expected:
                    missing_keys = sorted(expected.difference(observed), key=str)
                    extra_keys = sorted(observed.difference(expected), key=str)
                    raise ValueError(
                        f"incomplete dataset coverage for model {model!r} "
                        f"(missing={missing_keys}, extra={extra_keys})"
                    )
    combined = (
        frame.groupby(model_column, sort=False, as_index=False)
        .agg(
            total_log_evidence=(log_evidence_column, "sum"),
            n_datasets=(log_evidence_column, "size"),
        )
        .rename(columns={model_column: "model_key"})
    )
    best = float(combined["total_log_evidence"].max())
    combined["delta_log_evidence_vs_best"] = (
        combined["total_log_evidence"] - best
    )
    combined["log10_BF_best_vs_model"] = (
        -combined["delta_log_evidence_vs_best"] / LOG_10
    )
    combined["is_best"] = np.isclose(combined["total_log_evidence"], best)
    return combined.sort_values(
        ["total_log_evidence", "model_key"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def smc_log_evidence(idata: Any) -> float:
    """Return the mean final SMC log evidence across finite chains."""

    from .diagnostics import smc_log_evidence_by_chain

    table = smc_log_evidence_by_chain(idata)
    values = table.loc[table["is_finite"], "log_evidence"].to_numpy(float)
    if not values.size:
        raise RuntimeError("SMC evidence contains no finite chain estimates")
    return float(np.mean(values))


def evidence_from_inference_data(
    idatas: Mapping[str, Any],
) -> pd.DataFrame:
    """Build a ranked evidence table directly from ``InferenceData`` objects."""

    if not isinstance(idatas, Mapping):
        raise TypeError("idatas must be a mapping of model names to InferenceData")
    columns = [
        "model_key",
        "log_evidence",
        "delta_log_evidence_vs_best",
        "log10_BF_model_vs_best",
        "log10_BF_best_vs_model",
        "is_best",
    ]
    if not idatas:
        return pd.DataFrame(columns=columns)
    rows = [
        {"model_key": str(model), "log_evidence": smc_log_evidence(idata)}
        for model, idata in idatas.items()
    ]
    table = pd.DataFrame(rows)
    best = float(table["log_evidence"].max())
    table["delta_log_evidence_vs_best"] = table["log_evidence"] - best
    table["log10_BF_model_vs_best"] = (
        table["delta_log_evidence_vs_best"] / LOG_10
    )
    table["log10_BF_best_vs_model"] = -table["log10_BF_model_vs_best"]
    table["is_best"] = np.isclose(table["log_evidence"], best)
    return table.loc[:, columns].sort_values(
        ["log_evidence", "model_key"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def savage_dickey_ratio(
    idata: Any,
    parameter: str,
    *,
    reference: float = 0.0,
    bandwidth: str | float | None = None,
) -> SavageDickeyResult:
    """Estimate a Savage--Dickey Bayes factor from prior/posterior draws.

    ``idata`` must contain scalar draws for ``parameter`` in both its ``prior``
    and ``posterior`` groups.  Gaussian kernel density estimates are evaluated
    at ``reference``.  For bounded parameters or boundary nulls, use a method
    designed for boundary correction instead of this helper.
    """

    name = str(parameter).strip()
    if not name:
        raise ValueError("parameter must not be empty")
    point = _finite_number(reference, "reference")
    prior_group = getattr(idata, "prior", None)
    posterior_group = getattr(idata, "posterior", None)
    if prior_group is None:
        raise ValueError("idata must contain prior samples")
    if posterior_group is None:
        raise ValueError("idata must contain posterior samples")
    if name not in prior_group:
        raise ValueError(f"{name!r} is not present in idata.prior")
    if name not in posterior_group:
        raise ValueError(f"{name!r} is not present in idata.posterior")
    prior = np.asarray(prior_group[name], dtype=float).reshape(-1)
    posterior = np.asarray(posterior_group[name], dtype=float).reshape(-1)
    prior = prior[np.isfinite(prior)]
    posterior = posterior[np.isfinite(posterior)]
    if prior.size < 2 or posterior.size < 2:
        raise ValueError("at least two finite prior and posterior draws are required")
    if np.ptp(prior) == 0 or np.ptp(posterior) == 0:
        raise ValueError("prior and posterior draws must have non-zero variance")

    prior_density = float(gaussian_kde(prior, bw_method=bandwidth)(point)[0])
    posterior_density = float(
        gaussian_kde(posterior, bw_method=bandwidth)(point)[0]
    )
    if prior_density < 0 or posterior_density < 0:
        raise RuntimeError("kernel density estimation returned a negative density")
    bf_01 = math.inf if prior_density == 0 else posterior_density / prior_density
    bf_10 = math.inf if posterior_density == 0 else prior_density / posterior_density
    if prior_density == 0 and posterior_density == 0:
        bf_01 = math.nan
        bf_10 = math.nan
        log_bf_10 = math.nan
    elif posterior_density == 0:
        log_bf_10 = math.inf
    elif prior_density == 0:
        log_bf_10 = -math.inf
    else:
        log_bf_10 = float(np.log(prior_density) - np.log(posterior_density))
    return SavageDickeyResult(
        parameter=name,
        reference=point,
        prior_density=prior_density,
        posterior_density=posterior_density,
        bf_01=float(bf_01),
        bf_10=float(bf_10),
        log_bf_10=float(log_bf_10),
    )


def history_effect_bayes_factors(
    idata: Any,
    *,
    parameters: Sequence[str] = ("beta_f", "beta_s"),
    reference: float = 0.0,
    bandwidth: str | float | None = None,
) -> pd.DataFrame:
    """Evaluate point-null Bayes factors for trajectory history effects.

    Public names ``beta_f`` and ``beta_s`` are translated to the research
    backend's ``beta_x`` and ``beta_y`` variables when necessary.
    Parameters that are absent from either the prior or posterior are omitted,
    which makes the function safe across history-independent model results.
    """

    public_to_backend = {"beta_f": "beta_x", "beta_s": "beta_y"}
    prior_group = getattr(idata, "prior", {})
    posterior_group = getattr(idata, "posterior", {})
    rows: list[dict[str, Any]] = []
    for public_name in parameters:
        public = str(public_name).strip()
        backend = public_to_backend.get(public, public)
        if backend not in prior_group or backend not in posterior_group:
            continue
        result = savage_dickey_ratio(
            idata,
            backend,
            reference=reference,
            bandwidth=bandwidth,
        )
        row = result.as_dict()
        row["parameter"] = public
        row["backend_parameter"] = backend
        rows.append(row)
    columns = [
        "parameter",
        "backend_parameter",
        "reference",
        "prior_density",
        "posterior_density",
        "bf_01",
        "bf_10",
        "log_bf_10",
    ]
    return pd.DataFrame(rows, columns=columns)


__all__ = [
    "SavageDickeyResult",
    "bayes_factor",
    "classify_bayes_factor",
    "combine_independent_evidence",
    "evidence_from_inference_data",
    "history_effect_bayes_factors",
    "log_bayes_factor",
    "pairwise_bayes_factors",
    "posterior_model_probabilities",
    "savage_dickey_ratio",
    "smc_log_evidence",
]
