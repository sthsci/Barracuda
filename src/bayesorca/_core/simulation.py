"""Synthetic datasets for validating the four event count models."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import gamma as gamma_distribution
from scipy.stats import lognorm, truncnorm

from .data import MAX_CELLS, MIN_CELLS, validate_observation_time


MODEL_ALIASES: Final[dict[str, str]] = {
    "homo": "homo",
    "homogeneous": "homo",
    "poisson": "homo",
    "z2p": "z2p",
    "zero_inflated": "z2p",
    "zero-inflated": "z2p",
    "zero_inflated_poisson": "z2p",
    "dis2p": "dis2p",
    "distributed": "dis2p",
    "gamma_poisson": "dis2p",
    "hetero3": "hetero3",
    "full": "hetero3",
    "zero_inflated_gamma_poisson": "hetero3",
}

MODEL_LABELS: Final[dict[str, str]] = {
    "homo": "𝓜_homo · Homogeneous Poisson",
    "z2p": "𝓜_ZI · Zero inflated Poisson",
    "dis2p": "𝓜_Γ · Heterogeneous Gamma Poisson",
    "hetero3": "𝓜_ZIΓ · Zero inflated heterogeneous Gamma Poisson",
}

RATE_DISTRIBUTION_LABELS: Final[dict[str, str]] = {
    "fixed": "One shared rate",
    "gamma": "Gamma (used in the paper)",
    "lognormal": "Lognormal (in development)",
    "truncated_normal": "Normal truncated at zero (in development)",
}

PAPER_RATE_DISTRIBUTIONS: Final[dict[str, str]] = {
    "homo": "fixed",
    "z2p": "fixed",
    "dis2p": "gamma",
    "hetero3": "gamma",
}

_RATE_DISTRIBUTION_ALIASES: Final[dict[str, str]] = {
    "fixed": "fixed",
    "constant": "fixed",
    "gamma": "gamma",
    "lognormal": "lognormal",
    "log_normal": "lognormal",
    "normal": "truncated_normal",
    "truncated_normal": "truncated_normal",
    "truncated-normal": "truncated_normal",
}


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_float(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer or None")
    seed = int(seed)
    if seed < 0 or seed > np.iinfo(np.uint32).max:
        raise ValueError("seed must be between 0 and 2**32 - 1")
    return seed


def _canonical_model_key(model_key: str) -> str:
    normalized = str(model_key).strip().lower().replace(" ", "_")
    try:
        return MODEL_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(MODEL_LABELS)
        raise ValueError(f"unknown model_key {model_key!r}; choose one of: {supported}") from exc


def _canonical_rate_distribution(value: str) -> str:
    normalized = str(value).strip().lower().replace(" ", "_")
    try:
        return _RATE_DISTRIBUTION_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(RATE_DISTRIBUTION_LABELS)
        raise ValueError(
            f"unknown rate_distribution {value!r}; choose one of: {supported}"
        ) from exc


def paper_rate_distribution_for_model(model_key: str) -> str:
    """Return the rate distribution fixed by one of the four paper models."""

    return PAPER_RATE_DISTRIBUTIONS[_canonical_model_key(model_key)]


def _truncated_normal_parameters(mean: float, standard_deviation: float) -> tuple[float, float, float]:
    """Return location, scale and lower standardised bound for a positive Normal.

    The returned truncated Normal has the requested mean and standard deviation.
    A positive distribution cannot attain a coefficient of variation of one or
    greater, so those settings are rejected with a useful interface message.
    """

    coefficient_of_variation = standard_deviation / mean
    if not 0 < coefficient_of_variation < 1:
        raise ValueError(
            "For a Normal distribution truncated at zero, σλ must be smaller "
            "than μλ. Choose a smaller σλ or use Gamma or Lognormal."
        )

    if coefficient_of_variation < 0.025:
        location = mean
        scale = standard_deviation
        return location, scale, -location / scale

    def cv_difference(lower_bound: float) -> float:
        truncated_mean, truncated_variance = truncnorm.stats(
            lower_bound,
            np.inf,
            moments="mv",
        )
        truncated_cv = np.sqrt(float(truncated_variance)) / (
            float(truncated_mean) - lower_bound
        )
        return truncated_cv - coefficient_of_variation

    lower_bound = float(brentq(cv_difference, -40.0, 40.0))
    standard_mean, standard_variance = truncnorm.stats(
        lower_bound,
        np.inf,
        moments="mv",
    )
    scale = standard_deviation / np.sqrt(float(standard_variance))
    location = mean - scale * float(standard_mean)
    return location, scale, lower_bound


def rate_distribution_curve(
    rate_distribution: str,
    mu_lambda: float,
    sigma_lambda: float,
    *,
    points: int = 320,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a stable density curve for the engaging-cell rate distribution."""

    distribution = _canonical_rate_distribution(rate_distribution)
    mean_rate = _finite_float(mu_lambda, "mu_lambda")
    rate_sd = _finite_float(sigma_lambda, "sigma_lambda")
    if mean_rate <= 0:
        raise ValueError("μλ must be greater than zero")
    if rate_sd < 0:
        raise ValueError("σλ must be greater than or equal to zero")
    if distribution == "fixed" or rate_sd == 0:
        return np.asarray([mean_rate]), np.asarray([1.0])
    if points < 50:
        raise ValueError("points must be at least 50")

    if distribution == "gamma":
        shape = (mean_rate / rate_sd) ** 2
        scale = rate_sd**2 / mean_rate
        law = gamma_distribution(a=shape, scale=scale)
    elif distribution == "lognormal":
        log_sd = np.sqrt(np.log1p((rate_sd / mean_rate) ** 2))
        log_mean = np.log(mean_rate) - 0.5 * log_sd**2
        law = lognorm(s=log_sd, scale=np.exp(log_mean))
    elif distribution == "truncated_normal":
        location, scale, lower_bound = _truncated_normal_parameters(
            mean_rate,
            rate_sd,
        )
        law = truncnorm(lower_bound, np.inf, loc=location, scale=scale)
    else:  # pragma: no cover - canonicalisation makes this unreachable
        raise AssertionError(distribution)

    lower = max(0.0, float(law.ppf(0.0005)))
    upper = float(law.ppf(0.9995))
    if not np.isfinite(upper) or upper <= lower:
        upper = mean_rate + 6.0 * rate_sd
    x = np.linspace(lower, upper, int(points))
    density = np.asarray(law.pdf(x), dtype=float)
    finite = np.isfinite(density)
    if not finite.all():
        density[~finite] = np.nanmax(density[finite]) if finite.any() else 0.0
    return x, density


def _draw_engaging_rates(
    rng: np.random.Generator,
    distribution: str,
    cells: int,
    mean_rate: float,
    rate_sd: float,
) -> np.ndarray:
    if distribution == "fixed" or rate_sd == 0:
        return np.full(cells, mean_rate, dtype=float)
    if distribution == "gamma":
        shape = (mean_rate / rate_sd) ** 2
        scale = rate_sd**2 / mean_rate
        return rng.gamma(shape=shape, scale=scale, size=cells)
    if distribution == "lognormal":
        log_sd = np.sqrt(np.log1p((rate_sd / mean_rate) ** 2))
        log_mean = np.log(mean_rate) - 0.5 * log_sd**2
        return rng.lognormal(mean=log_mean, sigma=log_sd, size=cells)
    if distribution == "truncated_normal":
        location, scale, lower_bound = _truncated_normal_parameters(
            mean_rate,
            rate_sd,
        )
        return np.asarray(
            truncnorm.rvs(
                lower_bound,
                np.inf,
                loc=location,
                scale=scale,
                size=cells,
                random_state=rng,
            ),
            dtype=float,
        )
    raise AssertionError(distribution)  # pragma: no cover


def simulate_event_counts(
    model_key: str,
    n_cells: int,
    obs_time: float,
    mu_lambda: float,
    sigma_lambda: float,
    p_zero: float,
    seed: int | None,
    *,
    rate_distribution: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int | float | str | None | bool]]:
    """Simulate one canonical event count frame and its ground truth.

    The selected paper model determines the default rate distribution: the
    homogeneous models use one shared rate and the heterogeneous models use a
    Gamma distribution parameterized by its mean and standard deviation.
    Alternative positive distributions remain available to development code,
    but are not exposed by the public web workflow. Zero inflated structures
    set a fraction of rates to exactly zero before Poisson counts are sampled.
    Parameters not used by the selected model are recorded as zero in ``truth``.
    Passing ``seed=None`` requests a fresh random stream.
    """

    key = _canonical_model_key(model_key)
    cells = _positive_integer(n_cells, "n_cells")
    if not MIN_CELLS <= cells <= MAX_CELLS:
        raise ValueError(
            f"n_cells must be between {MIN_CELLS} and {MAX_CELLS:,} for this web application"
        )
    observation_time = validate_observation_time(obs_time)
    mean_rate = _finite_float(mu_lambda, "mu_lambda")
    rate_sd = _finite_float(sigma_lambda, "sigma_lambda")
    zero_probability = _finite_float(p_zero, "p_zero")
    random_seed = _validate_seed(seed)
    requested_distribution = (
        paper_rate_distribution_for_model(key)
        if rate_distribution is None
        else _canonical_rate_distribution(rate_distribution)
    )

    if mean_rate <= 0:
        raise ValueError("mu_lambda must be greater than zero")
    if rate_sd < 0:
        raise ValueError("sigma_lambda must be greater than or equal to zero")
    if not 0 <= zero_probability <= 1:
        raise ValueError("p_zero must be between zero and one")

    uses_distribution = key in {"dis2p", "hetero3"}
    uses_zero_inflation = key in {"z2p", "hetero3"}
    effective_sd = rate_sd if uses_distribution else 0.0
    effective_p_zero = zero_probability if uses_zero_inflation else 0.0
    effective_distribution = requested_distribution if uses_distribution else "fixed"
    if uses_distribution and effective_distribution == "fixed":
        effective_distribution = "gamma"

    rng = np.random.default_rng(random_seed)
    rates = _draw_engaging_rates(
        rng,
        effective_distribution,
        cells,
        mean_rate,
        effective_sd,
    )
    engaging_rates = rates.copy()

    structural_zeros = np.zeros(cells, dtype=bool)
    if effective_p_zero > 0:
        structural_zeros = rng.random(cells) < effective_p_zero
        rates[structural_zeros] = 0.0

    counts = rng.poisson(rates * observation_time).astype(np.int64)
    width = max(4, len(str(cells)))
    frame = pd.DataFrame(
        {
            "cell_id": [
                f"cell_{index:0{width}d}" for index in range(1, cells + 1)
            ],
            "count": counts,
        }
    )
    paper_model = effective_distribution in {"fixed", "gamma"}
    distribution_name = RATE_DISTRIBUTION_LABELS[effective_distribution].split(" (")[0]
    model_label = MODEL_LABELS[key]
    if not paper_model:
        prefix = "Zero inflated " if uses_zero_inflation else ""
        model_label = f"Exploratory {prefix}{distribution_name} Poisson generator"

    truth: dict[str, int | float | str | None | bool] = {
        "model_key": key,
        "model_label": model_label,
        "rate_distribution": effective_distribution,
        "rate_distribution_label": RATE_DISTRIBUTION_LABELS[effective_distribution],
        "is_paper_model": paper_model,
        "n_cells": cells,
        "observation_time": observation_time,
        "mu_lambda": mean_rate,
        "sigma_lambda": effective_sd,
        "p_zero": effective_p_zero,
        "seed": random_seed,
        "realized_structural_zeros": int(structural_zeros.sum()),
        "realized_engaging_mean_rate": float(engaging_rates.mean()),
        "realized_engaging_sd_rate": float(engaging_rates.std(ddof=0)),
        "realized_mean_rate": float(rates.mean()),
        "realized_mean_count": float(counts.mean()),
    }
    return frame, truth
