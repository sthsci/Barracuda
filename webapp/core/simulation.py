"""Synthetic datasets for validating the four event-count models."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

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
    "homo": "Homogeneous Poisson",
    "z2p": "Zero-inflated Poisson",
    "dis2p": "Distributed-rate Gamma–Poisson",
    "hetero3": "Zero-inflated Gamma–Poisson",
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


def simulate_event_counts(
    model_key: str,
    n_cells: int,
    obs_time: float,
    mu_lambda: float,
    sigma_lambda: float,
    p_zero: float,
    seed: int | None,
) -> tuple[pd.DataFrame, dict[str, int | float | str | None]]:
    """Simulate one canonical event-count frame and its ground truth.

    The distributed-rate models draw per-cell rates from a Gamma distribution
    parameterized by its mean and standard deviation. Zero-inflated models set
    a fraction of cell rates to exactly zero before Poisson counts are sampled.
    Parameters not used by the selected model are recorded as zero in ``truth``.
    Passing ``seed=None`` requests a fresh, non-deterministic random stream.
    """

    key = _canonical_model_key(model_key)
    cells = _positive_integer(n_cells, "n_cells")
    if not MIN_CELLS <= cells <= MAX_CELLS:
        raise ValueError(
            f"n_cells must be between {MIN_CELLS} and {MAX_CELLS:,} for this demo"
        )
    observation_time = validate_observation_time(obs_time)
    mean_rate = _finite_float(mu_lambda, "mu_lambda")
    rate_sd = _finite_float(sigma_lambda, "sigma_lambda")
    zero_probability = _finite_float(p_zero, "p_zero")
    random_seed = _validate_seed(seed)

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

    rng = np.random.default_rng(random_seed)
    if effective_sd == 0:
        rates = np.full(cells, mean_rate, dtype=float)
    else:
        shape = (mean_rate / effective_sd) ** 2
        scale = effective_sd**2 / mean_rate
        rates = rng.gamma(shape=shape, scale=scale, size=cells)

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
    truth: dict[str, int | float | str | None] = {
        "model_key": key,
        "model_label": MODEL_LABELS[key],
        "n_cells": cells,
        "observation_time": observation_time,
        "mu_lambda": mean_rate,
        "sigma_lambda": effective_sd,
        "p_zero": effective_p_zero,
        "seed": random_seed,
        "realized_structural_zeros": int(structural_zeros.sum()),
        "realized_mean_rate": float(rates.mean()),
        "realized_mean_count": float(counts.mean()),
    }
    return frame, truth
