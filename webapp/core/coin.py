from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.stats import beta as beta_distribution


def simulate_coin_tosses(
    probability_heads: float,
    n_tosses: int,
    *,
    seed: int | None = None,
) -> NDArray[np.int8]:
    """Simulate heads (1) and tails (0) from a coin with known bias."""
    if not 0.0 <= probability_heads <= 1.0:
        raise ValueError("The probability of heads must be between 0 and 1.")
    if n_tosses < 1:
        raise ValueError("The number of tosses must be at least 1.")

    generator = np.random.default_rng(seed)
    return generator.binomial(1, probability_heads, size=n_tosses).astype(np.int8)


def uniform_prior_posterior(heads: int, n_tosses: int) -> tuple[int, int]:
    """Return Beta posterior parameters under a fixed Uniform(0, 1) prior."""
    if n_tosses < 1:
        raise ValueError("The number of tosses must be at least 1.")
    if not 0 <= heads <= n_tosses:
        raise ValueError("Heads must be between 0 and the number of tosses.")

    tails = n_tosses - heads
    return heads + 1, tails + 1


def beta_highest_density_interval(
    alpha: float,
    beta: float,
    probability: float = 0.95,
) -> tuple[float, float]:
    """Return the shortest interval containing the requested Beta probability mass.

    The posterior used by the coin lesson is always unimodal or monotone.  Boundary
    cases therefore have an exact one-sided interval; interior modes are handled by
    minimising the width between two Beta quantiles.
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("Beta shape parameters must be positive.")
    if not 0 < probability < 1:
        raise ValueError("The interval probability must be between 0 and 1.")

    if alpha == 1 and beta == 1:
        tail = (1.0 - probability) / 2.0
        return tail, 1.0 - tail
    if alpha <= 1 < beta:
        return 0.0, float(beta_distribution.ppf(probability, alpha, beta))
    if beta <= 1 < alpha:
        return float(beta_distribution.ppf(1.0 - probability, alpha, beta)), 1.0

    maximum_lower_probability = 1.0 - probability

    def interval_width(lower_probability: float) -> float:
        lower = beta_distribution.ppf(lower_probability, alpha, beta)
        upper = beta_distribution.ppf(lower_probability + probability, alpha, beta)
        return float(upper - lower)

    result = minimize_scalar(
        interval_width,
        bounds=(0.0, maximum_lower_probability),
        method="bounded",
        options={"xatol": 1e-12},
    )
    lower_probability = float(result.x)
    lower = float(beta_distribution.ppf(lower_probability, alpha, beta))
    upper = float(beta_distribution.ppf(lower_probability + probability, alpha, beta))
    return lower, upper
