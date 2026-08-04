from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


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
