from __future__ import annotations

import numpy as np
import pytest

from webapp.core.coin import simulate_coin_tosses, uniform_prior_posterior


def test_coin_toss_simulation_is_reproducible_and_binary() -> None:
    first = simulate_coin_tosses(0.7, 40, seed=2026)
    second = simulate_coin_tosses(0.7, 40, seed=2026)

    np.testing.assert_array_equal(first, second)
    assert len(first) == 40
    assert set(first).issubset({0, 1})


def test_uniform_prior_updates_to_expected_beta_posterior() -> None:
    assert uniform_prior_posterior(heads=7, n_tosses=10) == (8, 4)


@pytest.mark.parametrize(
    ("probability_heads", "n_tosses"),
    [(-0.01, 10), (1.01, 10), (0.5, 0)],
)
def test_coin_toss_simulation_rejects_invalid_inputs(
    probability_heads: float,
    n_tosses: int,
) -> None:
    with pytest.raises(ValueError):
        simulate_coin_tosses(probability_heads, n_tosses)
