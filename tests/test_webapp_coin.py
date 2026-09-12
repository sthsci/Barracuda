from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import beta as beta_distribution

from webapp.core.coin import beta_highest_density_interval, simulate_coin_tosses, uniform_prior_posterior


def test_coin_toss_simulation_is_reproducible_and_binary() -> None:
    first = simulate_coin_tosses(0.7, 40, seed=2026)
    second = simulate_coin_tosses(0.7, 40, seed=2026)

    np.testing.assert_array_equal(first, second)
    assert len(first) == 40
    assert set(first).issubset({0, 1})
    np.testing.assert_array_equal(first, simulate_coin_tosses(0.7, 80, seed=2026)[:40])
    assert not np.array_equal(first, simulate_coin_tosses(0.7, 40, seed=2027))


def test_uniform_prior_updates_to_expected_beta_posterior() -> None:
    assert uniform_prior_posterior(heads=7, n_tosses=10) == (8, 4)


def test_highest_density_interval_handles_symmetric_and_boundary_posteriors() -> None:
    symmetric = beta_highest_density_interval(6, 6, 0.95)
    boundary = beta_highest_density_interval(13, 1, 0.95)

    assert symmetric[0] == pytest.approx(1 - symmetric[1], abs=1e-7)
    assert boundary == pytest.approx((0.05 ** (1 / 13), 1.0), abs=1e-7)
    assert beta_distribution.cdf(symmetric[1], 6, 6) - beta_distribution.cdf(symmetric[0], 6, 6) == pytest.approx(0.95)


def test_smaller_hdi_probability_produces_a_narrower_interval() -> None:
    interval_80 = beta_highest_density_interval(8, 4, 0.8)
    interval_95 = beta_highest_density_interval(8, 4, 0.95)

    assert interval_80[1] - interval_80[0] < interval_95[1] - interval_95[0]


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


def test_exact_coin_figures_match_beta_update_and_sequential_credible_mass() -> None:
    from scipy.integrate import trapezoid
    from webapp.pages.bayes_101 import _coin_figure, _coin_frequency_figure

    observations = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1])
    density, metrics = _coin_figure(0.7, 10, 0, 90, outcomes=observations)
    same_data, same_metrics = _coin_figure(0.1, 10, 0, 90, outcomes=observations)
    x, y = density.data[1].x, density.data[1].y
    assert trapezoid(y, x) == pytest.approx(1, abs=1e-6)
    assert trapezoid(x * y, x) == pytest.approx(8 / 12, abs=1e-6)
    np.testing.assert_array_equal(y, same_data.data[1].y)
    assert metrics == same_metrics  # Generating truth does not enter inference.

    sequential = _coin_frequency_figure(0.7, observations, 90)
    lower, upper, posterior_mean, mle = [trace.y for trace in sequential.data[:4]]
    n = np.arange(11)
    h = np.r_[0, np.cumsum(observations)]
    np.testing.assert_allclose(posterior_mean, (h + 1) / (n + 2))
    np.testing.assert_allclose(beta_distribution.cdf(upper, h + 1, n - h + 1) - beta_distribution.cdf(lower, h + 1, n - h + 1), 0.9)
    assert (lower[0], upper[0], posterior_mean[0]) == pytest.approx((0.05, 0.95, 0.5))
    assert mle[-1] == 0.7
