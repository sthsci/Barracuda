from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from webapp.core.data import validate_count_frame
from webapp.core.simulation import simulate_event_counts


@pytest.mark.parametrize("model_key", ["homo", "z2p", "dis2p", "hetero3"])
def test_all_four_models_generate_reproducible_canonical_counts(model_key: str) -> None:
    kwargs = dict(
        model_key=model_key,
        n_cells=40,
        obs_time=2.0,
        mu_lambda=3.0,
        sigma_lambda=1.2,
        p_zero=0.25,
        seed=2026,
    )
    first, first_truth = simulate_event_counts(**kwargs)
    second, second_truth = simulate_event_counts(**kwargs)

    pd.testing.assert_frame_equal(first, second)
    assert first_truth == second_truth
    assert first_truth["model_key"] == model_key
    assert len(validate_count_frame(first)) == 40


def test_irrelevant_parameters_are_zeroed_in_ground_truth() -> None:
    _, homogeneous = simulate_event_counts("homo", 5, 1, 2, 9, 0.8, 1)
    _, zero_inflated = simulate_event_counts("z2p", 5, 1, 2, 9, 0.8, 1)
    _, distributed = simulate_event_counts("dis2p", 5, 1, 2, 9, 0.8, 1)

    assert homogeneous["sigma_lambda"] == 0
    assert homogeneous["p_zero"] == 0
    assert zero_inflated["sigma_lambda"] == 0
    assert zero_inflated["p_zero"] == 0.8
    assert distributed["sigma_lambda"] == 9
    assert distributed["p_zero"] == 0


def test_zero_inflation_at_one_produces_only_zeros() -> None:
    frame, truth = simulate_event_counts("hetero3", 20, 3, 4, 2, 1, 7)
    assert frame["count"].eq(0).all()
    assert truth["realized_structural_zeros"] == 20


def test_none_seed_is_supported_and_recorded() -> None:
    frame, truth = simulate_event_counts("homo", 5, 1, 2, 0, 0, None)
    assert len(frame) == 5
    assert truth["seed"] is None


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"model_key": "unknown"}, "unknown model_key"),
        ({"n_cells": 0}, "positive integer"),
        ({"n_cells": 4}, "between 5"),
        ({"n_cells": 1_001}, "between 5"),
        ({"obs_time": 0}, "greater than zero"),
        ({"mu_lambda": 0}, "greater than zero"),
        ({"sigma_lambda": -0.1}, "greater than or equal"),
        ({"p_zero": 1.1}, "between zero and one"),
        ({"seed": -1}, "between 0"),
    ],
)
def test_simulation_rejects_invalid_controls(overrides, message: str) -> None:
    kwargs = {
        "model_key": "hetero3",
        "n_cells": 10,
        "obs_time": 1.0,
        "mu_lambda": 2.0,
        "sigma_lambda": 1.0,
        "p_zero": 0.2,
        "seed": 1,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        simulate_event_counts(**kwargs)


def test_distributed_model_has_nonconstant_realized_rates() -> None:
    _, truth = simulate_event_counts("dis2p", 200, 1, 3, 2, 0, 9)
    assert np.isfinite(truth["realized_mean_rate"])
    assert truth["realized_mean_rate"] > 0
