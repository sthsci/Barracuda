from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

import bayesorca
from bayesorca import event_counts, trajectories


def test_top_level_package_exposes_the_three_public_workflows():
    assert bayesorca.__version__ == "0.2.0"
    assert callable(bayesorca.run_count_models)
    assert callable(bayesorca.run_donor_models)
    assert bayesorca.run_donor_ignorant_models is bayesorca.run_count_models
    assert bayesorca.run_donor_aware_models is bayesorca.run_donor_models
    assert callable(bayesorca.run_trajectory_conditions)
    assert callable(bayesorca.run_count_bf_scan)
    assert callable(bayesorca.run_trajectory_validation)
    assert callable(bayesorca.pairwise_bayes_factors)
    assert callable(bayesorca.simulate_donor_event_counts)
    assert set(event_counts.MODEL_SPECS) == {"homo", "z2p", "dis2p", "hetero3"}
    assert len(trajectories.TRAJECTORY_MODEL_SPECS) == 4


def test_event_count_simulation_uses_the_paper_model_and_schema():
    frame, truth = event_counts.simulate_event_counts(
        model_key="hetero3",
        n_cells=24,
        obs_time=1.0,
        mu_lambda=4.0,
        sigma_lambda=2.0,
        p_zero=0.2,
        seed=123,
    )

    assert list(frame.columns) == ["cell_id", "count"]
    assert len(frame) == 24
    assert frame["count"].ge(0).all()
    assert truth["model_key"] == "hetero3"
    pd.testing.assert_frame_equal(
        event_counts.validate_count_frame(frame),
        frame,
    )


def test_donor_and_condition_validation_are_available_without_dash():
    rows = []
    for condition in ("Control", "Treatment"):
        for donor in ("D1", "D2"):
            for index, count in enumerate((0, 1, 2), start=1):
                rows.append(
                    {
                        "cell_id": f"{condition}_{donor}_{index}",
                        "donor_id": donor,
                        "condition": condition,
                        "count": count,
                    }
                )
    frame = pd.DataFrame(rows)

    validated = event_counts.validate_condition_frame(
        frame,
        donor_aware=True,
    )
    groups = event_counts.split_condition_frame(validated, donor_aware=True)

    assert list(groups) == ["Control", "Treatment"]
    assert all(list(group.columns) == ["cell_id", "donor_id", "count"] for group in groups.values())


def test_trajectory_simulation_preserves_public_paper_notation():
    frame, truth = trajectories.simulate_trajectory_frame(
        n_cells=20,
        observation_time=1.0,
        mu_lambda=2.0,
        sigma_lambda=0.5,
        p0=0.25,
        sigma_eta=0.4,
        beta_f=0.3,
        beta_s=-0.2,
        seed=321,
    )

    assert list(frame.columns) == ["cell_id", "condition", "history"]
    condition_truth = truth["Synthetic"]
    assert condition_truth["beta_f"] == 0.3
    assert condition_truth["beta_s"] == -0.2
    assert "beta_x" not in condition_truth
    assert "beta_y" not in condition_truth
    pd.testing.assert_frame_equal(
        trajectories.validate_trajectory_frame(frame),
        frame,
    )


def test_packaged_backend_module_names_resolve_after_install_or_mapping():
    event_backend = importlib.import_module(
        "bayesorca._backends.event_counts.inference"
    )
    donor_backend = importlib.import_module(
        "bayesorca._backends.donor.inference_donor_relative"
    )
    trajectory_backend = importlib.import_module(
        "bayesorca._backends.trajectories.inference"
    )

    assert callable(event_backend.inference_homo)
    assert callable(donor_backend.inference_homo)
    assert callable(trajectory_backend.build_model)


def test_trajectory_quadrature_ceiling_remains_numerically_finite():
    settings = trajectories.TrajectorySettings(n_quad=80)
    _nodes, weights = np.polynomial.hermite.hermgauss(settings.n_quad)
    assert np.isfinite(weights).all()
    assert (weights > 0).all()
    with pytest.raises(ValueError, match="n_quad must be at most 80"):
        trajectories.TrajectorySettings(n_quad=81)
