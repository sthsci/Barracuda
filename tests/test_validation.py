from __future__ import annotations

from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest

from barracuda import validation


def _idata(**posterior):
    arrays = {
        name: np.asarray([values], dtype=float)
        for name, values in posterior.items()
    }
    return az.from_dict(posterior=arrays)


def test_canonical_scenarios_are_typed_lowercase_and_scientifically_consistent():
    assert [scenario.true_model for scenario in validation.COUNT_SCENARIOS] == [
        "hetero3",
        "z2p",
        "dis2p",
        "homo",
    ]
    assert [scenario.true_model for scenario in validation.TRAJECTORY_SCENARIOS] == [
        "heterogeneous_history_dependent",
        "heterogeneous_history_independent",
        "homogeneous_history_dependent",
        "homogeneous_history_independent",
    ]
    assert all(
        isinstance(scenario, validation.EventCountScenario)
        for scenario in validation.COUNT_SCENARIOS
    )
    assert all(
        isinstance(scenario, validation.TrajectoryScenario)
        for scenario in validation.TRAJECTORY_SCENARIOS
    )
    assert validation.TRAJECTORY_SCENARIOS[0].mu_eta == pytest.approx(
        np.log(0.25 / 0.75)
    )

    with pytest.raises(ValueError, match="true_model must be 'homo'"):
        validation.EventCountScenario(
            "bad", "bad", 4.0, 0.0, 0.0, "hetero3"
        )


def test_stable_seed_is_mapping_order_independent_and_uint32_safe():
    first = validation.stable_seed("run", {"b": 2, "a": 1}, 3.5)
    second = validation.stable_seed("run", {"a": 1, "b": 2}, 3.5)

    assert first == second
    assert first != validation.stable_seed("run", {"a": 1, "b": 3}, 3.5)
    assert 1 <= first <= 2**32 - 1
    assert first != validation.stable_seed(
        "run", {"a": 1, "b": 2}, 3.5, namespace="other"
    )


def test_posterior_recovery_table_uses_hdi_error_coverage_and_aliases():
    idata = _idata(beta_x=[0.6, 0.8, 1.0, 1.2], sigma_eta=[0.1, 0.2, 0.3, 0.4])
    table = validation.posterior_recovery_table(
        idata,
        {"beta_f": 0.9, "sigma_eta": 0.0},
        parameters=["beta_f", "sigma_eta"],
        parameter_map={"beta_f": "beta_x", "sigma_eta": "sigma_eta"},
        model_key="MODEL",
        condition="Control",
        hdi_prob=0.75,
    ).set_index("parameter")

    assert table.loc["beta_f", "posterior_variable"] == "beta_x"
    assert table.loc["beta_f", "mean"] == pytest.approx(0.9)
    assert table.loc["beta_f", "error"] == pytest.approx(0.0)
    assert bool(table.loc["beta_f", "covered"])
    assert np.isnan(table.loc["sigma_eta", "relative_error"])
    assert table.loc["sigma_eta", "absolute_error"] == pytest.approx(0.25)
    assert table.loc["beta_f", "model_key"] == "model"
    assert table.loc["beta_f", "condition"] == "Control"
    assert table.loc["beta_f", "n_draws"] == 4


def test_model_recovery_tables_translate_count_and_trajectory_parameter_names():
    count_results = {
        "z2p": SimpleNamespace(
            model_key="z2p",
            idata=_idata(
                **{"lambda": [3.8, 4.0, 4.2], "p_zero": [0.1, 0.2, 0.3]}
            ),
        ),
        "dis2p": SimpleNamespace(
            model_key="dis2p",
            idata=_idata(
                mu_lambda=[3.7, 4.0, 4.3],
                sigma_lambda=[1.5, 2.0, 2.5],
            ),
        ),
    }
    count = validation.event_count_recovery_table(
        count_results,
        {"mu_lambda": 4.0, "sigma_lambda": 2.0, "p_zero": 0.2},
    )
    assert set(map(tuple, count[["model_key", "parameter"]].to_numpy())) == {
        ("z2p", "mu_lambda"),
        ("z2p", "p_zero"),
        ("dis2p", "mu_lambda"),
        ("dis2p", "sigma_lambda"),
    }
    assert count.loc[count["parameter"] == "mu_lambda", "covered"].all()

    trajectory_results = {
        "homogeneous_history_dependent": SimpleNamespace(
            model_key="homogeneous_history_dependent",
            idata=_idata(
                mu_lambda=[3.8, 4.0, 4.2],
                sigma_lambda=[1.8, 2.0, 2.2],
                mu_eta=[-1.2, -1.1, -1.0],
                beta_x=[0.6, 0.8, 1.0],
                beta_y=[-1.0, -0.8, -0.6],
            ),
        )
    }
    trajectory = validation.trajectory_recovery_table(
        trajectory_results,
        {
            "mu_lambda": 4.0,
            "sigma_lambda": 2.0,
            "mu_eta": np.log(0.25 / 0.75),
            "sigma_eta": 0.0,
            "beta_f": 0.8,
            "beta_s": -0.8,
        },
        condition="No3",
    )
    aliases = trajectory.set_index("parameter")["posterior_variable"].to_dict()
    assert aliases["beta_f"] == "beta_x"
    assert aliases["beta_s"] == "beta_y"
    assert set(trajectory["condition"]) == {"No3"}


def test_coverage_and_boundary_summaries_have_explicit_rates():
    recovery = pd.DataFrame(
        {
            "model_key": ["hetero3", "hetero3", "hetero3"],
            "parameter": ["sigma_lambda"] * 3,
            "truth": [0.0, 0.0, 0.0],
            "mean": [0.02, 0.08, 0.4],
            "covered": [True, True, False],
            "error": [0.02, 0.08, 0.4],
            "absolute_error": [0.02, 0.08, 0.4],
            "hdi_lower": [0.0, 0.0, 0.1],
            "hdi_upper": [0.2, 0.3, 0.7],
        }
    )

    coverage = validation.coverage_summary(
        recovery, group_by=("model_key", "parameter")
    ).iloc[0]
    assert coverage["n_runs"] == 3
    assert coverage["coverage_rate"] == pytest.approx(2 / 3)
    assert coverage["mean_error"] == pytest.approx(0.5 / 3)
    assert coverage["rmse"] == pytest.approx(
        np.sqrt(np.mean(np.square([0.02, 0.08, 0.4])))
    )

    boundary = validation.boundary_recovery_summary(
        recovery,
        estimate_tolerance=0.1,
    ).iloc[0]
    assert boundary["boundary"] == 0.0
    assert boundary["boundary_coverage_rate"] == pytest.approx(2 / 3)
    assert boundary["boundary_exclusion_rate"] == pytest.approx(1 / 3)
    assert boundary["estimate_within_tolerance_rate"] == pytest.approx(2 / 3)


def test_exact_superiority_and_rope_probabilities_match_cartesian_enumeration():
    first = np.array([0.0, 1.0, 2.0])
    second = np.array([0.0, 1.5])
    differences = first[:, None] - second[None, :]

    assert validation.posterior_superiority_probability(
        first, second, margin=0.25
    ) == pytest.approx(np.mean(differences > 0.25))
    result = validation.posterior_rope_probabilities(
        first,
        second,
        rope=(-0.5, 0.5),
    )
    assert result.probability_below == pytest.approx(np.mean(differences < -0.5))
    assert result.probability_in_rope == pytest.approx(
        np.mean((differences >= -0.5) & (differences <= 0.5))
    )
    assert result.probability_above == pytest.approx(np.mean(differences > 0.5))
    assert result.n_pairs == differences.size
    assert sum(
        (
            result.probability_below,
            result.probability_in_rope,
            result.probability_above,
        )
    ) == pytest.approx(1.0)


def test_single_event_count_validation_uses_module_level_fit_adapter(monkeypatch):
    captured = {}

    def fake_simulator(**kwargs):
        captured["simulation"] = kwargs
        frame = pd.DataFrame(
            {"cell_id": [f"cell_{index}" for index in range(kwargs["n_cells"])], "count": 1}
        )
        return frame, {
            "model_key": "homo",
            "mu_lambda": 4.0,
            "sigma_lambda": 0.0,
            "p_zero": 0.0,
        }

    def fake_fit(frame, observation_time, **kwargs):
        captured["fit"] = (frame.copy(), observation_time, kwargs)
        return {
            "homo": SimpleNamespace(
                model_key="homo",
                log_evidence=-10.0,
                idata=_idata(**{"lambda": [3.8, 4.0, 4.2]}),
            )
        }

    monkeypatch.setattr(validation, "simulate_event_count_data", fake_simulator)
    monkeypatch.setattr(validation, "fit_event_count_models", fake_fit)
    result = validation.run_event_count_validation(
        validation.COUNT_SCENARIOS[3],
        12,
        model_keys=["homo"],
        base_seed=99,
    )

    assert isinstance(result, validation.EventCountValidationResult)
    assert len(result.frame) == 12
    assert result.evidence.loc[0, "log_bf_model_vs_true"] == 0.0
    assert result.recovery.loc[0, "parameter"] == "mu_lambda"
    assert captured["simulation"]["seed"] == result.simulation_seed
    assert captured["fit"][2]["settings"].seed == result.inference_seed


def test_single_trajectory_validation_uses_public_names_and_nested_fit(monkeypatch):
    scenario = validation.TRAJECTORY_SCENARIOS[3]

    def fake_simulator(**kwargs):
        frame = pd.DataFrame(
            {
                "cell_id": [f"cell_{index}" for index in range(kwargs["n_cells"])],
                "condition": kwargs["condition"],
                "history": [(0, 1)] * kwargs["n_cells"],
            }
        )
        return frame, {
            kwargs["condition"]: {
                "mu_lambda": 4.0,
                "sigma_lambda": 2.0,
                "p0": 0.25,
                "mu_eta": np.log(0.25 / 0.75),
                "sigma_eta": 0.0,
                "beta_f": 0.0,
                "beta_s": 0.0,
            }
        }

    def fake_fit(frame, observation_time, **kwargs):
        condition = frame["condition"].iloc[0]
        fit = SimpleNamespace(
            model_key="homogeneous_history_independent",
            log_evidence=-4.0,
            idata=_idata(
                mu_lambda=[3.8, 4.0, 4.2],
                sigma_lambda=[1.8, 2.0, 2.2],
                mu_eta=[-1.2, np.log(0.25 / 0.75), -1.0],
            ),
        )
        return {condition: {fit.model_key: fit}}

    monkeypatch.setattr(validation, "simulate_trajectory_data", fake_simulator)
    monkeypatch.setattr(validation, "fit_trajectory_models", fake_fit)
    result = validation.run_trajectory_validation(
        scenario,
        8,
        model_keys=[scenario.true_model],
        base_seed=123,
    )

    assert isinstance(result, validation.TrajectoryValidationResult)
    assert set(result.recovery["parameter"]) == {
        "mu_lambda",
        "sigma_lambda",
        "mu_eta",
    }
    assert set(result.recovery["condition"]) == {"No4"}
    assert result.evidence.loc[0, "is_best"]
