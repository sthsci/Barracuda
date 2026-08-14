from __future__ import annotations

import math

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from bayesorca.diagnostics import (
    diagnostic_flags,
    population_p0_draws,
    population_p0_summary,
    posterior_diagnostics,
    smc_evidence_summary,
    smc_log_evidence_by_chain,
    trajectory_state_summary,
)
from bayesorca.evidence import (
    bayes_factor,
    classify_bayes_factor,
    combine_independent_evidence,
    evidence_from_inference_data,
    history_effect_bayes_factors,
    log_bayes_factor,
    pairwise_bayes_factors,
    posterior_model_probabilities,
    savage_dickey_ratio,
    smc_log_evidence,
)


def _trajectory_idata() -> az.InferenceData:
    rng = np.random.default_rng(4)
    posterior = xr.Dataset(
        {
            "mu_eta": (("chain", "draw"), rng.normal(0.0, 0.1, (2, 120))),
            "sigma_eta": (("chain", "draw"), np.full((2, 120), 0.5)),
            "beta_x": (("chain", "draw"), rng.normal(0.8, 0.3, (2, 120))),
            "beta_y": (("chain", "draw"), rng.normal(-0.4, 0.3, (2, 120))),
        }
    )
    prior = xr.Dataset(
        {
            "beta_x": (("chain", "draw"), rng.normal(0.0, 1.0, (2, 300))),
            "beta_y": (("chain", "draw"), rng.normal(0.0, 1.0, (2, 300))),
        }
    )
    return az.InferenceData(posterior=posterior, prior=prior)


def test_bayes_factor_helpers_keep_direction_and_overflow_explicit():
    assert log_bayes_factor(-10.0, -12.5) == 2.5
    assert bayes_factor(-10.0, -12.5) == pytest.approx(math.exp(2.5))
    assert math.isinf(bayes_factor(1_000.0, 0.0))
    assert bayes_factor(-1_000.0, 0.0) == 0.0
    assert classify_bayes_factor(0.2) == "negligible"
    assert classify_bayes_factor(-2.0) == "positive"
    assert classify_bayes_factor(4.0) == "strong"
    assert classify_bayes_factor(7.0) == "very strong"


def test_pairwise_bayes_factors_and_probabilities_are_stable():
    log_evidence = {"homo": -20.0, "dis2p": -15.0, "hetero3": -16.0}
    pairwise = pairwise_bayes_factors(log_evidence)
    assert len(pairwise) == 3
    comparison = pairwise.loc[
        (pairwise["model_1"] == "homo") & (pairwise["model_2"] == "dis2p")
    ].iloc[0]
    assert comparison["log_BF_1_vs_2"] == -5.0
    assert comparison["favoured_model"] == "dis2p"

    probabilities = posterior_model_probabilities(log_evidence)
    assert probabilities["posterior_probability"].sum() == pytest.approx(1.0)
    assert probabilities.iloc[0]["model_key"] == "dis2p"
    weighted = posterior_model_probabilities(
        log_evidence,
        prior_probabilities={"homo": 100.0, "dis2p": 1.0, "hetero3": 1.0},
    )
    assert weighted["posterior_probability"].sum() == pytest.approx(1.0)


def test_independent_evidence_combination_sums_log_scale():
    table = pd.DataFrame(
        {
            "condition": ["A", "A", "B", "B"],
            "model_key": ["homo", "dis2p", "homo", "dis2p"],
            "log_evidence": [-3.0, -2.0, -5.0, -1.0],
        }
    )
    combined = combine_independent_evidence(table)
    assert combined.iloc[0]["model_key"] == "dis2p"
    assert combined.iloc[0]["total_log_evidence"] == -3.0
    assert combined.set_index("model_key").loc["homo", "total_log_evidence"] == -8.0

    with pytest.raises(ValueError, match="incomplete dataset coverage"):
        combine_independent_evidence(table.iloc[:-1])

    with pytest.raises(ValueError, match="duplicate dataset-model"):
        combine_independent_evidence(pd.concat([table, table.iloc[[0]]], ignore_index=True))


def test_savage_dickey_and_history_aliases_use_explicit_bf_direction():
    idata = _trajectory_idata()
    result = savage_dickey_ratio(idata, "beta_x")
    assert result.parameter == "beta_x"
    assert np.isfinite(result.log_bf_10)
    assert result.bf_01 * result.bf_10 == pytest.approx(1.0)

    table = history_effect_bayes_factors(idata)
    assert table["parameter"].tolist() == ["beta_f", "beta_s"]
    assert table["backend_parameter"].tolist() == ["beta_x", "beta_y"]


def test_smc_evidence_diagnostics_preserve_incomplete_chains():
    idata = az.InferenceData(
        sample_stats=xr.Dataset(
            {
                "log_marginal_likelihood": (
                    ("chain", "stage"),
                    np.asarray([[-14.0, -10.0], [-13.0, -11.0], [np.nan, np.nan]]),
                )
            }
        )
    )
    chains = smc_log_evidence_by_chain(idata)
    assert chains["log_evidence"].iloc[:2].tolist() == [-10.0, -11.0]
    assert not bool(chains.iloc[2]["is_finite"])
    summary = smc_evidence_summary(idata)
    assert summary["n_chains"] == 3
    assert summary["n_finite_chains"] == 2
    assert summary["mean_log_evidence"] == -10.5
    assert smc_log_evidence(idata) == -10.5


def test_smc_stage_only_layout_is_one_chain_with_a_final_stage():
    idata = az.InferenceData(
        sample_stats=xr.Dataset(
            {
                "log_marginal_likelihood": (
                    ("stage",),
                    np.asarray([-14.0, -12.0, -10.0]),
                )
            }
        )
    )

    chains = smc_log_evidence_by_chain(idata)
    assert len(chains) == 1
    assert chains.loc[0, "log_evidence"] == -10.0
    assert chains.loc[0, "n_finite_stages"] == 3
    summary = smc_evidence_summary(idata)
    assert summary["mean_log_evidence"] == -10.0
    assert smc_log_evidence(idata) == -10.0

    ranked = evidence_from_inference_data(
        {
            "first": idata,
            "second": az.InferenceData(
                sample_stats=xr.Dataset(
                    {
                        "log_marginal_likelihood": (
                            ("chain",),
                            np.asarray([-9.0, -9.0]),
                        )
                    }
                )
            ),
        }
    )
    assert ranked.iloc[0]["model_key"] == "second"
    assert ranked.iloc[0]["is_best"]


def test_posterior_diagnostic_flags_distinguish_missing_rhat():
    rng = np.random.default_rng(8)
    idata = az.from_dict(posterior={"theta": rng.normal(size=(1, 240))})
    table = posterior_diagnostics(idata)
    assert table.iloc[0]["parameter"] == "theta"
    flagged = diagnostic_flags(table, min_ess_bulk=10, min_ess_tail=10)
    assert not bool(flagged.iloc[0]["r_hat_available"])
    assert flagged.iloc[0]["diagnostic_status"] in {"limited", "review"}


def test_population_p0_draws_are_reproducible_and_bounded():
    idata = _trajectory_idata()
    first = population_p0_draws(
        idata,
        n_parameter_draws=20,
        n_population_draws=30,
        seed=19,
    )
    second = population_p0_draws(
        idata,
        n_parameter_draws=20,
        n_population_draws=30,
        seed=19,
    )
    np.testing.assert_array_equal(first, second)
    assert first.shape == (20, 30)
    assert np.all((first > 0) & (first < 1))
    summary = population_p0_summary(
        idata,
        n_parameter_draws=20,
        n_population_draws=30,
        seed=19,
    )
    assert summary["n_values"] == 600
    assert 0 < summary["mean"] < 1


def test_trajectory_state_summary_uses_precontact_history():
    frame = pd.DataFrame(
        {
            "cell_id": ["c1", "c2"],
            "condition": ["Control", "Control"],
            "history": [(0, 1), (1,)],
        }
    )
    summary = trajectory_state_summary(frame)
    origin = summary.loc[(summary["x_before"] == 0) & (summary["y_before"] == 0)].iloc[0]
    assert origin["n_contacts"] == 2
    assert origin["n_lethal"] == 1
    assert origin["empirical_lethal_probability"] == 0.5


def test_matplotlib_plot_helpers_return_axes():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from bayesorca.plotting import (
        plot_bayes_factor_scan,
        plot_event_count_distribution,
        plot_model_evidence,
        plot_parameter_recovery,
        plot_posterior_pair,
        plot_trajectory_state_map,
    )

    counts = pd.DataFrame({"cell_id": ["a", "b", "c"], "count": [0, 1, 1]})
    assert plot_event_count_distribution(counts).get_xlabel() == "Events per cell"

    evidence = pd.DataFrame(
        {
            "model_key": ["homo", "dis2p"],
            "log10_BF_best_vs_model": [2.0, 0.0],
        }
    )
    assert "BF" in plot_model_evidence(evidence).get_xlabel()

    scan = pd.DataFrame(
        {
            "scenario": ["S"] * 8,
            "replicate": [1, 2] * 4,
            "n_cells": [10, 10, 20, 20] * 2,
            "model_key": ["homo"] * 4 + ["dis2p"] * 4,
            "log10_bf_model_vs_true": [0, 0.1, 1, 1.1, 0, 0, 0, 0],
        }
    )
    assert plot_bayes_factor_scan(scan, scenario="S").get_xlabel().startswith("Cumulative")

    recovery = pd.DataFrame(
        {
            "model_key": ["dis2p"],
            "parameter": ["mu_lambda"],
            "truth": [4.0],
            "posterior_mean": [3.8],
            "hdi_lower": [3.2],
            "hdi_upper": [4.4],
        }
    )
    assert plot_parameter_recovery(recovery).get_ylabel() == "mu_lambda"

    draws = pd.DataFrame(
        {
            "model_key": ["dis2p"] * 3,
            "mu_lambda": [3.0, 4.0, 5.0],
            "sigma_lambda": [1.0, 2.0, 1.5],
        }
    )
    assert plot_posterior_pair(draws, "mu_lambda", "sigma_lambda").get_xlabel() == "mu_lambda"

    histories = pd.DataFrame(
        {
            "cell_id": ["c1", "c2"],
            "condition": ["Control", "Control"],
            "history": [(0, 1), (1,)],
        }
    )
    assert plot_trajectory_state_map(histories).get_ylabel() == "Previous lethal contacts"

    import matplotlib.pyplot as plt

    plt.close("all")
