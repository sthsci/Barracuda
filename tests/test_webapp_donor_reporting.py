from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from webapp.donor_reporting import (
    BF3_LOG10,
    aggregate_model_evidence,
    bayes_factor_figure,
    cartesian_contrast_draws,
    condition_control_contrast_frame,
    condition_model_evidence,
    condition_results_donor_frame,
    condition_results_log_evidence,
    condition_results_population_frame,
    decompose_population_heterogeneity,
    donor_joint_posterior_figure,
    donor_frame_condition_contrasts,
    donor_frame_condition_contrast_figure,
    donor_posterior_frame,
    population_joint_posterior_figure,
    population_posterior_frame,
    summarize_contrast_draws,
    variance_decomposition_summary,
)


def _dis2p_fit(
    mu_donor: np.ndarray,
    sigma_donor: np.ndarray,
    *,
    mu_population: np.ndarray | None = None,
    sigma_population: np.ndarray | None = None,
) -> az.InferenceData:
    mu = np.asarray(mu_donor, dtype=float)
    sigma = np.asarray(sigma_donor, dtype=float)
    if mu.ndim != 3 or sigma.shape != mu.shape:
        raise ValueError("test donor arrays must have shape (chain, draw, donor)")
    weights = np.array([0.25, 0.75])
    if mu_population is None:
        mu_population = np.sum(mu * weights[None, None, :], axis=2)
    if sigma_population is None:
        second = np.sum(weights[None, None, :] * (sigma**2 + mu**2), axis=2)
        sigma_population = np.sqrt(second - np.asarray(mu_population) ** 2)
    return az.from_dict(
        posterior={
            "mu_lambda_donor": mu,
            "sigma_lambda_donor": sigma,
            "mu_lambda_population": np.asarray(mu_population, dtype=float),
            "sigma_lambda_population": np.asarray(sigma_population, dtype=float),
        },
        coords={"donor": ["D1", "D2"]},
        dims={
            "mu_lambda_donor": ["donor"],
            "sigma_lambda_donor": ["donor"],
        },
    )


def _posterior_mapping() -> dict[tuple[str, str, str], az.InferenceData]:
    control = _dis2p_fit(
        np.array([[[1.0, 10.0], [2.0, 20.0]]]),
        np.array([[[0.5, 5.0], [1.0, 10.0]]]),
    )
    treatment = _dis2p_fit(
        np.array([[[3.0, 30.0], [5.0, 50.0]]]),
        np.array([[[1.5, 15.0], [2.5, 25.0]]]),
    )
    return {
        ("No treatment", "contacts", "Dis2P"): control,
        ("Rituximab", "contacts", "Dis2P"): treatment,
    }


def test_condition_and_aggregate_evidence_follow_smc_logml_rules() -> None:
    evidence = pd.DataFrame(
        {
            "condition": ["Control", "Control", "Treatment", "Treatment"],
            "outcome": ["contacts"] * 4,
            "model": ["homo", "Dis2P", "homo", "Dis2P"],
            "logml": [-12.0, -10.0, -20.0, -21.0],
        }
    )

    condition = condition_model_evidence(evidence)
    control_homo = condition.loc[
        (condition["condition"] == "Control") & (condition["model"] == "homo")
    ].iloc[0]
    assert control_homo["delta_logml_vs_best"] == -2.0
    np.testing.assert_allclose(control_homo["log10_BF_best_vs_model"], 2.0 / np.log(10.0))
    assert control_homo["best_model"] == "dis2p"

    aggregate = aggregate_model_evidence(
        evidence,
        conditions=("Control", "Treatment"),
    )
    summed = dict(zip(aggregate["model"], aggregate["total_log_evidence"]))
    assert summed == {"homo": -32.0, "dis2p": -31.0}
    assert aggregate.loc[aggregate["is_best"], "model"].tolist() == ["dis2p"]
    assert set(aggregate["n_conditions"]) == {2}


def test_aggregate_evidence_rejects_incomplete_condition_coverage() -> None:
    evidence = pd.DataFrame(
        {
            "condition": ["Control", "Treatment", "Control"],
            "outcome": ["contacts"] * 3,
            "model": ["homo", "homo", "Dis2P"],
            "logml": [-12.0, -20.0, -10.0],
        }
    )
    with pytest.raises(ValueError, match="incomplete condition coverage"):
        aggregate_model_evidence(evidence, conditions=("Control", "Treatment"))


def test_bayes_factor_plot_uses_true_log10_scale_and_labels_best_model() -> None:
    comparison = condition_model_evidence(
        pd.DataFrame(
            {
                "condition": ["Control"] * 4,
                "outcome": ["contacts"] * 4,
                "model": ["homo", "Z2P", "Dis2P", "hetero3"],
                "logml": [-110.0, -12.0, -11.0, -10.0],
            }
        )
    )
    figure = bayes_factor_figure(comparison)

    plotted = np.asarray(figure.data[0].x, dtype=float)
    expected = comparison.sort_values(
        "log10_BF_best_vs_model", ascending=False
    )["log10_BF_best_vs_model"].to_numpy(dtype=float)
    np.testing.assert_allclose(plotted, expected)
    assert plotted.max() > 40.0
    assert any("Best model" in str(label) for label in figure.data[0].y)
    assert "Best model · 0.00" in list(figure.data[0].text)

    band_spans = [
        (float(shape.x0), float(shape.x1))
        for shape in figure.layout.shapes
        if shape.type == "rect"
    ]
    np.testing.assert_allclose(
        band_spans[:3],
        [(0.0, BF3_LOG10), (BF3_LOG10, 1.0), (1.0, 2.0)],
    )
    assert band_spans[3][0] == 2.0
    assert band_spans[3][1] > plotted.max()


def test_population_and_donor_frames_preserve_chain_draw_pairing() -> None:
    mapping = _posterior_mapping()
    population = population_posterior_frame(
        mapping,
        outcome="contacts",
        model="dis2p",
    )
    assert len(population) == 4
    assert set(population["condition"]) == {"No treatment", "Rituximab"}

    donors = donor_posterior_frame(
        mapping,
        outcome="contacts",
        model="dis2p",
        donor_labels={"D1": "Donor one", "D2": "Donor two"},
    )
    assert len(donors) == 8
    paired = donors.loc[
        (donors["condition"] == "No treatment")
        & (donors["draw"] == 1)
        & (donors["donor_id"] == "Donor two")
    ].iloc[0]
    assert paired["mu_lambda_donor"] == 20.0
    assert paired["sigma_lambda_donor"] == 10.0


def test_web_condition_result_adapters_apply_each_fits_donor_labels() -> None:
    mapping = _posterior_mapping()
    results = {
        condition: {
            "dis2p": SimpleNamespace(
                donor_aware=True,
                idata=idata,
                donor_labels=("Alice", "Bob"),
                log_evidence=-10.0 - index,
            )
        }
        for index, ((condition, _outcome, _model), idata) in enumerate(mapping.items())
    }
    evidence = condition_results_log_evidence(results)
    assert list(evidence.columns) == ["condition", "outcome", "model", "logml"]
    assert set(evidence["condition"]) == {"No treatment", "Rituximab"}

    population = condition_results_population_frame(results, model="Dis2P")
    assert len(population) == 4
    donors = condition_results_donor_frame(results, model="Dis2P")
    assert set(donors["donor_id"]) == {"Alice", "Bob"}
    assert len(donors) == 8


def test_paired_subsampling_keeps_every_donor_for_selected_draws() -> None:
    donors = donor_posterior_frame(
        _posterior_mapping(),
        outcome="contacts",
        model="Dis2P",
        conditions=("No treatment",),
        max_draws_per_fit=1,
        random_seed=2,
    )
    assert len(donors) == 2
    assert donors[["chain", "draw"]].drop_duplicates().shape[0] == 1
    assert set(donors["donor_coordinate"]) == {"D1", "D2"}


def test_population_variance_decomposition_matches_mixture_moments() -> None:
    mu = np.array([[[1.0, 3.0], [1.0, 3.0]]])
    sigma = np.array([[[0.5, 1.0], [0.5, 1.0]]])
    fit = _dis2p_fit(
        mu,
        sigma,
        mu_population=np.array([[2.5, 2.5]]),
        sigma_population=np.array([[1.25, 1.25]]),
    )
    decomposition = decompose_population_heterogeneity(fit, [1, 3])

    np.testing.assert_allclose(decomposition["variance_within_donor"], 0.8125)
    np.testing.assert_allclose(decomposition["variance_between_donor"], 0.75)
    np.testing.assert_allclose(decomposition["variance_total"], 1.5625)
    np.testing.assert_allclose(decomposition["sigma_lambda_population"], 1.25)
    np.testing.assert_allclose(
        decomposition["fraction_within_donor"]
        + decomposition["fraction_between_donor"],
        1.0,
    )

    summary = variance_decomposition_summary(decomposition, hdi_prob=0.8)
    assert set(summary["component"]) == {
        "variance_within_donor",
        "variance_between_donor",
    }


def test_variance_decomposition_detects_inconsistent_saved_population_sd() -> None:
    fit = _dis2p_fit(
        np.array([[[1.0, 3.0], [1.0, 3.0]]]),
        np.array([[[0.5, 1.0], [0.5, 1.0]]]),
        mu_population=np.array([[2.5, 2.5]]),
        sigma_population=np.array([[9.0, 9.0]]),
    )
    with pytest.raises(ValueError, match="reconstructed sigma_lambda_population"):
        decompose_population_heterogeneity(fit, [1, 3])


def test_condition_contrasts_use_all_independent_draw_pairs_not_means() -> None:
    treatment = np.array([[10.0, 100.0], [20.0, 200.0]])
    control = np.array([[1.0, 10.0], [3.0, 30.0]])
    contrasts, metadata = cartesian_contrast_draws(treatment, control)

    np.testing.assert_allclose(
        contrasts,
        [[9.0, 90.0], [7.0, 70.0], [19.0, 190.0], [17.0, 170.0]],
    )
    assert metadata["exact_cartesian"] is True
    assert metadata["possible_pairs"] == 4
    assert len(contrasts) == 4


def test_percentage_contrasts_use_fixed_control_posterior_mean() -> None:
    treatment = np.array([[4.0, 40.0]])
    control = np.array([[1.0, 10.0], [3.0, 30.0]])
    contrasts, metadata = cartesian_contrast_draws(
        treatment,
        control,
        scale="percent_of_control_mean",
    )

    # The denominator is fixed at [2, 20] for both Cartesian pairs.
    np.testing.assert_allclose(contrasts, [[150.0, 150.0], [50.0, 50.0]])
    np.testing.assert_allclose(metadata["control_mean"], [2.0, 20.0])


def test_collected_donor_frame_contrast_uses_independent_condition_pairs() -> None:
    donors = donor_posterior_frame(
        _posterior_mapping(),
        outcome="contacts",
        model="dis2p",
    )
    contrasts = donor_frame_condition_contrasts(
        donors,
        treatment="Rituximab",
        control="No treatment",
        scale="absolute",
    )
    assert len(contrasts) == 8
    assert set(contrasts["possible_pairs"]) == {4}
    assert set(contrasts["exact_cartesian"]) == {True}
    d1 = contrasts.loc[contrasts["donor_id"] == "D1"]
    assert sorted(d1["delta_mu_lambda"].tolist()) == [1.0, 2.0, 3.0, 4.0]

    figure = donor_frame_condition_contrast_figure(
        donors,
        treatment="Rituximab",
        control="No treatment",
    )
    assert sum(trace.type == "contour" for trace in figure.data) == 2
    assert all(trace.type != "histogram2dcontour" for trace in figure.data)
    assert {trace.legendgroup for trace in figure.data if trace.showlegend} == {"D1", "D2"}


def test_large_condition_contrast_uses_reproducible_independent_mc_pairs() -> None:
    treatment = np.arange(20.0).reshape(10, 2) + 10.0
    control = np.arange(16.0).reshape(8, 2) + 1.0
    first, metadata = cartesian_contrast_draws(
        treatment,
        control,
        max_exact_pairs=50,
        approximate_pairs=12,
        random_seed=17,
    )
    second, second_metadata = cartesian_contrast_draws(
        treatment,
        control,
        max_exact_pairs=50,
        approximate_pairs=12,
        random_seed=17,
    )
    np.testing.assert_allclose(first, second)
    assert metadata == second_metadata
    assert metadata["exact_cartesian"] is False
    assert metadata["possible_pairs"] == 80
    assert metadata["returned_pairs"] == 12


def test_condition_control_frame_and_summary_retain_draw_level_uncertainty() -> None:
    contrasts = condition_control_contrast_frame(
        _posterior_mapping(),
        treatment="Rituximab",
        control="No treatment",
        outcome="contacts",
        donor_labels={"D1": "Donor one", "D2": "Donor two"},
        scale="percent_of_control_mean",
    )
    assert len(contrasts) == 8
    assert set(contrasts["donor_id"]) == {"Donor one", "Donor two"}
    assert set(contrasts["possible_pairs"]) == {4}
    assert set(contrasts["control_mean_mu_lambda"]) == {1.5, 15.0}

    summary = summarize_contrast_draws(contrasts, hdi_prob=0.8)
    assert len(summary) == 4
    assert set(summary["parameter"]) == {
        "percent_delta_mu_lambda",
        "percent_delta_sigma_lambda",
    }
    assert set(summary["n_draws"]) == {4}


def test_joint_figures_use_paired_joint_contours_for_requested_grouping() -> None:
    mapping = _posterior_mapping()
    population = population_posterior_frame(mapping, outcome="contacts")
    population_figure = population_joint_posterior_figure(population)
    assert any(trace.type == "histogram2dcontour" for trace in population_figure.data)
    assert {
        trace.legendgroup
        for trace in population_figure.data
        if trace.showlegend
    } == {"No treatment", "Rituximab"}

    donors = donor_posterior_frame(
        mapping,
        outcome="contacts",
        conditions=("No treatment",),
    )
    donor_figure = donor_joint_posterior_figure(donors, group_by="donor_id")
    assert any(trace.type == "histogram2dcontour" for trace in donor_figure.data)
    assert {
        trace.legendgroup
        for trace in donor_figure.data
        if trace.showlegend
    } == {"D1", "D2"}
