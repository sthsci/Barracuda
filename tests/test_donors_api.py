from __future__ import annotations

from types import SimpleNamespace

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from bayesorca.donors import (
    DONOR_MODEL_KEYS,
    DonorSimulationSpec,
    canonical_donor_model_key,
    cartesian_contrast_draws,
    condition_contrast_frame,
    donor_posterior_frame,
    leave_one_donor_out_moments,
    population_posterior_frame,
    population_variance_decomposition,
    simulate_donor_event_counts,
    summarize_contrast_draws,
)


def _posterior_fixture(*, corrupt_population_sigma: bool = False) -> az.InferenceData:
    """Two donors with exactly reconstructable, nonconstant posterior draws."""

    chain = [2, 7]
    draw = [10, 20, 30]
    donor = [0, 1]
    base_mu = np.asarray(
        [
            [[2.0, 6.0], [3.0, 7.0], [4.0, 8.0]],
            [[2.5, 5.5], [3.5, 6.5], [4.5, 7.5]],
        ]
    )
    sigma = np.asarray(
        [
            [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]],
            [[0.8, 1.8], [0.9, 1.9], [1.0, 2.0]],
        ]
    )
    p_zero = np.asarray(
        [
            [[0.0, 0.5], [0.1, 0.4], [0.2, 0.3]],
            [[0.05, 0.45], [0.15, 0.35], [0.25, 0.25]],
        ]
    )
    raw_weights = np.asarray([0.25, 0.75])
    active_mass = raw_weights * (1.0 - p_zero)
    active_weights = active_mass / active_mass.sum(axis=2, keepdims=True)
    population_mu = np.sum(active_weights * base_mu, axis=2)
    population_variance = np.sum(
        active_weights * (sigma**2 + base_mu**2), axis=2
    ) - population_mu**2
    population_sigma = np.sqrt(population_variance)
    if corrupt_population_sigma:
        population_sigma = population_sigma + 0.25
    population_p_zero = np.sum(raw_weights * p_zero, axis=2)

    posterior = xr.Dataset(
        data_vars={
            "mu_lambda_donor": (("chain", "draw", "donor"), base_mu),
            "sigma_lambda_donor": (("chain", "draw", "donor"), sigma),
            "phi_0_donor": (("chain", "draw", "donor"), p_zero),
            "mu_lambda_population": (("chain", "draw"), population_mu),
            "sigma_lambda_population": (("chain", "draw"), population_sigma),
            "phi_0_population": (("chain", "draw"), population_p_zero),
        },
        coords={"chain": chain, "draw": draw, "donor": donor},
    )
    return az.InferenceData(posterior=posterior)


def test_model_keys_are_lowercase_and_accept_backend_aliases():
    assert DONOR_MODEL_KEYS == ("homo", "z2p", "dis2p", "hetero3")
    assert canonical_donor_model_key("Homogeneous Poisson") == "homo"
    assert canonical_donor_model_key("Z2P") == "z2p"
    assert canonical_donor_model_key("Gamma-Poisson") == "dis2p"
    assert canonical_donor_model_key("ZI_Gamma") == "hetero3"
    with pytest.raises(ValueError, match="unknown donor-aware model"):
        canonical_donor_model_key("not-a-model")


def test_simulation_spec_normalizes_unequal_donors_and_validates_models():
    spec = DonorSimulationSpec(
        donor_sizes={"D1": 7, "D2": 11, "D3": 5},
        mu_lambda={"D1": 2.0, "D2": 4.0, "D3": 8.0},
        sigma_lambda=(0.5, 1.0, 2.0),
        p_zero=0.2,
        model_key="ZI-Gamma",
        observation_time=1.5,
        seed=np.int64(42),
    )
    assert spec.model_key == "hetero3"
    assert spec.donor_ids == ("D1", "D2", "D3")
    assert spec.cells_per_donor == (7, 11, 5)
    assert spec.mu_lambda == (2.0, 4.0, 8.0)
    assert spec.p_zero == (0.2, 0.2, 0.2)
    assert spec.n_cells == 23

    with pytest.raises(ValueError, match="requires sigma_lambda=0"):
        DonorSimulationSpec({"D1": 4}, 2.0, model_key="z2p", sigma_lambda=1.0)
    with pytest.raises(ValueError, match="requires p_zero=0"):
        DonorSimulationSpec({"D1": 4}, 2.0, model_key="dis2p", p_zero=0.1)
    with pytest.raises(ValueError, match=r"p_zero must lie in \[0, 1\)"):
        DonorSimulationSpec({"D1": 4}, 2.0, p_zero=1.0)
    with pytest.raises(ValueError, match="name every donor exactly once"):
        DonorSimulationSpec(
            {"D1": 4, "D2": 5},
            {"D1": 2.0},
        )


def test_donor_simulation_is_reproducible_and_reports_exact_mixture_truths():
    spec = DonorSimulationSpec(
        donor_sizes={"small": 10, "large": 30},
        mu_lambda={"small": 2.0, "large": 6.0},
        sigma_lambda={"small": 1.0, "large": 2.0},
        p_zero={"small": 0.0, "large": 0.5},
        seed=123,
    )
    frame, truth = simulate_donor_event_counts(spec)
    repeated, repeated_truth = simulate_donor_event_counts(spec)

    pd.testing.assert_frame_equal(frame, repeated)
    assert truth == repeated_truth
    assert list(frame.columns) == ["cell_id", "donor_id", "count"]
    assert frame.groupby("donor_id", sort=False).size().to_dict() == {
        "small": 10,
        "large": 30,
    }
    assert frame["cell_id"].is_unique
    assert frame["count"].ge(0).all()
    assert pd.api.types.is_integer_dtype(frame["count"])

    # Raw weights are (.25, .75); active masses are (.25, .375), hence
    # engaging-cell donor weights are exactly (.4, .6).
    assert truth["donors"]["small"]["active_donor_weight"] == pytest.approx(0.4)
    assert truth["donors"]["large"]["active_donor_weight"] == pytest.approx(0.6)
    population = truth["population"]
    assert population["mu_lambda"] == pytest.approx(4.4)
    assert population["p_zero"] == pytest.approx(0.375)
    assert population["variance_within_donor"] == pytest.approx(2.8)
    assert population["variance_between_donor"] == pytest.approx(3.84)
    assert population["sigma_lambda"] ** 2 == pytest.approx(6.64)


def test_posterior_frames_accept_idata_and_result_and_preserve_particle_pairing():
    idata = _posterior_fixture()
    population = population_posterior_frame(idata, condition="Control")
    assert list(population[["chain", "draw"]].itertuples(index=False, name=None)) == [
        (2, 10),
        (2, 20),
        (2, 30),
        (7, 10),
        (7, 20),
        (7, 30),
    ]
    assert "p_zero_population" in population
    assert "phi_0_population" not in population
    assert population.loc[0, "mu_lambda_population"] == pytest.approx(4.4)
    assert population.loc[0, "p_zero_population"] == pytest.approx(0.375)

    result = SimpleNamespace(
        idata=idata,
        model_key="Hetero3",
        donor_labels=("Alice", "Bob"),
    )
    donors = donor_posterior_frame(result, condition="Treatment")
    assert donors.shape[0] == 2 * 3 * 2
    assert donors["model_key"].unique().tolist() == ["hetero3"]
    assert donors["donor_id"].unique().tolist() == ["Alice", "Bob"]
    assert "p_zero_donor" in donors
    assert "phi_0_donor" not in donors
    paired = donors.loc[
        (donors["chain"] == 2) & (donors["draw"] == 10)
    ].set_index("donor_id")
    assert paired.loc["Alice", "mu_lambda_donor"] == 2.0
    assert paired.loc["Alice", "sigma_lambda_donor"] == 1.0
    assert paired.loc["Bob", "mu_lambda_donor"] == 6.0
    assert paired.loc["Bob", "p_zero_donor"] == 0.5


def test_posterior_subsampling_selects_complete_reproducible_draw_pairs():
    result = SimpleNamespace(
        idata=_posterior_fixture(),
        model_key="hetero3",
        donor_labels=("A", "B"),
    )
    first = donor_posterior_frame(result, max_draws=2, random_seed=81)
    second = donor_posterior_frame(result, max_draws=2, random_seed=81)
    pd.testing.assert_frame_equal(first, second)
    counts = first.groupby(["chain", "draw"]).size()
    assert len(counts) == 2
    assert counts.eq(2).all()


def test_population_variance_decomposition_reconstructs_saved_moments_exactly():
    decomposition = population_variance_decomposition(
        _posterior_fixture(),
        donor_weights=[1, 3],
    )
    first = decomposition.sel(chain=2, draw=10)
    assert first["active_donor_weight"].values == pytest.approx([0.4, 0.6])
    assert first["mu_lambda_population"].item() == pytest.approx(4.4)
    assert first["p_zero_population"].item() == pytest.approx(0.375)
    assert first["variance_within_donor"].item() == pytest.approx(2.8)
    assert first["variance_between_donor"].item() == pytest.approx(3.84)
    assert first["variance_total"].item() == pytest.approx(6.64)
    np.testing.assert_allclose(
        decomposition["variance_total"],
        decomposition["variance_within_donor"]
        + decomposition["variance_between_donor"],
    )
    np.testing.assert_allclose(
        decomposition["sigma_lambda_population"] ** 2,
        decomposition["variance_total"],
    )

    with pytest.raises(ValueError, match="differs from the saved posterior"):
        population_variance_decomposition(
            _posterior_fixture(corrupt_population_sigma=True),
            [1, 3],
        )


def test_leave_one_donor_out_recomputes_mixtures_without_refitting():
    result = SimpleNamespace(
        idata=_posterior_fixture(),
        donor_labels=("Alice", "Bob"),
    )
    loo = leave_one_donor_out_moments(result, [1, 3])
    assert loo.excluded_donor.values.tolist() == ["Alice", "Bob"]
    assert "no refitting" in loo.attrs["interpretation"].lower()

    excluding_alice = loo.sel(excluded_donor="Alice", chain=2, draw=10)
    assert excluding_alice["mu_lambda_population"].item() == 6.0
    assert excluding_alice["sigma_lambda_population"].item() == 2.0
    assert excluding_alice["p_zero_population"].item() == 0.5
    assert excluding_alice["variance_within_donor"].item() == 4.0
    assert excluding_alice["variance_between_donor"].item() == 0.0

    excluding_bob = loo.sel(excluded_donor="Bob", chain=2, draw=10)
    assert excluding_bob["mu_lambda_population"].item() == 2.0
    assert excluding_bob["sigma_lambda_population"].item() == 1.0
    assert excluding_bob["p_zero_population"].item() == 0.0


def test_cartesian_contrasts_are_exact_or_reproducible_monte_carlo():
    treatment = np.asarray([[3.0, 30.0], [5.0, 50.0]])
    control = np.asarray([[1.0, 10.0], [2.0, 20.0]])
    exact, exact_metadata = cartesian_contrast_draws(treatment, control)
    np.testing.assert_allclose(
        exact,
        [[2.0, 20.0], [1.0, 10.0], [4.0, 40.0], [3.0, 30.0]],
    )
    assert exact_metadata["exact_cartesian"] is True
    assert exact_metadata["possible_pairs"] == 4

    percent, metadata = cartesian_contrast_draws(
        treatment,
        control,
        scale="percent_of_control_mean",
    )
    np.testing.assert_allclose(percent[0], [200.0 / 1.5, 2000.0 / 15.0])
    np.testing.assert_allclose(metadata["control_mean"], [1.5, 15.0])

    approximate_1, metadata_1 = cartesian_contrast_draws(
        treatment,
        control,
        max_exact_pairs=1,
        approximate_pairs=17,
        random_seed=99,
    )
    approximate_2, metadata_2 = cartesian_contrast_draws(
        treatment,
        control,
        max_exact_pairs=1,
        approximate_pairs=17,
        random_seed=99,
    )
    np.testing.assert_array_equal(approximate_1, approximate_2)
    assert metadata_1 == metadata_2
    assert metadata_1["exact_cartesian"] is False
    assert approximate_1.shape == (17, 2)


def test_condition_contrasts_compare_independent_draws_per_donor():
    donor_draws = pd.DataFrame(
        {
            "condition": ["Control"] * 4 + ["Treatment"] * 4,
            "donor_id": ["A", "A", "B", "B"] * 2,
            "chain": [0] * 8,
            "draw": [0, 1, 0, 1] * 2,
            "mu_lambda_donor": [1.0, 2.0, 10.0, 20.0, 3.0, 5.0, 15.0, 25.0],
        }
    )
    contrasts = condition_contrast_frame(
        donor_draws,
        treatment="Treatment",
        control="Control",
        parameter_columns=["mu_lambda_donor"],
    )
    assert contrasts.groupby("donor_id").size().to_dict() == {"A": 4, "B": 4}
    a_values = contrasts.loc[
        contrasts["donor_id"] == "A", "delta_mu_lambda_donor"
    ].to_numpy()
    np.testing.assert_allclose(a_values, [2.0, 1.0, 4.0, 3.0])
    assert contrasts["exact_cartesian"].all()
    assert contrasts["possible_pairs"].eq(4).all()


def test_condition_contrasts_keep_models_separate_by_default():
    draws = pd.DataFrame(
        {
            "condition": ["Control", "Treatment"] * 2,
            "model_key": ["homo", "homo", "dis2p", "dis2p"],
            "mu_lambda_population": [1.0, 3.0, 10.0, 14.0],
        }
    )
    contrasts = condition_contrast_frame(
        draws,
        treatment="Treatment",
        control="Control",
        parameter_columns=["mu_lambda_population"],
    )

    assert set(contrasts["model_key"]) == {"homo", "dis2p"}
    assert contrasts.groupby("model_key")["delta_mu_lambda_population"].first().to_dict() == {
        "homo": 2.0,
        "dis2p": 4.0,
    }
    summary = summarize_contrast_draws(contrasts)
    assert set(summary["model_key"]) == {"homo", "dis2p"}


def test_contrast_summary_reports_hdi_and_both_sign_probabilities():
    contrasts = pd.DataFrame(
        {
            "donor_id": ["A"] * 4 + ["B"] * 3,
            "delta_mu_lambda_donor": [-1.0, 1.0, 2.0, 3.0, -3.0, -2.0, -1.0],
        }
    )
    summary = summarize_contrast_draws(
        contrasts,
        group_columns=["donor_id"],
        hdi_prob=0.8,
    ).set_index("donor_id")
    assert summary.loc["A", "mean"] == pytest.approx(1.25)
    assert summary.loc["A", "probability_above_zero"] == 0.75
    assert summary.loc["A", "probability_below_zero"] == 0.25
    assert summary.loc["A", "n_draws"] == 4
    assert summary.loc["B", "probability_above_zero"] == 0.0
    assert summary.loc["B", "probability_below_zero"] == 1.0
