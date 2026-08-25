from __future__ import annotations

import numpy as np

from webapp.pages import bayes_101


def test_two_parameter_surfaces_share_a_grid_and_obey_bayes_rule() -> None:
    means, scales, likelihood, prior, unnormalised, posterior, evidence = bayes_101._two_parameter_surfaces()

    expected_shape = (len(scales), len(means))
    assert likelihood.shape == prior.shape == unnormalised.shape == posterior.shape == expected_shape
    assert evidence > 0
    assert all(np.isfinite(surface).all() for surface in (likelihood, prior, unnormalised, posterior))
    np.testing.assert_allclose(unnormalised, likelihood * prior)
    np.testing.assert_allclose(posterior, unnormalised / evidence)


def test_likelihood_peak_tracks_the_sample_mean_and_spread() -> None:
    means, scales, likelihood, *_ = bayes_101._two_parameter_surfaces()
    scale_index, mean_index = np.unravel_index(np.argmax(likelihood), likelihood.shape)
    maximum_likelihood_mean = means[mean_index]
    maximum_likelihood_scale = scales[scale_index]

    np.testing.assert_allclose(maximum_likelihood_mean, bayes_101.TWO_PARAMETER_DATA.mean(), atol=0.03)
    np.testing.assert_allclose(maximum_likelihood_scale, bayes_101.TWO_PARAMETER_DATA.std(ddof=0), atol=0.03)


def test_mcmc_animation_contains_accepted_and_rejected_updates_in_valid_space() -> None:
    figure = bayes_101._mcmc_figure()
    statuses = [frame.data[5].text[0].lower() for frame in figure.frames]
    current_pairs = [(float(frame.data[2].x[0]), float(frame.data[2].y[0])) for frame in figure.frames]

    assert any("accepted" in status for status in statuses)
    assert any("rejected" in status for status in statuses)
    assert "marginals ready" in statuses[-1]
    assert {trace.type for trace in figure.data} >= {"contour", "histogram", "scatter"}
    assert all(bayes_101.MEAN_BOUNDS[0] <= mean <= bayes_101.MEAN_BOUNDS[1] for mean, _ in current_pairs)
    assert all(bayes_101.SCALE_BOUNDS[0] <= scale <= bayes_101.SCALE_BOUNDS[1] for _, scale in current_pairs)


def test_smc_moves_a_constant_particle_population_from_prior_to_posterior() -> None:
    temperatures, states = bayes_101._smc_particle_states()
    particle_counts = {len(means) for _, means, _ in states}
    posterior = bayes_101._two_parameter_surfaces()[5]

    assert temperatures[0] == 0
    assert temperatures[-1] == 1
    assert np.all(np.diff(temperatures) > 0)
    assert particle_counts == {46}
    np.testing.assert_allclose(bayes_101._tempered_surface(1.0), posterior / posterior.max())


def test_update_and_smc_figures_end_at_readable_joint_and_marginal_views() -> None:
    update = bayes_101._bayes_update_figure()
    smc = bayes_101._smc_figure()

    assert [frame.name for frame in update.frames] == ["likelihood", "prior", "multiply", "normalise"]
    assert "total area one" in update.frames[-1].data[1].text[0]
    assert "marginals ready" in smc.frames[-1].data[-1].text[0]
    assert sum(trace.type == "histogram" for trace in smc.data) == 2
