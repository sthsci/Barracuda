from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from webapp.pages import bayes_101


def test_two_parameter_surfaces_share_a_grid_and_obey_bayes_rule() -> None:
    means, scales, likelihood, prior, unnormalised, posterior, evidence = bayes_101._two_parameter_surfaces()

    expected_shape = (len(scales), len(means))
    assert likelihood.shape == prior.shape == unnormalised.shape == posterior.shape == expected_shape
    assert evidence > 0
    assert all(np.isfinite(surface).all() for surface in (likelihood, prior, unnormalised, posterior))
    np.testing.assert_allclose(unnormalised, likelihood * prior)
    np.testing.assert_allclose(posterior, unnormalised / evidence)
    np.testing.assert_allclose(prior, 1.0 / (np.ptp(bayes_101.MEAN_BOUNDS) * np.ptp(bayes_101.SCALE_BOUNDS)))
    np.testing.assert_allclose(trapezoid(trapezoid(prior, means, axis=1), scales), 1.0)
    np.testing.assert_allclose(trapezoid(trapezoid(posterior, means, axis=1), scales), 1.0)
    np.testing.assert_allclose(posterior / posterior.max(), likelihood / likelihood.max())
    np.testing.assert_allclose(np.exp(bayes_101._parameter_log_prior(means[0], scales[-1])), prior[0, 0])
    assert bayes_101._parameter_log_prior(means[0] - 0.1, scales[0]) == -np.inf
    assert bayes_101._parameter_log_prior(means[0], scales[0] - 0.01) == -np.inf


def test_likelihood_peak_tracks_the_sample_mean_and_spread() -> None:
    means, scales, likelihood, *_ = bayes_101._two_parameter_surfaces()
    scale_index, mean_index = np.unravel_index(np.argmax(likelihood), likelihood.shape)
    maximum_likelihood_mean = means[mean_index]
    maximum_likelihood_scale = scales[scale_index]

    np.testing.assert_allclose(maximum_likelihood_mean, bayes_101.TWO_PARAMETER_DATA.mean(), atol=0.03)
    np.testing.assert_allclose(maximum_likelihood_scale, bayes_101.TWO_PARAMETER_DATA.std(ddof=0), atol=0.03)


def test_mcmc_animation_contains_accepted_and_rejected_updates_in_valid_space() -> None:
    figure = bayes_101._mcmc_figure()
    statuses = [frame.data[6].text[0].lower() for frame in figure.frames]
    current_pairs = [(float(frame.data[3].x[0]), float(frame.data[3].y[0])) for frame in figure.frames]

    assert any("accepted" in status for status in statuses)
    assert any("rejected" in status for status in statuses)
    assert len(figure.frames[-1].data[1].x) == bayes_101.MCMC_DRAWS
    assert "retained 1,000/1,000" in statuses[-1]
    assert "sample complete" in statuses[-1]
    assert len(figure.frames[-1].data[2].x) == 0
    assert sum(trace.type == "bar" for trace in figure.data) == 2
    assert all(trace.type != "histogram" for trace in figure.data)
    assert np.count_nonzero(figure.frames[-1].data[0].y) > 1
    assert np.count_nonzero(figure.frames[-1].data[5].x) > 1
    assert all(np.array_equal(frame.data[0].x, figure.frames[0].data[0].x) for frame in figure.frames)
    assert all(np.array_equal(frame.data[5].y, figure.frames[0].data[5].y) for frame in figure.frames)
    assert all(bayes_101.MEAN_BOUNDS[0] <= mean <= bayes_101.MEAN_BOUNDS[1] for mean, _ in current_pairs)
    assert all(bayes_101.SCALE_BOUNDS[0] <= scale <= bayes_101.SCALE_BOUNDS[1] for _, scale in current_pairs)

    teaching = [frame for frame in figure.frames if frame.group == "teaching"]
    assert len(teaching) == 2 * bayes_101.MCMC_TEACHING_DRAWS
    assert len(figure.frames[0].data[1].x) == 0
    previous_pair = current_pairs[0]
    for draw, (proposal, decision) in enumerate(zip(teaching[::2], teaching[1::2], strict=True), start=1):
        assert "1/2 propose" in proposal.data[6].text[0]
        assert "2/2" in decision.data[6].text[0]
        assert len(proposal.data[1].x) == draw - 1
        assert len(decision.data[1].x) == draw
        np.testing.assert_allclose([proposal.data[3].x[0], proposal.data[3].y[0]], previous_pair)
        next_pair = (decision.data[3].x[0], decision.data[3].y[0])
        if "rejected" in decision.data[6].text[0]:
            np.testing.assert_allclose(next_pair, previous_pair)
        else:
            np.testing.assert_allclose(next_pair, [proposal.data[4].x[-1], proposal.data[4].y[-1]])
        previous_pair = next_pair
    for frame in figure.frames:
        np.testing.assert_allclose(frame.data[0].y, bayes_101._relative_histogram(frame.data[1].x, bayes_101.MEAN_MARGINAL_EDGES))
        np.testing.assert_allclose(frame.data[5].x, bayes_101._relative_histogram(frame.data[1].y, bayes_101.SCALE_MARGINAL_EDGES))
        assert frame.traces == (0, 3, 4, 5, 6, 7, 9)
    controls = figure.layout.updatemenus[0]
    assert controls.buttons[0].args[1]["frame"]["duration"] >= 1000
    assert controls.y > figure.layout.sliders[0].y

    grid_means, grid_scales, *_, posterior, _ = bayes_101._two_parameter_surfaces()
    posterior_mass = posterior / posterior.sum()
    final_sample = figure.frames[-1].data[1]
    np.testing.assert_allclose(np.mean(final_sample.x), np.sum(posterior_mass * grid_means[None, :]), atol=0.05)
    np.testing.assert_allclose(np.mean(final_sample.y), np.sum(posterior_mass * grid_scales[:, None]), atol=0.05)


def test_smc_moves_a_constant_particle_population_from_prior_to_posterior() -> None:
    temperatures, states = bayes_101._smc_particle_states()
    particle_counts = {len(means) for _, _, means, _, _ in states}
    phases = {phase for _, phase, _, _, _ in states}
    posterior = bayes_101._two_parameter_surfaces()[5]

    assert temperatures[0] == 0
    assert temperatures[-1] == 1
    assert np.all(np.diff(temperatures) > 0)
    assert particle_counts == {bayes_101.SMC_PARTICLES}
    assert phases == {"prior", "reweight", "resample", "move"}
    assert states[-1][0:2] == (1.0, "move")
    assert all(np.isclose(weights.sum(), 1.0) for _, _, _, _, weights in states)
    np.testing.assert_allclose(bayes_101._tempered_surface(1.0), posterior / posterior.max())

    for prior_state, reweighted, resampled, moved in zip(states[:-1:3], states[1::3], states[2::3], states[3::3], strict=True):
        assert reweighted[0] == resampled[0] == moved[0]
        np.testing.assert_array_equal(reweighted[2], prior_state[2])
        np.testing.assert_array_equal(reweighted[3], prior_state[3])
        available = set(zip(reweighted[2], reweighted[3], strict=True))
        assert all(pair in available for pair in zip(resampled[2], resampled[3], strict=True))
        assert np.allclose(resampled[4], 1 / bayes_101.SMC_PARTICLES)
        assert not (np.array_equal(resampled[2], moved[2]) and np.array_equal(resampled[3], moved[3]))

    reweighted_states = [state for state in states if state[1] == "reweight"]
    target_ess = bayes_101.SMC_ESS_FRACTION * bayes_101.SMC_PARTICLES
    assert all(abs(1 / np.sum(state[4] ** 2) - target_ess) < 1 for state in reweighted_states[:-1])
    assert 1 / np.sum(reweighted_states[-1][4] ** 2) >= target_ess - 1

    means, scales, *_ = bayes_101._two_parameter_surfaces()
    posterior_mass = posterior / posterior.sum()
    final_means, final_scales = states[-1][2:4]
    np.testing.assert_allclose(final_means.mean(), np.sum(posterior_mass * means[None, :]), atol=0.04)
    np.testing.assert_allclose(final_scales.mean(), np.sum(posterior_mass * scales[:, None]), atol=0.04)


def test_sampler_figures_end_at_readable_joint_and_fixed_marginal_views() -> None:
    smc = bayes_101._smc_figure()

    assert "posterior" in smc.frames[-1].layout.annotations[0].text
    assert sum(trace.type == "bar" for trace in smc.data) == 2
    assert all(trace.type != "histogram" for trace in smc.data)
    assert sum(trace.name == "Grid stage target" for trace in smc.data) == 1
    assert all(np.array_equal(frame.data[0].x, smc.frames[0].data[0].x) for frame in smc.frames)
    assert all(np.array_equal(frame.data[4].y, smc.frames[0].data[4].y) for frame in smc.frames)
    assert len(smc.layout.shapes) == 3
    assert len(bayes_101._mcmc_figure().layout.shapes) == 3
    # Resampling must expose multiplicity rather than silently overplot copies.
    _, states = bayes_101._smc_particle_states()
    for index, (frame, state) in enumerate(zip(smc.frames, states, strict=True)):
        buttons = {button.label: button for button in frame.layout.updatemenus[0].buttons}
        assert buttons["Next"].args[0] == (f"smc-{min(index + 1, len(states) - 1)}",)
        assert buttons["Back"].args[0] == (f"smc-{max(index - 1, 0)}",)
        assert buttons["Play slowly"].args[1]["frame"]["duration"] >= 2000
        if index == len(states) - 1:
            assert len(buttons["Play slowly"].args[0]) == len(states)
        if state[1] == "resample":
            copies = np.asarray(frame.data[3].customdata).ravel()
            assert copies.sum() == bayes_101.SMC_PARTICLES
            assert copies.max() > 1
            assert len(copies) < bayes_101.SMC_PARTICLES
        elif state[1] == "reweight":
            np.testing.assert_allclose(frame.data[3].x, states[index - 1][2])
            np.testing.assert_allclose(frame.data[0].y, bayes_101._relative_histogram(state[2], bayes_101.MEAN_MARGINAL_EDGES, weights=state[4]))


def test_bayesian_update_maps_use_uniform_prior_and_honest_shared_axes() -> None:
    _, _, likelihood, prior, _, posterior, _ = bayes_101._two_parameter_surfaces()
    figures = [bayes_101._bayesian_update_figure(stage) for stage in ("prior", "likelihood", "posterior")]

    for figure, surface in zip(figures, (prior, likelihood, posterior), strict=True):
        assert tuple(figure.layout.xaxis.range) == bayes_101.MEAN_BOUNDS
        assert tuple(figure.layout.yaxis.range) == bayes_101.SCALE_BOUNDS
        np.testing.assert_allclose(figure.data[0].customdata, surface)
        np.testing.assert_allclose(figure.data[0].z, surface / surface.max())
        assert "panel’s peak" in figure.data[0].hovertemplate
    np.testing.assert_allclose(figures[0].data[0].z, 1.0)
    np.testing.assert_allclose(figures[1].data[0].z, figures[2].data[0].z)
