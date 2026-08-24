"""Interactive introduction to Bayesian inference."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from scipy.stats import beta as beta_distribution

from webapp.core.coin import (
    beta_highest_density_interval,
    simulate_coin_tosses,
    uniform_prior_posterior,
)
from webapp.palette import CONDITION_BISPECIFIC, DONOR_RUST, DONOR_TEAL, MODEL_ZERO_INFLATED_GAMMA, PAPER_SPINE
from webapp.ui import hero, markdown, metrics, note, step_card


PATH = "/bayesian-101"
TITLE = "Bayesian inference 101"

BDA3_URL = "https://sites.stat.columbia.edu/gelman/book/BDA3.pdf"
SEEING_THEORY_URL = "https://seeing-theory.brown.edu/bayesian-inference/index.html"
MCMC_GALLERY_URL = "https://chi-feng.github.io/mcmc-demo/"
PYMC_SMC_URL = "https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.smc.sample_smc.html"
PYMC_BF_URL = "https://www.pymc.io/projects/examples/en/latest/diagnostics_and_criticism/Bayes_factor.html"
THOMAS_BAYES_URL = "https://en.wikipedia.org/wiki/Thomas_Bayes"
THOMAS_BAYES_PAPER_URL = "https://doi.org/10.1098/rstl.1763.0053"
THOMAS_BAYES_PORTRAIT_URL = "https://commons.wikimedia.org/wiki/File:Thomas_Bayes.gif"

BOOK_INK = "#25231F"
BOOK_PAPER = "#FAF8F2"
BOOK_SHEET = "#FFFEFA"
BOOK_RULE = "#887B66"
BOOK_GRID = "#D8D0C2"
BOOK_SERIF = "Iowan Old Style, Baskerville, Palatino Linotype, Palatino, Georgia, serif"

TWO_PARAMETER_DATA = np.array([4.8, 4.9, 5.0, 5.1, 5.3])
MEAN_BOUNDS = (4.35, 5.85)
SCALE_BOUNDS = (0.08, 0.85)
MEAN_PRIOR_LOCATION = 5.45
MEAN_PRIOR_SCALE = 0.15
SCALE_PRIOR_SCALE = 0.45
SURFACE_COLORS = [[0.0, "#F5F1E8"], [0.35, "#B7CDC2"], [0.72, "#486857"], [1.0, "#20372E"]]


def _external_link(label: str, href: str, *, class_name: str | None = None) -> html.A:
    return html.A(label, href=href, target="_blank", rel="noreferrer", className=class_name)


def _plot_layout(
    *,
    height: int,
    bottom_margin: int = 64,
    top_margin: int = 72,
    left_margin: int = 58,
    right_margin: int = 24,
) -> dict:
    return {
        "template": "none",
        "height": height,
        "paper_bgcolor": BOOK_SHEET,
        "plot_bgcolor": BOOK_PAPER,
        "font": {"family": BOOK_SERIF, "color": BOOK_INK, "size": 13},
        "margin": {"l": left_margin, "r": right_margin, "t": top_margin, "b": bottom_margin},
        "hoverlabel": {
            "bgcolor": BOOK_SHEET,
            "bordercolor": BOOK_RULE,
            "font_family": BOOK_SERIF,
            "font_color": BOOK_INK,
        },
    }


def _coin_figure(
    probability_heads: float,
    n_tosses: int,
    toss_round: int,
    hdi_percent: int = 95,
    *,
    outcomes: np.ndarray | None = None,
) -> tuple[go.Figure, list[tuple[str, str]]]:
    """Plot the exact Beta posterior and its selected highest density interval."""
    if outcomes is None:
        outcomes = simulate_coin_tosses(probability_heads, n_tosses, seed=2026 + toss_round)
    heads = int(outcomes.sum())
    tails = n_tosses - heads
    observed = heads / n_tosses
    posterior_alpha, posterior_beta = uniform_prior_posterior(heads, n_tosses)
    interval_mass = float(hdi_percent) / 100.0
    interval = beta_highest_density_interval(posterior_alpha, posterior_beta, interval_mass)

    x = np.linspace(0.0, 1.0, 700)
    posterior = beta_distribution.pdf(x, posterior_alpha, posterior_beta)
    density_limit = max(float(np.max(posterior)) * 1.08, 1.15)
    in_interval = (x >= interval[0]) & (x <= interval[1])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x,
            y=np.ones_like(x),
            mode="lines",
            name="Uniform prior",
            line={"color": PAPER_SPINE, "width": 2},
            hovertemplate="Prior density: 1<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=posterior,
            mode="lines",
            name=f"Posterior, Beta({posterior_alpha}, {posterior_beta})",
            line={"color": DONOR_TEAL, "width": 3},
            hovertemplate="P(head): %{x:.3f}<br>Density: %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x[in_interval],
            y=posterior[in_interval],
            mode="lines",
            name=f"{hdi_percent}% HDI",
            fill="tozeroy",
            fillcolor="rgba(0,133,133,0.18)",
            line={"color": "rgba(0,133,133,0)"},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[probability_heads, probability_heads],
            y=[0.0, density_limit],
            mode="lines",
            name=f"True probability {probability_heads:.2f}",
            line={"color": CONDITION_BISPECIFIC, "width": 2},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[observed, observed],
            y=[0.0, density_limit],
            mode="lines",
            name=f"Observed frequency {observed:.2f}",
            line={"color": MODEL_ZERO_INFLATED_GAMMA, "width": 2, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        **_plot_layout(height=430, bottom_margin=66),
        xaxis={
            "title": "Probability of heads",
            "range": [0, 1],
            "gridcolor": BOOK_GRID,
            "linecolor": BOOK_RULE,
            "ticks": "outside",
            "tickcolor": BOOK_RULE,
            "zeroline": False,
            "automargin": True,
        },
        yaxis={
            "title": "Probability density",
            "range": [0, density_limit],
            "gridcolor": BOOK_GRID,
            "linecolor": BOOK_RULE,
            "ticks": "outside",
            "tickcolor": BOOK_RULE,
            "zeroline": False,
            "automargin": True,
        },
        legend={
            "orientation": "h",
            "y": 1.02,
            "yanchor": "bottom",
            "x": 0,
            "xanchor": "left",
            "bgcolor": "rgba(255,254,250,0.88)",
            "font": {"size": 11},
        },
        hovermode="closest",
    )

    posterior_mean = posterior_alpha / (posterior_alpha + posterior_beta)
    values = [
        ("Tosses observed", f"{heads} heads · {tails} tails"),
        ("Observed P(head)", f"{observed:.3f}"),
        ("Posterior mean P(head)", f"{posterior_mean:.3f}"),
        (f"Posterior {hdi_percent}% HDI", f"{interval[0]:.3f}–{interval[1]:.3f}"),
    ]
    return figure, values


def _coin_frequency_figure(
    probability_heads: float,
    outcomes: Sequence[int] | np.ndarray,
) -> go.Figure:
    """Plot cumulative empirical frequencies for heads and tails."""
    tosses = np.arange(1, len(outcomes) + 1)
    heads = np.cumsum(np.asarray(outcomes, dtype=float)) / tosses
    tails = 1.0 - heads
    true_heads = float(probability_heads)
    true_tails = 1.0 - true_heads

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=tosses,
            y=heads,
            mode="lines+markers" if len(tosses) <= 40 else "lines",
            name="Empirical heads",
            line={"color": DONOR_TEAL, "width": 2.5},
            marker={"size": 5},
            hovertemplate="Toss %{x}<br>Heads: %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=tosses,
            y=tails,
            mode="lines+markers" if len(tosses) <= 40 else "lines",
            name="Empirical tails",
            line={"color": DONOR_RUST, "width": 2.5},
            marker={"size": 5},
            hovertemplate="Toss %{x}<br>Tails: %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[1, max(1, len(tosses))],
            y=[true_heads, true_heads],
            mode="lines",
            name=f"True heads {true_heads:.2f}",
            line={"color": DONOR_TEAL, "width": 1.5, "dash": "dot"},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[1, max(1, len(tosses))],
            y=[true_tails, true_tails],
            mode="lines",
            name=f"True tails {true_tails:.2f}",
            line={"color": DONOR_RUST, "width": 1.5, "dash": "dot"},
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        **_plot_layout(height=410, bottom_margin=64),
        xaxis={
            "title": "Number of tosses",
            "range": [1, max(2, len(tosses))],
            "gridcolor": BOOK_GRID,
            "linecolor": BOOK_RULE,
            "ticks": "outside",
            "tickcolor": BOOK_RULE,
            "zeroline": False,
            "automargin": True,
        },
        yaxis={
            "title": "Empirical frequency",
            "range": [0, 1],
            "tickformat": ".0%",
            "gridcolor": BOOK_GRID,
            "linecolor": BOOK_RULE,
            "ticks": "outside",
            "tickcolor": BOOK_RULE,
            "zeroline": False,
            "automargin": True,
        },
        legend={
            "orientation": "h",
            "y": 1.02,
            "yanchor": "bottom",
            "x": 0,
            "xanchor": "left",
            "bgcolor": "rgba(255,254,250,0.88)",
            "font": {"size": 11},
        },
        hovermode="x unified",
    )
    return figure


def _recent_outcomes(outcomes: Sequence[int] | np.ndarray) -> list[html.Span]:
    recent = list(outcomes)[-24:]
    return [
        html.Span("H" if outcome else "T", className=f"barracuda-outcome {'heads' if outcome else 'tails'}")
        for outcome in recent
    ]


def _relative_density(log_density: np.ndarray) -> np.ndarray:
    """Scale a log density to the interval zero to one for comparison plots."""
    return np.exp(log_density - np.nanmax(log_density))


@lru_cache(maxsize=1)
def _two_parameter_surfaces() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    """Return a shared grid for the likelihood, prior and posterior example."""
    means = np.linspace(*MEAN_BOUNDS, 72)
    scales = np.linspace(*SCALE_BOUNDS, 62)
    mean_grid, scale_grid = np.meshgrid(means, scales)
    squared_error = sum((observation - mean_grid) ** 2 for observation in TWO_PARAMETER_DATA)
    log_likelihood = (
        -len(TWO_PARAMETER_DATA) * np.log(scale_grid * np.sqrt(2.0 * np.pi))
        - squared_error / (2.0 * scale_grid**2)
    )
    log_mean_prior = -np.log(MEAN_PRIOR_SCALE * np.sqrt(2.0 * np.pi)) - 0.5 * ((mean_grid - MEAN_PRIOR_LOCATION) / MEAN_PRIOR_SCALE) ** 2
    log_scale_prior = np.log(np.sqrt(2.0 / np.pi) / SCALE_PRIOR_SCALE) - 0.5 * (scale_grid / SCALE_PRIOR_SCALE) ** 2
    likelihood = np.exp(log_likelihood)
    prior = np.exp(log_mean_prior + log_scale_prior)
    unnormalised_posterior = likelihood * prior
    mean_step = float(means[1] - means[0])
    scale_step = float(scales[1] - scales[0])
    evidence = float(unnormalised_posterior.sum() * mean_step * scale_step)
    posterior = unnormalised_posterior / evidence
    return means, scales, likelihood, prior, unnormalised_posterior, posterior, evidence


def _parameter_log_likelihood(mean: float, scale: float) -> float:
    if not MEAN_BOUNDS[0] <= mean <= MEAN_BOUNDS[1] or not SCALE_BOUNDS[0] <= scale <= SCALE_BOUNDS[1]:
        return -np.inf
    squared_error = float(np.sum((TWO_PARAMETER_DATA - mean) ** 2))
    return -len(TWO_PARAMETER_DATA) * np.log(scale) - squared_error / (2.0 * scale**2)


def _parameter_log_prior(mean: float, scale: float) -> float:
    if not MEAN_BOUNDS[0] <= mean <= MEAN_BOUNDS[1] or not SCALE_BOUNDS[0] <= scale <= SCALE_BOUNDS[1]:
        return -np.inf
    return -0.5 * ((mean - MEAN_PRIOR_LOCATION) / MEAN_PRIOR_SCALE) ** 2 - 0.5 * (scale / SCALE_PRIOR_SCALE) ** 2


@lru_cache(maxsize=4)
def _parameter_surface_figure(surface_name: str, quantity: str) -> go.Figure:
    means, scales, likelihood, prior, unnormalised, posterior, _ = _two_parameter_surfaces()
    surfaces = {
        "likelihood": likelihood,
        "prior": prior,
        "unnormalised": unnormalised,
        "posterior": posterior,
    }
    surface = surfaces[surface_name]
    relative_surface = surface / float(np.max(surface))
    figure = go.Figure(
        go.Contour(
            x=means,
            y=scales,
            z=relative_surface,
            customdata=surface,
            zmin=0,
            zmax=1,
            colorscale=SURFACE_COLORS,
            contours={"coloring": "heatmap", "showlines": False},
            showscale=False,
            hovertemplate=f"Mean μ: %{{x:.2f}}<br>SD σ: %{{y:.2f}}<br>{quantity}: %{{customdata:.3g}}<extra></extra>",
        )
    )
    figure.update_layout(
        **_plot_layout(height=300, bottom_margin=58, top_margin=12, left_margin=56, right_margin=12),
        xaxis={"title": "Mean μ", "range": list(MEAN_BOUNDS), "gridcolor": BOOK_GRID, "linecolor": BOOK_RULE, "automargin": True},
        yaxis={"title": "SD σ", "range": list(SCALE_BOUNDS), "gridcolor": BOOK_GRID, "linecolor": BOOK_RULE, "automargin": True},
        hovermode="closest",
    )
    return figure


def _posterior_contour_trace(*, name: str = "Posterior surface") -> go.Contour:
    means, scales, _, _, _, posterior, _ = _two_parameter_surfaces()
    relative_posterior = posterior / float(np.max(posterior))
    return go.Contour(
        x=means,
        y=scales,
        z=relative_posterior,
        customdata=posterior,
        zmin=0,
        zmax=1,
        name=name,
        colorscale=SURFACE_COLORS,
        contours={"coloring": "heatmap", "showlines": False},
        showscale=False,
        hovertemplate="Mean μ: %{x:.2f}<br>SD σ: %{y:.2f}<br>Posterior density: %{customdata:.3g}<extra></extra>",
    )


@lru_cache(maxsize=1)
def _mcmc_figure() -> go.Figure:
    """Animate a two-parameter random-walk Metropolis chain."""
    rng = np.random.default_rng(401)
    current_mean, current_scale = 5.66, 0.68
    means = [current_mean]
    scales = [current_scale]
    proposed_means = [current_mean]
    proposed_scales = [current_scale]
    accepted_updates = [True]
    decisions = ["Starting pair"]

    for _ in range(59):
        proposed_mean = current_mean + rng.normal(0.0, 0.075)
        proposed_scale = current_scale + rng.normal(0.0, 0.045)
        current_log_target = _parameter_log_likelihood(current_mean, current_scale) + _parameter_log_prior(current_mean, current_scale)
        proposed_log_target = _parameter_log_likelihood(proposed_mean, proposed_scale) + _parameter_log_prior(proposed_mean, proposed_scale)
        accepted = np.log(rng.random()) < min(0.0, proposed_log_target - current_log_target)
        proposed_means.append(proposed_mean)
        proposed_scales.append(proposed_scale)
        accepted_updates.append(bool(accepted))
        if accepted:
            current_mean, current_scale = proposed_mean, proposed_scale
        means.append(current_mean)
        scales.append(current_scale)
        decisions.append("Accepted proposal" if accepted else "Rejected proposal; stayed here")

    initial_status = "Step 0 · starting pair"
    figure = go.Figure(
        data=[
            _posterior_contour_trace(),
            go.Scatter(x=[means[0]], y=[scales[0]], mode="lines+markers", name="Chain path", line={"color": PAPER_SPINE, "width": 1.5}, marker={"color": PAPER_SPINE, "size": 5}),
            go.Scatter(x=[means[0]], y=[scales[0]], mode="markers", name="Current pair", marker={"color": CONDITION_BISPECIFIC, "size": 11, "line": {"color": BOOK_INK, "width": 1}}),
            go.Scatter(x=[proposed_means[0]], y=[proposed_scales[0]], mode="markers", name="Proposed pair", marker={"color": DONOR_TEAL, "size": 10, "symbol": "diamond-open", "line": {"width": 2}}),
            go.Scatter(x=[MEAN_BOUNDS[0] + 0.22], y=[SCALE_BOUNDS[1] - 0.055], mode="text", text=[initial_status], textposition="middle right", textfont={"family": BOOK_SERIF, "size": 13, "color": BOOK_INK}, showlegend=False, hoverinfo="skip"),
        ],
        frames=[
            go.Frame(
                name=f"mcmc-{index}",
                traces=[1, 2, 3, 4],
                data=[
                    go.Scatter(x=means[: index + 1], y=scales[: index + 1], mode="lines+markers", line={"color": PAPER_SPINE, "width": 1.5}, marker={"color": PAPER_SPINE, "size": 5}),
                    go.Scatter(x=[mean], y=[scale], mode="markers", marker={"color": CONDITION_BISPECIFIC, "size": 11, "line": {"color": BOOK_INK, "width": 1}}),
                    go.Scatter(
                        x=[proposed_means[index]],
                        y=[proposed_scales[index]],
                        mode="markers",
                        marker={
                            "color": DONOR_TEAL if accepted_updates[index] else DONOR_RUST,
                            "size": 10,
                            "symbol": "diamond-open" if accepted_updates[index] else "x-open",
                            "line": {"width": 2},
                        },
                    ),
                    go.Scatter(x=[MEAN_BOUNDS[0] + 0.22], y=[SCALE_BOUNDS[1] - 0.055], mode="text", text=[f"Step {index} · {decisions[index].lower()}"], textposition="middle right", textfont={"family": BOOK_SERIF, "size": 13, "color": BOOK_INK}, showlegend=False, hoverinfo="skip"),
                ],
            )
            for index, (mean, scale) in enumerate(zip(means, scales, strict=True))
        ],
    )
    figure.update_layout(
        **_plot_layout(height=430, bottom_margin=92, top_margin=58, right_margin=20),
        xaxis={"title": "Mean μ", "range": list(MEAN_BOUNDS), "gridcolor": BOOK_GRID, "linecolor": BOOK_RULE, "automargin": True},
        yaxis={"title": "SD σ", "range": list(SCALE_BOUNDS), "gridcolor": BOOK_GRID, "linecolor": BOOK_RULE, "automargin": True},
        legend={"orientation": "h", "y": 1.03, "yanchor": "bottom", "x": 0, "font": {"size": 11}},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0,
                "y": -0.29,
                "xanchor": "left",
                "yanchor": "top",
                "showactive": False,
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 150, "redraw": False}, "transition": {"duration": 0}, "fromcurrent": True}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]},
                ],
            }
        ],
    )
    return figure


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


@lru_cache(maxsize=1)
def _smc_particle_states() -> tuple[np.ndarray, list[tuple[float, np.ndarray, np.ndarray]]]:
    """Generate deterministic tempered SMC states for the visual explanation."""
    rng = np.random.default_rng(902)
    temperatures = np.linspace(0.0, 1.0, 7)
    n_particles = 46

    def draw_within_bounds(draw, bounds: tuple[float, float]) -> np.ndarray:
        accepted: list[float] = []
        while len(accepted) < n_particles:
            candidates = np.asarray(draw(n_particles * 2))
            accepted.extend(candidates[(candidates >= bounds[0]) & (candidates <= bounds[1])].tolist())
        return np.asarray(accepted[:n_particles])

    means = draw_within_bounds(lambda size: rng.normal(MEAN_PRIOR_LOCATION, MEAN_PRIOR_SCALE, size), MEAN_BOUNDS)
    scales = draw_within_bounds(lambda size: np.abs(rng.normal(0.0, SCALE_PRIOR_SCALE, size)), SCALE_BOUNDS)
    states = [(0.0, means.copy(), scales.copy())]

    for previous_temperature, temperature in zip(temperatures[:-1], temperatures[1:], strict=True):
        log_likelihoods = np.array([_parameter_log_likelihood(mean, scale) for mean, scale in zip(means, scales, strict=True)])
        incremental_log_weights = (temperature - previous_temperature) * log_likelihoods
        incremental_log_weights -= np.max(incremental_log_weights)
        weights = np.exp(incremental_log_weights)
        weights /= weights.sum()
        ancestors = _systematic_resample(weights, rng)
        means = means[ancestors]
        scales = scales[ancestors]

        for _ in range(3):
            for particle in range(n_particles):
                proposed_mean = means[particle] + rng.normal(0.0, 0.055)
                proposed_scale = scales[particle] + rng.normal(0.0, 0.032)
                current_target = _parameter_log_prior(means[particle], scales[particle]) + temperature * _parameter_log_likelihood(means[particle], scales[particle])
                proposed_target = _parameter_log_prior(proposed_mean, proposed_scale) + temperature * _parameter_log_likelihood(proposed_mean, proposed_scale)
                if np.log(rng.random()) < min(0.0, proposed_target - current_target):
                    means[particle] = proposed_mean
                    scales[particle] = proposed_scale
        states.append((float(temperature), means.copy(), scales.copy()))
    return temperatures, states


def _tempered_surface(temperature: float) -> np.ndarray:
    _, _, likelihood, prior, _, _, _ = _two_parameter_surfaces()
    log_likelihood = np.log(np.clip(likelihood, 1e-300, None))
    log_prior = np.log(np.clip(prior, 1e-300, None))
    return _relative_density(log_prior + temperature * log_likelihood)


@lru_cache(maxsize=1)
def _smc_figure() -> go.Figure:
    """Animate two-parameter particles through tempered distributions."""
    _, states = _smc_particle_states()
    initial_temperature, initial_means, initial_scales = states[0]

    def frame_data(temperature: float, means: np.ndarray, scales: np.ndarray) -> list[go.BaseTraceType]:
        grid_means, grid_scales, _, _, _, _, _ = _two_parameter_surfaces()
        return [
            go.Contour(
                x=grid_means,
                y=grid_scales,
                z=_tempered_surface(temperature),
                zmin=0,
                zmax=1,
                colorscale=SURFACE_COLORS,
                contours={"coloring": "heatmap", "showlines": False},
                showscale=False,
                hoverinfo="skip",
            ),
            go.Scatter(x=means, y=scales, mode="markers", marker={"color": CONDITION_BISPECIFIC, "size": 7, "opacity": 0.76, "line": {"color": BOOK_INK, "width": 0.45}}),
            go.Scatter(x=[MEAN_BOUNDS[0] + 0.18], y=[SCALE_BOUNDS[1] - 0.055], mode="text", text=[f"β = {temperature:.2f}"], textposition="middle right", textfont={"family": BOOK_SERIF, "size": 14, "color": BOOK_INK}, showlegend=False, hoverinfo="skip"),
        ]

    initial_data = frame_data(initial_temperature, initial_means, initial_scales)
    initial_data[0].name = "Tempered target"
    initial_data[1].name = "Particles"
    figure = go.Figure(
        data=initial_data,
        frames=[go.Frame(name=f"smc-{temperature:.2f}", data=frame_data(temperature, means, scales)) for temperature, means, scales in states],
    )
    figure.update_layout(
        **_plot_layout(height=430, bottom_margin=92, top_margin=58, right_margin=20),
        xaxis={"title": "Mean μ", "range": list(MEAN_BOUNDS), "gridcolor": BOOK_GRID, "linecolor": BOOK_RULE, "automargin": True},
        yaxis={"title": "SD σ", "range": list(SCALE_BOUNDS), "gridcolor": BOOK_GRID, "linecolor": BOOK_RULE, "automargin": True},
        legend={"orientation": "h", "y": 1.03, "yanchor": "bottom", "x": 0, "font": {"size": 11}},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0,
                "y": -0.29,
                "xanchor": "left",
                "yanchor": "top",
                "showactive": False,
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": 620, "redraw": True}, "transition": {"duration": 180}, "fromcurrent": True}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}]},
                ],
            }
        ],
    )
    return figure


def _source_note(*children) -> html.P:
    return html.P(children, className="barracuda-source-note")


def _contents() -> html.Nav:
    items = [
        ("01", "Bayes theorem", "#bayes-theorem"),
        ("02", "Coin experiment", "#coin-experiment"),
        ("03", "MCMC and SMC", "#computation"),
        ("04", "Bayes factors", "#bayes-factors"),
        ("05", "Thomas Bayes", "#thomas-bayes"),
    ]
    return html.Nav(
        [
            html.Span("On this page", className="barracuda-section-label"),
            html.Ol(
                [html.Li(html.A([html.Span(number), label], href=href)) for number, label, href in items],
                className="barracuda-toc-list",
            ),
        ],
        className="barracuda-toc",
        **{"aria-label": "Bayesian inference lesson contents"},
    )


def layout() -> html.Div:
    probability = 0.5
    n_tosses = 20
    toss_round = 0
    hdi_percent = 95
    initial_outcomes = simulate_coin_tosses(probability, n_tosses, seed=2026 + toss_round)
    initial_figure, initial_metrics = _coin_figure(
        probability,
        n_tosses,
        toss_round,
        hdi_percent,
        outcomes=initial_outcomes,
    )
    initial_frequency = _coin_frequency_figure(probability, initial_outcomes)
    initial_face = "Heads" if initial_outcomes[-1] else "Tails"
    evidence = _two_parameter_surfaces()[-1]

    return html.Div(
        [
            hero(
                "Foundations",
                "Bayesian inference, from probability to computation",
                "Start with conditional probability, update a coin model by hand, then see how MCMC and SMC solve larger Bayesian problems.",
                badge="Interactive lesson · no data upload required",
            ),
            _contents(),
            html.Section(
                [
                    html.Span("01 · Bayes theorem", className="barracuda-section-label"),
                    html.H2("The update at the heart of Bayesian inference"),
                    html.P(
                        "Conditional probability describes how the probability of one event changes after another event is known. "
                        "Write the same joint event in two ways, then rearrange.",
                        className="barracuda-section-lead",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("1", className="barracuda-derivation-number"),
                                    html.H3("Condition on B"),
                                    markdown(r"$$P(A\mid B)=\frac{P(A\cap B)}{P(B)}$$", class_name="barracuda-equation small", mathjax=True),
                                ],
                                className="barracuda-derivation-card",
                            ),
                            html.Div(
                                [
                                    html.Span("2", className="barracuda-derivation-number"),
                                    html.H3("Reverse the condition"),
                                    markdown(r"$$P(B\mid A)=\frac{P(A\cap B)}{P(A)}$$", class_name="barracuda-equation small", mathjax=True),
                                ],
                                className="barracuda-derivation-card",
                            ),
                            html.Div(
                                [
                                    html.Span("3", className="barracuda-derivation-number"),
                                    html.H3("Use the joint probability"),
                                    markdown(r"$$P(A\cap B)=P(B\mid A)P(A)$$", class_name="barracuda-equation small", mathjax=True),
                                ],
                                className="barracuda-derivation-card",
                            ),
                        ],
                        className="barracuda-derivation-grid",
                    ),
                    html.Div(
                        [
                            html.Span("Bayes’ theorem", className="barracuda-equation-label"),
                            markdown(
                                r"$$P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)},\qquad P(B)>0$$",
                                class_name="barracuda-equation barracuda-equation-feature",
                                mathjax=True,
                            ),
                            html.P("Updated probability = compatibility with the observation × initial probability ÷ probability of the observation.", className="barracuda-equation-caption"),
                        ],
                        className="barracuda-theorem-panel",
                    ),
                    html.H3("From events to model parameters"),
                    html.P(
                        "For data y, parameter θ and model M, the same rule becomes the Bayesian update used in statistics.",
                        className="barracuda-copy",
                    ),
                    markdown(
                        r"$$p(\theta\mid y,M)=\frac{p(y\mid\theta,M)p(\theta\mid M)}{p(y\mid M)}$$",
                        class_name="barracuda-equation",
                        mathjax=True,
                    ),
                    html.Div(
                        [
                            step_card("01", "Prior", "What parameter values are plausible before the current data are observed."),
                            step_card("02", "Likelihood", "How compatible the observed data are with each possible parameter value."),
                            step_card("03", "Posterior", "The updated uncertainty after the prior and likelihood are combined."),
                            step_card("04", "Evidence", "The average likelihood under the prior. It normalises the posterior."),
                        ],
                        className="barracuda-card-grid four barracuda-bayes-terms",
                    ),
                    _source_note(
                        "Sources: Miller and Miller, ",
                        html.Em("John E. Freund’s Mathematical Statistics with Applications"),
                        ", 8th ed., Ch. 2 §§6 and 8; Gelman et al., ",
                        _external_link("Bayesian Data Analysis, 3rd ed.", BDA3_URL),
                        ", §1.3.",
                    ),
                ],
                id="bayes-theorem",
                className="barracuda-lesson-section",
            ),
            html.Section(
                [
                    html.Span("02 · Coin experiment", className="barracuda-section-label"),
                    html.H2("Learn the unknown bias of a coin"),
                    html.P(
                        "Suppose each toss is independent and has an unknown chance θ of heads. With a uniform Beta(1, 1) prior, h heads and t tails produce an exact Beta posterior.",
                        className="barracuda-section-lead",
                    ),
                    html.Div(
                        [
                            markdown(r"$$h\mid\theta,n\sim\operatorname{Binomial}(n,\theta)$$", class_name="barracuda-equation small", mathjax=True),
                            markdown(r"$$\theta\sim\operatorname{Beta}(1,1)$$", class_name="barracuda-equation small", mathjax=True),
                            markdown(r"$$\theta\mid h,t\sim\operatorname{Beta}(1+h,1+t)$$", class_name="barracuda-equation small", mathjax=True),
                        ],
                        className="barracuda-equation-triptych",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label(
                                        [
                                            html.Span("True probability of heads", className="barracuda-field-label"),
                                            dcc.Slider(
                                                id="coin-probability",
                                                min=0,
                                                max=1,
                                                step=0.01,
                                                value=probability,
                                                marks={0: "0", 0.5: "0.5", 1: "1"},
                                                tooltip={"placement": "bottom"},
                                            ),
                                        ],
                                        className="barracuda-field",
                                    ),
                                    html.Div(id="coin-ground-truth", children="Ground truth: P(head) = 0.50 · P(tail) = 0.50", className="barracuda-help"),
                                    html.Label(
                                        [
                                            html.Span("Number of tosses", className="barracuda-field-label"),
                                            dcc.Slider(
                                                id="coin-tosses",
                                                min=1,
                                                max=500,
                                                step=1,
                                                value=n_tosses,
                                                marks={1: "1", 100: "100", 250: "250", 500: "500"},
                                                tooltip={"placement": "bottom"},
                                            ),
                                        ],
                                        className="barracuda-field",
                                    ),
                                    html.Label(
                                        [
                                            html.Span("Highest density interval", className="barracuda-field-label"),
                                            dcc.Slider(
                                                id="coin-hdi-percent",
                                                min=50,
                                                max=99,
                                                step=1,
                                                value=hdi_percent,
                                                marks={50: "50%", 75: "75%", 99: "99%"},
                                                tooltip={"placement": "bottom"},
                                            ),
                                        ],
                                        className="barracuda-field",
                                    ),
                                    html.P("Choose how much posterior probability the HDI should contain.", className="barracuda-help"),
                                    html.Div([html.Strong("Fixed prior"), html.Br(), "P(head) ~ Beta(1, 1)"], className="barracuda-fixed-prior"),
                                    html.Button("Toss the coin again", id="coin-toss-again", n_clicks=0, className="barracuda-button primary full"),
                                ],
                                className="barracuda-control-panel",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span("✋", className="barracuda-toss-hand", **{"aria-hidden": "true"}),
                                            html.Span("H" if initial_face == "Heads" else "T", id="coin-visual-face", className="barracuda-toss-coin", **{"aria-hidden": "true"}),
                                        ],
                                        id="coin-toss-scene",
                                        className="barracuda-coin-stage",
                                        **{"aria-label": "Animated hand tossing a coin"},
                                    ),
                                    html.Div([html.Span("Latest toss", className="barracuda-mini-label"), html.Strong(initial_face, id="coin-face")], className="barracuda-coin-result"),
                                    html.Div(
                                        [
                                            html.Span("Most recent outcomes", className="barracuda-mini-label"),
                                            html.Div(_recent_outcomes(initial_outcomes), id="coin-outcomes", className="barracuda-outcome-strip"),
                                        ],
                                        className="barracuda-recent-outcomes",
                                    ),
                                ],
                                className="barracuda-coin-demo",
                            ),
                        ],
                        className="barracuda-coin-lab-grid",
                    ),
                    html.Div(id="coin-metrics", children=metrics(initial_metrics)),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Prior and posterior for the chance of heads"),
                                    dcc.Graph(id="coin-figure", figure=initial_figure, config={"displaylogo": False, "responsive": True}, className="barracuda-coin-plot"),
                                ],
                                className="barracuda-coin-chart",
                            ),
                            html.Div(
                                [
                                    html.H3("Cumulative empirical frequencies"),
                                    dcc.Graph(id="coin-frequency-figure", figure=initial_frequency, config={"displaylogo": False, "responsive": True}, className="barracuda-coin-plot"),
                                ],
                                className="barracuda-coin-chart",
                            ),
                        ],
                        className="barracuda-coin-plot-grid",
                    ),
                    note(
                        "How to read the HDI",
                        "For this one dimensional Beta posterior, the selected HDI is the shortest interval containing the chosen percentage of posterior probability. It describes parameter uncertainty, not the long run frequency of future intervals.",
                        tone="teal",
                    ),
                    html.P(
                        [
                            "As the number of tosses grows, the empirical frequencies usually settle near the true probabilities and the posterior becomes more concentrated. Try the same update visually in ",
                            _external_link("Seeing Theory’s Bayesian inference lesson", SEEING_THEORY_URL),
                            ".",
                        ],
                        className="barracuda-copy",
                    ),
                    _source_note(
                        "Sources: Miller and Miller, Ch. 5 §4 and Ch. 10 §9; Gelman et al., ",
                        _external_link("Bayesian Data Analysis, 3rd ed.", BDA3_URL),
                        ", Ch. 2 §§2.1–2.3.",
                    ),
                ],
                id="coin-experiment",
                className="barracuda-lesson-section barracuda-interactive-section barracuda-bayes-lab",
            ),
            html.Section(
                [
                    html.Span("03 · Computation", className="barracuda-section-label"),
                    html.H2("The model defines the posterior; computation finds it"),
                    html.P(
                        "The coin example has a closed form posterior because its prior and likelihood are conjugate. Bayes’ theorem still defines a posterior when no familiar formula is available, but we then need a numerical way to explore it.",
                        className="barracuda-section-lead",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("Observed data", className="barracuda-mini-label"),
                                    html.Strong("y = 4.8, 4.9, 5.0, 5.1, 5.3"),
                                ],
                                className="barracuda-example-data",
                            ),
                            html.Div(
                                [
                                    html.P("Assume five measurements come from a Normal population with an unknown mean μ and unknown standard deviation σ."),
                                    markdown(r"$$y_i\mid\mu,\sigma\sim\operatorname{Normal}(\mu,\sigma^2),\qquad \sigma>0$$", class_name="barracuda-equation small", mathjax=True),
                                ],
                                className="barracuda-example-model",
                            ),
                        ],
                        className="barracuda-two-parameter-example",
                    ),
                    html.Div(
                        [
                            html.Div([html.Strong("μ · mean"), html.P("Moves the centre of the population left or right.")]),
                            html.Div([html.Strong("σ · standard deviation"), html.P("Controls how tightly the measurements cluster around μ.")]),
                        ],
                        className="barracuda-parameter-grid",
                    ),
                    html.H3("Four views of the same parameter space"),
                    html.P(
                        "Every point in the maps below is one candidate pair (μ, σ). The axes stay fixed so you can see exactly what the data, the prior and Bayes’ theorem each contribute.",
                        className="barracuda-copy",
                    ),
                    html.Div(
                        [
                            html.Article(
                                [
                                    html.Span("1", className="barracuda-surface-step"),
                                    html.H3("Likelihood: score each pair using the data"),
                                    html.P("Hold the observations fixed. For each candidate (μ, σ), ask how compatible those observations would be with that pair."),
                                    markdown(r"$$L(\mu,\sigma;y)=\prod_i p(y_i\mid\mu,\sigma)$$", class_name="barracuda-equation small", mathjax=True),
                                    dcc.Graph(id="likelihood-surface", figure=_parameter_surface_figure("likelihood", "Likelihood"), config={"displaylogo": False, "responsive": True}, className="barracuda-surface-plot", style={"height": "300px"}),
                                    html.P("The likelihood uses the sampling model and these data. It does not use the prior.", className="barracuda-help"),
                                ],
                                className="barracuda-surface-card",
                            ),
                            html.Article(
                                [
                                    html.Span("2", className="barracuda-surface-step"),
                                    html.H3("Prior: describe uncertainty before these data"),
                                    html.P("The prior is a probability density over parameter pairs before the five measurements are used."),
                                    markdown(r"$$\mu\sim\operatorname{Normal}(5.45,0.15^2),\qquad \sigma\sim\operatorname{HalfNormal}(0.45)$$", class_name="barracuda-equation small", mathjax=True),
                                    dcc.Graph(id="prior-surface", figure=_parameter_surface_figure("prior", "Prior density"), config={"displaylogo": False, "responsive": True}, className="barracuda-surface-plot", style={"height": "300px"}),
                                    html.P("The prior can favour some pairs even before the likelihood is applied.", className="barracuda-help"),
                                ],
                                className="barracuda-surface-card",
                            ),
                            html.Article(
                                [
                                    html.Span("3", className="barracuda-surface-step"),
                                    html.H3("Multiply: obtain the posterior shape"),
                                    html.P("Multiply the likelihood and prior at every pair. A pair remains prominent only when both sources support it."),
                                    markdown(r"$$q(\mu,\sigma)=L(\mu,\sigma;y)\,p(\mu,\sigma)$$", class_name="barracuda-equation small", mathjax=True),
                                    dcc.Graph(id="unnormalised-posterior-surface", figure=_parameter_surface_figure("unnormalised", "Unnormalised density q"), config={"displaylogo": False, "responsive": True}, className="barracuda-surface-plot", style={"height": "300px"}),
                                    html.P("q has the correct posterior shape, but its total area is not generally one.", className="barracuda-help"),
                                ],
                                className="barracuda-surface-card",
                            ),
                            html.Article(
                                [
                                    html.Span("4", className="barracuda-surface-step"),
                                    html.H3("Normalise: obtain the posterior distribution"),
                                    html.P("Integrate q to obtain the evidence Z, then divide by Z so the posterior density has total area one."),
                                    markdown(r"$$Z=\iint q(\mu,\sigma)\,d\mu\,d\sigma,\qquad p(\mu,\sigma\mid y)=\frac{q(\mu,\sigma)}{Z}$$", class_name="barracuda-equation small", mathjax=True),
                                    dcc.Graph(id="posterior-surface", figure=_parameter_surface_figure("posterior", "Posterior density"), config={"displaylogo": False, "responsive": True}, className="barracuda-surface-plot", style={"height": "300px"}),
                                    html.P(f"On this finite grid, Z ≈ {evidence:.3g}. Normalisation changes the density scale, not the contour shape.", className="barracuda-help"),
                                ],
                                className="barracuda-surface-card",
                            ),
                        ],
                        className="barracuda-surface-grid",
                    ),
                    html.P("Darker regions have higher relative values within each panel. Hover over a map to inspect its numerical value.", className="barracuda-surface-legend"),
                    note(
                        "Likelihood is not posterior probability",
                        "Once the observations are fixed, the likelihood ranks candidate parameter pairs. It does not have to integrate to one over μ and σ. The prior and posterior are probability densities over those parameters; the posterior combines both the likelihood and the prior.",
                        tone="teal",
                    ),
                    html.H3("Why a numerical method becomes necessary"),
                    html.P(
                        "With two parameters, a grid can approximate the normalising integral. A grid becomes impossible surprisingly quickly: checking K values for each of d parameters requires Kᵈ evaluations. Hierarchical models may also contain latent variables and parameter dependencies that prevent a closed form calculation.",
                        className="barracuda-copy",
                    ),
                    html.Div(
                        [
                            html.Div([html.Strong("2 parameters"), html.Span("100² = 10,000 grid points")]),
                            html.Div([html.Strong("10 parameters"), html.Span("100¹⁰ = 10²⁰ grid points")]),
                        ],
                        className="barracuda-dimension-contrast",
                    ),
                    note(
                        "The posterior stays fixed",
                        "MCMC iterations and SMC temperature steps do not perform new Bayesian updates with new data. The likelihood and prior already define one posterior. The algorithms provide different numerical routes to that same target.",
                        tone="navy",
                    ),
                    html.Div(
                        [
                            html.Article(
                                [
                                    html.Span("MCMC", className="barracuda-sampler-tag"),
                                    html.H3("Explore one parameter pair at a time"),
                                    html.P(
                                        "MCMC moves a chain across the fixed posterior surface. Each computational update changes the chain’s current pair, not the posterior itself.",
                                    ),
                                    html.Ol(
                                        [
                                            html.Li("Propose a new pair (μ′, σ′)."),
                                            html.Li("Evaluate its likelihood using the same five observations."),
                                            html.Li("Multiply by its prior density and compare the two q scores."),
                                            html.Li("Accept or reject, then repeat. Retained pairs form a posterior sample."),
                                        ],
                                        className="barracuda-sampler-steps",
                                    ),
                                    markdown(r"$$a=\min\left(1,\frac{q(\mu',\sigma')}{q(\mu,\sigma)}\right)$$", class_name="barracuda-equation small", mathjax=True),
                                    dcc.Graph(id="mcmc-animation", figure=_mcmc_figure(), config={"displaylogo": False, "responsive": True}, className="barracuda-sampler-plot", style={"height": "430px"}),
                                    html.P("The unknown Z cancels in this ratio, so MCMC only needs the unnormalised posterior q.", className="barracuda-help"),
                                    html.P(
                                        [
                                            "Explore several algorithms in the ",
                                            _external_link("interactive MCMC gallery", MCMC_GALLERY_URL),
                                            ".",
                                        ],
                                        className="barracuda-help",
                                    ),
                                ],
                                className="barracuda-sampler-card",
                            ),
                            html.Article(
                                [
                                    html.Span("SMC", className="barracuda-sampler-tag"),
                                    html.H3("Turn on the likelihood for many pairs"),
                                    html.P(
                                        "SMC begins with a population drawn from the prior. It gradually increases the influence of the same likelihood until the particles represent the posterior.",
                                    ),
                                    markdown(r"$$\pi_\beta(\theta)\propto p(y\mid\theta)^\beta p(\theta),\qquad 0\leq\beta\leq1$$", class_name="barracuda-equation small", mathjax=True),
                                    html.Ol(
                                        [
                                            html.Li("At β = 0, particles represent the prior."),
                                            html.Li("Increase β and reweight particles by their likelihood."),
                                            html.Li("Resample well weighted particles and move them to restore diversity."),
                                            html.Li("At β = 1, the population represents the posterior."),
                                        ],
                                        className="barracuda-sampler-steps",
                                    ),
                                    dcc.Graph(id="smc-animation", figure=_smc_figure(), config={"displaylogo": False, "responsive": True}, className="barracuda-sampler-plot", style={"height": "430px"}),
                                    html.P("The temperature stages are computational bridges, not additional observations.", className="barracuda-help"),
                                    html.P(
                                        ["Read the implementation details in the ", _external_link("official PyMC SMC documentation", PYMC_SMC_URL), "."],
                                        className="barracuda-help",
                                    ),
                                ],
                                className="barracuda-sampler-card",
                            ),
                        ],
                        className="barracuda-sampler-grid",
                    ),
                    note(
                        "What more computation can and cannot do",
                        "More draws or particles can reduce Monte Carlo error. They cannot add information that is absent from the data, identify an unidentified parameter or repair a poor model.",
                        tone="amber",
                    ),
                    _source_note(
                        "Sources: Gelman et al., ",
                        _external_link("Bayesian Data Analysis, 3rd ed.", BDA3_URL),
                        ", Ch. 11 for MCMC; ",
                        _external_link("PyMC Sequential Monte Carlo", PYMC_SMC_URL),
                        " for tempered SMC.",
                    ),
                ],
                id="computation",
                className="barracuda-lesson-section",
            ),
            html.Section(
                [
                    html.Span("04 · Model comparison", className="barracuda-section-label"),
                    html.H2("Bayes factors from SMC"),
                    html.P(
                        "A marginal likelihood measures how well a model predicted the observed data on average over the parameter values allowed by its prior.",
                        className="barracuda-section-lead",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Evidence for one model"),
                                    markdown(r"$$Z_M=p(y\mid M)=\int p(y\mid\theta,M)p(\theta\mid M)\,d\theta$$", class_name="barracuda-equation small", mathjax=True),
                                ],
                                className="barracuda-concept-panel",
                            ),
                            html.Div(
                                [
                                    html.H3("Compare two models"),
                                    markdown(r"$$BF_{12}=\frac{Z_1}{Z_2},\qquad \log BF_{12}=\log Z_1-\log Z_2$$", class_name="barracuda-equation small", mathjax=True),
                                ],
                                className="barracuda-concept-panel",
                            ),
                        ],
                        className="barracuda-card-grid two",
                    ),
                    html.H3("How SMC estimates the evidence"),
                    html.P(
                        "Each change in temperature contributes an incremental likelihood weight. Adding those contributions on the log scale gives an estimate of the log marginal likelihood.",
                        className="barracuda-copy",
                    ),
                    markdown(
                        r"$$\log \widehat Z_M=\sum_t\log\left[\frac{1}{N}\sum_{i=1}^{N}p(y\mid\theta_i,M)^{\,\beta_t-\beta_{t-1}}\right]$$",
                        class_name="barracuda-equation",
                        mathjax=True,
                    ),
                    html.Div(
                        [
                            html.Div([html.Span("1"), html.Strong("Temper"), html.P("Increase β from the prior towards the posterior.")]),
                            html.Div([html.Span("2"), html.Strong("Reweight"), html.P("Score particles by the new likelihood increment.")]),
                            html.Div([html.Span("3"), html.Strong("Accumulate"), html.P("Add the normalising contributions to log Z.")]),
                            html.Div([html.Span("4"), html.Strong("Compare"), html.P("Subtract log evidences to obtain the log Bayes factor.")]),
                        ],
                        className="barracuda-bf-flow",
                    ),
                    html.Div(
                        [
                            html.Table(
                                [
                                    html.Thead(html.Tr([html.Th("Bayes factor BF₁₂"), html.Th("Likelihood of the data under M₁ relative to M₂")])),
                                    html.Tbody(
                                        [
                                            html.Tr([html.Td("100"), html.Td("100 to 1")]),
                                            html.Tr([html.Td("10"), html.Td("10 to 1")]),
                                            html.Tr([html.Td("1"), html.Td("Equal")]),
                                            html.Tr([html.Td("0.1"), html.Td("1 to 10")]),
                                            html.Tr([html.Td("0.01"), html.Td("1 to 100")]),
                                        ]
                                    ),
                                ],
                                className="barracuda-simple-table",
                            )
                        ],
                        className="barracuda-simple-table-wrap",
                    ),
                    note(
                        "Interpret with the priors in view",
                        "A Bayes factor compares only the models and priors that were specified. Marginal likelihoods can be sensitive to prior choices, so model assumptions and predictive checks remain essential.",
                        tone="amber",
                    ),
                    html.P(
                        [
                            "PyMC records an estimated log marginal likelihood as an SMC byproduct. See the official ",
                            _external_link("Bayes factor and marginal likelihood example", PYMC_BF_URL),
                            " for a worked calculation and practical cautions.",
                        ],
                        className="barracuda-copy",
                    ),
                    _source_note(
                        "Sources: Gelman et al., ",
                        _external_link("Bayesian Data Analysis, 3rd ed.", BDA3_URL),
                        ", Ch. 7 §7.4; ",
                        _external_link("PyMC’s SMC API", PYMC_SMC_URL),
                        " and ",
                        _external_link("Bayes factor example", PYMC_BF_URL),
                        ".",
                    ),
                ],
                id="bayes-factors",
                className="barracuda-lesson-section",
            ),
            html.Section(
                [
                    html.Span("05 · Historical context", className="barracuda-section-label"),
                    html.H2("Thomas Bayes and the theorem that bears his name"),
                    html.Div(
                        [
                            html.Figure(
                                [
                                    html.Img(
                                        src="/assets/thomas_bayes.png",
                                        alt="Engraving commonly attributed to Thomas Bayes",
                                    ),
                                    html.Figcaption(
                                        [
                                            "Commonly attributed to Thomas Bayes; the identity of the sitter is uncertain. Public domain image via ",
                                            _external_link("Wikimedia Commons", THOMAS_BAYES_PORTRAIT_URL),
                                            ".",
                                        ]
                                    ),
                                ],
                                className="barracuda-bayes-portrait",
                            ),
                            html.Div(
                                [
                                    html.P(
                                        "Thomas Bayes (c. 1701–1761) was an English nonconformist minister and a Fellow of the Royal Society. His work on inverse probability was unfinished when he died. Richard Price edited and presented it, and the paper appeared in 1763.",
                                        className="barracuda-bayes-biography",
                                    ),
                                    html.P(
                                        "The familiar theorem is now written in a compact form that Bayes himself did not use. Its lasting idea is to reverse a probability statement by combining the observed evidence with what was plausible beforehand.",
                                        className="barracuda-copy",
                                    ),
                                    html.Div(
                                        [
                                            _external_link("Read the Thomas Bayes biography", THOMAS_BAYES_URL, class_name="barracuda-button secondary"),
                                            _external_link("Read the 1763 paper", THOMAS_BAYES_PAPER_URL, class_name="barracuda-button secondary"),
                                        ],
                                        className="barracuda-bayes-actions",
                                    ),
                                ]
                            ),
                        ],
                        className="barracuda-bayes-history-grid",
                    ),
                    html.Div(
                        [
                            html.Strong("References used for this lesson"),
                            html.Ul(
                                [
                                    html.Li("Miller, I. and Miller, M. John E. Freund’s Mathematical Statistics with Applications, 8th ed., Pearson."),
                                    html.Li(["Gelman, A. et al. ", _external_link("Bayesian Data Analysis, 3rd ed.", BDA3_URL), ". CRC Press."]),
                                    html.Li([_external_link("Seeing Theory: Bayesian inference", SEEING_THEORY_URL), ". Brown University."]),
                                    html.Li([_external_link("PyMC: Sequential Monte Carlo", PYMC_SMC_URL), " and ", _external_link("Bayes factors and marginal likelihood", PYMC_BF_URL), "."]),
                                ]
                            ),
                        ],
                        className="barracuda-reference-box",
                    ),
                ],
                id="thomas-bayes",
                className="barracuda-lesson-section",
            ),
        ]
    )


def register_callbacks(app) -> None:
    @app.callback(
        Output("coin-figure", "figure"),
        Output("coin-frequency-figure", "figure"),
        Output("coin-metrics", "children"),
        Output("coin-ground-truth", "children"),
        Output("coin-toss-scene", "className"),
        Output("coin-visual-face", "children"),
        Output("coin-face", "children"),
        Output("coin-outcomes", "children"),
        Input("coin-probability", "value"),
        Input("coin-tosses", "value"),
        Input("coin-hdi-percent", "value"),
        Input("coin-toss-again", "n_clicks"),
    )
    def update_coin(probability: float, tosses: int, hdi_percent: int, clicks: int):
        probability = float(probability)
        tosses = int(tosses)
        hdi_percent = int(hdi_percent)
        toss_round = int(clicks or 0)
        outcomes = simulate_coin_tosses(probability, tosses, seed=2026 + toss_round)
        figure, values = _coin_figure(
            probability,
            tosses,
            toss_round,
            hdi_percent,
            outcomes=outcomes,
        )
        frequency_figure = _coin_frequency_figure(probability, outcomes)
        face = "Heads" if outcomes[-1] else "Tails"
        animation_class = f"barracuda-coin-stage is-tossing toss-{'a' if toss_round % 2 == 0 else 'b'}"
        ground_truth = f"Ground truth: P(head) = {probability:.2f} · P(tail) = {1 - probability:.2f}"
        return (
            figure,
            frequency_figure,
            metrics(values),
            ground_truth,
            animation_class,
            "H" if face == "Heads" else "T",
            face,
            _recent_outcomes(outcomes),
        )
