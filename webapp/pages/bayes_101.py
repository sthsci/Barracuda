"""Interactive introduction to Bayesian inference."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from plotly.subplots import make_subplots
from scipy.stats import beta as beta_distribution

from webapp.core.coin import (
    beta_highest_density_interval,
    simulate_coin_tosses,
    uniform_prior_posterior,
)
from webapp.palette import CONDITION_BISPECIFIC, DONOR_RUST, DONOR_TEAL, MODEL_ZERO_INFLATED_GAMMA, PAPER_SPINE
from webapp.ui import markdown, metrics, note, page_header, step_card


PATH = "/bayesian-101"
TITLE = "Bayesian inference 101"

BDA3_URL = "https://sites.stat.columbia.edu/gelman/book/BDA3.pdf"
SEEING_THEORY_URL = "https://seeing-theory.brown.edu/bayesian-inference/index.html"
MCMC_GUIDE_URL = "https://mc-stan.org/docs/cmdstan-guide/mcmc_config.html"
PYMC_SMC_URL = "https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.smc.sample_smc.html"
PYMC_BF_URL = "https://www.pymc.io/projects/examples/en/latest/diagnostics_and_criticism/Bayes_factor.html"
THOMAS_BAYES_URL = "https://en.wikipedia.org/wiki/Thomas_Bayes"
THOMAS_BAYES_PAPER_URL = "https://doi.org/10.1098/rstl.1763.0053"
THOMAS_BAYES_PORTRAIT_URL = "https://commons.wikimedia.org/wiki/File:Thomas_Bayes.gif"

BOOK_INK = "#17272C"
BOOK_PAPER = "#F3ECDF"
BOOK_SHEET = "#FFFDF8"
BOOK_RULE = "#7E9299"
BOOK_GRID = "#D2CEC4"
BOOK_SERIF = "Iowan Old Style, Baskerville, Palatino Linotype, Palatino, Georgia, serif"
BOOK_MONO = "SFMono-Regular, Menlo, Monaco, Consolas, Liberation Mono, monospace"
IMPERIAL_BLUE = "#00548F"
IMPERIAL_SKY = "#549DC5"
OXIDE_RED = "#B44E37"
MCMC_WARMUP = 200
MCMC_DRAWS = 1_000
MCMC_STATES = MCMC_WARMUP + MCMC_DRAWS
MCMC_FRAME_STEP = 10
MCMC_RECENT_PATH = 40

TWO_PARAMETER_DATA = np.array([4.8, 4.9, 5.0, 5.1, 5.3])
MEAN_BOUNDS = (4.35, 5.85)
SCALE_BOUNDS = (0.08, 0.85)
MEAN_PRIOR_LOCATION = 5.45
MEAN_PRIOR_SCALE = 0.15
SCALE_PRIOR_SCALE = 0.45
MEAN_MARGINAL_EDGES = np.linspace(*MEAN_BOUNDS, 25)
SCALE_MARGINAL_EDGES = np.linspace(*SCALE_BOUNDS, 21)
SMC_PARTICLES = 180
SMC_ESS_FRACTION = 0.70
SMC_MOVE_STEPS = 8
SURFACE_COLORS = [[0.0, BOOK_PAPER], [0.35, "#C4D8DC"], [0.72, IMPERIAL_SKY], [1.0, "#003E6B"]]
SAMPLER_PLATE_SHAPES = [
    {"type": "rect", "xref": "paper", "yref": "paper", "x0": -0.01, "x1": 0.772, "y0": 0.758, "y1": 1.01, "fillcolor": BOOK_SHEET, "line": {"color": BOOK_RULE, "width": 1}, "layer": "below"},
    {"type": "rect", "xref": "paper", "yref": "paper", "x0": -0.01, "x1": 0.772, "y0": -0.01, "y1": 0.744, "fillcolor": "#E6EEF1", "line": {"color": BOOK_RULE, "width": 1}, "layer": "below"},
    {"type": "rect", "xref": "paper", "yref": "paper", "x0": 0.787, "x1": 1.01, "y0": -0.01, "y1": 0.744, "fillcolor": BOOK_SHEET, "line": {"color": BOOK_RULE, "width": 1}, "layer": "below"},
]


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


@lru_cache(maxsize=32)
def _tempered_surface(temperature: float) -> np.ndarray:
    _, _, likelihood, prior, _, _, _ = _two_parameter_surfaces()
    log_likelihood = np.log(np.clip(likelihood, 1e-300, None))
    log_prior = np.log(np.clip(prior, 1e-300, None))
    return _relative_density(log_prior + float(temperature) * log_likelihood)


def _relative_histogram(
    values: Sequence[float] | np.ndarray,
    edges: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return fixed-bin frequencies scaled to their largest bin."""

    if len(values) == 0:
        return np.zeros(len(edges) - 1)
    counts, _ = np.histogram(values, bins=edges, weights=weights)
    maximum = float(np.max(counts))
    return counts / maximum if maximum > 0 else np.zeros_like(counts, dtype=float)


@lru_cache(maxsize=32)
def _target_marginals(temperature: float) -> tuple[np.ndarray, np.ndarray]:
    target = _tempered_surface(float(temperature))
    mean_marginal = target.sum(axis=0)
    scale_marginal = target.sum(axis=1)
    return mean_marginal / mean_marginal.max(), scale_marginal / scale_marginal.max()


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
    """Animate warm-up and retained Metropolis draws with fixed marginals."""
    rng = np.random.default_rng(401)
    current_mean, current_scale = 5.66, 0.68
    means = [current_mean]
    scales = [current_scale]
    proposed_means = [current_mean]
    proposed_scales = [current_scale]
    accepted_updates = [True]
    decisions = ["Starting pair"]

    for _ in range(MCMC_STATES - 1):
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

    grid_means, grid_scales, *_ = _two_parameter_surfaces()
    target_mean, target_scale = _target_marginals(1.0)
    mean_centres = (MEAN_MARGINAL_EDGES[:-1] + MEAN_MARGINAL_EDGES[1:]) / 2.0
    scale_centres = (SCALE_MARGINAL_EDGES[:-1] + SCALE_MARGINAL_EDGES[1:]) / 2.0

    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "xy"}, None], [{"type": "xy"}, {"type": "xy"}]],
        row_heights=[0.24, 0.76],
        column_widths=[0.79, 0.21],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.035,
        vertical_spacing=0.035,
    )
    figure.add_trace(go.Bar(x=mean_centres, y=np.zeros_like(mean_centres), width=np.diff(MEAN_MARGINAL_EDGES) * 0.9, name="Retained marginal", marker={"color": IMPERIAL_SKY, "line": {"color": BOOK_SHEET, "width": 0.7}}, opacity=0.72, hovertemplate="Mean μ: %{x:.3f}<br>Relative frequency: %{y:.3f}<extra>Retained marginal</extra>"), row=1, col=1)
    figure.add_trace(go.Scatter(x=grid_means, y=target_mean, mode="lines", name="Grid target marginal", line={"color": IMPERIAL_BLUE, "width": 2}), row=1, col=1)
    figure.add_trace(_posterior_contour_trace(), row=2, col=1)
    figure.add_trace(go.Scatter(x=[], y=[], mode="markers", name="Retained draws", marker={"color": IMPERIAL_BLUE, "size": 4, "opacity": 0.32}), row=2, col=1)
    figure.add_trace(go.Scatter(x=[means[0]], y=[scales[0]], mode="lines+markers", name="Recent chain path", line={"color": IMPERIAL_BLUE, "width": 2.1}, marker={"color": IMPERIAL_BLUE, "size": 4}), row=2, col=1)
    figure.add_trace(go.Scatter(x=[means[0]], y=[scales[0]], mode="markers", name="Current pair", marker={"color": OXIDE_RED, "size": 12, "line": {"color": BOOK_SHEET, "width": 2}}), row=2, col=1)
    figure.add_trace(go.Scatter(x=[proposed_means[0]], y=[proposed_scales[0]], mode="markers", name="Proposal", marker={"color": BOOK_SHEET, "size": 10, "symbol": "diamond", "line": {"color": IMPERIAL_SKY, "width": 2}}), row=2, col=1)
    figure.add_trace(go.Bar(x=np.zeros_like(scale_centres), y=scale_centres, width=np.diff(SCALE_MARGINAL_EDGES) * 0.9, orientation="h", name="Retained marginal", marker={"color": IMPERIAL_SKY, "line": {"color": BOOK_SHEET, "width": 0.7}}, opacity=0.72, showlegend=False, hovertemplate="SD σ: %{y:.3f}<br>Relative frequency: %{x:.3f}<extra>Retained marginal</extra>"), row=2, col=2)
    figure.add_trace(go.Scatter(x=target_scale, y=grid_scales, mode="lines", name="Grid target marginal", line={"color": IMPERIAL_BLUE, "width": 2}, showlegend=False), row=2, col=2)
    figure.add_trace(go.Scatter(x=[MEAN_BOUNDS[0] + 0.04], y=[SCALE_BOUNDS[1] - 0.04], mode="text", text=[f"Warm-up 1/{MCMC_WARMUP:,} · starting pair"], textposition="middle right", textfont={"family": BOOK_SERIF, "size": 14, "color": BOOK_INK}, showlegend=False, hoverinfo="skip"), row=2, col=1)
    frame_indices = [0, *range(MCMC_FRAME_STEP - 1, len(means), MCMC_FRAME_STEP)]
    figure.frames = [
        go.Frame(
            name=f"mcmc-{index}",
            traces=[0, 3, 4, 5, 6, 7, 9],
            data=[
                go.Bar(x=mean_centres, y=_relative_histogram(retained_means, MEAN_MARGINAL_EDGES), width=np.diff(MEAN_MARGINAL_EDGES) * 0.9),
                go.Scatter(x=retained_means, y=retained_scales, mode="markers", marker={"color": IMPERIAL_BLUE, "size": 4, "opacity": 0.32}),
                go.Scatter(x=means[path_start : index + 1], y=scales[path_start : index + 1], mode="lines+markers", line={"color": IMPERIAL_BLUE, "width": 2.1}, marker={"color": IMPERIAL_BLUE, "size": 4}),
                go.Scatter(x=[mean], y=[scale], mode="markers", marker={"color": OXIDE_RED, "size": 12, "line": {"color": BOOK_SHEET, "width": 2}}),
                go.Scatter(
                    x=[proposed_means[index]],
                    y=[proposed_scales[index]],
                    mode="markers",
                    marker={
                        "color": BOOK_SHEET if accepted_updates[index] else OXIDE_RED,
                        "size": 10,
                        "symbol": "diamond" if accepted_updates[index] else "x-open",
                        "line": {"color": IMPERIAL_SKY if accepted_updates[index] else OXIDE_RED, "width": 2},
                    },
                ),
                go.Bar(x=_relative_histogram(retained_scales, SCALE_MARGINAL_EDGES), y=scale_centres, width=np.diff(SCALE_MARGINAL_EDGES) * 0.9, orientation="h"),
                go.Scatter(
                    x=[MEAN_BOUNDS[0] + 0.04],
                    y=[SCALE_BOUNDS[1] - 0.04],
                    mode="text",
                    text=[status],
                    textposition="middle right",
                    textfont={"family": BOOK_SERIF, "size": 14, "color": BOOK_INK},
                    showlegend=False,
                    hoverinfo="skip",
                ),
            ],
        )
        for index in frame_indices
        for mean, scale in [(means[index], scales[index])]
        for retained_means, retained_scales in [
            (
                means[MCMC_WARMUP : index + 1] if index >= MCMC_WARMUP else [],
                scales[MCMC_WARMUP : index + 1] if index >= MCMC_WARMUP else [],
            )
        ]
        for path_start in [max(0, index - MCMC_RECENT_PATH + 1)]
        for retained_count in [len(retained_means)]
        for status in [
            (
                f"Warm-up {index + 1:,}/{MCMC_WARMUP:,} · {decisions[index].lower()}"
                if index < MCMC_WARMUP
                else f"Retained {retained_count:,}/{MCMC_DRAWS:,} · {decisions[index].lower()}"
            )
        ]
    ]
    figure.update_layout(
        **_plot_layout(height=590, bottom_margin=96, top_margin=34, left_margin=64, right_margin=24),
        barmode="overlay",
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0, "font": {"size": 11}},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0,
                "y": -0.11,
                "xanchor": "left",
                "yanchor": "top",
                "showactive": False,
                "bgcolor": BOOK_INK,
                "bordercolor": BOOK_INK,
                "font": {"family": BOOK_MONO, "color": BOOK_PAPER, "size": 12},
                "buttons": [
                    {"label": f"Run {MCMC_STATES:,} chain states", "method": "animate", "args": [None, {"frame": {"duration": 85, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}]},
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.26,
                "y": -0.08,
                "len": 0.74,
                "pad": {"t": 4},
                "steps": [
                    {
                        "label": f"{index + 1:,}" if index == 0 or (index + 1) % 200 == 0 else "",
                        "method": "animate",
                        "args": [[f"mcmc-{index}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
                    }
                    for index in frame_indices
                ],
            }
        ],
    )
    figure.update_layout(plot_bgcolor="rgba(0,0,0,0)", shapes=SAMPLER_PLATE_SHAPES)
    figure.update_xaxes(range=list(MEAN_BOUNDS), showgrid=False, showticklabels=False, row=1, col=1)
    figure.update_yaxes(title="Relative marginal of μ", range=[0, 1.05], showgrid=False, showticklabels=False, row=1, col=1)
    figure.update_xaxes(title="Mean μ", range=list(MEAN_BOUNDS), gridcolor=BOOK_GRID, linecolor=BOOK_RULE, row=2, col=1)
    figure.update_yaxes(title="SD σ", range=list(SCALE_BOUNDS), gridcolor=BOOK_GRID, linecolor=BOOK_RULE, row=2, col=1)
    figure.update_xaxes(title="Relative marginal of σ", range=[0, 1.05], showgrid=False, showticklabels=False, row=2, col=2)
    figure.update_yaxes(range=list(SCALE_BOUNDS), showgrid=False, showticklabels=False, row=2, col=2)
    return figure


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


def _importance_weights(log_likelihoods: np.ndarray, temperature_step: float) -> np.ndarray:
    log_weights = temperature_step * log_likelihoods
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    return weights / weights.sum()


def _next_smc_temperature(current: float, log_likelihoods: np.ndarray) -> float:
    """Choose the next temperature so incremental-weight ESS stays usable."""
    target_ess = SMC_ESS_FRACTION * len(log_likelihoods)
    full_step_weights = _importance_weights(log_likelihoods, 1.0 - current)
    if 1.0 / np.sum(full_step_weights**2) >= target_ess:
        return 1.0

    lower, upper = current, 1.0
    for _ in range(40):
        candidate = (lower + upper) / 2.0
        weights = _importance_weights(log_likelihoods, candidate - current)
        if 1.0 / np.sum(weights**2) < target_ess:
            upper = candidate
        else:
            lower = candidate
    return lower


@lru_cache(maxsize=1)
def _smc_particle_states() -> tuple[np.ndarray, list[tuple[float, str, np.ndarray, np.ndarray, np.ndarray]]]:
    """Generate deterministic adaptive-tempering SMC states."""
    rng = np.random.default_rng(902)

    def draw_within_bounds(draw, bounds: tuple[float, float]) -> np.ndarray:
        accepted: list[float] = []
        while len(accepted) < SMC_PARTICLES:
            candidates = np.asarray(draw(SMC_PARTICLES * 2))
            accepted.extend(candidates[(candidates >= bounds[0]) & (candidates <= bounds[1])].tolist())
        return np.asarray(accepted[:SMC_PARTICLES])

    means = draw_within_bounds(lambda size: rng.normal(MEAN_PRIOR_LOCATION, MEAN_PRIOR_SCALE, size), MEAN_BOUNDS)
    scales = draw_within_bounds(lambda size: np.abs(rng.normal(0.0, SCALE_PRIOR_SCALE, size)), SCALE_BOUNDS)
    uniform_weights = np.full(SMC_PARTICLES, 1.0 / SMC_PARTICLES)
    states = [(0.0, "prior", means.copy(), scales.copy(), uniform_weights.copy())]
    temperatures = [0.0]

    while temperatures[-1] < 1.0:
        previous_temperature = temperatures[-1]
        log_likelihoods = np.array([_parameter_log_likelihood(mean, scale) for mean, scale in zip(means, scales, strict=True)])
        temperature = _next_smc_temperature(previous_temperature, log_likelihoods)
        weights = _importance_weights(log_likelihoods, temperature - previous_temperature)
        temperatures.append(temperature)
        states.append((temperature, "reweight", means.copy(), scales.copy(), weights.copy()))

        particles = np.column_stack((means, scales))
        proposal_covariance = np.cov(particles, rowvar=False, aweights=weights, ddof=0)
        proposal_covariance = proposal_covariance * (2.38**2 / 2.0) + np.diag([1e-5, 1e-5])
        ancestors = _systematic_resample(weights, rng)
        means = means[ancestors]
        scales = scales[ancestors]
        states.append((temperature, "resample", means.copy(), scales.copy(), uniform_weights.copy()))

        for _ in range(SMC_MOVE_STEPS):
            proposals = np.column_stack((means, scales)) + rng.multivariate_normal([0.0, 0.0], proposal_covariance, SMC_PARTICLES)
            for particle, (proposed_mean, proposed_scale) in enumerate(proposals):
                current_target = _parameter_log_prior(means[particle], scales[particle]) + temperature * _parameter_log_likelihood(means[particle], scales[particle])
                proposed_target = _parameter_log_prior(proposed_mean, proposed_scale) + temperature * _parameter_log_likelihood(proposed_mean, proposed_scale)
                if np.log(rng.random()) < min(0.0, proposed_target - current_target):
                    means[particle] = proposed_mean
                    scales[particle] = proposed_scale
        states.append((temperature, "move", means.copy(), scales.copy(), uniform_weights.copy()))
    return np.asarray(temperatures), states


@lru_cache(maxsize=1)
def _smc_figure() -> go.Figure:
    """Animate adaptive tempering, resampling, and mutation."""
    _, states = _smc_particle_states()
    initial_temperature, initial_phase, initial_means, initial_scales, initial_weights = states[0]
    grid_means, grid_scales, *_ = _two_parameter_surfaces()
    mean_centres = (MEAN_MARGINAL_EDGES[:-1] + MEAN_MARGINAL_EDGES[1:]) / 2.0
    scale_centres = (SCALE_MARGINAL_EDGES[:-1] + SCALE_MARGINAL_EDGES[1:]) / 2.0

    def contour(temperature: float) -> go.Contour:
        return go.Contour(
            x=grid_means,
            y=grid_scales,
            z=_tempered_surface(temperature),
            zmin=0,
            zmax=1,
            colorscale=SURFACE_COLORS,
            contours={"coloring": "heatmap", "showlines": False},
            showscale=False,
            hoverinfo="skip",
            name="Tempered grid target",
        )

    def particle_marker(phase: str, weights: np.ndarray) -> dict:
        marker = {"size": 6, "opacity": 0.78, "line": {"color": BOOK_SHEET, "width": 0.8}}
        if phase == "reweight":
            relative_weights = weights / weights.max()
            marker.update(
                color=relative_weights,
                size=4 + 8 * np.sqrt(relative_weights),
                colorscale=[[0.0, IMPERIAL_SKY], [1.0, OXIDE_RED]],
                cmin=0,
                cmax=1,
                showscale=False,
            )
        else:
            marker["color"] = IMPERIAL_SKY if phase == "resample" else OXIDE_RED
        return marker

    def status_text(temperature: float, phase: str, weights: np.ndarray) -> str:
        if phase == "prior":
            return f"β = 0.00 · {SMC_PARTICLES} particles drawn from the prior"
        if phase == "reweight":
            effective_particles = 1.0 / float(np.sum(weights**2))
            return f"β = {temperature:.3f} · 1/3 reweight · ESS {effective_particles:.0f}/{SMC_PARTICLES}"
        if phase == "resample":
            return f"β = {temperature:.3f} · 2/3 resample · equal weights restored"
        if temperature == 1.0:
            return "β = 1.000 · finite particle approximation to the posterior"
        return f"β = {temperature:.3f} · 3/3 mutate · mutation step complete"

    def marginal_traces(temperature: float, means: np.ndarray, scales: np.ndarray, weights: np.ndarray) -> tuple[go.Bar, go.Scatter, go.Bar, go.Scatter]:
        target_mean, target_scale = _target_marginals(temperature)
        return (
            go.Bar(x=mean_centres, y=_relative_histogram(means, MEAN_MARGINAL_EDGES, weights=weights), width=np.diff(MEAN_MARGINAL_EDGES) * 0.9),
            go.Scatter(x=grid_means, y=target_mean, mode="lines", line={"color": IMPERIAL_BLUE, "width": 2}),
            go.Bar(x=_relative_histogram(scales, SCALE_MARGINAL_EDGES, weights=weights), y=scale_centres, width=np.diff(SCALE_MARGINAL_EDGES) * 0.9, orientation="h"),
            go.Scatter(x=target_scale, y=grid_scales, mode="lines", line={"color": IMPERIAL_BLUE, "width": 2}),
        )

    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "xy"}, None], [{"type": "xy"}, {"type": "xy"}]],
        row_heights=[0.24, 0.76],
        column_widths=[0.79, 0.21],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.035,
        vertical_spacing=0.035,
    )
    initial_mean_bar, initial_mean_target, initial_scale_bar, initial_scale_target = marginal_traces(initial_temperature, initial_means, initial_scales, initial_weights)
    initial_mean_bar.update(name="Particle marginal", marker={"color": IMPERIAL_SKY, "line": {"color": BOOK_SHEET, "width": 0.7}}, opacity=0.72, hovertemplate="Mean μ: %{x:.3f}<br>Relative frequency: %{y:.3f}<extra>Particle marginal</extra>")
    initial_mean_target.update(name="Grid stage target", line={"color": IMPERIAL_BLUE, "width": 2})
    initial_scale_bar.update(marker={"color": IMPERIAL_SKY, "line": {"color": BOOK_SHEET, "width": 0.7}}, opacity=0.72, showlegend=False, hovertemplate="SD σ: %{y:.3f}<br>Relative frequency: %{x:.3f}<extra>Particle marginal</extra>")
    initial_scale_target.update(line={"color": IMPERIAL_BLUE, "width": 2}, showlegend=False)
    figure.add_trace(initial_mean_bar, row=1, col=1)
    figure.add_trace(initial_mean_target, row=1, col=1)
    figure.add_trace(contour(initial_temperature), row=2, col=1)
    figure.add_trace(go.Scatter(x=initial_means, y=initial_scales, mode="markers", name="Particles", marker=particle_marker(initial_phase, initial_weights), hovertemplate="Mean μ: %{x:.3f}<br>SD σ: %{y:.3f}<extra>Particle</extra>"), row=2, col=1)
    figure.add_trace(initial_scale_bar, row=2, col=2)
    figure.add_trace(initial_scale_target, row=2, col=2)
    figure.add_trace(go.Scatter(x=[MEAN_BOUNDS[0] + 0.04], y=[SCALE_BOUNDS[1] - 0.04], mode="text", text=[status_text(initial_temperature, initial_phase, initial_weights)], textposition="middle right", textfont={"family": BOOK_SERIF, "size": 14, "color": BOOK_INK}, showlegend=False, hoverinfo="skip"), row=2, col=1)
    figure.frames = [
        go.Frame(
            name=f"smc-{index}",
            data=[
                mean_bar,
                mean_target,
                contour(temperature),
                go.Scatter(x=means, y=scales, mode="markers", marker=particle_marker(phase, weights)),
                scale_bar,
                scale_target,
                go.Scatter(
                    x=[MEAN_BOUNDS[0] + 0.04],
                    y=[SCALE_BOUNDS[1] - 0.04],
                    mode="text",
                    text=[status_text(temperature, phase, weights)],
                    textposition="middle right",
                    textfont={"family": BOOK_SERIF, "size": 14, "color": BOOK_INK},
                    showlegend=False,
                    hoverinfo="skip",
                ),
            ],
        )
        for index, (temperature, phase, means, scales, weights) in enumerate(states)
        for mean_bar, mean_target, scale_bar, scale_target in [marginal_traces(temperature, means, scales, weights)]
    ]
    figure.update_layout(
        **_plot_layout(height=590, bottom_margin=96, top_margin=34, left_margin=64, right_margin=24),
        barmode="overlay",
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0, "font": {"size": 11}},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0,
                "y": -0.11,
                "xanchor": "left",
                "yanchor": "top",
                "showactive": False,
                "bgcolor": BOOK_INK,
                "bordercolor": BOOK_INK,
                "font": {"family": BOOK_MONO, "color": BOOK_PAPER, "size": 12},
                "buttons": [
                    {"label": "Run SMC stages", "method": "animate", "args": [None, {"frame": {"duration": 340, "redraw": True}, "transition": {"duration": 140}, "fromcurrent": True}]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}]},
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.26,
                "y": -0.08,
                "len": 0.74,
                "pad": {"t": 4},
                "steps": [
                    {
                        "label": f"β {temperature:.2f}" if phase in {"prior", "move"} else "",
                        "method": "animate",
                        "args": [[f"smc-{index}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 80}}],
                    }
                    for index, (temperature, phase, _, _, _) in enumerate(states)
                ],
            }
        ],
    )
    figure.update_layout(plot_bgcolor="rgba(0,0,0,0)", shapes=SAMPLER_PLATE_SHAPES)
    figure.update_xaxes(range=list(MEAN_BOUNDS), showgrid=False, showticklabels=False, row=1, col=1)
    figure.update_yaxes(title="Relative marginal of μ", range=[0, 1.05], showgrid=False, showticklabels=False, row=1, col=1)
    figure.update_xaxes(title="Mean μ", range=list(MEAN_BOUNDS), gridcolor=BOOK_GRID, linecolor=BOOK_RULE, row=2, col=1)
    figure.update_yaxes(title="SD σ", range=list(SCALE_BOUNDS), gridcolor=BOOK_GRID, linecolor=BOOK_RULE, row=2, col=1)
    figure.update_xaxes(title="Relative marginal of σ", range=[0, 1.05], showgrid=False, showticklabels=False, row=2, col=2)
    figure.update_yaxes(range=list(SCALE_BOUNDS), showgrid=False, showticklabels=False, row=2, col=2)
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

    return html.Div(
        [
            page_header(
                "Learn",
                "Bayesian inference for BARRACUDA",
                "Learn how observations update parameter uncertainty and how model evidence compares alternative biological explanations.",
                badge="Interactive lesson · no data upload required",
                crumb="Bayesian inference",
                educational=True,
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
                            html.P("Posterior probability = likelihood × prior probability ÷ evidence.", className="barracuda-equation-caption"),
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
                        "Assume independent tosses with an unknown probability θ of heads. A uniform Beta(1, 1) prior and observations of h heads and t tails yield an exact Beta posterior.",
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
                                            html.Span("Simulation truth", className="barracuda-field-label"),
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
                                    html.Div(id="coin-ground-truth", children="Simulation truth: P(head) = 0.50 · P(tail) = 0.50. This value would be unknown in real inference.", className="barracuda-help"),
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
                                    html.Button("Toss the coin", id="coin-toss-again", n_clicks=0, className="barracuda-button primary full"),
                                ],
                                className="barracuda-control-panel",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span("Release", className="barracuda-coin-station start", **{"aria-hidden": "true"}),
                                            html.Span("Read", className="barracuda-coin-station finish", **{"aria-hidden": "true"}),
                                            html.Span(className="barracuda-coin-track", **{"aria-hidden": "true"}),
                                            html.Span(className="barracuda-coin-shadow", **{"aria-hidden": "true"}),
                                            html.Span("H" if initial_face == "Heads" else "T", id="coin-visual-face", className="barracuda-toss-coin", **{"aria-hidden": "true"}),
                                        ],
                                        id="coin-toss-scene",
                                        className="barracuda-coin-stage",
                                        role="img",
                                        **{"aria-label": "Animated coin toss between release and read stations"},
                                    ),
                                    html.Div([html.Span("Latest toss", className="barracuda-mini-label"), html.Strong(initial_face, id="coin-face")], className="barracuda-coin-result", **{"aria-live": "polite"}),
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
                        "For this one-dimensional Beta posterior, the HDI is the shortest interval containing the selected posterior probability. It describes uncertainty in θ, not the long-run frequency of future intervals.",
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
                        "The coin model has a closed-form posterior because its prior and likelihood are conjugate. For other models, Bayes’ theorem still defines the posterior, but numerical methods are needed to explore it.",
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
                    html.Span("Bayesian updating", className="barracuda-section-label"),
                    html.H3("From likelihood and prior to posterior"),
                    html.P(
                        "Bayesian updating combines what the data support with what was plausible before these data were observed.",
                        className="barracuda-copy",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("01"),
                                    html.Strong("Likelihood"),
                                    html.P("The likelihood asks which pairs of μ and σ are compatible with the five measurements. It uses the data but not the prior."),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Span("02"),
                                    html.Strong("Prior"),
                                    html.P("The prior assigns greater density to values considered plausible before these measurements were observed."),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Span("03"),
                                    html.Strong("Combine"),
                                    html.P("Multiplication retains parameter pairs supported by both the likelihood and the prior."),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Span("04"),
                                    html.Strong("Normalise"),
                                    html.P("Dividing by the integral Z gives a posterior density with total probability one. It changes the scale, not the relative support."),
                                ]
                            ),
                        ],
                        className="barracuda-bf-flow",
                    ),
                    note(
                        "Likelihood is not posterior probability",
                        "With fixed observations, the likelihood ranks candidate parameter pairs; it need not integrate to one over μ and σ. The prior and posterior are densities over these parameters.",
                        tone="teal",
                    ),
                    html.H3("Why a numerical method becomes necessary"),
                    html.P(
                        "A grid can approximate the normalising integral for two parameters. Evaluating K values for each of d parameters requires Kᵈ points, so grids become impractical as dimension increases. Hierarchical models can also include latent variables or dependencies that prevent a closed-form calculation.",
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
                        "MCMC iterations and SMC temperatures do not introduce new data or redefine the posterior. The prior and likelihood already define one target; the algorithms provide different numerical routes to it.",
                        tone="navy",
                    ),
                    html.H3("One target, two sampling routes"),
                    html.P(
                        "Choose a method, press play, or drag the timeline. The large panel shows the joint distribution of μ and σ; the top and right strips show their marginal distributions.",
                        className="barracuda-copy",
                    ),
                    dcc.Tabs(
                        id="sampler-tabs",
                        value="mcmc",
                        className="barracuda-sampler-tabs",
                        content_className="barracuda-sampler-tab-content",
                        children=[
                            dcc.Tab(
                                label="MCMC · one chain",
                                value="mcmc",
                                className="barracuda-sampler-tab",
                                selected_className="barracuda-sampler-tab selected",
                                children=html.Div(
                                    [
                                        html.Aside(
                                            [
                                                html.Span("MCMC", className="barracuda-sampler-tag"),
                                                html.H3("Follow one chain through the posterior"),
                                                html.P("The first 200 chain states are discarded as warm-up. The next 1,000 retained states form the displayed sample."),
                                                html.Ol(
                                                    [
                                                        html.Li("Use a symmetric random walk to propose (μ′, σ′)."),
                                                        html.Li("Compare its unnormalised posterior q with the current pair."),
                                                        html.Li("If accepted, move to the proposal; if rejected, retain the current pair again."),
                                                        html.Li("Add each retained state to the joint cloud and fixed-bin marginal bars."),
                                                    ],
                                                    className="barracuda-sampler-steps",
                                                ),
                                                markdown(r"$$a=\min\left(1,\frac{q(\mu',\sigma')}{q(\mu,\sigma)}\right)$$", class_name="barracuda-equation small", mathjax=True),
                                                html.P(["The blue lines show grid-based reference marginals; the bars show the retained sample. Their agreement is useful visually, but formal assessment requires diagnostics such as effective sample size. See ", _external_link("Stan’s MCMC guidance", MCMC_GUIDE_URL), "."], className="barracuda-help"),
                                            ]
                                        ),
                                        html.Div(
                                            dcc.Graph(
                                                id="mcmc-animation",
                                                figure=_mcmc_figure(),
                                                config={"displaylogo": False, "responsive": True},
                                                className="barracuda-sampler-plot",
                                                style={"height": "590px"},
                                            ),
                                            role="group",
                                            **{"aria-label": "Interactive MCMC animation with a joint posterior sample and aligned marginal distributions for mean and standard deviation"},
                                        ),
                                    ],
                                    className="barracuda-sampler-workbench",
                                ),
                            ),
                            dcc.Tab(
                                label="SMC · particle population",
                                value="smc",
                                className="barracuda-sampler-tab",
                                selected_className="barracuda-sampler-tab selected",
                                children=html.Div(
                                    [
                                        html.Aside(
                                            [
                                                html.Span("SMC", className="barracuda-sampler-tag"),
                                                html.H3("Move a population from prior to posterior"),
                                                html.P("The temperature β controls the contribution of the fixed likelihood. The blue contour and lines show a grid-based target at each stage."),
                                                html.Ol(
                                                    [
                                                        html.Li(f"At β = 0, draw {SMC_PARTICLES} particles from the prior."),
                                                        html.Li(f"Choose the largest next β that keeps effective sample size near {SMC_ESS_FRACTION:.0%}; finish at β = 1 when it remains above that threshold."),
                                                        html.Li("Resample supported particles and reset their weights."),
                                                        html.Li(f"Attempt {SMC_MOVE_STEPS} Metropolis mutation steps per particle at the current tempered target."),
                                                        html.Li("At β = 1, compare the finite particle approximation with the posterior target."),
                                                    ],
                                                    className="barracuda-sampler-steps",
                                                ),
                                                markdown(r"$$\pi_\beta(\theta)\propto p(y\mid\theta)^\beta p(\theta),\qquad 0\leq\beta\leq1$$", class_name="barracuda-equation small", mathjax=True),
                                                html.P(["The bars are weighted during reweighting and use fixed bins throughout. This sequence follows the stages documented for ", _external_link("PyMC Sequential Monte Carlo", PYMC_SMC_URL), "."], className="barracuda-help"),
                                            ]
                                        ),
                                        html.Div(
                                            dcc.Graph(
                                                id="smc-animation",
                                                figure=_smc_figure(),
                                                config={"displaylogo": False, "responsive": True},
                                                className="barracuda-sampler-plot",
                                                style={"height": "590px"},
                                            ),
                                            role="group",
                                            **{"aria-label": "Interactive SMC animation with tempered particles, a joint posterior sample and aligned marginal distributions"},
                                        ),
                                    ],
                                    className="barracuda-sampler-workbench",
                                ),
                            ),
                        ],
                    ),
                    note(
                        "What additional computation changes",
                        "More draws or particles can reduce Monte Carlo error, but cannot add missing information, identify an unidentified parameter, or repair a poor model.",
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
                        "A marginal likelihood averages the likelihood of the observed data over the parameter values allowed by a model's prior.",
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
                        "Each temperature change produces an incremental likelihood weight. Summing the log contributions estimates the log marginal likelihood.",
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
                                        width=304,
                                        height=326,
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
        ],
        className="barracuda-bayes-page",
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
        ground_truth = f"Simulation truth: P(head) = {probability:.2f} · P(tail) = {1 - probability:.2f}. This value would be unknown in real inference."
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
