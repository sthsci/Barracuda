"""Interactive introduction to Bayesian inference."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from scipy.integrate import trapezoid
from scipy.stats import beta as beta_distribution

from webapp.core.coin import (
    beta_highest_density_interval,
    simulate_coin_tosses,
    uniform_prior_posterior,
)
from webapp.palette import CONDITION_BISPECIFIC, DONOR_RUST, DONOR_TEAL, MODEL_ZERO_INFLATED_GAMMA, PAPER_SPINE
from webapp.pages.bayes_resources import learning_resources
from webapp.ui import markdown, metrics, note, page_header, step_card


PATH = "/bayesian-101"
TITLE = "Bayesian inference 101"

BDA3_URL = "https://sites.stat.columbia.edu/gelman/book/BDA3.pdf"
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
MCMC_TEACHING_DRAWS = 12
MCMC_STEP_DURATION_MS = 1_400

TWO_PARAMETER_DATA = np.array([4.8, 4.9, 5.0, 5.1, 5.3])
MEAN_BOUNDS = (4.35, 5.85)
SCALE_BOUNDS = (0.08, 0.85)
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

    x = np.unique(np.r_[np.linspace(0.0, 1.0, 700), interval])
    posterior = beta_distribution.pdf(x, posterior_alpha, posterior_beta)
    density_limit = max(float(np.max(posterior)) * 1.08, 1.15)
    in_interval = (x >= interval[0]) & (x <= interval[1])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x,
            y=np.ones_like(x),
            mode="lines",
            name="Prior: Beta(1, 1)",
            line={"color": PAPER_SPINE, "width": 2},
            hovertemplate="Prior density: 1<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=posterior,
            mode="lines",
            name=f"Posterior: Beta({posterior_alpha}, {posterior_beta})",
            line={"color": DONOR_TEAL, "width": 3},
            hovertemplate="θ: %{x:.3f}<br>Posterior density: %{y:.3f}<extra></extra>",
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
            name=f"Generating θ₀ = {probability_heads:.2f}",
            line={"color": CONDITION_BISPECIFIC, "width": 2, "dash": "dot"},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[observed, observed],
            y=[0.0, density_limit],
            mode="lines",
            name=f"MLE h/n = {observed:.2f}",
            line={"color": MODEL_ZERO_INFLATED_GAMMA, "width": 2, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        **_plot_layout(height=430, bottom_margin=66),
        xaxis={
            "title": "Success probability θ",
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
        ("Observed data", f"{heads} heads · {tails} tails"),
        ("Maximum likelihood · h/n", f"{observed:.3f}"),
        ("Posterior mean · E[θ | y]", f"{posterior_mean:.3f}"),
        (f"Posterior {hdi_percent}% HDI", f"{interval[0]:.3f}–{interval[1]:.3f}"),
    ]
    return figure, values


def _coin_frequency_figure(
    probability_heads: float,
    outcomes: Sequence[int] | np.ndarray,
    interval_percent: int = 95,
) -> go.Figure:
    """Show exact sequential posterior means and pointwise equal-tailed intervals."""
    tosses = np.arange(len(outcomes) + 1)
    heads = np.r_[0, np.cumsum(outcomes)]
    alpha, beta = heads + 1, tosses - heads + 1
    posterior_mean = alpha / (alpha + beta)
    tail = (1 - interval_percent / 100) / 2
    lower = beta_distribution.ppf(tail, alpha, beta)
    upper = beta_distribution.ppf(1 - tail, alpha, beta)

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=tosses, y=lower, mode="lines", line={"width": 0},
        name="Lower credible limit", showlegend=False, hoverinfo="skip",
    ))
    figure.add_trace(go.Scatter(
        x=tosses, y=upper, mode="lines", line={"width": 0},
        fill="tonexty", fillcolor="rgba(0,133,133,0.18)",
        name=f"{interval_percent}% equal-tailed interval", hoverinfo="skip",
    ))
    figure.add_trace(go.Scatter(
        x=tosses, y=posterior_mean,
        customdata=np.column_stack((lower, upper)),
        mode="lines+markers" if len(tosses) <= 40 else "lines",
        name="Posterior mean", line={"color": DONOR_TEAL, "width": 2.5},
        marker={"size": 4},
        hovertemplate="n = %{x}<br>E[θ | y]: %{y:.3f}<br>Credible interval: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=tosses[1:], y=heads[1:] / tosses[1:],
        mode="lines+markers" if len(tosses) <= 40 else "lines",
        name="MLE h/n", line={"color": MODEL_ZERO_INFLATED_GAMMA, "width": 1.5, "dash": "dash"},
        marker={"size": 4},
        hovertemplate="n = %{x}<br>h/n: %{y:.3f}<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=[0, len(outcomes)], y=[probability_heads, probability_heads],
        mode="lines", name=f"Generating θ₀ = {probability_heads:.2f}",
        line={"color": CONDITION_BISPECIFIC, "width": 2, "dash": "dot"},
        hoverinfo="skip",
    ))
    figure.update_layout(
        **_plot_layout(height=430, bottom_margin=66),
        xaxis={
            "title": "Observations included, n (0 = prior)",
            "range": [0, max(2, len(outcomes))],
            "gridcolor": BOOK_GRID, "linecolor": BOOK_RULE,
            "ticks": "outside", "tickcolor": BOOK_RULE,
            "zeroline": False, "automargin": True,
        },
        yaxis={
            "title": "Success probability θ", "range": [0, 1],
            "gridcolor": BOOK_GRID, "linecolor": BOOK_RULE,
            "ticks": "outside", "tickcolor": BOOK_RULE,
            "zeroline": False, "automargin": True,
        },
        legend={
            "orientation": "h", "y": 1.02, "yanchor": "bottom",
            "x": 0, "xanchor": "left", "font": {"size": 11},
        },
        hovermode="x unified",
    )
    return figure


def _recent_outcomes(outcomes: Sequence[int] | np.ndarray) -> list[html.Span]:
    return [
        html.Span(
            str(int(outcome)),
            className=f"barracuda-outcome {'heads' if outcome else 'tails'}",
            title=f"Observation {index + 1}: {'head' if outcome else 'tail'}",
        )
        for index, outcome in enumerate(outcomes[:24])
    ]


def _coin_experiment_summary(outcomes: np.ndarray, probability_heads: float, seed: int) -> list:
    """Keep the observed data, exact update and predictive calculation together."""
    n = len(outcomes)
    heads = int(outcomes.sum())
    alpha, beta = uniform_prior_posterior(heads, n)
    predictive = alpha / (alpha + beta)
    return [
        _coin_toss_scene(int(outcomes[-1]), n, f"{seed}:{probability_heads}:{n}"),
        html.Span("Observed sample → inference → prediction", className="barracuda-section-label"),
        html.H3("One dataset, an exact update"),
        html.P(
            f"n = {n}; h = {heads}; n − h = {n - heads}. The count h is sufficient for θ under the independent, constant-probability model.",
            className="barracuda-bernoulli-copy",
        ),
        html.P(f"First {min(n, 24)} observations · head = 1, tail = 0", className="barracuda-mini-label"),
        html.Div(_recent_outcomes(outcomes), id="coin-outcomes", className="barracuda-outcome-strip"),
        markdown(
            rf"$$p(\theta\mid y)=\frac{{\theta^{{{heads}}}(1-\theta)^{{{n-heads}}}}}{{B({alpha},{beta})}}$$",
            class_name="barracuda-equation small barracuda-bernoulli-equation", mathjax=True,
        ),
        html.P(
            f"Exact posterior: Beta({alpha}, {beta}). B is the beta function, which normalizes the density; no posterior sampling is required.",
            className="barracuda-bernoulli-copy",
        ),
        html.Div([
            html.Span("Posterior predictive · next trial", className="barracuda-mini-label"),
            html.Strong(f"P(next head | y) = {predictive:.3f}", id="coin-predictive"),
            html.P("Averaging θ over its posterior gives (h + 1)/(n + 2). This probability is the posterior mean, even when h/n is 0 or 1."),
        ], className="barracuda-bernoulli-predictive"),
        html.Details([
            html.Summary(f"Reproduce this dataset · seed {seed}"),
            html.P(f"NumPy {np.__version__} Generator(PCG64) generates the Bernoulli observations. The posterior depends only on the observations and prior; θ₀ is used only to generate and assess the simulation."),
            html.Pre(
                f"import numpy as np\ntheta_0 = {probability_heads!r}\n"
                f"rng = np.random.default_rng({seed})\ny = rng.binomial(1, theta_0, size={n})",
            ),
            html.P("Complete observed sequence (in order):"),
            html.Pre(", ".join(str(int(outcome)) for outcome in outcomes), id="coin-observed-sequence"),
        ], className="barracuda-bernoulli-reproducibility"),
    ]


def _coin_toss_scene(outcome: int, n: int, dataset_key: str) -> html.Figure:
    """A keyed scene replays for a new dataset, while interval edits preserve it."""
    side = "heads" if outcome else "tails"
    return html.Figure([
        html.Div([
            html.Span("Toss", className="barracuda-coin-station start"),
            html.Span("Observe", className="barracuda-coin-station finish"),
            html.Div(className="barracuda-coin-ground"),
            html.Div(className="barracuda-coin-shadow"),
            html.Div([
                html.Div([
                    html.Img(src="/assets/coin-head-portrait.jpeg", alt="Heads: the supplied portrait", className="barracuda-coin-portrait"),
                    html.Span("H", className="barracuda-coin-letter"),
                ], className="barracuda-coin-face front"),
                html.Div([
                    html.Img(src="/assets/barracuda-abstract-consistent-posterior-mark.png", alt="Tails: the barracuda", className="barracuda-coin-fish"),
                    html.Span("T", className="barracuda-coin-letter"),
                ], className="barracuda-coin-face back"),
            ], className="barracuda-toss-coin"),
        ], className=f"barracuda-coin-stage lands-{side}", role="img",
            **{"aria-label": f"Coin toss {n} lands on {side}: {'portrait' if outcome else 'barracuda'}.", "data-outcome": outcome}),
        html.Figcaption([
            html.Div([
                html.Span(f"Final observed toss · {n}", className="barracuda-mini-label"),
                html.Strong(f"{side.capitalize()} · {outcome}"),
            ], className="barracuda-coin-result"),
            html.P("Heads = portrait · tails = barracuda. This animation shows the last of the n observations used below."),
        ]),
    ], id="coin-toss-scene", key=dataset_key, className="barracuda-coin-demo")


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
    likelihood = np.exp(log_likelihood)
    prior = np.full_like(likelihood, 1.0 / (np.ptp(MEAN_BOUNDS) * np.ptp(SCALE_BOUNDS)))
    unnormalised_posterior = likelihood * prior
    evidence = float(trapezoid(trapezoid(unnormalised_posterior, means, axis=1), scales))
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
    return -np.log(np.ptp(MEAN_BOUNDS) * np.ptp(SCALE_BOUNDS))


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
    means, scales, *_ = _two_parameter_surfaces()
    mean_marginal = trapezoid(target, scales, axis=0)
    scale_marginal = trapezoid(target, means, axis=1)
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
    """Show individual proposal/decision steps before building fixed-bin marginals."""
    rng = np.random.default_rng(401)
    current_mean, current_scale = 5.66, 0.68
    means = [current_mean]
    scales = [current_scale]
    proposed_means = [current_mean]
    proposed_scales = [current_scale]
    accepted_updates = [True]
    acceptance_probabilities = [1.0]
    uniforms = [0.0]

    for _ in range(MCMC_STATES - 1):
        proposed_mean = current_mean + rng.normal(0.0, 0.12)
        proposed_scale = current_scale + rng.normal(0.0, 0.10)
        current_log_target = _parameter_log_likelihood(current_mean, current_scale) + _parameter_log_prior(current_mean, current_scale)
        proposed_log_target = _parameter_log_likelihood(proposed_mean, proposed_scale) + _parameter_log_prior(proposed_mean, proposed_scale)
        acceptance_probability = float(np.exp(min(0.0, proposed_log_target - current_log_target)))
        uniform = float(rng.random())
        accepted = uniform < acceptance_probability
        proposed_means.append(proposed_mean)
        proposed_scales.append(proposed_scale)
        accepted_updates.append(accepted)
        acceptance_probabilities.append(acceptance_probability)
        uniforms.append(uniform)
        if accepted:
            current_mean, current_scale = proposed_mean, proposed_scale
        means.append(current_mean)
        scales.append(current_scale)

    grid_means, grid_scales, *_ = _two_parameter_surfaces()
    target_mean, target_scale = _target_marginals(1.0)
    mean_centres = (MEAN_MARGINAL_EDGES[:-1] + MEAN_MARGINAL_EDGES[1:]) / 2.0
    scale_centres = (SCALE_MARGINAL_EDGES[:-1] + SCALE_MARGINAL_EDGES[1:]) / 2.0
    initial_index = MCMC_WARMUP - 1

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
    figure.add_trace(go.Bar(x=mean_centres, y=np.zeros_like(mean_centres), width=np.diff(MEAN_MARGINAL_EDGES) * 0.9, name="Retained marginal", marker={"color": IMPERIAL_SKY, "line": {"color": BOOK_SHEET, "width": 0.7}}, opacity=0.72, hovertemplate="Mean μ: %{x:.3f}<br>Fraction of largest bin: %{y:.3f}<extra>Retained marginal</extra>"), row=1, col=1)
    figure.add_trace(go.Scatter(x=grid_means, y=target_mean, mode="lines", name="Grid target marginal", line={"color": IMPERIAL_BLUE, "width": 2}), row=1, col=1)
    figure.add_trace(_posterior_contour_trace(), row=2, col=1)
    figure.add_trace(go.Scatter(x=[], y=[], mode="markers", name="Retained draws", marker={"color": IMPERIAL_BLUE, "size": 4, "opacity": 0.32}), row=2, col=1)
    figure.add_trace(go.Scatter(x=[means[initial_index]], y=[scales[initial_index]], mode="lines+markers", name="Recent chain path", line={"color": IMPERIAL_BLUE, "width": 2.1}, marker={"color": IMPERIAL_BLUE, "size": 4}), row=2, col=1)
    figure.add_trace(go.Scatter(x=[means[initial_index]], y=[scales[initial_index]], mode="markers", name="Current pair", marker={"color": OXIDE_RED, "size": 12, "line": {"color": BOOK_SHEET, "width": 2}}), row=2, col=1)
    figure.add_trace(go.Scatter(x=[], y=[], mode="lines+markers", name="Proposal", line={"color": IMPERIAL_SKY, "width": 2, "dash": "dot"}, marker={"color": BOOK_SHEET, "size": 10, "symbol": "diamond", "line": {"color": IMPERIAL_SKY, "width": 2}}), row=2, col=1)
    figure.add_trace(go.Bar(x=np.zeros_like(scale_centres), y=scale_centres, width=np.diff(SCALE_MARGINAL_EDGES) * 0.9, orientation="h", name="Retained marginal", marker={"color": IMPERIAL_SKY, "line": {"color": BOOK_SHEET, "width": 0.7}}, opacity=0.72, showlegend=False, hovertemplate="SD σ: %{y:.3f}<br>Fraction of largest bin: %{x:.3f}<extra>Retained marginal</extra>"), row=2, col=2)
    figure.add_trace(go.Scatter(x=target_scale, y=grid_scales, mode="lines", name="Grid target marginal", line={"color": IMPERIAL_BLUE, "width": 2}, showlegend=False), row=2, col=2)
    figure.add_trace(go.Scatter(x=[MEAN_BOUNDS[0] + 0.04], y=[SCALE_BOUNDS[1] - 0.075], mode="text", text=[f"{MCMC_WARMUP} warm-up states discarded<br>Watch a proposal, then its decision."], textposition="middle right", textfont={"family": BOOK_SERIF, "size": 13, "color": BOOK_INK}, showlegend=False, hoverinfo="skip"), row=2, col=1)
    frames = [go.Frame(name="mcmc-start", traces=[0, 3, 4, 5, 6, 7, 9], data=[figure.data[index] for index in [0, 3, 4, 5, 6, 7, 9]])]
    teaching_indices = range(MCMC_WARMUP, MCMC_WARMUP + MCMC_TEACHING_DRAWS)
    sample_indices = [*range(MCMC_WARMUP + MCMC_TEACHING_DRAWS + MCMC_FRAME_STEP - 1, MCMC_STATES - 1, MCMC_FRAME_STEP), MCMC_STATES - 1]
    for index in [*teaching_indices, *sample_indices]:
        teaching = index in teaching_indices
        for phase in (["propose", "retain"] if teaching else ["retain"]):
            shown_index = index - 1 if phase == "propose" else index
            retained_means = means[MCMC_WARMUP : shown_index + 1]
            retained_scales = scales[MCMC_WARMUP : shown_index + 1]
            path_start = max(initial_index, shown_index - MCMC_RECENT_PATH + 1) if index < MCMC_STATES - 1 else shown_index + 1
            accepted = accepted_updates[index]
            if phase == "propose":
                probability_text = "Outside the prior bounds: a = 0." if acceptance_probabilities[index] == 0 else f"Accept with probability a = {acceptance_probabilities[index]:.0%}."
                status = f"Draw {len(retained_means) + 1:,}/{MCMC_DRAWS:,} · 1/2 propose<br>{probability_text}"
                proposal_x = [means[index - 1], proposed_means[index]]
                proposal_y = [scales[index - 1], proposed_scales[index]]
            elif teaching:
                outcome = "accepted: move" if accepted else "rejected: stay and count again"
                status = f"Retained {len(retained_means):,}/{MCMC_DRAWS:,} · 2/2 decide<br>{outcome}<br>Random u = {uniforms[index]:.2f} {'<' if accepted else '≥'} a = {acceptance_probabilities[index]:.2f}."
                proposal_x, proposal_y = [proposed_means[index]], [proposed_scales[index]]
            else:
                progress = "Sample complete" if index == MCMC_STATES - 1 else "Building the sample"
                status = f"Retained {len(retained_means):,}/{MCMC_DRAWS:,}<br>{progress}; repeats count."
                proposal_x, proposal_y = [], []
            frames.append(go.Frame(
                name=f"mcmc-{index}-{phase}",
                group="teaching" if teaching else "sample",
                traces=[0, 3, 4, 5, 6, 7, 9],
                data=[
                    go.Bar(x=mean_centres, y=_relative_histogram(retained_means, MEAN_MARGINAL_EDGES), width=np.diff(MEAN_MARGINAL_EDGES) * 0.9),
                    go.Scatter(x=retained_means, y=retained_scales, mode="markers", marker={"color": IMPERIAL_BLUE, "size": 4, "opacity": 0.32}),
                    go.Scatter(x=means[path_start : shown_index + 1], y=scales[path_start : shown_index + 1], mode="lines+markers", line={"color": IMPERIAL_BLUE, "width": 2.1}, marker={"color": IMPERIAL_BLUE, "size": 4}),
                    go.Scatter(x=[means[shown_index]], y=[scales[shown_index]], mode="markers", marker={"color": OXIDE_RED, "size": 12, "line": {"color": BOOK_SHEET, "width": 2}}),
                    go.Scatter(x=proposal_x, y=proposal_y, mode="lines+markers", line={"color": IMPERIAL_SKY, "width": 2, "dash": "dot"}, marker={"color": BOOK_SHEET if phase == "propose" or accepted else OXIDE_RED, "size": [0, 11] if phase == "propose" else 11, "symbol": "diamond" if phase == "propose" or accepted else "x-open", "line": {"color": IMPERIAL_SKY if phase == "propose" or accepted else OXIDE_RED, "width": 2}}),
                    go.Bar(x=_relative_histogram(retained_scales, SCALE_MARGINAL_EDGES), y=scale_centres, width=np.diff(SCALE_MARGINAL_EDGES) * 0.9, orientation="h"),
                    go.Scatter(x=[MEAN_BOUNDS[0] + 0.04], y=[SCALE_BOUNDS[1] - 0.075], mode="text", text=[status], textposition="middle right", textfont={"family": BOOK_SERIF, "size": 13, "color": BOOK_INK}, showlegend=False, hoverinfo="skip"),
                ],
            ))
    figure.frames = frames
    instant = {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}
    figure.update_layout(
        **_plot_layout(height=650, bottom_margin=145, top_margin=48, left_margin=64, right_margin=24),
        barmode="overlay",
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0, "font": {"size": 11}},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0,
                "y": -0.17,
                "xanchor": "left",
                "yanchor": "top",
                "showactive": False,
                "bgcolor": BOOK_INK,
                "bordercolor": BOOK_INK,
                "font": {"family": BOOK_MONO, "color": BOOK_PAPER, "size": 11},
                "buttons": [
                    {"label": f"Watch {MCMC_TEACHING_DRAWS}", "method": "animate", "args": [["mcmc-start", *[frame.name for frame in frames if frame.group == "teaching"]], {"frame": {"duration": MCMC_STEP_DURATION_MS, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}]},
                    {"label": f"Build {MCMC_DRAWS:,}", "method": "animate", "args": [[frame.name for frame in frames if frame.name.endswith("-retain")], {"frame": {"duration": 240, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}, "fromcurrent": True}]},
                    {"label": "Pause", "method": "animate", "args": [[None], instant]},
                    {"label": "Reset", "method": "animate", "args": [["mcmc-start"], instant]},
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0,
                "y": -0.29,
                "len": 1,
                "pad": {"t": 4},
                "currentvalue": {"visible": False},
                "steps": [
                    {
                        "label": "Start" if frame.name == "mcmc-start" else (f"{len(frame.data[1].x):,}" if frame.name.endswith("-retain") and (len(frame.data[1].x) == MCMC_TEACHING_DRAWS or len(frame.data[1].x) >= MCMC_DRAWS or (len(frame.data[1].x) - MCMC_TEACHING_DRAWS) % 200 == 0) else ""),
                        "method": "animate",
                        "args": [[frame.name], instant],
                    }
                    for frame in frames
                ],
            }
        ],
    )
    figure.update_layout(plot_bgcolor="rgba(0,0,0,0)", shapes=SAMPLER_PLATE_SHAPES)
    figure.update_xaxes(range=list(MEAN_BOUNDS), showgrid=False, showticklabels=False, row=1, col=1)
    figure.update_yaxes(title="Relative marginal of μ", range=[0, 1.05], showgrid=False, showticklabels=False, row=1, col=1)
    figure.update_xaxes(title="Mean μ", range=list(MEAN_BOUNDS), gridcolor=BOOK_GRID, linecolor=BOOK_RULE, row=2, col=1)
    figure.update_yaxes(title="SD σ", range=list(SCALE_BOUNDS), gridcolor=BOOK_GRID, linecolor=BOOK_RULE, row=2, col=1)
    figure.update_xaxes(title={"text": "σ marginal", "font": {"size": 13}}, range=[0, 1.05], showgrid=False, showticklabels=False, row=2, col=2)
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

    means = rng.uniform(*MEAN_BOUNDS, SMC_PARTICLES)
    scales = rng.uniform(*SCALE_BOUNDS, SMC_PARTICLES)
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

    def contour(temperature: float) -> go.Heatmap:
        return go.Heatmap(
            x=grid_means,
            y=grid_scales,
            z=_tempered_surface(temperature),
            zmin=0,
            zmax=1,
            colorscale=SURFACE_COLORS,
            zsmooth="best",
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
            return f"<b>Start · uniform prior · β = 0</b><br>{SMC_PARTICLES} candidate pairs, all equally weighted.<br>Press Next to add data support."
        if phase == "reweight":
            effective_particles = 1.0 / float(np.sum(weights**2))
            return f"<b>1 · Reweight · β = {temperature:.3f}</b><br>Same positions; larger, redder = more weight.<br>Effective sample size: {effective_particles:.0f}/{SMC_PARTICLES}."
        if phase == "resample":
            return f"<b>2 · Resample · β = {temperature:.3f}</b><br>Copy high-weight pairs; drop others.<br>Larger dots now mean more copies here."
        if temperature == 1.0:
            return "<b>3 · Move · β = 1 · posterior reached</b><br>Random-walk updates spread copies apart.<br>Particles now approximate the posterior."
        return f"<b>3 · Move · β = {temperature:.3f}</b><br>Random-walk updates spread copies apart.<br>Lines: 24 example moves. Then repeat."

    def particles(phase: str, means: np.ndarray, scales: np.ndarray, weights: np.ndarray) -> go.Scatter:
        marker = particle_marker(phase, weights)
        customdata = weights[:, None]
        hover = "Weight: %{customdata[0]:.3f}"
        if phase == "resample":
            pairs, counts = np.unique(np.column_stack((means, scales)), axis=0, return_counts=True)
            means, scales = pairs.T
            marker.update(size=6 * np.sqrt(counts), color=IMPERIAL_SKY)
            customdata = counts[:, None]
            hover = "Copies here: %{customdata[0]}<br>Each copy has equal weight"
        return go.Scatter(x=means, y=scales, mode="markers", name="Particles", marker=marker,
                          customdata=customdata, hovertemplate="Mean μ: %{x:.3f}<br>SD σ: %{y:.3f}<br>" + hover + "<extra></extra>")

    def controls(index: int) -> list[dict]:
        instant = {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}
        return [{
            "type": "buttons", "direction": "left", "x": 0, "y": -0.20,
            "xanchor": "left", "yanchor": "top", "showactive": False,
            "bgcolor": BOOK_INK, "bordercolor": BOOK_INK,
            "font": {"family": BOOK_MONO, "color": BOOK_PAPER, "size": 11},
            "buttons": [
                {"label": "Back", "method": "animate", "args": [[f"smc-{max(0, index - 1)}"], instant]},
                {"label": "Next", "method": "animate", "args": [[f"smc-{min(len(states) - 1, index + 1)}"], instant]},
                {"label": "Play slowly", "method": "animate", "args": [[f"smc-{i}" for i in range(index + 1, len(states))] or [f"smc-{i}" for i in range(len(states))], {"frame": {"duration": 2500, "redraw": True}, "transition": {"duration": 0}, "mode": "immediate"}]},
                {"label": "Pause", "method": "animate", "args": [[None], instant]},
                {"label": "Reset", "method": "animate", "args": [["smc-0"], instant]},
            ],
        }]

    def stage_annotation(temperature: float, phase: str, weights: np.ndarray) -> list[dict]:
        return [{"xref": "paper", "yref": "paper", "x": 0, "y": 1.16,
                 "xanchor": "left", "yanchor": "bottom", "align": "left", "showarrow": False,
                 "text": status_text(temperature, phase, weights), "font": {"size": 13, "color": BOOK_INK}}]

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
    initial_mean_bar.update(name="Particle marginal", marker={"color": IMPERIAL_SKY, "line": {"color": BOOK_SHEET, "width": 0.7}}, opacity=0.72, hovertemplate="Mean μ: %{x:.3f}<br>Fraction of largest bin: %{y:.3f}<extra>Particle marginal</extra>")
    initial_mean_target.update(name="Grid stage target", line={"color": IMPERIAL_BLUE, "width": 2})
    initial_scale_bar.update(marker={"color": IMPERIAL_SKY, "line": {"color": BOOK_SHEET, "width": 0.7}}, opacity=0.72, showlegend=False, hovertemplate="SD σ: %{y:.3f}<br>Fraction of largest bin: %{x:.3f}<extra>Particle marginal</extra>")
    initial_scale_target.update(line={"color": IMPERIAL_BLUE, "width": 2}, showlegend=False)
    figure.add_trace(initial_mean_bar, row=1, col=1)
    figure.add_trace(initial_mean_target, row=1, col=1)
    figure.add_trace(contour(initial_temperature), row=2, col=1)
    figure.add_trace(particles(initial_phase, initial_means, initial_scales, initial_weights), row=2, col=1)
    figure.add_trace(initial_scale_bar, row=2, col=2)
    figure.add_trace(initial_scale_target, row=2, col=2)
    figure.add_trace(go.Scatter(x=[], y=[], mode="lines", line={"color": BOOK_RULE, "width": 1}, showlegend=False, hoverinfo="skip"), row=2, col=1)
    figure.frames = [
        go.Frame(
            name=f"smc-{index}",
            traces=list(range(7)),
            layout={"annotations": stage_annotation(temperature, phase, weights), "updatemenus": controls(index)},
            data=[
                mean_bar,
                mean_target,
                contour(temperature),
                particles(phase, means, scales, weights),
                scale_bar,
                scale_target,
                go.Scatter(
                    x=[value for particle in range(24) for value in (states[index - 1][2][particle], means[particle], None)] if phase == "move" else [],
                    y=[value for particle in range(24) for value in (states[index - 1][3][particle], scales[particle], None)] if phase == "move" else [],
                    mode="lines",
                    line={"color": BOOK_RULE, "width": 1},
                    showlegend=False,
                    hoverinfo="skip",
                ),
            ],
        )
        for index, (temperature, phase, means, scales, weights) in enumerate(states)
        for mean_bar, mean_target, scale_bar, scale_target in [marginal_traces(temperature, means, scales, weights)]
    ]
    figure.update_layout(
        **_plot_layout(height=650, bottom_margin=140, top_margin=116, left_margin=64, right_margin=24),
        barmode="overlay",
        legend={"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0, "font": {"size": 11}},
        annotations=stage_annotation(initial_temperature, initial_phase, initial_weights),
        updatemenus=controls(0),
        sliders=[
            {
                "active": 0,
                "x": 0,
                "y": -0.34,
                "len": 1,
                "currentvalue": {"visible": False},
                "pad": {"t": 4},
                "steps": [
                    {
                        "label": f"β {temperature:.2f}" if phase in {"prior", "move"} else "",
                        "method": "animate",
                        "args": [[f"smc-{index}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
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
    figure.update_xaxes(title={"text": "σ marginal", "font": {"size": 13}}, range=[0, 1.05], showgrid=False, showticklabels=False, row=2, col=2)
    figure.update_yaxes(range=list(SCALE_BOUNDS), showgrid=False, showticklabels=False, row=2, col=2)
    return figure


@lru_cache(maxsize=3)
def _bayesian_update_figure(stage: str) -> go.Figure:
    """Show each update on the same grid, with explicitly peak-relative colour."""
    means, scales, likelihood, prior, _, posterior, _ = _two_parameter_surfaces()
    surface = {"prior": prior, "likelihood": likelihood, "posterior": posterior}[stage]
    value_label = {"prior": "Prior density", "likelihood": "Likelihood", "posterior": "Posterior density"}[stage]
    figure = go.Figure(
        go.Heatmap(
            x=means,
            y=scales,
            z=surface / surface.max(),
            customdata=surface,
            zmin=0,
            zmax=1,
            colorscale=SURFACE_COLORS,
            showscale=False,
            hovertemplate=f"Mean μ: %{{x:.2f}}<br>SD σ: %{{y:.2f}}<br>{value_label}: %{{customdata:.3g}}<br>Fraction of this panel’s peak: %{{z:.2f}}<extra></extra>",
        )
    )
    if stage == "prior":
        figure.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
            text="Equal-area patches<br>have equal probability",
            font={"color": BOOK_SHEET, "size": 13},
        )
    figure.update_layout(**_plot_layout(height=330, top_margin=14, bottom_margin=56, left_margin=48, right_margin=12))
    figure.update_xaxes(title="Mean μ", range=list(MEAN_BOUNDS), tickvals=[4.5, 5.0, 5.5], showgrid=False, linecolor=BOOK_RULE, fixedrange=True)
    figure.update_yaxes(title="SD σ", range=list(SCALE_BOUNDS), tickvals=[0.2, 0.4, 0.6, 0.8], showgrid=False, linecolor=BOOK_RULE, fixedrange=True)
    return figure


def _source_note(*children) -> html.P:
    return html.P(children, className="barracuda-source-note")


def _contents() -> html.Nav:
    items = [
        ("01", "Bayes theorem", "#bayes-theorem"),
        ("02", "Bernoulli model", "#coin-experiment"),
        ("03", "MCMC and SMC", "#computation"),
        ("04", "Bayes factors", "#bayes-factors"),
        ("05", "Thomas Bayes", "#thomas-bayes"),
        ("06", "Further learning", "#learning-resources"),
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

    return html.Div(
        [
            page_header(
                "Learn",
                "Bayesian inference for BARRACUDA",
                "A scientific introduction to probability models, posterior uncertainty and computational inference, with reproducible experiments and an annotated reading pathway.",
                badge="Interactive methods companion · simulated data",
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
                        "For events with positive conditioning probabilities, write the same joint event in two ways, then rearrange.",
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
                        "For data y, parameter θ and model M, the same rule becomes the Bayesian update used in statistics. Here p denotes a probability mass function or density, as appropriate; every inference is conditional on the specified model and prior.",
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
                        "Source: Gelman et al., ",
                        _external_link("Bayesian Data Analysis, 3rd ed.", BDA3_URL),
                        ", §1.3.",
                    ),
                ],
                id="bayes-theorem",
                className="barracuda-lesson-section",
            ),
            html.Section(
                [
                    html.Span("02 · Exact inference", className="barracuda-section-label"),
                    html.H2("A controlled Bernoulli experiment"),
                    html.P(
                        "A coin provides a simple statistical model: each observation is a head (1) or tail (0). Assume independent trials with the same unknown success probability θ. We generate a dataset, compute its exact posterior and distinguish parameter uncertainty from the randomness of the next observation.",
                        className="barracuda-section-lead",
                    ),
                    html.Div(
                        [
                            markdown(r"$$Y_i\mid\theta\overset{\mathrm{iid}}{\sim}\operatorname{Bernoulli}(\theta)$$", class_name="barracuda-equation small", mathjax=True),
                            markdown(r"$$\theta\sim\operatorname{Beta}(1,1)$$", class_name="barracuda-equation small", mathjax=True),
                            markdown(r"$$\theta\mid y\sim\operatorname{Beta}(h+1,n-h+1)$$", class_name="barracuda-equation small", mathjax=True),
                        ],
                        className="barracuda-equation-triptych",
                    ),
                    html.P(
                        "Here n is the sample size and h = Σᵢyᵢ is the number of heads. The count H | θ follows Binomial(n, θ); its likelihood is proportional to θʰ(1 − θ)ⁿ⁻ʰ. Multiplying by the uniform prior gives the Beta posterior above.",
                        className="barracuda-copy",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("Experiment settings", className="barracuda-section-label"),
                                    html.Label(
                                        [
                                            html.Span("Generating probability θ₀", className="barracuda-field-label"),
                                            dcc.Slider(
                                                id="coin-probability", min=0, max=1, step=0.01,
                                                value=probability, marks={0: "0", 0.5: "0.5", 1: "1"},
                                                tooltip={"placement": "bottom"},
                                            ),
                                        ],
                                        className="barracuda-field",
                                    ),
                                    html.Div(id="coin-ground-truth", children="θ₀ = 0.50 is known only because this is a simulation; it is not supplied to the posterior calculation.", className="barracuda-help"),
                                    html.Label(
                                        [
                                            html.Span("Sample size n", className="barracuda-field-label"),
                                            dcc.Slider(
                                                id="coin-tosses", min=1, max=500, step=1,
                                                value=n_tosses, marks={1: "1", 100: "100", 250: "250", 500: "500"},
                                                tooltip={"placement": "bottom"},
                                            ),
                                        ],
                                        className="barracuda-field",
                                    ),
                                    html.Label(
                                        [
                                            html.Span("Posterior interval probability (%)", className="barracuda-field-label"),
                                            dcc.Slider(
                                                id="coin-hdi-percent", min=50, max=99, step=1,
                                                value=hdi_percent, marks={50: "50%", 75: "75%", 99: "99%"},
                                                tooltip={"placement": "bottom"},
                                            ),
                                        ],
                                        className="barracuda-field",
                                    ),
                                    html.P("Changing interval probability leaves the observed data unchanged.", className="barracuda-help"),
                                    html.Div([html.Strong("Fixed prior: θ ∼ Beta(1, 1)"), html.P("Equal-length intervals in θ have equal prior probability. Uniformity is specific to this parameterization.")], className="barracuda-bernoulli-prior"),
                                    html.P("At fixed θ₀ and seed, changing n reveals a longer or shorter prefix of the same sequence. Changing θ₀ generates a different experiment.", className="barracuda-help"),
                                ],
                                className="barracuda-control-panel barracuda-bernoulli-controls",
                            ),
                            html.Div(
                                [
                                    html.Button("Toss the coin · new dataset", id="coin-toss-again", n_clicks=0, className="barracuda-button primary full"),
                                    html.P("Generates all n observations using the next seed; the coin shows the final toss.", className="barracuda-help"),
                                    html.Div(
                                        _coin_experiment_summary(initial_outcomes, probability, 2026 + toss_round),
                                        id="coin-summary", **{"aria-live": "polite"},
                                    ),
                                ],
                                className="barracuda-bernoulli-summary",
                            ),
                        ],
                        className="barracuda-bernoulli-lab-grid",
                    ),
                    html.Div(id="coin-metrics", children=metrics(initial_metrics)),
                    html.Div(
                        [
                            html.Figure(
                                [
                                    html.H3("a · Prior and exact posterior"),
                                    dcc.Graph(id="coin-figure", figure=initial_figure, config={"displayModeBar": False, "responsive": True}, className="barracuda-coin-plot", style={"height": "430px"}),
                                    html.Figcaption("Each density integrates to one. Shading marks the selected highest-density interval (HDI). The dashed line is the observed proportion h/n, which maximizes the likelihood; the dotted line is the generating θ₀."),
                                ],
                                className="barracuda-coin-chart barracuda-bernoulli-figure",
                            ),
                            html.Figure(
                                [
                                    html.H3("b · Learning from successive observations"),
                                    dcc.Graph(id="coin-frequency-figure", figure=initial_frequency, config={"displayModeBar": False, "responsive": True}, className="barracuda-coin-plot", style={"height": "430px"}),
                                    html.Figcaption("At each n, the solid line is E[θ | y₁,…,yₙ] and the shaded band is a pointwise equal-tailed credible interval with the selected probability. Equal-tailed limits use posterior quantiles and can differ from the HDI in panel a."),
                                ],
                                className="barracuda-coin-chart barracuda-bernoulli-figure",
                            ),
                        ],
                        className="barracuda-coin-plot-grid",
                    ),
                    note(
                        "What the uncertainty means",
                        "A 95% credible interval contains 95% of the posterior probability for θ, conditional on this dataset, prior and model. It is not an interval for individual 0/1 outcomes, and does not assert 95% repeated-sampling coverage. The sequential band is pointwise, not a simultaneous statement about the entire curve.",
                        tone="teal",
                    ),
                    html.P(
                        "Under the assumed model, more informative data generally concentrate the posterior, although a realized interval need not shrink after every observation. The prior regularizes small samples: h/n may be 0 or 1 while the posterior predictive probability remains between them. More data do not repair dependence, selection bias or a success probability that changes between trials.",
                        className="barracuda-copy",
                    ),
                    _source_note(
                        "Derivation and interpretation: Gelman et al., ",
                        _external_link("Bayesian Data Analysis, 3rd ed., Chapter 2", BDA3_URL),
                        ". Numerical implementation: SciPy evaluates the Beta density and quantiles; a one-dimensional optimization locates the HDI. The simulation and inference are separate steps.",
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
                                    html.P("Assume five measurements are conditionally independent and identically distributed given an unknown population mean μ and standard deviation σ."),
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
                    html.H3("From an even starting point to a focused posterior"),
                    html.P(
                        "Imagine each small patch of this map as a possible combination of mean and spread. Start with equal probability in equal-area patches, then let the five measurements favour the combinations that explain them.",
                        className="barracuda-copy",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(number, className="barracuda-section-label"),
                                    html.H4(title),
                                    html.P(description, className="barracuda-help"),
                                    dcc.Graph(
                                        id=f"bayesian-update-{stage}",
                                        figure=_bayesian_update_figure(stage),
                                        config={"displayModeBar": False, "responsive": True},
                                        style={"height": "330px"},
                                    ),
                                ],
                                className="barracuda-concept-panel barracuda-update-card",
                            )
                            for number, stage, title, description in [
                                ("01 · Before the data", "prior", "Start with a uniform prior", f"Every equal-area patch is equally plausible within μ = {MEAN_BOUNDS[0]}–{MEAN_BOUNDS[1]} and σ = {SCALE_BOUNDS[0]}–{SCALE_BOUNDS[1]}. The prior is zero outside this teaching range."),
                                ("02 · Read the data", "likelihood", "Score how well each pair fits", "The measurements cluster near 5 with a small spread. Pairs near the dark region make these data more likely; pale regions explain them poorly."),
                                ("03 · After the data", "posterior", "Reweight, then make the total 1", "Multiply each prior density by its likelihood, then divide by their total integral. The resulting posterior concentrates probability in the region supported by the data."),
                            ]
                        ],
                        className="barracuda-card-grid three",
                    ),
                    html.P("All maps use the same axes. Darker blue means a larger value relative to that panel’s own peak, not equal absolute density across panels. Hover to read the actual values.", className="barracuda-help"),
                    markdown(r"$$\text{posterior}=\frac{\text{uniform prior}\times\text{likelihood}}{\text{total integral}}$$", class_name="barracuda-equation small", mathjax=True),
                    note(
                        "Why do the last two maps have the same shape?",
                        "The uniform prior gives every pair the same multiplier, so the likelihood determines the posterior’s shape inside the bounds. Normalising changes only the vertical scale: posterior probability must total 1, whereas likelihood scores have no such requirement. An informative prior could change the shape.",
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
                    html.P(
                        "Posterior means, event probabilities and predictions require integrating over parameter uncertainty. Sampling replaces such integrals with averages over draws; SMC uses weighted averages. The number of effectively independent draws, rather than the number of plotted points alone, controls Monte Carlo precision.",
                        className="barracuda-copy",
                    ),
                    markdown(
                        r"$$\mathbb{E}[g(\theta)\mid y]=\int g(\theta)p(\theta\mid y)\,d\theta\ \approx\ \frac{1}{N}\sum_{s=1}^{N}g(\theta^{(s)})$$",
                        class_name="barracuda-equation small",
                        mathjax=True,
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
                                                html.P("Warm-up has already run: 200 states were discarded. Watch 12 individual steps, then build the 1,000-draw sample."),
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
                                                html.P(["The blue lines show grid-based reference marginals; the bars show the retained sample. Bars and reference curves are each scaled to their own peak, so compare their shapes. Formal assessment also requires multiple-chain diagnostics, effective sample sizes and Monte Carlo standard errors. See ", _external_link("Stan’s MCMC guidance", MCMC_GUIDE_URL), "."], className="barracuda-help"),
                                            ]
                                        ),
                                        html.Div(
                                            dcc.Graph(
                                                id="mcmc-animation",
                                                figure=_mcmc_figure(),
                                                config={"displayModeBar": False, "responsive": True},
                                                className="barracuda-sampler-plot",
                                                style={"height": "650px"},
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
                                                html.P("Start with equally plausible pairs. Press Next to inspect one operation at a time, or Play slowly for a guided sequence."),
                                                html.Ol(
                                                    [
                                                        html.Li(f"At β = 0, draw {SMC_PARTICLES} particles from the prior."),
                                                        html.Li("Reweight: increase β to give the data more influence. Positions stay fixed; better-supported pairs get larger weights."),
                                                        html.Li("Resample: copy pairs according to their weights. Each copy has equal weight; dot size shows the number of copies."),
                                                        html.Li(f"Move: try {SMC_MOVE_STEPS} Metropolis updates per particle to spread the copies out. Repeat until β = 1."),
                                                        html.Li("At β = 1, compare the finite particle approximation with the posterior target."),
                                                    ],
                                                    className="barracuda-sampler-steps",
                                                ),
                                                markdown(r"$$\pi_\beta(\theta)\propto p(y\mid\theta)^\beta p(\theta),\qquad 0\leq\beta\leq1$$", class_name="barracuda-equation small", mathjax=True),
                                                html.P([f"β = 0 uses only the prior; β = 1 uses the full likelihood. Each increase keeps effective sample size (ESS) near {SMC_ESS_FRACTION:.0%} unless the posterior is already reachable. Bars show weighted marginals; blue lines show the current target. Each is scaled to its own peak to compare shape. See ", _external_link("PyMC Sequential Monte Carlo", PYMC_SMC_URL), "."], className="barracuda-help"),
                                            ]
                                        ),
                                        html.Div(
                                            dcc.Graph(
                                                id="smc-animation",
                                                figure=_smc_figure(),
                                                config={"displayModeBar": False, "responsive": True},
                                                className="barracuda-sampler-plot",
                                                style={"height": "650px"},
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
                        "Each temperature change produces an incremental likelihood weight. Its weighted average estimates a ratio of successive normalizing constants. With a normalized, proper prior and β progressing from 0 to 1, multiplying these ratios estimates the marginal likelihood.",
                        className="barracuda-copy",
                    ),
                    markdown(
                        r"$$\log \widehat Z_M=\sum_t\log\left[\sum_{i=1}^{N}W_{t-1}^{(i)}p(y\mid\theta_{t-1}^{(i)},M)^{\,\beta_t-\beta_{t-1}}\right],\quad \sum_i W_{t-1}^{(i)}=1$$",
                        class_name="barracuda-equation",
                        mathjax=True,
                    ),
                    html.P(
                        "W denotes the normalized particle weight before reweighting; immediately after resampling, W = 1/N. Retain all likelihood normalization constants when estimating evidence. The logarithm of a finite-sample evidence estimate is not generally unbiased; assess stability across independent runs.",
                        className="barracuda-help",
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
                                    html.Thead(html.Tr([html.Th("Bayes factor BF₁₂"), html.Th("Prior-averaged support for the data under M₁ relative to M₂")])),
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
                    markdown(
                        r"$$\frac{P(M_1\mid y)}{P(M_2\mid y)}=BF_{12}\,\frac{P(M_1)}{P(M_2)}$$",
                        class_name="barracuda-equation small",
                        mathjax=True,
                    ),
                    html.P(
                        "The Bayes factor multiplies prior model odds to give posterior model odds; it is not itself a posterior model probability. These ratios are numerical comparisons, not universal thresholds for scientific discovery.",
                        className="barracuda-copy",
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
                ],
                id="thomas-bayes",
                className="barracuda-lesson-section",
            ),
            learning_resources(),
        ],
        className="barracuda-bayes-page",
    )


def register_callbacks(app) -> None:
    @app.callback(
        Output("coin-figure", "figure"),
        Output("coin-frequency-figure", "figure"),
        Output("coin-metrics", "children"),
        Output("coin-ground-truth", "children"),
        Output("coin-summary", "children"),
        Input("coin-probability", "value"),
        Input("coin-tosses", "value"),
        Input("coin-hdi-percent", "value"),
        Input("coin-toss-again", "n_clicks"),
        State("coin-toss-scene", "key"),
    )
    def update_coin(probability: float, tosses: int, hdi_percent: int, clicks: int, displayed_dataset_key: str | None = None):
        try:
            probability = float(probability)
            tosses = int(tosses)
            hdi_percent = int(hdi_percent)
            toss_round = int(clicks or 0)
        except (TypeError, ValueError, OverflowError):
            raise PreventUpdate
        if not (0 <= probability <= 1 and 1 <= tosses <= 500 and 50 <= hdi_percent <= 99 and 0 <= toss_round < 2**32 - 2026):
            raise PreventUpdate
        outcomes = simulate_coin_tosses(probability, tosses, seed=2026 + toss_round)
        figure, values = _coin_figure(
            probability,
            tosses,
            toss_round,
            hdi_percent,
            outcomes=outcomes,
        )
        frequency_figure = _coin_frequency_figure(probability, outcomes, hdi_percent)
        ground_truth = f"θ₀ = {probability:.2f} is known only because this is a simulation; it is not supplied to the posterior calculation."
        dataset_key = f"{2026 + toss_round}:{probability}:{tosses}"
        return (
            figure,
            frequency_figure,
            metrics(values),
            ground_truth,
            no_update if displayed_dataset_key == dataset_key else _coin_experiment_summary(outcomes, probability, 2026 + toss_round),
        )
