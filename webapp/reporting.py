"""Reference-style posterior and Bayes-factor figures for event-count results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import gc
from io import BytesIO

import arviz as az
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from webapp.core.inference import InferenceResult, MODEL_SPECS, evidence_table, posterior_draw_table
from webapp.palette import (
    MODEL_GAMMA,
    MODEL_HOMOGENEOUS,
    MODEL_ZERO_INFLATED,
    MODEL_ZERO_INFLATED_GAMMA,
)


COMMON_PARAMETERS = ("mu_lambda", "sigma_lambda", "p_zero")
COMMON_PARAMETER_LABELS = {
    "mu_lambda": "Mean event rate among engaging cells, μλ",
    "sigma_lambda": "Continuous cell-to-cell heterogeneity in event rates, σλ",
    "p_zero": "Fraction of nonengaging cells, φ₀",
}
PLOTLY_PARAMETER_LABELS = {
    "mu_lambda": "Mean event rate among<br>engaging cells, μλ",
    "sigma_lambda": "Continuous cell-to-cell heterogeneity<br>in event rates, σλ",
    "p_zero": "Fraction of nonengaging cells, φ₀",
}
MATPLOTLIB_PARAMETER_LABELS = {
    "mu_lambda": "Mean event rate among\nengaging cells, μλ",
    "sigma_lambda": "Continuous cell-to-cell heterogeneity\nin event rates, σλ",
    "p_zero": "Fraction of nonengaging cells,\nφ₀",
}
MODEL_PARAMETERS = {
    "homo": ("mu_lambda",),
    "z2p": ("mu_lambda", "p_zero"),
    "dis2p": ("mu_lambda", "sigma_lambda"),
    "hetero3": COMMON_PARAMETERS,
}
MODEL_COLOURS = {
    "homo": MODEL_HOMOGENEOUS,
    "z2p": MODEL_ZERO_INFLATED,
    "dis2p": MODEL_GAMMA,
    "hetero3": MODEL_ZERO_INFLATED_GAMMA,
}
MATPLOTLIB_MODEL_LABELS = {
    "homo": r"$\mathcal{M}_{\mathrm{homo}}$",
    "z2p": r"$\mathcal{M}_{\mathrm{ZI}}$",
    "dis2p": r"$\mathcal{M}_{\Gamma}$",
    "hetero3": r"$\mathcal{M}_{\mathrm{ZI}\Gamma}$",
}
INK = "#25231F"
PAPER = "#F3EDDF"
SHEET = "#FBF7ED"
RULE = "#887B66"
GRID = "#D6CCBA"
TRUTH = "#9A4938"
SERIF = "Iowan Old Style, Baskerville, Palatino Linotype, Palatino, Georgia, serif"
BF3_LOG10 = float(np.log10(3.0))
BF_BAND_COLOURS = {
    "Anecdotal": "#F7F7F7",
    "Moderate": "#FFF3B0",
    "Strong": "#FFD28A",
    "Extreme": "#E76F51",
}
BF_BAND_DEFINITIONS = (
    ("Anecdotal", "BF 1–3", 0.0, BF3_LOG10),
    ("Moderate", "BF 3–10", BF3_LOG10, 1.0),
    ("Strong", "BF 10–100", 1.0, 2.0),
    ("Extreme", "BF ≥100", 2.0, None),
)


def _truth_values(truth: Mapping[str, object] | None) -> dict[str, float]:
    if truth is None:
        return {}
    values: dict[str, float] = {}
    for name in COMMON_PARAMETERS:
        if name not in truth:
            continue
        try:
            value = float(truth[name])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values[name] = value
    return values


def _truth_title(
    truth: Mapping[str, object] | None,
    parameters: Sequence[str] = COMMON_PARAMETERS,
) -> str:
    values = _truth_values(truth)
    if not values:
        return ""
    parts = []
    for name, symbol in (("mu_lambda", "μλ"), ("sigma_lambda", "σλ"), ("p_zero", "φ₀")):
        if name in values and name in parameters:
            parts.append(f"{symbol} = {values[name]:g}")
    return "Ground truth: " + ", ".join(parts)


def _model_frame(draws: pd.DataFrame, model_key: str) -> pd.DataFrame:
    return draws.loc[draws["model_key"] == model_key].reset_index(drop=True)


def posterior_parameters_for_models(model_keys: Sequence[str]) -> list[str]:
    """Return the ordered union of parameters represented by fitted models."""

    selected = {str(key) for key in model_keys}
    unknown = selected.difference(MODEL_PARAMETERS)
    if unknown:
        raise ValueError(f"unknown event count model(s): {', '.join(sorted(unknown))}")
    return [
        parameter
        for parameter in COMMON_PARAMETERS
        if any(parameter in MODEL_PARAMETERS[key] for key in selected)
    ]


def posterior_store_payload(
    draws: pd.DataFrame,
    model_keys: Sequence[str],
    truth: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Serialise paired posterior draws for the client-side model filter."""

    requested = list(dict.fromkeys(str(key) for key in model_keys))
    available = set(draws.get("model_key", pd.Series(dtype=str)).astype(str))
    model_order = [key for key in requested if key in MODEL_PARAMETERS and key in available]
    models: dict[str, object] = {}
    for model_key in model_order:
        parameters = [
            parameter
            for parameter in MODEL_PARAMETERS[model_key]
            if parameter in draws.columns
        ]
        frame = _model_frame(draws, model_key)[parameters].apply(
            pd.to_numeric,
            errors="coerce",
        )
        if parameters:
            finite = np.isfinite(frame.to_numpy(dtype=float)).all(axis=1)
            frame = frame.loc[finite]
        models[model_key] = {
            "parameters": parameters,
            "draws": frame.to_numpy(dtype=float).tolist(),
        }
    return {
        "schema_version": 1,
        "model_order": model_order,
        "truth": _truth_values(truth),
        "models": models,
    }


def posterior_draws_from_store(
    payload: Mapping[str, object],
    model_keys: Sequence[str],
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    """Reconstruct paired draws for a valid subset of stored fitted models."""

    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported posterior data format")
    model_order = [str(key) for key in payload.get("model_order", [])]
    model_payloads = payload.get("models", {})
    if not isinstance(model_payloads, Mapping):
        raise ValueError("posterior model data are unavailable")
    chosen = {str(key) for key in model_keys}
    selected = [key for key in model_order if key in chosen and key in MODEL_PARAMETERS]
    if not selected:
        raise ValueError("select at least one fitted model")

    frames: list[pd.DataFrame] = []
    for model_key in selected:
        stored = model_payloads.get(model_key)
        if not isinstance(stored, Mapping):
            continue
        parameters = [
            str(parameter)
            for parameter in stored.get("parameters", [])
            if str(parameter) in COMMON_PARAMETERS
        ]
        rows = stored.get("draws", [])
        try:
            frame = pd.DataFrame(rows, columns=parameters, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid posterior draws for {model_key}") from exc
        frame.insert(0, "posterior_draw", np.arange(len(frame), dtype=int))
        frame.insert(0, "model", MODEL_SPECS[model_key].label)
        frame.insert(0, "model_key", model_key)
        frames.append(frame)
    if not frames:
        raise ValueError("posterior draws are unavailable for the selected models")

    columns = ["model_key", "model", "posterior_draw", *COMMON_PARAMETERS]
    truth = _truth_values(
        payload.get("truth") if isinstance(payload.get("truth"), Mapping) else None
    )
    return pd.concat(frames, ignore_index=True).reindex(columns=columns), selected, truth


def _evidence_axis_upper(values: Sequence[float] | np.ndarray) -> float:
    """Return a padded upper limit on the untransformed log10 BF scale."""

    raw = np.maximum(np.asarray(values, dtype=float), 0.0)
    maximum = float(np.max(raw)) if raw.size else 0.0
    return max(3.0, maximum * 1.08)


def _evidence_axis_ticks(upper: float) -> tuple[list[float], list[str]]:
    """Keep evidence thresholds exact while adding readable large-value ticks."""

    if upper <= 4.0:
        tick_values = [0.0, BF3_LOG10, 1.0, 2.0]
        tick_labels = ["0", "log₁₀ 3", "1", "2"]
        for value in (3.0, 4.0):
            if value <= upper + 1e-9:
                tick_values.append(value)
                tick_labels.append(f"{value:g}")
        return tick_values, tick_labels

    # Once the range is large, threshold labels at 0.477, 1 and 2 would
    # overlap. Their exact values remain encoded by the band boundaries and
    # legend, while the axis uses evenly spaced raw log10 BF ticks.
    tick_values = [0.0]
    tick_labels = ["0"]
    rough_step = max(upper / 5.0, 1e-9)
    magnitude = 10.0 ** np.floor(np.log10(rough_step))
    normalised = rough_step / magnitude
    if normalised <= 1.0:
        multiplier = 1.0
    elif normalised <= 2.0:
        multiplier = 2.0
    elif normalised <= 5.0:
        multiplier = 5.0
    else:
        multiplier = 10.0
    step = multiplier * magnitude
    value = step
    while value <= upper + 1e-9:
        tick_values.append(float(value))
        tick_labels.append(f"{value:g}")
        value += step
    return tick_values, tick_labels


def joint_posterior_figure_from_draws(
    draws: pd.DataFrame,
    model_keys: Sequence[str],
    truth: Mapping[str, object] | None = None,
    *,
    parameters: Sequence[str] | None = None,
) -> go.Figure:
    """Build a joint posterior figure from a selectable subset of fitted draws."""

    selected_models = list(dict.fromkeys(str(key) for key in model_keys))
    if not selected_models:
        raise ValueError("select at least one fitted model")
    if not isinstance(draws, pd.DataFrame) or draws.empty:
        raise ValueError("no scalar event count posterior draws are available")
    available_models = set(draws["model_key"].astype(str))
    unavailable = set(selected_models).difference(available_models)
    if unavailable:
        raise ValueError(f"posterior draws are unavailable for: {', '.join(sorted(unavailable))}")

    if parameters is None:
        params = posterior_parameters_for_models(selected_models)
    else:
        params = [name for name in parameters if name in COMMON_PARAMETERS]
    if not params:
        raise ValueError("at least one common parameter is required")

    filtered_draws = draws.loc[
        draws["model_key"].astype(str).isin(selected_models)
    ].reset_index(drop=True)

    size = len(params)
    figure = make_subplots(
        rows=size,
        cols=size,
        horizontal_spacing=0.07,
        vertical_spacing=0.07,
    )
    legend_drawn: set[str] = set()
    truth_legend_drawn = False
    truths = _truth_values(truth)

    for row_index, row_parameter in enumerate(params, start=1):
        for column_index, column_parameter in enumerate(params, start=1):
            if column_index > row_index:
                figure.update_xaxes(visible=False, row=row_index, col=column_index)
                figure.update_yaxes(visible=False, row=row_index, col=column_index)
                continue

            for model_key in selected_models:
                model_draws = _model_frame(filtered_draws, model_key)
                colour = MODEL_COLOURS[model_key]
                if row_index == column_index:
                    values = model_draws[row_parameter].dropna().to_numpy(dtype=float)
                    if not len(values):
                        continue
                    show_legend = model_key not in legend_drawn
                    figure.add_trace(
                        go.Histogram(
                            x=values,
                            nbinsx=30,
                            histnorm="probability density",
                            name=MODEL_SPECS[model_key].short_label,
                            legendgroup=model_key,
                            showlegend=show_legend,
                            opacity=0.20,
                            marker={
                                "color": colour,
                                "line": {"color": colour, "width": 2},
                            },
                            hovertemplate=(
                                f"{MODEL_SPECS[model_key].short_label}<br>"
                                "%{x:.4g}<br>Density %{y:.4g}<extra></extra>"
                            ),
                        ),
                        row=row_index,
                        col=column_index,
                    )
                    legend_drawn.add(model_key)
                    if len(values) > 1:
                        lower, upper = np.asarray(az.hdi(values, hdi_prob=0.95), dtype=float)
                        figure.add_vrect(
                            x0=float(lower),
                            x1=float(upper),
                            fillcolor=colour,
                            opacity=0.055,
                            line_width=0,
                            layer="below",
                            row=row_index,
                            col=column_index,
                        )
                else:
                    paired = model_draws[[column_parameter, row_parameter]].dropna()
                    if len(paired) < 4:
                        continue
                    figure.add_trace(
                        go.Histogram2dContour(
                            x=paired[column_parameter],
                            y=paired[row_parameter],
                            name=MODEL_SPECS[model_key].short_label,
                            legendgroup=model_key,
                            showlegend=False,
                            showscale=False,
                            ncontours=6,
                            contours={"coloring": "none", "showlabels": False},
                            line={"color": colour, "width": 2},
                            hoverinfo="skip",
                        ),
                        row=row_index,
                        col=column_index,
                    )

            if row_index == column_index and row_parameter in truths:
                figure.add_vline(
                    x=truths[row_parameter],
                    line={"color": TRUTH, "width": 2, "dash": "dash"},
                    row=row_index,
                    col=column_index,
                )
            elif column_parameter in truths and row_parameter in truths:
                figure.add_vline(
                    x=truths[column_parameter],
                    line={"color": TRUTH, "width": 1.5, "dash": "dash"},
                    row=row_index,
                    col=column_index,
                )
                figure.add_hline(
                    y=truths[row_parameter],
                    line={"color": TRUTH, "width": 1.5, "dash": "dash"},
                    row=row_index,
                    col=column_index,
                )
                figure.add_trace(
                    go.Scatter(
                        x=[truths[column_parameter]],
                        y=[truths[row_parameter]],
                        mode="markers",
                        name="Ground truth",
                        showlegend=not truth_legend_drawn,
                        legendgroup="truth",
                        marker={
                            "symbol": "star",
                            "size": 12,
                            "color": TRUTH,
                            "line": {"color": INK, "width": 1},
                        },
                        hovertemplate="Ground truth<extra></extra>",
                    ),
                    row=row_index,
                    col=column_index,
                )
                truth_legend_drawn = True

            if row_index == size:
                figure.update_xaxes(
                    title_text=PLOTLY_PARAMETER_LABELS[column_parameter],
                    row=row_index,
                    col=column_index,
                )
            if column_index == 1:
                y_title = "Posterior density" if row_index == 1 else PLOTLY_PARAMETER_LABELS[row_parameter]
                figure.update_yaxes(title_text=y_title, row=row_index, col=column_index)

    figure.update_layout(
        template="none",
        barmode="overlay",
        height=max(430, 285 * size),
        paper_bgcolor=SHEET,
        plot_bgcolor=PAPER,
        font={"family": SERIF, "color": INK, "size": 13},
        title={
            "text": _truth_title(truth, params),
            "x": 0.01,
            "xanchor": "left",
            "y": 0.985,
            "yanchor": "top",
            "font": {"size": 16},
        },
        margin={"l": 92, "r": 32, "t": 118 if truth else 86, "b": 94},
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.075,
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 12},
        },
        hoverlabel={"bgcolor": SHEET, "bordercolor": RULE, "font_family": SERIF},
    )
    figure.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor=RULE,
        gridcolor=GRID,
        ticks="outside",
        tickcolor=RULE,
        zeroline=False,
        automargin=True,
    )
    figure.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor=RULE,
        gridcolor=GRID,
        ticks="outside",
        tickcolor=RULE,
        zeroline=False,
        automargin=True,
    )
    return figure


def joint_posterior_figure(
    results: Mapping[str, InferenceResult],
    truth: Mapping[str, object] | None = None,
    *,
    parameters: Sequence[str] | None = None,
) -> go.Figure:
    """Build the notebook-style lower-triangle joint posterior figure."""

    draws = posterior_draw_table(results, max_draws_per_model=5_000)
    return joint_posterior_figure_from_draws(
        draws,
        list(results),
        truth,
        parameters=parameters,
    )


def bayes_factor_figure(
    results: Mapping[str, InferenceResult],
    truth: Mapping[str, object] | None = None,
) -> go.Figure:
    """Plot log10 BF(best/model) using the convention in the demo notebook."""

    evidence = evidence_table(results).sort_values(
        "log10_BF_best_vs_model", ascending=True
    )
    raw_values = evidence["log10_BF_best_vs_model"].to_numpy(dtype=float)
    raw_values = np.where(np.isclose(raw_values, 0.0), 0.0, raw_values)
    upper = _evidence_axis_upper(raw_values)
    model_keys = evidence["model_key"].astype(str).tolist()
    best_flags = evidence["is_best"].astype(bool).tolist()
    labels = [
        f"{MODEL_SPECS[key].short_label} · Best model"
        if is_best
        else MODEL_SPECS[key].short_label
        for key, is_best in zip(model_keys, best_flags)
    ]
    colours = [
        INK if bool(best) else MODEL_COLOURS[key]
        for key, best in zip(model_keys, best_flags)
    ]
    figure = go.Figure(
        go.Bar(
            x=raw_values,
            y=labels,
            orientation="h",
            marker={"color": colours, "line": {"color": INK, "width": 1}},
            text=[
                f"Best model · {value:.2f}" if is_best else f"{value:.2f}"
                for value, is_best in zip(raw_values, best_flags)
            ],
            textposition="outside",
            cliponaxis=False,
            customdata=np.column_stack(
                [raw_values, evidence["log_evidence"].to_numpy(dtype=float)]
            ),
            hovertemplate=(
                "%{y}<br>log10 BF(best/model) = %{customdata[0]:.4g}"
                "<br>log evidence = %{customdata[1]:.4g}<extra></extra>"
            ),
        )
    )
    for label, range_label, lower, definition_upper in BF_BAND_DEFINITIONS:
        band_upper = upper if definition_upper is None else definition_upper
        figure.add_vrect(
            x0=lower,
            x1=band_upper,
            fillcolor=BF_BAND_COLOURS[label],
            opacity=0.48,
            line_width=0,
            layer="below",
        )
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"{label} · {range_label}",
                marker={
                    "symbol": "square",
                    "size": 11,
                    "color": BF_BAND_COLOURS[label],
                    "line": {"color": RULE, "width": 0.8},
                },
                hoverinfo="skip",
                showlegend=True,
            )
        )
    tick_values, tick_labels = _evidence_axis_ticks(upper)
    figure.update_layout(
        template="none",
        height=max(330, 84 * len(evidence) + 150),
        paper_bgcolor=SHEET,
        plot_bgcolor=PAPER,
        font={"family": SERIF, "color": INK, "size": 14},
        title={"text": _truth_title(truth), "x": 0.01, "xanchor": "left", "font": {"size": 16}},
        margin={"l": 96, "r": 74, "t": 118 if truth else 88, "b": 76},
        xaxis_title="log₁₀ BF(𝓜<sub>best</sub> / 𝓜)",
        yaxis_title="Fitted model",
        showlegend=True,
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.08,
            "xanchor": "left",
            "yanchor": "bottom",
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"size": 11},
        },
    )
    figure.update_xaxes(
        range=[0, upper],
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_labels,
        gridcolor=GRID,
        zeroline=True,
        zerolinecolor=INK,
        automargin=True,
    )
    figure.update_yaxes(gridcolor="rgba(0,0,0,0)", automargin=True)
    return figure


def _matplotlib_joint(
    draws: pd.DataFrame,
    results: Mapping[str, InferenceResult],
    truth: Mapping[str, object] | None,
):
    from matplotlib.figure import Figure

    params = posterior_parameters_for_models(list(results))
    size = len(params)
    figure = Figure(figsize=(max(5.5, 3.65 * size), max(5.5, 3.65 * size)))
    axes = figure.subplots(size, size, squeeze=False)
    truths = _truth_values(truth)

    for row_index, row_parameter in enumerate(params):
        for column_index, column_parameter in enumerate(params):
            axis = axes[row_index, column_index]
            axis.set_facecolor("white")
            axis.grid(alpha=0.2)
            if column_index > row_index:
                axis.axis("off")
                continue
            for model_key in results:
                model_draws = _model_frame(draws, model_key)
                colour = MODEL_COLOURS[model_key]
                if row_index == column_index:
                    values = model_draws[row_parameter].dropna().to_numpy(dtype=float)
                    if not len(values):
                        continue
                    axis.hist(
                        values,
                        bins=30,
                        density=True,
                        histtype="step",
                        linewidth=1.8,
                        color=colour,
                        label=MATPLOTLIB_MODEL_LABELS[model_key],
                    )
                    if len(values) > 1:
                        lower, upper = np.asarray(az.hdi(values, hdi_prob=0.95), dtype=float)
                        axis.axvspan(lower, upper, color=colour, alpha=0.07, linewidth=0)
                else:
                    paired = model_draws[[column_parameter, row_parameter]].dropna()
                    if len(paired) < 8:
                        continue
                    x_values = paired[column_parameter].to_numpy(dtype=float)
                    y_values = paired[row_parameter].to_numpy(dtype=float)
                    try:
                        x_range = np.quantile(x_values, [0.005, 0.995])
                        y_range = np.quantile(y_values, [0.005, 0.995])
                        counts, x_edges, y_edges = np.histogram2d(
                            x_values,
                            y_values,
                            bins=34,
                            range=[x_range, y_range],
                        )
                        # A small separable NumPy blur gives stable density
                        # contours without invoking SciPy's KDE inside Dash's
                        # forked macOS background worker.
                        kernel = np.asarray([1.0, 2.0, 1.0]) / 4.0
                        smoothed = counts
                        for _ in range(2):
                            smoothed = np.apply_along_axis(
                                lambda values: np.convolve(
                                    values, kernel, mode="same"
                                ),
                                0,
                                smoothed,
                            )
                            smoothed = np.apply_along_axis(
                                lambda values: np.convolve(
                                    values, kernel, mode="same"
                                ),
                                1,
                                smoothed,
                            )
                        positive = smoothed[smoothed > 0]
                        if not len(positive):
                            raise ValueError("empty posterior density")
                        levels = np.unique(
                            np.quantile(
                                positive,
                                [0.35, 0.5, 0.65, 0.78, 0.88, 0.95],
                            )
                        )
                        x_centres = (x_edges[:-1] + x_edges[1:]) / 2
                        y_centres = (y_edges[:-1] + y_edges[1:]) / 2
                        axis.contour(
                            x_centres,
                            y_centres,
                            smoothed.T,
                            levels=levels,
                            colors=[colour],
                            linewidths=1.1,
                        )
                    except (IndexError, ValueError):
                        axis.scatter(x_values, y_values, s=3, color=colour, alpha=0.15)

            if row_index == column_index and row_parameter in truths:
                axis.axvline(truths[row_parameter], color=TRUTH, linestyle="--", linewidth=1.8)
            elif column_parameter in truths and row_parameter in truths:
                axis.axvline(truths[column_parameter], color=TRUTH, linestyle="--", linewidth=1.3)
                axis.axhline(truths[row_parameter], color=TRUTH, linestyle="--", linewidth=1.3)
                axis.scatter(
                    truths[column_parameter],
                    truths[row_parameter],
                    marker="*",
                    s=90,
                    color=TRUTH,
                    edgecolors=INK,
                    linewidths=0.6,
                    zorder=10,
                )
            if row_index == size - 1:
                axis.set_xlabel(MATPLOTLIB_PARAMETER_LABELS[column_parameter])
            if column_index == 0:
                axis.set_ylabel(
                    "Posterior density"
                    if row_index == 0
                    else MATPLOTLIB_PARAMETER_LABELS[row_parameter]
                )
            axis.tick_params(labelsize=9)

    title = _truth_title(truth, params)
    if title:
        figure.suptitle(title, fontsize=15, y=0.985)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955 if title else 0.985),
            ncol=max(1, len(handles)),
            frameon=False,
        )
    figure.tight_layout(rect=(0, 0, 1, 0.90 if title else 0.93))
    return figure


def _matplotlib_bayes_factor(
    results: Mapping[str, InferenceResult],
    truth: Mapping[str, object] | None,
):
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    evidence = evidence_table(results).sort_values("log10_BF_best_vs_model", ascending=True)
    raw_values = evidence["log10_BF_best_vs_model"].to_numpy(dtype=float)
    raw_values = np.where(np.isclose(raw_values, 0.0), 0.0, raw_values)
    upper = _evidence_axis_upper(raw_values)
    keys = evidence["model_key"].astype(str).tolist()
    best_flags = evidence["is_best"].astype(bool).tolist()
    labels = [
        f"{MATPLOTLIB_MODEL_LABELS[key]}  ·  Best model"
        if is_best
        else MATPLOTLIB_MODEL_LABELS[key]
        for key, is_best in zip(keys, best_flags)
    ]
    colours = [
        INK if bool(best) else MODEL_COLOURS[key]
        for key, best in zip(keys, best_flags)
    ]
    figure = Figure(figsize=(9.2, 5.2))
    axis = figure.subplots()
    legend_handles = []
    for label, range_label, lower, definition_upper in BF_BAND_DEFINITIONS:
        band_upper = upper if definition_upper is None else definition_upper
        colour = BF_BAND_COLOURS[label]
        axis.axvspan(lower, band_upper, color=colour, alpha=0.48, linewidth=0)
        legend_handles.append(
            Patch(
                facecolor=colour,
                edgecolor=RULE,
                linewidth=0.7,
                label=f"{label} · {range_label}",
            )
        )
    bars = axis.barh(
        labels,
        raw_values,
        color=colours,
        edgecolor=INK,
        linewidth=1.0,
        zorder=3,
    )
    for bar, raw_value, is_best in zip(bars, raw_values, best_flags):
        axis.text(
            raw_value + upper * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"Best model · {raw_value:.2f}" if is_best else f"{raw_value:.2f}",
            va="center",
            zorder=4,
        )
    axis.set_xlim(0, upper)
    tick_values, tick_labels = _evidence_axis_ticks(upper)
    axis.set_xticks(tick_values, tick_labels)
    axis.set_xlabel(
        r"$\log_{10}\mathrm{BF}(\mathcal{M}_{\mathrm{best}}/\mathcal{M})$"
    )
    axis.set_ylabel("Fitted model")
    axis.grid(axis="x", alpha=0.25, zorder=1)
    axis.axvline(0, color=INK, linewidth=1.2)
    title = _truth_title(truth)
    if title:
        figure.suptitle(title, y=0.98, fontsize=14)
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91 if title else 0.97),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )
    figure.subplots_adjust(
        left=0.14,
        right=0.97,
        bottom=0.16,
        top=0.80 if title else 0.86,
    )
    return figure


def _save_figure_bytes(figure, file_format: str) -> bytes:
    if file_format not in {"png", "pdf"}:
        raise ValueError("file_format must be 'png' or 'pdf'")
    buffer = BytesIO()
    figure.savefig(
        buffer,
        format=file_format,
        dpi=170 if file_format == "png" else None,
        bbox_inches="tight",
        facecolor="white",
    )
    return buffer.getvalue()


def validation_figure_artifacts(
    results: Mapping[str, InferenceResult],
    truth: Mapping[str, object] | None,
) -> dict[str, bytes]:
    """Return reference-style PNG and PDF figure files for downloads."""

    draws = posterior_draw_table(results, max_draws_per_model=8_000)
    artifacts: dict[str, bytes] = {}

    # PyMC's compiled state is still resident in this background worker.
    # Render and release one canvas at a time so constrained web processes do
    # not lose a completed inference while preparing downloadable figures.
    joint = _matplotlib_joint(draws, results, truth)
    artifacts["figures/joint_posterior.png"] = _save_figure_bytes(joint, "png")
    artifacts["figures/joint_posterior.pdf"] = _save_figure_bytes(joint, "pdf")
    joint.clear()
    del joint
    gc.collect()

    bayes_factor = _matplotlib_bayes_factor(results, truth)
    artifacts["figures/bayes_factors.png"] = _save_figure_bytes(
        bayes_factor, "png"
    )
    artifacts["figures/bayes_factors.pdf"] = _save_figure_bytes(
        bayes_factor, "pdf"
    )
    bayes_factor.clear()
    del bayes_factor
    gc.collect()
    return artifacts
