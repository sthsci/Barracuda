"""Condition-aware evidence and posterior figures for event count analyses."""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import dcc, html

from webapp.analysis_ui import MODEL_LABELS, csv_download_link
from webapp.core.condition_inference import (
    ConditionResults,
    build_condition_results_zip,
)
from webapp.core.inference import (
    InferenceResult,
    InferenceSettings,
    MODEL_SPECS,
    evidence_table,
    posterior_draw_table,
)
from webapp.reporting import (
    BF3_LOG10,
    BF_BAND_COLOURS,
    BF_BAND_DEFINITIONS,
    GRID,
    INK,
    MODEL_COLOURS,
    PAPER,
    PLOTLY_PARAMETER_LABELS,
    RULE,
    SERIF,
    SHEET,
    _evidence_axis_ticks,
    _evidence_axis_upper,
    posterior_parameters_for_models,
)
from webapp.ui import note


def condition_evidence_table(results: Mapping[str, Mapping[str, InferenceResult]]) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for condition, condition_results in results.items():
        table = evidence_table(condition_results)
        table.insert(0, "condition", str(condition))
        tables.append(table)
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def condition_bayes_factor_figure(
    results: Mapping[str, Mapping[str, InferenceResult]],
) -> go.Figure:
    """Plot all condition-specific Bayes factors on one true log10 scale."""

    evidence = condition_evidence_table(results)
    if evidence.empty:
        raise ValueError("Bayes factors require at least one completed condition")
    evidence = evidence.sort_values(
        ["condition", "log10_BF_best_vs_model"],
        ascending=[True, True],
        kind="stable",
    )
    raw = evidence["log10_BF_best_vs_model"].to_numpy(dtype=float)
    raw = np.where(np.isclose(raw, 0.0), 0.0, raw)
    upper = _evidence_axis_upper(raw)
    conditions = evidence["condition"].astype(str).tolist()
    model_keys = evidence["model_key"].astype(str).tolist()
    best_flags = evidence["is_best"].astype(bool).tolist()
    y_labels = [
        f"{condition} · {MODEL_SPECS[key].short_label}"
        + (" · Best model" if best else "")
        for condition, key, best in zip(conditions, model_keys, best_flags)
    ]
    colours = [
        INK if best else MODEL_COLOURS[key]
        for key, best in zip(model_keys, best_flags)
    ]
    figure = go.Figure(
        go.Bar(
            x=raw,
            y=y_labels,
            orientation="h",
            marker={"color": colours, "line": {"color": INK, "width": 1}},
            text=["Best model" if best else f"{value:.2f}" for value, best in zip(raw, best_flags)],
            textposition="outside",
            cliponaxis=False,
            customdata=np.column_stack(
                [
                    evidence["log_evidence"].to_numpy(dtype=float),
                    np.asarray(conditions, dtype=object),
                ]
            ),
            hovertemplate=(
                "%{y}<br>log10 BF(best/model) = %{x:.4g}"
                "<br>log evidence = %{customdata[0]:.4g}<extra></extra>"
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
    ticks, tick_labels = _evidence_axis_ticks(upper)
    figure.update_layout(
        template="none",
        height=max(420, 54 * len(evidence) + 175),
        paper_bgcolor=SHEET,
        plot_bgcolor=PAPER,
        font={"family": SERIF, "color": INK, "size": 13},
        margin={"l": 190, "r": 82, "t": 92, "b": 78},
        xaxis_title="log₁₀ BF(𝓜<sub>best</sub> / 𝓜)",
        yaxis_title="Condition and candidate model",
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.04,
            "xanchor": "left",
            "yanchor": "bottom",
            "font": {"size": 11},
        },
    )
    figure.update_xaxes(
        range=[0, upper],
        tickmode="array",
        tickvals=ticks,
        ticktext=tick_labels,
        gridcolor=GRID,
        zeroline=True,
        zerolinecolor=INK,
        automargin=True,
    )
    figure.update_yaxes(gridcolor="rgba(0,0,0,0)", automargin=True)
    return figure


def _model_condition_draws(
    results: Mapping[str, Mapping[str, InferenceResult]],
    model_key: str,
) -> dict[str, pd.DataFrame]:
    output: dict[str, pd.DataFrame] = {}
    for condition, condition_results in results.items():
        if model_key not in condition_results:
            continue
        frame = posterior_draw_table(
            {model_key: condition_results[model_key]},
            max_draws_per_model=5_000,
        )
        if not frame.empty:
            output[str(condition)] = frame
    return output


def condition_model_posterior_figure(
    results: Mapping[str, Mapping[str, InferenceResult]],
    model_key: str,
    condition_colours: Mapping[str, str],
) -> go.Figure:
    """Overlay experimental conditions for one fitted donor-ignorant model."""

    if model_key not in MODEL_SPECS:
        raise ValueError(f"unknown model {model_key!r}")
    frames = _model_condition_draws(results, model_key)
    if not frames:
        raise ValueError(f"no posterior draws are available for {model_key}")
    parameters = posterior_parameters_for_models([model_key])
    size = len(parameters)
    figure = make_subplots(
        rows=size,
        cols=size,
        horizontal_spacing=0.08,
        vertical_spacing=0.08,
    )
    legend_drawn: set[str] = set()
    for row_index, row_parameter in enumerate(parameters, start=1):
        for column_index, column_parameter in enumerate(parameters, start=1):
            if column_index > row_index:
                figure.update_xaxes(visible=False, row=row_index, col=column_index)
                figure.update_yaxes(visible=False, row=row_index, col=column_index)
                continue
            for condition, frame in frames.items():
                colour = condition_colours.get(condition, "#007AFF")
                if row_index == column_index:
                    values = frame[row_parameter].dropna().to_numpy(dtype=float)
                    if not len(values):
                        continue
                    figure.add_trace(
                        go.Histogram(
                            x=values,
                            nbinsx=30,
                            histnorm="probability density",
                            name=condition,
                            legendgroup=condition,
                            showlegend=condition not in legend_drawn,
                            opacity=0.22,
                            marker={
                                "color": colour,
                                "line": {"color": colour, "width": 2},
                            },
                            hovertemplate=(
                                f"{condition}<br>%{{x:.4g}}"
                                "<br>Density %{y:.4g}<extra></extra>"
                            ),
                        ),
                        row=row_index,
                        col=column_index,
                    )
                    legend_drawn.add(condition)
                else:
                    paired = frame[[column_parameter, row_parameter]].dropna()
                    if len(paired) < 4:
                        continue
                    figure.add_trace(
                        go.Histogram2dContour(
                            x=paired[column_parameter],
                            y=paired[row_parameter],
                            name=condition,
                            legendgroup=condition,
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
            if row_index == size:
                figure.update_xaxes(
                    title_text=PLOTLY_PARAMETER_LABELS[column_parameter],
                    row=row_index,
                    col=column_index,
                )
            if column_index == 1:
                figure.update_yaxes(
                    title_text=(
                        "Posterior density"
                        if row_index == 1
                        else PLOTLY_PARAMETER_LABELS[row_parameter]
                    ),
                    row=row_index,
                    col=column_index,
                )
    figure.update_layout(
        template="none",
        barmode="overlay",
        height=max(430, 285 * size),
        paper_bgcolor=SHEET,
        plot_bgcolor=PAPER,
        font={"family": SERIF, "color": INK, "size": 13},
        title={
            "text": MODEL_SPECS[model_key].label,
            "x": 0.01,
            "xanchor": "left",
            "font": {"size": 16},
        },
        margin={"l": 98, "r": 32, "t": 108, "b": 94},
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.06,
            "font": {"size": 12},
        },
    )
    figure.update_xaxes(
        showline=True,
        linecolor=RULE,
        gridcolor=GRID,
        ticks="outside",
        zeroline=False,
        automargin=True,
    )
    figure.update_yaxes(
        showline=True,
        linecolor=RULE,
        gridcolor=GRID,
        ticks="outside",
        zeroline=False,
        automargin=True,
    )
    return figure


def _zip_download(content: bytes, filename: str) -> html.A:
    encoded = base64.b64encode(content).decode("ascii")
    return html.A(
        "Download all results and InferenceData files",
        href=f"data:application/zip;base64,{encoded}",
        download=filename,
        className="barracuda-button primary download",
    )


def render_condition_results(
    results: ConditionResults,
    *,
    data: pd.DataFrame,
    observation_time: float,
    settings: InferenceSettings,
    condition_colours: Mapping[str, str],
    prefix: str,
    donor_aware: bool = False,
) -> tuple[html.Div, html.A]:
    """Render evidence first, followed by user-selectable fitted models."""

    if donor_aware:
        raise ValueError("use the donor-aware condition renderer")
    fitted_models = list(next(iter(results.values()))) if results else []
    evidence = condition_evidence_table(results).round(6)
    bayes_figure = condition_bayes_factor_figure(results)
    model_panels: list[html.Div] = []
    for model_key in fitted_models:
        figure = condition_model_posterior_figure(
            results,
            model_key,
            condition_colours,
        )
        model_panels.append(
            html.Div(
                [
                    dcc.Graph(
                        figure=figure,
                        config={
                            "displaylogo": False,
                            "responsive": True,
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": f"barracuda_{model_key}_condition_posteriors",
                                "scale": 2,
                            },
                        },
                        responsive=True,
                        className="barracuda-joint-posterior-plot",
                        style={"height": f"{int(figure.layout.height)}px"},
                    )
                ],
                id={"type": f"{prefix}-model-panel", "index": model_key},
                className="barracuda-model-result-panel",
            )
        )
    archive = build_condition_results_zip(
        results,
        data,
        observation_time,
        settings,
        donor_aware=False,
    )
    content = html.Div(
        [
            note(
                "Inference complete",
                "Inference was run independently for each condition with the same model and prior settings.",
                tone="teal",
            ),
            html.Section(
                [
                    html.Span("Model evidence", className="barracuda-section-label"),
                    html.H3("Bayes factors by experimental condition"),
                    html.P(
                        "Every condition has its own best model. Bars use the raw log₁₀ BF(best model / candidate model) scale; the boundaries are exactly log₁₀(3), 1 and 2.",
                        className="barracuda-help",
                    ),
                    dcc.Graph(
                        figure=bayes_figure,
                        config={
                            "displaylogo": False,
                            "responsive": True,
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": "barracuda_condition_bayes_factors",
                                "scale": 2,
                            },
                        },
                        responsive=True,
                        className="barracuda-bayes-factor-plot",
                    ),
                    csv_download_link(
                        evidence,
                        "barracuda_condition_model_evidence.csv",
                        "Download Bayes factor CSV",
                    ),
                ],
                className="barracuda-result-section barracuda-figure-result",
            ),
            html.Section(
                [
                    html.Span("Posterior results", className="barracuda-section-label"),
                    html.H3("Choose inference results to visualise"),
                    dcc.Checklist(
                        id=f"{prefix}-model-view",
                        options=[
                            {"label": MODEL_LABELS[key], "value": key}
                            for key in fitted_models
                        ],
                        value=fitted_models,
                        inline=True,
                        className="barracuda-posterior-model-options",
                        inputClassName="barracuda-check-input",
                        labelClassName="barracuda-posterior-model-option",
                    ),
                    html.P(
                        "Condition colours are the choices made above. Each model uses only the parameters it contains, so the grid changes from one to three dimensions automatically.",
                        className="barracuda-help",
                    ),
                    html.Div(model_panels, className="barracuda-condition-model-panels"),
                ],
                className="barracuda-result-section",
            ),
        ],
        className="barracuda-results barracuda-condition-results",
    )
    return content, _zip_download(archive, "barracuda_condition_analysis.zip")


def model_panel_styles(
    selected: Sequence[str] | None,
    panel_ids: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, str]]:
    chosen = {str(value) for value in (selected or [])}
    return [
        {} if str(panel_id.get("index")) in chosen else {"display": "none"}
        for panel_id in (panel_ids or [])
    ]
