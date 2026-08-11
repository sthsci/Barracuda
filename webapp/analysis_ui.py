"""Dash components and UI-neutral result rendering for Orca analyses."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_ag_grid as dag
from dash import dcc, html

from webapp.core.inference import (
    InferenceResult,
    InferenceSettings,
    MODEL_SPECS,
    build_results_zip,
    evidence_table,
    posterior_draw_table,
    summary_table,
)
from webapp.core.simulation import RATE_DISTRIBUTION_LABELS, rate_distribution_curve
from webapp.palette import (
    DECISION_TEAL,
    DONOR_GOLD,
    DONOR_RUST,
    DONOR_SAGE,
    DONOR_TEAL,
    MODEL_GAMMA,
    MODEL_HOMOGENEOUS,
    MODEL_ZERO_INFLATED,
    MODEL_ZERO_INFLATED_GAMMA,
)
from webapp.ui import metric, note, research_warning
from webapp.reporting import (
    bayes_factor_figure,
    joint_posterior_figure_from_draws,
    posterior_store_payload,
    validation_figure_artifacts,
)


MODEL_LABELS = {key: spec.label for key, spec in MODEL_SPECS.items()}
MODEL_HELP = {key: spec.description for key, spec in MODEL_SPECS.items()}
MODEL_NAMES = {
    "homo": "Homogeneous Poisson",
    "z2p": "Zero inflated Poisson",
    "dis2p": "Heterogeneous Gamma Poisson",
    "hetero3": "Zero inflated heterogeneous Gamma Poisson",
}
MODEL_COLOURS = {
    "homo": MODEL_HOMOGENEOUS,
    "z2p": MODEL_ZERO_INFLATED,
    "dis2p": MODEL_GAMMA,
    "hetero3": MODEL_ZERO_INFLATED_GAMMA,
}
DONOR_COLOURS = [DONOR_TEAL, DONOR_SAGE, DONOR_GOLD, DONOR_RUST]
PROFILE_VALUES = {
    "preview": (64, 1, 1),
    "demo": (256, 2, 2),
    "custom": (128, 2, 1),
}

BOOK_INK = "#25231F"
BOOK_PAPER = "#F3EDDF"
BOOK_SHEET = "#FBF7ED"
BOOK_RULE = "#887B66"
BOOK_GRID = "#D6CCBA"
BOOK_SERIF = "Iowan Old Style, Baskerville, Palatino Linotype, Palatino, Georgia, serif"

PARAMETER_LABELS = {
    "lambda": "Shared event rate among engaging cells, λ",
    "mu_lambda": "Mean event rate among engaging cells, μλ",
    "sigma_lambda": "Continuous cell-to-cell heterogeneity in event rates, σλ",
    "p_zero": "Fraction of nonengaging cells, φ₀",
    "mu_lambda_population": "Population mean event rate, μλ",
    "sigma_lambda_population": "Population continuous cell-to-cell heterogeneity, σλ",
    "phi_0_population": "Population fraction of nonengaging cells, φ₀",
    "mu_lambda_donor": "Donor mean event rate, μλ,d",
    "sigma_lambda_donor": "Donor continuous cell-to-cell heterogeneity, σλ,d",
    "phi_0_donor": "Donor fraction of nonengaging cells, φ₀,d",
}


def model_title(key: str):
    """Render a paper model symbol together with its full text name."""

    subscripts = {
        "homo": "homo",
        "z2p": "ZI",
        "dis2p": "Γ",
        "hetero3": "ZIΓ",
    }
    return html.Span(
        [
            html.Span("M", className="orca-model-script", **{"aria-hidden": "true"}),
            html.Sub(subscripts[key], **{"aria-hidden": "true"}),
            html.Span(f" · {MODEL_NAMES[key]}"),
        ],
        className="orca-model-title",
        **{"aria-label": MODEL_LABELS[key]},
    )


def parse_optional_seed(raw: object) -> int | None:
    value = "" if raw is None else str(raw).strip()
    if not value:
        return None
    try:
        seed = int(value)
    except ValueError as exc:
        raise ValueError("Seed must be a whole number or left blank.") from exc
    if not 0 <= seed <= 4_294_967_295:
        raise ValueError("Seed must be between 0 and 4,294,967,295.")
    return seed


def _help(text: str) -> html.Small:
    return html.Small(text, className="orca-help")


def field(label: str, control, help_text: str | None = None) -> html.Label:
    children: list = [html.Span(label, className="orca-field-label"), control]
    if help_text:
        children.append(_help(help_text))
    return html.Label(children, className="orca-field")


def model_selector(prefix: str, default: list[str] | None = None) -> html.Div:
    defaults = default or list(MODEL_LABELS)
    return html.Div(
        [
            html.Div("Models to fit", className="orca-field-label"),
            dcc.Checklist(
                id=f"{prefix}-models",
                options=[
                    {
                        "label": html.Span(
                            [html.Strong(model_title(key)), html.Small(MODEL_HELP[key])],
                            className="orca-model-option-copy",
                        ),
                        "value": key,
                    }
                    for key in MODEL_LABELS
                ],
                value=defaults,
                className="orca-model-checklist",
                inputClassName="orca-check-input",
                labelClassName="orca-model-option",
            ),
            _help("Fit one model for parameter recovery, or at least two models for a Bayes factor comparison."),
        ],
        className="orca-field",
    )


def inference_controls(prefix: str, *, donor_aware: bool = False) -> html.Div:
    lower, upper = (-1.5, 1.5)
    threshold = 0.6 if donor_aware else 0.5
    sigma_prior = 3.0
    donor_fields: list = []
    if donor_aware:
        donor_fields = [
            html.H4("Donor deviation prior scales"),
            html.P(
                "These hierarchical priors affect shrinkage and marginal likelihoods. Record them when reporting a Bayes factor analysis.",
                className="orca-help",
            ),
            html.Div(
                [
                    field(
                        "Mean log rate deviation τμ",
                        dcc.Input(id=f"{prefix}-donor-mean-scale", type="number", min=0.05, max=2, step=0.05, value=0.3),
                    ),
                    field(
                        "Heterogeneity log deviation τσ",
                        dcc.Input(id=f"{prefix}-donor-sigma-scale", type="number", min=0.05, max=2, step=0.05, value=0.3),
                    ),
                    field(
                        "Nonengaging logit deviation τφ",
                        dcc.Input(id=f"{prefix}-donor-zero-scale", type="number", min=0.05, max=3, step=0.05, value=1.0),
                    ),
                ],
                className="orca-form-grid three",
            ),
        ]

    return html.Div(
        [
            research_warning(),
            field(
                "Compute profile",
                dcc.Dropdown(
                    id=f"{prefix}-profile",
                    options=[
                        {"label": "Preview · quickest", "value": "preview"},
                        {"label": "Demo · more repeatability", "value": "demo"},
                        {"label": "Custom", "value": "custom"},
                    ],
                    value="preview",
                    clearable=False,
                ),
                "Preview is illustrative. Demo uses more particles and independent chains.",
            ),
            html.Details(
                [
                    html.Summary("Inference settings and what they mean"),
                    html.Div(
                        [
                            field(
                                "SMC particles per chain",
                                dcc.Input(id=f"{prefix}-particles", type="number", min=32, max=1000, step=32, value=64),
                                "More particles reduce Monte Carlo noise but increase runtime and memory use.",
                            ),
                            field(
                                "Independent chains",
                                dcc.Input(id=f"{prefix}-chains", type="number", min=1, max=2, step=1, value=1),
                                "Independent runs help assess repeatability.",
                            ),
                            field(
                                "CPU cores",
                                dcc.Input(id=f"{prefix}-cores", type="number", min=1, max=2, step=1, value=1),
                                "Cores can shorten runtime; they add no information.",
                            ),
                        ],
                        className="orca-form-grid three",
                    ),
                    field(
                        "Inference seed (optional)",
                        dcc.Input(
                            id=f"{prefix}-seed",
                            type="text",
                            placeholder="Blank = a new random run",
                            value="",
                        ),
                        "A fixed seed makes the computation reproducible; it does not improve inference.",
                    ),
                    html.Div(
                        [
                            field(
                                "Tempering threshold",
                                dcc.Slider(id=f"{prefix}-threshold", min=0.1, max=0.9, step=0.05, value=threshold, marks=None, tooltip={"placement": "bottom", "always_visible": True}),
                                "Higher values generally create more intermediate SMC stages.",
                            ),
                            field(
                                "Mutation correlation threshold",
                                dcc.Input(id=f"{prefix}-correlation", type="number", min=0.001, max=0.2, step=0.005, value=0.01),
                                "Smaller values generally request more particle mutation effort.",
                            ),
                        ],
                        className="orca-form-grid two",
                    ),
                    field(
                        "log10 rate prior bounds",
                        dcc.RangeSlider(id=f"{prefix}-prior-bounds", min=-6, max=3, step=0.5, value=[lower, upper], marks={value: str(value) for value in range(-6, 4, 3)}, tooltip={"placement": "bottom", "always_visible": True}),
                        "Uniform bounds on the base-10 logarithm of the event rate.",
                    ),
                    field(
                        "Heterogeneity prior scale",
                        dcc.Input(id=f"{prefix}-sigma-prior", type="number", min=0.1, max=10, step=0.1, value=sigma_prior),
                        "Scale of the half normal prior for continuous rate heterogeneity. Bayes factors can be sensitive to it.",
                    ),
                    *donor_fields,
                ],
                className="orca-details",
            ),
        ],
        className="orca-inference-controls",
    )


def settings_from_values(
    particles: object,
    chains: object,
    cores: object,
    seed_raw: object,
    threshold: object,
    correlation: object,
    prior_bounds: object,
    sigma_prior: object,
    *,
    donor_aware: bool = False,
    donor_scales: tuple[object, object, object] = (0.3, 0.3, 1.0),
) -> InferenceSettings:
    bounds = list(prior_bounds or [])
    if len(bounds) != 2:
        raise ValueError("Choose both lower and upper rate prior bounds.")
    parsed_chains = int(chains)
    return InferenceSettings(
        draws=int(particles),
        chains=parsed_chains,
        cores=min(int(cores), parsed_chains),
        seed=parse_optional_seed(seed_raw),
        threshold=float(threshold),
        correlation_threshold=float(correlation),
        lambda_prior_bounds=(float(bounds[0]), float(bounds[1])),
        p_prior_bounds=(1.0, 1.0),
        std_prior_factor=float(sigma_prior),
        donor_deviation_prior=tuple(float(value) for value in donor_scales),
    )


def table_records(frame: pd.DataFrame) -> list[dict]:
    clean = frame.replace({np.nan: None})
    return clean.to_dict("records")


def data_table(
    frame: pd.DataFrame,
    *,
    table_id: str | None = None,
    editable: bool = False,
    max_rows: int = 12,
) -> dag.AgGrid:
    kwargs = {}
    if table_id is not None:
        kwargs["id"] = table_id
    return dag.AgGrid(
        rowData=table_records(frame),
        columnDefs=[
            {
                "headerName": str(column).replace("_", " ").title(),
                "field": column,
                "editable": editable,
                "wrapText": True,
                "autoHeight": True,
            }
            for column in frame.columns
        ],
        defaultColDef={"sortable": True, "resizable": True, "minWidth": 120, "flex": 1},
        dashGridOptions={
            "pagination": len(frame) > max_rows,
            "paginationPageSize": max_rows,
            "paginationPageSizeSelector": False,
            "domLayout": "autoHeight",
        },
        className="ag-theme-quartz orca-data-grid",
        style={"width": "100%"},
        **kwargs,
    )


def _plot_layout(figure: go.Figure, *, x_title: str, y_title: str) -> go.Figure:
    figure.update_layout(
        template="none",
        paper_bgcolor=BOOK_SHEET,
        plot_bgcolor=BOOK_PAPER,
        font={"family": BOOK_SERIF, "color": BOOK_INK, "size": 13},
        margin={"l": 54, "r": 24, "t": 30, "b": 52},
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend={"orientation": "h", "y": 1.12, "x": 0, "bgcolor": "rgba(0,0,0,0)"},
        hoverlabel={"bgcolor": BOOK_SHEET, "bordercolor": BOOK_RULE, "font_family": BOOK_SERIF, "font_color": BOOK_INK},
    )
    figure.update_xaxes(showline=True, linewidth=1, linecolor=BOOK_RULE, gridcolor=BOOK_GRID, ticks="outside", tickcolor=BOOK_RULE, zeroline=False)
    figure.update_yaxes(showline=True, linewidth=1, linecolor=BOOK_RULE, gridcolor=BOOK_GRID, ticks="outside", tickcolor=BOOK_RULE, zeroline=False)
    return figure


def count_figure(frame: pd.DataFrame) -> go.Figure:
    frequency = frame["count"].value_counts().sort_index()
    figure = go.Figure(
        go.Bar(
            x=frequency.index.astype(int),
            y=frequency.values.astype(int),
            marker={"color": DONOR_TEAL, "line": {"color": BOOK_INK, "width": 1}},
            hovertemplate="Event count %{x}<br>Cells %{y}<extra></extra>",
        )
    )
    return _plot_layout(figure, x_title="Event count", y_title="Number of cells")


def rate_distribution_figure(
    rate_distribution: str,
    mu_lambda: float,
    sigma_lambda: float,
    phi_zero: float,
) -> go.Figure:
    """Plot the full population distribution of the cell rate λᵢ."""

    x, density = rate_distribution_curve(
        rate_distribution,
        mu_lambda,
        sigma_lambda,
    )
    label = RATE_DISTRIBUTION_LABELS[rate_distribution].split(" (")[0]
    zero_fraction = float(phi_zero)
    if not 0 <= zero_fraction <= 1:
        raise ValueError("φ₀ must be between zero and one")

    if rate_distribution == "fixed" or len(x) == 1:
        locations = [float(x[0])]
        masses = [1.0 - zero_fraction]
        colours = [DECISION_TEAL]
        labels = ["Engaging cells"]
        if zero_fraction > 0:
            locations.insert(0, 0.0)
            masses.insert(0, zero_fraction)
            colours.insert(0, DONOR_RUST)
            labels.insert(0, "Nonengaging cells")
        figure = go.Figure(
            go.Bar(
                x=locations,
                y=masses,
                width=[max(float(x[0]) * 0.08, 0.08)] * len(locations),
                marker={"color": colours, "line": {"color": BOOK_INK, "width": 1}},
                customdata=labels,
                hovertemplate="%{customdata}<br>λᵢ = %{x:.4g}<br>Population mass = %{y:.3f}<extra></extra>",
            )
        )
        figure = _plot_layout(
            figure,
            x_title="Cell-specific event rate λᵢ",
            y_title="Population probability mass",
        )
        figure.update_yaxes(range=[0, 1.05])
        return figure.update_layout(
            height=340,
            margin={"l": 70, "r": 24, "t": 30, "b": 58},
            showlegend=False,
        )

    population_density = (1.0 - zero_fraction) * density
    if zero_fraction > 0:
        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.28, 0.72],
        )
        bar_width = max(float(np.nanmax(x)) * 0.025, 0.05)
        figure.add_trace(
            go.Bar(
                x=[0.0],
                y=[zero_fraction],
                width=[bar_width],
                name="Point mass at zero",
                marker={"color": DONOR_RUST, "line": {"color": BOOK_INK, "width": 1}},
                hovertemplate="λᵢ = 0<br>Population mass φ₀ = %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        figure.add_annotation(
            x=0.0,
            y=zero_fraction,
            text=f"φ₀ = {zero_fraction:.2f}",
            showarrow=False,
            yshift=12,
            row=1,
            col=1,
        )
        figure.update_yaxes(title_text="Mass", range=[0, 1.05], row=1, col=1)
        density_row = 2
    else:
        figure = make_subplots(rows=1, cols=1)
        density_row = 1

    figure.add_trace(
        go.Scatter(
            x=x,
            y=population_density,
            mode="lines",
            fill="tozeroy",
            name=f"{label} population density",
            line={"color": DECISION_TEAL, "width": 3},
            fillcolor="rgba(28, 120, 117, 0.20)",
            hovertemplate="λᵢ = %{x:.4g}<br>Population density = %{y:.4g}<extra></extra>",
        ),
        row=density_row,
        col=1,
    )
    figure.update_yaxes(title_text="Population density", row=density_row, col=1)
    figure.update_xaxes(title_text="Cell-specific event rate λᵢ", row=density_row, col=1)

    figure.update_layout(
        template="none",
        paper_bgcolor=BOOK_SHEET,
        plot_bgcolor=BOOK_PAPER,
        font={"family": BOOK_SERIF, "color": BOOK_INK, "size": 13},
        margin={"l": 72, "r": 24, "t": 28, "b": 58},
        height=390 if zero_fraction > 0 else 340,
        showlegend=False,
        hoverlabel={
            "bgcolor": BOOK_SHEET,
            "bordercolor": BOOK_RULE,
            "font_family": BOOK_SERIF,
            "font_color": BOOK_INK,
        },
    )
    figure.update_xaxes(
        showline=True,
        linewidth=1,
        linecolor=BOOK_RULE,
        gridcolor=BOOK_GRID,
        ticks="outside",
        tickcolor=BOOK_RULE,
        zeroline=False,
        automargin=True,
    )
    figure.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor=BOOK_RULE,
        gridcolor=BOOK_GRID,
        ticks="outside",
        tickcolor=BOOK_RULE,
        zeroline=False,
        automargin=True,
    )
    return figure


def data_overview(frame: pd.DataFrame, *, donor_aware: bool = False) -> html.Div:
    metric_items = [
        metric("Cells", f"{len(frame):,}"),
        metric("Mean count", f"{frame['count'].mean():.2f}"),
        metric("Cells with zero counts", f"{(frame['count'] == 0).mean():.1%}"),
    ]
    if donor_aware:
        metric_items.append(metric("Donors", f"{frame['donor_id'].nunique():,}"))
    children: list = [
        html.Div(metric_items, className="orca-metrics"),
        dcc.Graph(figure=count_figure(frame), config={"displaylogo": False, "responsive": True}, className="orca-plot"),
        data_table(frame, max_rows=10),
    ]
    if donor_aware:
        donor_table = (
            frame.groupby("donor_id", sort=True)["count"]
            .agg(cells="size", mean_count="mean", median_count="median", zero_fraction=lambda values: (values == 0).mean())
            .reset_index()
        )
        donor_table["mean_count"] = donor_table["mean_count"].round(3)
        donor_table["median_count"] = donor_table["median_count"].round(3)
        donor_table["zero_fraction"] = donor_table["zero_fraction"].map(lambda value: f"{value:.1%}")
        children.extend([html.H4("Input summary by donor"), data_table(donor_table)])
        small = donor_table.loc[donor_table["cells"] < 20, "donor_id"]
        if not small.empty:
            children.append(
                note(
                    "Small donor groups",
                    "Estimates for individual donors may be weakly identified for: " + ", ".join(map(str, small.tolist())),
                    tone="amber",
                )
            )
    return html.Div(children, className="orca-overview")


def posterior_figure(result: InferenceResult, variable: str, truth: Mapping[str, object] | None = None) -> go.Figure:
    values = result.idata.posterior[variable]
    figure = go.Figure()
    donor_dim = next((dim for dim in values.dims if dim not in {"chain", "draw"}), None)
    if donor_dim is None:
        figure.add_trace(
            go.Histogram(
                x=np.asarray(values).reshape(-1),
                histnorm="probability density",
                nbinsx=35,
                name=variable,
                marker_color=MODEL_COLOURS[result.model_key],
                marker_line_color=BOOK_INK,
                marker_line_width=1,
                opacity=0.82,
                hovertemplate="%{x:.4g}<br>density %{y:.4g}<extra></extra>",
            )
        )
    else:
        coordinate_values = list(values.coords[donor_dim].values)
        for index, coordinate in enumerate(coordinate_values):
            label = result.donor_labels[index] if index < len(result.donor_labels) else str(coordinate)
            samples = np.asarray(values.isel({donor_dim: index})).reshape(-1)
            figure.add_trace(
                go.Histogram(
                    x=samples,
                    histnorm="probability density",
                    nbinsx=28,
                    name=str(label),
                    marker_color=DONOR_COLOURS[index % len(DONOR_COLOURS)],
                    marker_line_color=BOOK_INK,
                    marker_line_width=1,
                    opacity=0.52,
                )
            )
        figure.update_layout(barmode="overlay")
    truth_key = {"lambda": "mu_lambda", "mu_lambda": "mu_lambda", "sigma_lambda": "sigma_lambda", "p_zero": "p_zero"}.get(variable)
    if truth is not None and truth_key in truth:
        figure.add_vline(x=float(truth[truth_key]), line_color=DONOR_RUST, line_width=2, annotation_text="Ground truth")
    return _plot_layout(
        figure,
        x_title=PARAMETER_LABELS.get(variable, variable),
        y_title="Posterior density",
    )


def _recovery_table(summary: pd.DataFrame, truth: Mapping[str, object]) -> pd.DataFrame:
    parameter_truth_keys = {"lambda": "mu_lambda", "mu_lambda": "mu_lambda", "sigma_lambda": "sigma_lambda", "p_zero": "p_zero"}
    hdi_columns = [column for column in summary.columns if str(column).startswith("hdi_")]
    rows: list[dict] = []
    if len(hdi_columns) < 2:
        return pd.DataFrame()
    low, high = hdi_columns[:2]
    for row in summary.to_dict("records"):
        truth_key = parameter_truth_keys.get(str(row["parameter"]))
        if truth_key is None or truth_key not in truth:
            continue
        truth_value = float(truth[truth_key])
        rows.append(
            {
                "Fitted model": row["model"],
                "Parameter": PARAMETER_LABELS.get(str(row["parameter"]), row["parameter"]),
                "Ground truth": truth_value,
                "Posterior mean": row.get("mean"),
                "95% HDI lower": row[low],
                "95% HDI upper": row[high],
                "Truth in 95% HDI": float(row[low]) <= truth_value <= float(row[high]),
            }
        )
    return pd.DataFrame(rows)


def render_results(
    results: Mapping[str, InferenceResult],
    *,
    data: pd.DataFrame,
    observation_time: float,
    settings: InferenceSettings,
    truth: Mapping[str, object] | None = None,
    download_name: str = "orca_results.zip",
) -> tuple[html.Div, html.A]:
    evidence = evidence_table(results).round(5)
    summary = summary_table(results).round(5)
    display_summary = summary.copy()
    if not display_summary.empty:
        display_summary["parameter"] = display_summary["parameter"].map(
            lambda value: PARAMETER_LABELS.get(str(value), value)
        )
    best = str(evidence.iloc[0]["model"]) if not evidence.empty else "Unavailable"
    cards = html.Div(
        [
            metric("Best supported model", best, accent="teal"),
            metric("Models fitted", str(len(results)), accent="navy"),
            metric("Cells analysed", f"{len(data):,}"),
        ],
        className="orca-metrics",
    )

    tabs: list[dcc.Tab] = []
    for key, result in results.items():
        spec = MODEL_SPECS[key]
        requested = spec.donor_parameters if result.donor_aware else spec.count_parameters
        available = [name for name in requested if name in result.idata.posterior.data_vars]
        plots = [
            html.Div(
                [html.H4(PARAMETER_LABELS.get(variable, variable)), dcc.Graph(figure=posterior_figure(result, variable, truth), config={"displaylogo": False, "responsive": True})],
                className="orca-posterior-panel",
            )
            for variable in available
        ]
        tabs.append(dcc.Tab(label=MODEL_LABELS[key], children=html.Div(plots, className="orca-posterior-grid"), className="orca-tab", selected_className="orca-tab selected"))

    recovery = _recovery_table(summary, truth) if truth is not None else pd.DataFrame()
    recovery_component = (
        html.Div([html.H3("Ground truth recovery check"), data_table(recovery), html.P("This table compares the known generating values with the posterior interval for this dataset.", className="orca-help")], className="orca-result-section")
        if not recovery.empty
        else html.Div()
    )
    generator_note = html.Div()
    if truth is not None and truth.get("is_paper_model") is False:
        generator_note = note(
            "Exploratory rate distribution",
            "These data use "
            f"{truth.get('rate_distribution_label', 'an alternative distribution')} "
            "for the engaging-cell rates. The fitted heterogeneous models assume "
            "Gamma rates, so no fitted model is the exact generator.",
            tone="amber",
        )
    archive = build_results_zip(results, data, observation_time, settings, truth=dict(truth) if truth is not None else None)
    encoded = base64.b64encode(archive).decode("ascii")
    download = html.A(
        "Download results and configuration",
        href=f"data:application/zip;base64,{encoded}",
        download=download_name,
        className="orca-button primary download",
    )
    content = html.Div(
        [
            note("Inference complete", "All selected models completed successfully.", tone="teal"),
            generator_note,
            cards,
            html.Div([html.H3("Model comparison"), data_table(evidence), html.P("The best supported fitted model has log10 BF versus best equal to zero. Repeat small SMC runs across seeds and particle counts before drawing conclusions.", className="orca-help")], className="orca-result-section"),
            html.Div([html.H3("Posterior summaries"), data_table(display_summary, max_rows=15)], className="orca-result-section"),
            recovery_component,
            html.Div([html.H3("Posterior distributions"), dcc.Tabs(tabs, className="orca-tabs")], className="orca-result-section"),
        ],
        className="orca-results",
    )
    return content, download


def _download_link_bytes(
    content: bytes,
    filename: str,
    label: str,
    mime_type: str,
    *,
    primary: bool = False,
) -> html.A:
    encoded = base64.b64encode(content).decode("ascii")
    kind = "primary" if primary else "secondary"
    return html.A(
        label,
        href=f"data:{mime_type};base64,{encoded}",
        download=filename,
        className=f"orca-button {kind} download",
    )


def render_validation_results(
    results: Mapping[str, InferenceResult],
    *,
    data: pd.DataFrame,
    observation_time: float,
    settings: InferenceSettings,
    truth: Mapping[str, object],
    download_name: str = "orca_synthetic_validation.zip",
) -> tuple[html.Div, html.Div]:
    """Render the notebook-style synthetic-validation result release."""

    evidence = evidence_table(results).round(6)
    summary = summary_table(results).round(6)
    recovery = _recovery_table(summary, truth)
    posterior_draws = posterior_draw_table(results)
    plot_draws = posterior_draw_table(results, max_draws_per_model=5_000)
    fitted_models = list(results)
    joint_figure = joint_posterior_figure_from_draws(
        plot_draws,
        fitted_models,
        truth,
    )
    posterior_payload = posterior_store_payload(plot_draws, fitted_models, truth)
    bayes_figure = bayes_factor_figure(results, truth)
    artifacts = validation_figure_artifacts(results, truth)

    archive = build_results_zip(
        results,
        data,
        observation_time,
        settings,
        truth=dict(truth),
        artifacts=artifacts,
    )

    plot_config = {
        "displaylogo": False,
        "responsive": True,
        "toImageButtonOptions": {
            "format": "png",
            "scale": 2,
        },
    }
    joint_exports = html.Div(
        [
            _download_link_bytes(
                artifacts["figures/joint_posterior.png"],
                "orca_joint_posterior.png",
                "PNG",
                "image/png",
            ),
            _download_link_bytes(
                artifacts["figures/joint_posterior.pdf"],
                "orca_joint_posterior.pdf",
                "PDF",
                "application/pdf",
            ),
            csv_download_link(
                posterior_draws,
                "orca_posterior_samples.csv",
                "CSV",
            ),
        ],
        className="orca-figure-exports",
        **{"aria-label": "Export joint posterior figure"},
    )
    bayes_exports = html.Div(
        [
            _download_link_bytes(
                artifacts["figures/bayes_factors.png"],
                "orca_bayes_factors.png",
                "PNG",
                "image/png",
            ),
            _download_link_bytes(
                artifacts["figures/bayes_factors.pdf"],
                "orca_bayes_factors.pdf",
                "PDF",
                "application/pdf",
            ),
            csv_download_link(
                evidence,
                "orca_model_evidence.csv",
                "CSV",
            ),
        ],
        className="orca-figure-exports",
        **{"aria-label": "Export Bayes factor figure"},
    )

    content = html.Div(
        [
            note(
                "Inference complete",
                "The figures use paired posterior draws from the fitted PyMC models. Dashed rust lines and stars mark the known generating values.",
                tone="teal",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("Primary result", className="orca-section-label"),
                                    html.H3("Joint posterior distributions"),
                                ]
                            ),
                            joint_exports,
                        ],
                        className="orca-result-heading",
                    ),
                    html.P(
                        "Diagonal panels show marginal posterior densities and 95% HDIs. Lower panels retain dependence between parameters. For the homogeneous and zero inflated models, the mean-rate axis contains the shared rate λ; for the Gamma models it contains μλ.",
                        className="orca-help",
                    ),
                    dcc.Store(
                        id="synthetic-posterior-data",
                        data=posterior_payload,
                        storage_type="memory",
                    ),
                    html.Div(
                        [
                            html.Strong("Models shown in this plot"),
                            dcc.Checklist(
                                id="synthetic-posterior-model-filter",
                                options=[
                                    {
                                        "label": MODEL_LABELS[model_key],
                                        "value": model_key,
                                    }
                                    for model_key in fitted_models
                                ],
                                value=fitted_models,
                                inline=True,
                                className="orca-posterior-model-options",
                                inputClassName="orca-check-input",
                                labelClassName="orca-posterior-model-option",
                            ),
                            html.P(
                                f"{len(fitted_models)} model{'s' if len(fitted_models) != 1 else ''} shown · the grid contains every parameter represented by the selection.",
                                id="synthetic-posterior-selection-summary",
                                className="orca-help",
                                role="status",
                                **{"aria-live": "polite"},
                            ),
                        ],
                        className="orca-posterior-filter",
                        role="group",
                        **{"aria-label": "Choose fitted models for the joint posterior plot"},
                    ),
                    html.Div(
                        dcc.Graph(
                            id="synthetic-posterior-figure",
                            figure=joint_figure,
                            config={
                                **plot_config,
                                "toImageButtonOptions": {
                                    **plot_config["toImageButtonOptions"],
                                    "filename": "orca_joint_posterior",
                                    "width": 1400,
                                    "height": 1400,
                                },
                            },
                            responsive=True,
                            className="orca-joint-posterior-plot",
                            style={"height": f"{int(joint_figure.layout.height)}px"},
                        ),
                        className="orca-joint-plot-scroll",
                    ),
                    html.P(
                        "The PNG, PDF and CSV buttons above export all fitted models. The Plotly camera button exports the current on-screen selection.",
                        className="orca-help orca-export-scope",
                    ),
                ],
                className="orca-result-section orca-figure-result",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("Model evidence", className="orca-section-label"),
                                    html.H3("Bayes factors"),
                                ]
                            ),
                            bayes_exports,
                        ],
                        className="orca-result-heading",
                    ),
                    html.P(
                        "Bars use the untransformed linear log₁₀ BF(best model / fitted model) scale computed from the SMC log marginal likelihoods. The highest-evidence model is labelled Best model and sits at zero by definition. The background boundaries are exactly log₁₀(3) ≈ 0.477, 1 and 2, corresponding to Bayes factors of 3, 10 and 100.",
                        className="orca-help",
                    ),
                    dcc.Graph(
                        figure=bayes_figure,
                        config={
                            **plot_config,
                            "toImageButtonOptions": {
                                **plot_config["toImageButtonOptions"],
                                "filename": "orca_bayes_factors",
                                "width": 1100,
                                "height": 650,
                            },
                        },
                        responsive=True,
                        className="orca-bayes-factor-plot",
                        style={"height": "430px"},
                    ),
                ],
                className="orca-result-section orca-figure-result",
            ),
        ],
        className="orca-results orca-validation-results",
    )

    csv_links: list = [
        csv_download_link(evidence, "orca_model_evidence.csv", "Model evidence CSV"),
        csv_download_link(summary, "orca_posterior_summary.csv", "Posterior summary CSV"),
        csv_download_link(posterior_draws, "orca_posterior_samples.csv", "Posterior samples CSV"),
    ]
    if not recovery.empty:
        csv_links.append(
            csv_download_link(recovery, "orca_ground_truth_recovery.csv", "Ground truth recovery CSV")
        )
    downloads = html.Div(
        [
            _download_link_bytes(
                archive,
                download_name,
                "Download analysis and InferenceData",
                "application/zip",
                primary=True,
            ),
            html.P(
                "The ZIP contains one ArviZ InferenceData .nc file per fitted model, all CSV tables, both figures as PNG and PDF, the input data, exact settings and software versions.",
                className="orca-help",
            ),
            html.Details(
                [
                    html.Summary("Download individual CSV tables"),
                    html.Div(csv_links, className="orca-download-grid"),
                ],
                className="orca-details",
            ),
            html.Details(
                [
                    html.Summary("Open an InferenceData file in Python"),
                    html.Pre(
                        "import arviz as az\n\n"
                        "idata = az.from_netcdf('posterior_hetero3_smc.nc')\n"
                        "print(az.summary(idata))\n"
                        "draws = idata.posterior.to_dataframe()\n"
                        "az.plot_pair(\n"
                        "    idata,\n"
                        "    var_names=['mu_lambda', 'sigma_lambda', 'p_zero'],\n"
                        "    kind='kde',\n"
                        "    marginals=True,\n"
                        ")",
                        className="orca-code-block",
                    ),
                    html.P(
                        "Unzip the complete analysis first, then replace hetero3 with the model key shown in the filename.",
                        className="orca-help",
                    ),
                ],
                className="orca-details",
            ),
        ],
        className="orca-analysis-downloads",
    )
    return content, downloads


def read_uploaded_csv(contents: str | None) -> pd.DataFrame:
    if not contents:
        raise ValueError("Choose a CSV file first.")
    try:
        _, encoded = contents.split(",", 1)
        payload = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError("The uploaded file could not be decoded.") from exc
    if len(payload) > 1_000_000:
        raise ValueError("The demo accepts CSV files up to 1 MB.")
    try:
        raw = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The CSV must use UTF-8 text encoding.") from exc
    try:
        return pd.read_csv(StringIO(raw))
    except Exception as exc:
        raise ValueError(f"Could not read the CSV file: {exc}") from exc


def normalize_uploaded_frame(raw: pd.DataFrame, *, donor_aware: bool) -> tuple[pd.DataFrame, str]:
    if raw.empty:
        raise ValueError("The uploaded CSV is empty.")
    columns = list(raw.columns)
    required = ["cell_id", "donor_id", "count"] if donor_aware else ["cell_id", "count"]
    if all(column in columns for column in required):
        return raw.loc[:, required].copy(), "Recognised the standard Orca column names."
    needed = 3 if donor_aware else 2
    if len(columns) < needed:
        raise ValueError(f"The CSV needs at least {needed} columns: {', '.join(required)}.")
    if donor_aware:
        mapped = pd.DataFrame({"cell_id": raw.iloc[:, 0], "donor_id": raw.iloc[:, 1], "count": raw.iloc[:, 2]})
        message = f"Mapped {columns[0]!r} → cell_id, {columns[1]!r} → donor_id and {columns[2]!r} → count."
    else:
        mapped = pd.DataFrame({"cell_id": raw.iloc[:, 0], "count": raw.iloc[:, 1]})
        message = f"Mapped {columns[0]!r} → cell_id and {columns[1]!r} → count."
    return mapped, message


def csv_download_link(frame: pd.DataFrame, filename: str, label: str) -> html.A:
    encoded = base64.b64encode(frame.to_csv(index=False).encode("utf-8")).decode("ascii")
    return html.A(label, href=f"data:text/csv;base64,{encoded}", download=filename, className="orca-button secondary download")
