"""Interactive donor-aware condition contrasts for the Dash interface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import arviz as az
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, dcc, html

from webapp.analysis_ui import data_table
from webapp.core.condition_inference import ConditionResults
from webapp.core.inference import MODEL_SPECS
from webapp.palette import DONOR_GOLD, DONOR_RUST, DONOR_SAGE, DONOR_TEAL
from webapp.reporting import GRID, INK, PAPER, RULE, SERIF, SHEET
from webapp.ui import note


DONOR_PARAMETERS = (
    "mu_lambda_donor",
    "sigma_lambda_donor",
    "phi_0_donor",
)
ABSOLUTE_COLUMNS = {
    "mu_lambda_donor": "delta_mu_lambda",
    "sigma_lambda_donor": "delta_sigma_lambda",
    "phi_0_donor": "delta_phi_0",
}
PERCENT_COLUMNS = {
    "mu_lambda_donor": "percent_delta_mu_lambda",
    "sigma_lambda_donor": "percent_delta_sigma_lambda",
    "phi_0_donor": "percent_delta_phi_0",
}
CONTRAST_LABELS = {
    "delta_mu_lambda": "Difference in mean event rate, Δμλ,d",
    "delta_sigma_lambda": "Difference in continuous cell-to-cell heterogeneity, Δσλ,d",
    "delta_phi_0": "Difference in nonengaging fraction, Δφ₀,d",
    "percent_delta_mu_lambda": "Relative difference in mean event rate, %",
    "percent_delta_sigma_lambda": "Relative difference in continuous cell-to-cell heterogeneity, %",
    "percent_delta_phi_0": "Relative difference in nonengaging fraction, %",
}
CONTRAST_AXIS_LABELS = {
    "delta_mu_lambda": "Δμλ,d",
    "delta_sigma_lambda": "Δσλ,d",
    "delta_phi_0": "Δφ₀,d",
    "percent_delta_mu_lambda": "Δμλ,d (%)",
    "percent_delta_sigma_lambda": "Δσλ,d (%)",
    "percent_delta_phi_0": "Δφ₀,d (%)",
}
DONOR_COLOURS = (DONOR_TEAL, DONOR_SAGE, DONOR_GOLD, DONOR_RUST)


def _donor_dimension(values) -> str:
    dimensions = [
        dimension
        for dimension in values.dims
        if dimension not in {"chain", "draw"}
    ]
    if len(dimensions) != 1:
        raise ValueError("donor posterior variables must contain one donor dimension")
    return dimensions[0]


def donor_contrast_payload(
    results: ConditionResults,
    *,
    max_draws_per_fit: int = 2_000,
) -> dict[str, object]:
    """Serialise paired donor draws needed for browser-selected contrasts."""

    if max_draws_per_fit <= 0:
        raise ValueError("max_draws_per_fit must be positive")
    condition_order = list(results)
    if not condition_order:
        raise ValueError("at least one completed condition inference is required")
    model_order = list(next(iter(results.values())))
    models: dict[str, object] = {}
    for model_key in model_order:
        condition_payloads: dict[str, object] = {}
        for condition in condition_order:
            if model_key not in results[condition]:
                continue
            result = results[condition][model_key]
            posterior = result.idata.posterior
            parameters = [
                parameter
                for parameter in DONOR_PARAMETERS
                if parameter in posterior.data_vars
            ]
            if not parameters:
                continue
            first = posterior[parameters[0]]
            donor_dimension = _donor_dimension(first)
            n_chain = int(first.sizes["chain"])
            n_draw = int(first.sizes["draw"])
            total_draws = n_chain * n_draw
            selected = (
                np.linspace(0, total_draws - 1, max_draws_per_fit, dtype=int)
                if total_draws > max_draws_per_fit
                else np.arange(total_draws, dtype=int)
            )
            n_donors = int(first.sizes[donor_dimension])
            labels = list(result.donor_labels)
            if len(labels) != n_donors:
                labels = [str(value) for value in first.coords[donor_dimension].values]
            donor_draws: dict[str, list[list[float]]] = {}
            for donor_index, label in enumerate(labels):
                vectors: list[np.ndarray] = []
                for parameter in parameters:
                    values = posterior[parameter]
                    parameter_donor_dimension = _donor_dimension(values)
                    flattened = np.asarray(
                        values.transpose(
                            "chain",
                            "draw",
                            parameter_donor_dimension,
                        )
                    ).reshape(total_draws, n_donors)
                    vectors.append(flattened[selected, donor_index])
                matrix = np.column_stack(vectors).astype(float)
                matrix = matrix[np.isfinite(matrix).all(axis=1)]
                donor_draws[str(label)] = matrix.tolist()
            condition_payloads[str(condition)] = {
                "parameters": parameters,
                "donors": labels,
                "draws": donor_draws,
            }
        if condition_payloads:
            models[model_key] = {"conditions": condition_payloads}
    return {
        "schema_version": 1,
        "condition_order": condition_order,
        "model_order": [key for key in model_order if key in models],
        "models": models,
    }


def contrast_from_payload(
    payload: Mapping[str, object],
    *,
    model_key: str,
    comparison: str,
    reference: str,
    scale: str = "absolute",
    max_exact_pairs: int = 500_000,
    approximate_pairs: int = 100_000,
    random_seed: int = 307,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare independent condition posteriors using all or sampled pairs."""

    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("unsupported donor contrast data format")
    if comparison == reference:
        raise ValueError("choose two different experimental conditions")
    models = payload.get("models", {})
    if not isinstance(models, Mapping) or model_key not in models:
        raise ValueError("choose a donor aware model included in inference")
    model = models[model_key]
    if not isinstance(model, Mapping):
        raise ValueError("donor posterior data are unavailable")
    conditions = model.get("conditions", {})
    if not isinstance(conditions, Mapping):
        raise ValueError("donor posterior data are unavailable")
    if comparison not in conditions or reference not in conditions:
        raise ValueError("the selected conditions do not have inference results for this model")
    comparison_data = conditions[comparison]
    reference_data = conditions[reference]
    if not isinstance(comparison_data, Mapping) or not isinstance(reference_data, Mapping):
        raise ValueError("donor posterior data are unavailable")
    comparison_parameters = [str(value) for value in comparison_data.get("parameters", [])]
    reference_parameters = [str(value) for value in reference_data.get("parameters", [])]
    parameters = [
        parameter
        for parameter in DONOR_PARAMETERS
        if parameter in comparison_parameters and parameter in reference_parameters
    ]
    if not parameters:
        raise ValueError("the selected model has no comparable donor parameters")
    comparison_draws = comparison_data.get("draws", {})
    reference_draws = reference_data.get("draws", {})
    if not isinstance(comparison_draws, Mapping) or not isinstance(reference_draws, Mapping):
        raise ValueError("donor posterior draws are unavailable")
    common_donors = [
        str(label)
        for label in comparison_data.get("donors", [])
        if str(label) in reference_draws
    ]
    if not common_donors:
        raise ValueError("the two conditions have no donor labels in common")

    try:
        from webapp.donor_reporting import cartesian_contrast_draws
    except ImportError as exc:  # pragma: no cover - defensive packaging guard
        raise RuntimeError("donor contrast helpers are unavailable") from exc

    output_columns = ABSOLUTE_COLUMNS if scale == "absolute" else PERCENT_COLUMNS
    if scale not in {"absolute", "percent_of_reference_mean"}:
        raise ValueError("choose absolute or relative differences")
    frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, object]] = []
    for donor_index, donor in enumerate(common_donors):
        comparison_matrix = np.asarray(comparison_draws[donor], dtype=float)
        reference_matrix = np.asarray(reference_draws[donor], dtype=float)
        # Stored columns may differ between models. Select the shared columns
        # in canonical scientific order before constructing particle pairs.
        comparison_indices = [comparison_parameters.index(name) for name in parameters]
        reference_indices = [reference_parameters.index(name) for name in parameters]
        comparison_matrix = comparison_matrix[:, comparison_indices]
        reference_matrix = reference_matrix[:, reference_indices]
        contrast_scale = (
            "absolute" if scale == "absolute" else "percent_of_control_mean"
        )
        values, metadata = cartesian_contrast_draws(
            comparison_matrix,
            reference_matrix,
            scale=contrast_scale,
            max_exact_pairs=max_exact_pairs,
            approximate_pairs=approximate_pairs,
            random_seed=random_seed + donor_index,
        )
        columns = [output_columns[parameter] for parameter in parameters]
        frame = pd.DataFrame(values, columns=columns)
        frame.insert(0, "donor_id", donor)
        frames.append(frame)
        metadata_rows.append({"donor_id": donor, **metadata})
    return pd.concat(frames, ignore_index=True), {
        "comparison": comparison,
        "reference": reference,
        "model_key": model_key,
        "parameters": parameters,
        "scale": scale,
        "donors": metadata_rows,
    }


def contrast_summary(frame: pd.DataFrame, *, hdi_prob: float = 0.95) -> pd.DataFrame:
    value_columns = [column for column in frame.columns if column != "donor_id"]
    rows: list[dict[str, object]] = []
    for donor, group in frame.groupby("donor_id", sort=False):
        for parameter in value_columns:
            values = pd.to_numeric(group[parameter], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            interval = np.asarray(az.hdi(values, hdi_prob=hdi_prob), dtype=float)
            rows.append(
                {
                    "Donor": donor,
                    "Parameter": CONTRAST_LABELS.get(parameter, parameter),
                    "Posterior median": float(np.median(values)),
                    "95% HDI lower": float(interval[0]),
                    "95% HDI upper": float(interval[1]),
                    "P(difference > 0)": float(np.mean(values > 0)),
                    "Particle pairs": len(values),
                }
            )
    return pd.DataFrame(rows)


def contrast_figure(frame: pd.DataFrame, *, title: str) -> go.Figure:
    parameters = [column for column in frame.columns if column != "donor_id"]
    donors = list(dict.fromkeys(frame["donor_id"].astype(str)))
    size = len(parameters)
    figure = make_subplots(
        rows=size,
        cols=size,
        horizontal_spacing=0.07,
        vertical_spacing=0.08,
    )
    for row_index, row_parameter in enumerate(parameters, start=1):
        for column_index, column_parameter in enumerate(parameters, start=1):
            if column_index > row_index:
                figure.update_xaxes(visible=False, row=row_index, col=column_index)
                figure.update_yaxes(visible=False, row=row_index, col=column_index)
                continue
            for donor_index, donor in enumerate(donors):
                subset = frame.loc[frame["donor_id"].astype(str) == donor]
                colour = DONOR_COLOURS[donor_index % len(DONOR_COLOURS)]
                if row_index == column_index:
                    figure.add_trace(
                        go.Histogram(
                            x=subset[row_parameter],
                            histnorm="probability density",
                            nbinsx=32,
                            opacity=0.23,
                            marker={"color": colour},
                            name=donor,
                            legendgroup=donor,
                            showlegend=row_index == 1,
                        ),
                        row=row_index,
                        col=column_index,
                    )
                    figure.add_vline(
                        x=0,
                        line={"color": RULE, "width": 1.2, "dash": "dash"},
                        row=row_index,
                        col=column_index,
                    )
                else:
                    joint = subset[[column_parameter, row_parameter]].dropna()
                    figure.add_trace(
                        go.Histogram2dContour(
                            x=joint[column_parameter],
                            y=joint[row_parameter],
                            ncontours=5,
                            contours={"coloring": "none", "showlabels": False},
                            line={"color": colour, "width": 2},
                            name=donor,
                            legendgroup=donor,
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row_index,
                        col=column_index,
                    )
                    figure.add_vline(
                        x=0,
                        line={"color": RULE, "width": 1, "dash": "dash"},
                        row=row_index,
                        col=column_index,
                    )
                    figure.add_hline(
                        y=0,
                        line={"color": RULE, "width": 1, "dash": "dash"},
                        row=row_index,
                        col=column_index,
                    )
            if row_index == size:
                figure.update_xaxes(
                    title_text=CONTRAST_AXIS_LABELS.get(column_parameter, column_parameter),
                    row=row_index,
                    col=column_index,
                )
            if column_index == 1:
                figure.update_yaxes(
                    title_text=(
                        "Posterior density"
                        if row_index == column_index
                        else CONTRAST_AXIS_LABELS.get(row_parameter, row_parameter)
                    ),
                    row=row_index,
                    col=column_index,
                )
    figure.update_layout(
        template="none",
        barmode="overlay",
        height=max(430, 320 * size),
        title=title,
        paper_bgcolor=SHEET,
        plot_bgcolor=PAPER,
        font={"family": SERIF, "color": INK, "size": 13},
        margin={"l": 100, "r": 32, "t": 105, "b": 96},
        legend={"orientation": "h", "x": 0, "y": 1.05},
    )
    figure.update_xaxes(gridcolor=GRID, automargin=True, zeroline=False)
    figure.update_yaxes(gridcolor=GRID, automargin=True, zeroline=False)
    return figure


def donor_contrast_section(results: ConditionResults, *, prefix: str) -> html.Section:
    payload = donor_contrast_payload(results)
    conditions = list(payload["condition_order"])
    model_order = list(payload["model_order"])
    if len(conditions) < 2:
        return html.Section(
            [
                html.Span("Condition comparison", className="barracuda-section-label"),
                html.H3("Compare any two experimental conditions"),
                note(
                    "Two conditions are required",
                    "Run inference for at least two experimental conditions to compare their donor posterior distributions.",
                    tone="navy",
                ),
            ],
            className="barracuda-result-section barracuda-donor-contrast-section",
        )
    options = [
        {"label": MODEL_SPECS[key].label, "value": key}
        for key in model_order
    ]
    default_reference = conditions[0]
    default_comparison = conditions[1] if len(conditions) > 1 else conditions[0]
    return html.Section(
        [
            html.Span("Condition comparison", className="barracuda-section-label"),
            html.H3("Compare any two experimental conditions"),
            html.P(
                "The comparison uses comparison minus reference for complete posterior particle distributions. Independent conditions do not share chain or draw positions. BARRACUDA uses every Cartesian particle pair when feasible and a reproducible uniform sample of independent pairs for larger runs. It does not reduce the comparison to a difference between two posterior means.",
                className="barracuda-help",
            ),
            dcc.Store(id=f"{prefix}-contrast-data", data=payload),
            html.Div(
                [
                    html.Label(
                        [
                            html.Span("Candidate model", className="barracuda-field-label"),
                            dcc.Dropdown(
                                id=f"{prefix}-contrast-model",
                                options=options,
                                value=model_order[0] if model_order else None,
                                clearable=False,
                            ),
                        ],
                        className="barracuda-field",
                    ),
                    html.Label(
                        [
                            html.Span("Reference condition", className="barracuda-field-label"),
                            dcc.Dropdown(
                                id=f"{prefix}-contrast-reference",
                                options=[{"label": value, "value": value} for value in conditions],
                                value=default_reference,
                                clearable=False,
                            ),
                        ],
                        className="barracuda-field",
                    ),
                    html.Label(
                        [
                            html.Span("Comparison condition", className="barracuda-field-label"),
                            dcc.Dropdown(
                                id=f"{prefix}-contrast-comparison",
                                options=[{"label": value, "value": value} for value in conditions],
                                value=default_comparison,
                                clearable=False,
                            ),
                        ],
                        className="barracuda-field",
                    ),
                    html.Label(
                        [
                            html.Span("Difference scale", className="barracuda-field-label"),
                            dcc.Dropdown(
                                id=f"{prefix}-contrast-scale",
                                options=[
                                    {"label": "Absolute difference", "value": "absolute"},
                                    {"label": "Percent of reference posterior mean", "value": "percent_of_reference_mean"},
                                ],
                                value="absolute",
                                clearable=False,
                            ),
                        ],
                        className="barracuda-field",
                    ),
                ],
                className="barracuda-form-grid two",
            ),
            html.Div(
                id=f"{prefix}-contrast-rule",
                role="status",
                **{"aria-live": "polite"},
            ),
            html.Div(
                dcc.Graph(
                    id=f"{prefix}-contrast-figure",
                    figure=go.Figure(),
                    config={"displaylogo": False, "responsive": True},
                    responsive=True,
                    className="barracuda-joint-posterior-plot",
                    style={"height": "430px"},
                ),
                className="barracuda-joint-plot-scroll",
            ),
            html.Div(id=f"{prefix}-contrast-summary"),
        ],
        className="barracuda-result-section barracuda-donor-contrast-section",
    )


def register_donor_contrast_callbacks(app, *, prefix: str) -> None:
    @app.callback(
        Output(f"{prefix}-contrast-figure", "figure"),
        Output(f"{prefix}-contrast-figure", "style"),
        Output(f"{prefix}-contrast-summary", "children"),
        Output(f"{prefix}-contrast-rule", "children"),
        Input(f"{prefix}-contrast-model", "value"),
        Input(f"{prefix}-contrast-reference", "value"),
        Input(f"{prefix}-contrast-comparison", "value"),
        Input(f"{prefix}-contrast-scale", "value"),
        State(f"{prefix}-contrast-data", "data"),
    )
    def update_condition_contrast(model_key, reference, comparison, scale, payload):
        try:
            frame, metadata = contrast_from_payload(
                payload or {},
                model_key=str(model_key),
                comparison=str(comparison),
                reference=str(reference),
                scale=str(scale),
            )
            summary = contrast_summary(frame)
            figure = contrast_figure(
                frame,
                title=f"{comparison} minus {reference} · {MODEL_SPECS[str(model_key)].short_label}",
            )
            exact = all(
                bool(row["exact_cartesian"])
                for row in metadata["donors"]
            )
            if exact:
                rule = "Exact Cartesian comparison: every posterior particle from the comparison inference run was paired with every particle from the reference inference run."
            else:
                returned = max(int(row["returned_pairs"]) for row in metadata["donors"])
                rule = f"Monte Carlo Cartesian comparison: {returned:,} uniformly sampled independent particle pairs per donor."
        except Exception as exc:
            figure = go.Figure()
            figure.add_annotation(
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                text="Choose two conditions with comparable inference results",
                showarrow=False,
            )
            figure.update_layout(
                template="none",
                height=360,
                paper_bgcolor=SHEET,
                plot_bgcolor=PAPER,
            )
            return figure, {"height": "360px"}, html.Div(), note("Comparison unavailable", str(exc), tone="amber")
        return figure, {"height": f"{int(figure.layout.height or 430)}px"}, data_table(summary, max_rows=24), note("Particle comparison rule", rule, tone="navy")
