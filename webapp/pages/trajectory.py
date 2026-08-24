"""Donor ignorant trajectory simulation and inference workflow."""

from __future__ import annotations

import base64
from collections.abc import Mapping

import pandas as pd
import plotly.graph_objects as go
from dash import ALL, MATCH, Input, Output, State, dcc, html, no_update

from webapp.analysis_ui import (
    data_table,
    field,
    parse_optional_seed,
)
from webapp.core.conditions import (
    APPLE_COLOUR_PRESETS,
    default_condition_colours,
    sanitize_condition_colours,
)
from webapp.core.trajectory import (
    TrajectorySettings,
    build_trajectory_archive,
    normalize_trajectory_frame,
    read_trajectory_csv,
    run_trajectory_conditions,
    simulate_trajectory_frame,
    trajectory_evidence_frame,
    trajectory_posterior_draws,
    validate_trajectory_frame,
)
from webapp.progress_ui import (
    pymc_progress,
    sampling_complete_payload,
    sampling_progress_payload,
)
from webapp.trajectory_reporting import (
    empirical_state_arrow_figure,
    empirical_state_encoding_legend,
    model_panel_styles,
    posterior_marginal_figure,
    render_trajectory_results,
)
from webapp.ui import hero, markdown, metric, note, research_warning


PATH = "/trajectory"
TITLE = "Trajectory inference"

MODEL_ORDER = (
    "homogeneous_history_independent",
    "homogeneous_history_dependent",
    "heterogeneous_history_independent",
    "heterogeneous_history_dependent",
)
MODEL_UI = {
    "homogeneous_history_independent": (
        "Hom-HI",
        "Homogeneous, history independent",
        "One baseline killing propensity; previous contacts have no effect.",
        "ση = 0 · βf = βs = 0",
    ),
    "homogeneous_history_dependent": (
        "Hom-HD",
        "Homogeneous, history dependent",
        "One baseline killing propensity with effects from previous contacts.",
        "ση = 0 · βf and βs inferred",
    ),
    "heterogeneous_history_independent": (
        "Het-HI",
        "Heterogeneous, history independent",
        "Baseline killing propensity varies between cells; history has no effect.",
        "ση inferred · βf = βs = 0",
    ),
    "heterogeneous_history_dependent": (
        "Het-HD",
        "Heterogeneous, history dependent",
        "Baseline propensity varies between cells and previous contacts can change later decisions.",
        "ση, βf and βs inferred",
    ),
}


def _empty_figure(message: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15, "color": "#6E675E"},
    )
    figure.update_layout(
        template="none",
        height=360,
        paper_bgcolor="#FFFEFA",
        plot_bgcolor="#FAF8F2",
        margin={"l": 30, "r": 30, "t": 30, "b": 30},
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return figure


def _workflow_option(title: str, description: str) -> html.Span:
    return html.Span(
        [html.Strong(title), html.Small(description)],
        className="barracuda-model-option-copy",
    )


def _model_option(model_key: str) -> html.Span:
    short, name, description, parameters = MODEL_UI[model_key]
    return html.Span(
        [
            html.Strong(f"{short} · {name}"),
            html.Small(description),
            html.Span(parameters, className="barracuda-trajectory-parameter-chip"),
        ],
        className="barracuda-model-option-copy",
    )


def _model_selector() -> html.Div:
    return html.Div(
        [
            html.Div("Models included in inference", className="barracuda-field-label"),
            dcc.Checklist(
                id="trajectory-models",
                options=[
                    {"label": _model_option(model_key), "value": model_key}
                    for model_key in MODEL_ORDER
                ],
                value=list(MODEL_ORDER),
                className="barracuda-model-checklist",
                inputClassName="barracuda-check-input",
                labelClassName="barracuda-model-option",
            ),
            html.Small(
                "Select one model for parameter inference or at least two for Bayes factor comparison.",
                className="barracuda-help",
            ),
        ],
        className="barracuda-field",
    )


def _inference_controls() -> html.Div:
    return html.Div(
        [
            research_warning(),
            field(
                "Compute profile",
                dcc.Dropdown(
                    id="trajectory-profile",
                    options=[
                        {"label": "Preview · quickest", "value": "preview"},
                        {"label": "Standard · more repeatability", "value": "demo"},
                        {"label": "Custom", "value": "custom"},
                    ],
                    value="preview",
                    clearable=False,
                ),
                "Trajectory inference is more expensive than event count inference. Start with Preview.",
            ),
            html.Details(
                [
                    html.Summary("Inference settings and priors"),
                    html.Div(
                        [
                            field(
                                "SMC particles per chain",
                                dcc.Input(id="trajectory-particles", type="number", min=32, max=1000, step=32, value=64),
                            ),
                            field(
                                "Independent chains",
                                dcc.Input(id="trajectory-chains", type="number", min=1, max=2, step=1, value=1),
                            ),
                            field(
                                "CPU cores",
                                dcc.Input(id="trajectory-cores", type="number", min=1, max=2, step=1, value=1),
                            ),
                        ],
                        className="barracuda-form-grid three",
                    ),
                    field(
                        "Inference seed (optional)",
                        dcc.Input(id="trajectory-seed", type="text", value="", placeholder="Blank = a new random run"),
                    ),
                    html.Div(
                        [
                            field(
                                "Tempering threshold",
                                dcc.Slider(
                                    id="trajectory-threshold",
                                    min=0.1,
                                    max=0.9,
                                    step=0.05,
                                    value=0.5,
                                    marks=None,
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                            ),
                            field(
                                "Mutation correlation threshold",
                                dcc.Input(id="trajectory-correlation", type="number", min=0.001, max=0.2, step=0.005, value=0.01),
                            ),
                        ],
                        className="barracuda-form-grid two",
                    ),
                    field(
                        "log10 mean contact rate prior bounds",
                        dcc.RangeSlider(
                            id="trajectory-lambda-prior-bounds",
                            min=-5,
                            max=2,
                            step=0.5,
                            value=[-1, 1.5],
                            marks={-5: "-5", -2: "-2", 0: "0", 2: "2"},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        "Figure 4 used Uniform(-1, 1.5); the real-data script used broader bounds.",
                    ),
                    html.Div(
                        [
                            field(
                                "Contact-rate SD prior scale",
                                dcc.Input(id="trajectory-sigma-lambda-prior", type="number", min=0.1, max=10, step=0.1, value=2.0),
                                "Half normal scale for σλ.",
                            ),
                            field(
                                "Baseline heterogeneity prior scale",
                                dcc.Input(id="trajectory-sigma-eta-prior", type="number", min=0.1, max=5, step=0.1, value=1.0),
                                "Half normal scale for ση.",
                            ),
                            field(
                                "History effect prior SD",
                                dcc.Input(id="trajectory-beta-prior-sd", type="number", min=0.1, max=5, step=0.1, value=1.0),
                                "Normal(0, SD) prior for βf and βs.",
                            ),
                            field(
                                "Quadrature points",
                                dcc.Input(id="trajectory-n-quad", type="number", min=5, max=60, step=5, value=20),
                                "More points improve the heterogeneous likelihood approximation but increase runtime.",
                            ),
                        ],
                        className="barracuda-form-grid two",
                    ),
                ],
                className="barracuda-details",
            ),
        ],
        className="barracuda-inference-controls",
    )


def _condition_colour_controls(labels: list[str]) -> html.Div:
    defaults = default_condition_colours(labels)
    options = [
        {"label": f"{name} · {colour}", "value": colour}
        for name, colour in APPLE_COLOUR_PRESETS
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.Strong(label),
                    dcc.Input(
                        id={"type": "trajectory-condition-colour", "index": label},
                        type="color",
                        value=defaults[label],
                        className="barracuda-colour-input",
                    ),
                    dcc.Dropdown(
                        id={"type": "trajectory-condition-preset", "index": label},
                        options=options,
                        value=defaults[label],
                        clearable=False,
                        searchable=False,
                        className="barracuda-condition-preset",
                    ),
                ],
                className="barracuda-condition-colour-card",
            )
            for label in labels
        ],
        className="barracuda-condition-colour-grid",
    )


def _serialise_frame(frame: pd.DataFrame) -> list[dict]:
    clean = frame.copy()
    clean["history"] = clean["history"].map(
        lambda values: ",".join(str(int(value)) for value in values)
        if not isinstance(values, str)
        else values
    )
    return clean.to_dict("records")


def _frame_from_records(records: list[dict] | None) -> pd.DataFrame:
    if not records:
        raise ValueError("Choose or provide trajectory data first.")
    return validate_trajectory_frame(pd.DataFrame(records))


def _data_summary(frame: pd.DataFrame) -> html.Div:
    contacts = int(sum(len(history) for history in frame["history"]))
    lethal = int(sum(sum(history) for history in frame["history"]))
    zero_contact = int(sum(len(history) == 0 for history in frame["history"]))
    conditions = int(frame["condition"].nunique())
    lethal_fraction = lethal / contacts if contacts else 0.0
    return html.Div(
        [
            metric("Cells", f"{len(frame):,}", accent="teal"),
            metric("Conditions", str(conditions), accent="navy"),
            metric("Observed contacts", f"{contacts:,}"),
            metric("Zero-contact cells", f"{zero_contact:,}"),
            metric("Lethal contact fraction", f"{lethal_fraction:.1%}"),
        ],
        className="barracuda-metrics",
    )


def _preview_table(frame: pd.DataFrame):
    preview = frame.copy()
    preview["history"] = preview["history"].map(
        lambda history: "[" + ", ".join(map(str, history)) + "]"
    )
    return data_table(preview, max_rows=8)


def layout() -> html.Div:
    return html.Div(
        [
            dcc.Store(id="trajectory-synthetic-data"),
            dcc.Store(id="trajectory-synthetic-truth"),
            dcc.Store(id="trajectory-upload-data"),
            dcc.Store(id="trajectory-active-data"),
            dcc.Store(id="trajectory-active-truth"),
            dcc.Store(id="trajectory-active-observation-time", data=1.0),
            hero(
                "Contact trajectories · Donor ignorant",
                "Trajectory inference",
                "Use ordered lethal and non-lethal contacts to separate stable differences between cells from changes caused by previous encounters.",
                badge="Synthetic validation • Up to four experimental conditions",
            ),
            html.Section(
                [
                    html.Span("Model", className="barracuda-section-label"),
                    html.H2("From ordered contacts to killing decisions"),
                    html.P(
                        "Every candidate model shares a Gamma-Poisson contact layer. They differ in whether baseline killing propensity varies between cells and whether previous lethal or non-lethal contacts change the next decision.",
                        className="barracuda-section-lead",
                    ),
                    html.Div(
                        [
                            markdown(
                                "$$z_{ij}\\sim\\mathrm{Bernoulli}(p_{ij}),\\qquad "
                                "\\mathrm{logit}(p_{ij})=\\eta_i+\\beta_f f_{ij}+\\beta_s s_{ij}$$",
                                class_name="barracuda-equation small",
                                mathjax=True,
                            ),
                            markdown(
                                "$$\\eta_i\\sim\\mathrm{Normal}(\\mu_\\eta,\\sigma_\\eta),\\qquad "
                                "\\lambda_i\\sim\\mathrm{Gamma}(\\mu_\\lambda,\\sigma_\\lambda)$$",
                                class_name="barracuda-equation small",
                                mathjax=True,
                            ),
                        ],
                        className="barracuda-equation-row",
                    ),
                    html.P(
                        "Here f is the number of previous non-lethal contacts and s is the number of previous lethal contacts. Positive β increases the odds of a later kill; negative β reduces them.",
                        className="barracuda-help",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Strong(short),
                                    html.H3(name),
                                    html.P(description),
                                    html.Small(parameters),
                                ],
                                className="barracuda-trajectory-model-card",
                            )
                            for short, name, description, parameters in MODEL_UI.values()
                        ],
                        className="barracuda-model-definition-grid",
                    ),
                ],
                className="barracuda-trajectory-intro",
            ),
            html.Section(
                [
                    html.Span("Start here", className="barracuda-section-label"),
                    html.H2("Which trajectory data do you want to use?"),
                    dcc.RadioItems(
                        id="trajectory-workflow",
                        options=[
                            {
                                "label": _workflow_option(
                                    "Synthetic data",
                                    "Set a known generating model and check recovery against ground truth.",
                                ),
                                "value": "synthetic",
                            },
                            {
                                "label": _workflow_option(
                                    "My own data",
                                    "Upload ordered histories for one to four experimental conditions.",
                                ),
                                "value": "upload",
                            },
                        ],
                        value=None,
                        className="barracuda-model-checklist",
                        inputClassName="barracuda-check-input",
                        labelClassName="barracuda-model-option",
                    ),
                    html.P(
                        "Choose one option to continue.",
                        id="trajectory-workflow-status",
                        className="barracuda-help",
                        role="status",
                        **{"aria-live": "polite"},
                    ),
                ],
                className="barracuda-workflow-panel",
            ),
            html.Div(
                [
                    html.Span("Synthetic data", className="barracuda-section-label"),
                    html.H2("Choose the ground truth"),
                    field(
                        "Generating model",
                        dcc.Dropdown(
                            id="trajectory-ground-model",
                            options=[
                                {"label": f"{MODEL_UI[key][0]} · {MODEL_UI[key][1]}", "value": key}
                                for key in MODEL_ORDER
                            ],
                            value="heterogeneous_history_dependent",
                            clearable=False,
                        ),
                    ),
                    html.Div(
                        [
                            field("Number of cells", dcc.Input(id="trajectory-n-cells", type="number", min=10, max=1000, step=10, value=100)),
                            field("Mean contact rate, μλ", dcc.Input(id="trajectory-mu-lambda", type="number", min=0.05, max=50, step=0.25, value=4.0)),
                            field("Contact-rate SD, σλ", dcc.Input(id="trajectory-sigma-lambda", type="number", min=0, max=30, step=0.25, value=2.0)),
                            field(
                                "Central baseline lethal probability",
                                dcc.Input(id="trajectory-baseline-probability", type="number", min=0.01, max=0.99, step=0.01, value=0.25),
                                "This is logit⁻¹(μη), not a nonengaging fraction.",
                            ),
                            field("Baseline heterogeneity, ση", dcc.Input(id="trajectory-sigma-eta", type="number", min=0, max=5, step=0.05, value=0.75)),
                            field("Previous non-lethal effect, βf", dcc.Input(id="trajectory-beta-f", type="number", min=-5, max=5, step=0.05, value=0.8)),
                            field("Previous lethal effect, βs", dcc.Input(id="trajectory-beta-s", type="number", min=-5, max=5, step=0.05, value=-0.8)),
                        ],
                        className="barracuda-form-grid three",
                    ),
                    html.Details(
                        [
                            html.Summary("Observation window and reproducibility"),
                            html.Div(
                                [
                                    field(
                                        "Observation time, T",
                                        dcc.Input(id="trajectory-observation-time", type="number", min=0.01, max=100, step=0.25, value=1.0),
                                        "T changes how the contact rate is interpreted; the default is one complete observation window.",
                                    ),
                                    field(
                                        "Simulation seed (optional)",
                                        dcc.Input(id="trajectory-simulation-seed", type="text", value="", placeholder="Blank = a new dataset"),
                                    ),
                                ],
                                className="barracuda-form-grid two",
                            ),
                        ],
                        className="barracuda-details barracuda-observation-details",
                    ),
                    html.Button("Generate synthetic trajectories", id="trajectory-generate", n_clicks=0, className="barracuda-button primary full"),
                    html.Div(id="trajectory-generate-status", role="status", **{"aria-live": "polite"}),
                ],
                id="trajectory-synthetic-panel",
                className="barracuda-workflow-panel is-hidden",
            ),
            html.Div(
                [
                    html.Span("Your data", className="barracuda-section-label"),
                    html.H2("Upload ordered contact histories"),
                    html.P(
                        "Use one row per cell with cell_id, condition and history. Write the ordered outcomes as a quoted comma-separated sequence such as \"0,1,0\". A blank history keeps a zero-contact cell in the contact-rate analysis.",
                    ),
                    dcc.Upload(
                        id="trajectory-upload",
                        children=html.Div([html.Strong("Drop a CSV here"), " or choose a file"]),
                        className="barracuda-upload",
                        multiple=False,
                    ),
                    html.Div(id="trajectory-upload-status", role="status", **{"aria-live": "polite"}),
                    html.A(
                        "Download the trajectory CSV template",
                        href="/assets/downloads/barracuda_trajectory_template.csv",
                        download="barracuda_trajectory_template.csv",
                        className="barracuda-button secondary",
                    ),
                    html.Details(
                        [
                            html.Summary("Observation window"),
                            field(
                                "Observation time, T",
                                dcc.Input(
                                    id="trajectory-upload-observation-time",
                                    type="number",
                                    min=0.01,
                                    max=100,
                                    step=0.25,
                                    value=1.0,
                                ),
                                "The default is one complete observation window. T changes the interpretation of contact rates, not the order of outcomes.",
                            ),
                        ],
                        className="barracuda-details barracuda-observation-details",
                    ),
                    html.P(
                        "The paper's wide format with Cell and numbered contact columns is also accepted. Do not include donor labels; this model is donor ignorant.",
                        className="barracuda-help",
                    ),
                ],
                id="trajectory-upload-panel",
                className="barracuda-workflow-panel is-hidden",
            ),
            html.Section(
                [
                    html.Span("01 · Empirical data", className="barracuda-section-label"),
                    html.H2("Observed contact-history states"),
                    html.Div(id="trajectory-data-summary"),
                    html.Div(
                        [
                            html.H3("Condition colours"),
                            html.P(
                                "These colours identify conditions in posterior plots. The state-map colour scale remains fixed to the empirical killing probability.",
                                className="barracuda-help",
                            ),
                            html.Div(id="trajectory-condition-colour-controls"),
                        ],
                        id="trajectory-condition-colours-section",
                        className="barracuda-condition-colours-section",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("Figure display", className="barracuda-section-label"),
                                    html.P(
                                        "Increase the canvas for dense state maps or reduce the arrows when neighbouring states look crowded. These settings change only the display, never the data or inference.",
                                        className="barracuda-help",
                                    ),
                                ],
                                className="barracuda-figure-display-copy",
                            ),
                            field(
                                "Figure height",
                                dcc.Slider(
                                    id="trajectory-figure-height",
                                    min=480,
                                    max=1200,
                                    step=40,
                                    value=700,
                                    marks={480: "480 px", 700: "700", 960: "960", 1200: "1200 px"},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                            ),
                            field(
                                "Arrow size",
                                dcc.Slider(
                                    id="trajectory-arrow-scale",
                                    min=0.4,
                                    max=1.6,
                                    step=0.1,
                                    value=1.0,
                                    marks={0.4: "40%", 1.0: "100%", 1.6: "160%"},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                            ),
                        ],
                        className="barracuda-figure-display-controls",
                    ),
                    html.Div(id="trajectory-empirical-legend"),
                    dcc.Graph(
                        id="trajectory-empirical-figure",
                        figure=_empty_figure("Choose data to draw the empirical state map."),
                        responsive=True,
                        config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "filename": "barracuda_empirical_trajectory_map", "scale": 2}},
                        className="barracuda-trajectory-empirical-plot",
                        style={"height": "700px"},
                    ),
                    html.P(
                        "Arrow direction and colour show the empirical probability that the next contact is lethal. Arrow-tail length shows the number of cells reaching the state on a log2 scale.",
                        className="barracuda-help",
                    ),
                    html.Details(
                        [html.Summary("Review the normalized cell histories"), html.Div(id="trajectory-data-preview")],
                        className="barracuda-details",
                    ),
                ],
                id="trajectory-data-section",
                className="barracuda-workflow-panel is-hidden",
            ),
            html.Section(
                [
                    html.Span("02 · Bayesian inference", className="barracuda-section-label"),
                    html.H2("Compare trajectory mechanisms"),
                    html.P(
                        "Inference runs independently for each experimental condition. Bayes factors compare candidate mechanisms within a condition, while posterior plots overlay conditions using your selected colours.",
                        className="barracuda-section-lead",
                    ),
                    note(
                        "How model evidence is computed",
                        "PyMC SMC estimates each model's marginal likelihood while moving particles from the prior to the posterior. Bayes factors compare those marginal likelihood estimates; they are not ratios of posterior means.",
                        tone="navy",
                    ),
                    html.Fieldset(
                        [
                            _model_selector(),
                            _inference_controls(),
                            html.P(id="trajectory-workload", className="barracuda-help"),
                            html.Button("Run trajectory inference", id="trajectory-run", n_clicks=0, disabled=True, className="barracuda-button primary full"),
                        ],
                        id="trajectory-inference-controls",
                        disabled=False,
                        className="barracuda-inference-fieldset",
                    ),
                    html.Div(id="trajectory-run-status", role="status", **{"aria-live": "polite"}),
                    pymc_progress("trajectory"),
                    dcc.Loading(
                        html.Div(
                            html.Div(
                                [
                                    html.Strong("Bayes factors and posterior distributions"),
                                    html.P("Run inference to compare model evidence and inspect marginal or joint posterior distributions."),
                                ],
                                className="barracuda-results-placeholder",
                            ),
                            id="trajectory-results",
                        ),
                        type="circle",
                        color="#304B3D",
                        className="barracuda-loading",
                    ),
                    html.Div(id="trajectory-download", className="barracuda-download-slot"),
                ],
                id="trajectory-inference",
                className="barracuda-workflow-panel is-hidden",
            ),
        ]
    )


def register_callbacks(app) -> None:
    @app.callback(
        Output("trajectory-synthetic-panel", "className"),
        Output("trajectory-upload-panel", "className"),
        Output("trajectory-workflow-status", "children"),
        Input("trajectory-workflow", "value"),
    )
    def choose_workflow(workflow):
        if workflow == "synthetic":
            return "barracuda-workflow-panel", "barracuda-workflow-panel is-hidden", "Set the ground truth, then generate trajectories."
        if workflow == "upload":
            return "barracuda-workflow-panel is-hidden", "barracuda-workflow-panel", "Upload a trajectory CSV to continue."
        return "barracuda-workflow-panel is-hidden", "barracuda-workflow-panel is-hidden", "Choose one option to continue."

    @app.callback(
        Output("trajectory-sigma-eta", "disabled"),
        Output("trajectory-sigma-eta", "value"),
        Output("trajectory-beta-f", "disabled"),
        Output("trajectory-beta-f", "value"),
        Output("trajectory-beta-s", "disabled"),
        Output("trajectory-beta-s", "value"),
        Input("trajectory-ground-model", "value"),
    )
    def set_ground_model_controls(model_key):
        heterogeneous = str(model_key).startswith("heterogeneous")
        history_dependent = str(model_key).endswith("history_dependent") and "independent" not in str(model_key)
        return (
            not heterogeneous,
            0.75 if heterogeneous else 0.0,
            not history_dependent,
            0.8 if history_dependent else 0.0,
            not history_dependent,
            -0.8 if history_dependent else 0.0,
        )

    @app.callback(
        Output("trajectory-synthetic-data", "data"),
        Output("trajectory-synthetic-truth", "data"),
        Output("trajectory-generate-status", "children"),
        Input("trajectory-generate", "n_clicks"),
        State("trajectory-ground-model", "value"),
        State("trajectory-n-cells", "value"),
        State("trajectory-mu-lambda", "value"),
        State("trajectory-sigma-lambda", "value"),
        State("trajectory-baseline-probability", "value"),
        State("trajectory-sigma-eta", "value"),
        State("trajectory-beta-f", "value"),
        State("trajectory-beta-s", "value"),
        State("trajectory-observation-time", "value"),
        State("trajectory-simulation-seed", "value"),
        prevent_initial_call=True,
    )
    def generate_trajectories(_clicks, model_key, n_cells, mu_lambda, sigma_lambda, baseline_probability, sigma_eta, beta_f, beta_s, observation_time, seed):
        try:
            frame, truth = simulate_trajectory_frame(
                n_cells=int(n_cells),
                condition="Synthetic",
                mu_lambda=float(mu_lambda),
                sigma_lambda=float(sigma_lambda),
                p0=float(baseline_probability),
                sigma_eta=float(sigma_eta),
                beta_f=float(beta_f),
                beta_s=float(beta_s),
                observation_time=float(observation_time),
                seed=parse_optional_seed(seed),
            )
            frame = validate_trajectory_frame(frame)
        except Exception as exc:
            return no_update, no_update, note("Synthetic data were not generated", str(exc), tone="amber")
        return _serialise_frame(frame), dict(truth), note("Synthetic trajectories generated", f"Created {len(frame):,} cell histories from {MODEL_UI[str(model_key)][0]}.", tone="teal")

    @app.callback(
        Output("trajectory-upload-data", "data"),
        Output("trajectory-upload-status", "children"),
        Input("trajectory-upload", "contents"),
        prevent_initial_call=True,
    )
    def upload_trajectories(contents):
        if not contents:
            return no_update, html.Div()
        try:
            _, encoded = contents.split(",", 1)
            payload = base64.b64decode(encoded, validate=True)
            if len(payload) > 1_000_000:
                raise ValueError("This web application accepts CSV files up to 1 MB.")
            raw = read_trajectory_csv(payload)
            normalized = normalize_trajectory_frame(raw)
            frame = validate_trajectory_frame(normalized)
        except Exception as exc:
            return no_update, note("CSV could not be used", str(exc), tone="amber")
        return _serialise_frame(frame), note(
            "Trajectory CSV loaded",
            f"Recognised {len(frame):,} cells across {frame['condition'].nunique()} condition(s).",
            tone="teal",
        )

    @app.callback(
        Output("trajectory-active-data", "data"),
        Output("trajectory-active-truth", "data"),
        Output("trajectory-active-observation-time", "data"),
        Output("trajectory-data-section", "className"),
        Output("trajectory-inference", "className"),
        Output("trajectory-data-summary", "children"),
        Output("trajectory-empirical-legend", "children"),
        Output("trajectory-empirical-figure", "figure"),
        Output("trajectory-empirical-figure", "style"),
        Output("trajectory-data-preview", "children"),
        Output("trajectory-condition-colour-controls", "children"),
        Output("trajectory-run", "disabled"),
        Input("trajectory-workflow", "value"),
        Input("trajectory-synthetic-data", "data"),
        Input("trajectory-upload-data", "data"),
        Input("trajectory-upload-observation-time", "value"),
        Input("trajectory-figure-height", "value"),
        Input("trajectory-arrow-scale", "value"),
        State("trajectory-synthetic-truth", "data"),
    )
    def activate_data(
        workflow,
        synthetic_records,
        upload_records,
        upload_time,
        figure_height,
        arrow_scale,
        synthetic_truth,
    ):
        records = synthetic_records if workflow == "synthetic" else upload_records if workflow == "upload" else None
        truth = synthetic_truth if workflow == "synthetic" else None
        if workflow == "synthetic" and isinstance(synthetic_truth, Mapping):
            truth_values = next(
                (values for values in synthetic_truth.values() if isinstance(values, Mapping)),
                {},
            )
            observation_time = truth_values.get("observation_time", 1.0)
        else:
            observation_time = upload_time
        if not records:
            return None, None, 1.0, "barracuda-workflow-panel is-hidden", "barracuda-workflow-panel is-hidden", html.Div(), html.Div(), _empty_figure("Choose data to draw the empirical state map."), {"height": f"{int(figure_height or 700)}px"}, html.Div(), html.Div(), True
        try:
            frame = _frame_from_records(records)
            observation_time = float(observation_time)
            if observation_time <= 0:
                raise ValueError("Observation time must be greater than zero.")
            labels = list(dict.fromkeys(frame["condition"].astype(str)))
            figure = empirical_state_arrow_figure(
                frame,
                arrow_scale=float(arrow_scale or 1.0),
                figure_height=int(figure_height or 700),
            )
            encoding_legend = empirical_state_encoding_legend(
                frame,
                arrow_scale=float(arrow_scale or 1.0),
            )
        except Exception as exc:
            return None, None, 1.0, "barracuda-workflow-panel is-hidden", "barracuda-workflow-panel is-hidden", note("Data are not ready", str(exc), tone="amber"), html.Div(), _empty_figure(str(exc)), {"height": f"{int(figure_height or 700)}px"}, html.Div(), html.Div(), True
        return (
            _serialise_frame(frame),
            truth,
            observation_time,
            "barracuda-workflow-panel",
            "barracuda-workflow-panel",
            _data_summary(frame),
            encoding_legend,
            figure,
            {"height": f"{int(figure.layout.height or 430)}px"},
            _preview_table(frame),
            _condition_colour_controls(labels),
            False,
        )

    @app.callback(
        Output({"type": "trajectory-condition-colour", "index": MATCH}, "value"),
        Input({"type": "trajectory-condition-preset", "index": MATCH}, "value"),
        prevent_initial_call=True,
    )
    def apply_colour_preset(value):
        return value

    @app.callback(
        Output("trajectory-workload", "children"),
        Input("trajectory-active-data", "data"),
        Input("trajectory-models", "value"),
        Input("trajectory-chains", "value"),
    )
    def show_workload(records, models, chains):
        if not records:
            return "Choose data to calculate the inference workload."
        try:
            conditions = _frame_from_records(records)["condition"].nunique()
            model_count = len(models or [])
            chain_count = int(chains or 1)
        except Exception:
            return "The inference workload is not available."
        runs = conditions * model_count
        return f"Workload: {conditions} condition{'s' if conditions != 1 else ''} × {model_count} model{'s' if model_count != 1 else ''} = {runs} SMC inference run{'s' if runs != 1 else ''}, with {chain_count} chain{'s' if chain_count != 1 else ''} each."

    @app.callback(
        Output("trajectory-results", "children"),
        Output("trajectory-download", "children"),
        Output("trajectory-run-status", "children"),
        Input("trajectory-run", "n_clicks"),
        State("trajectory-active-data", "data"),
        State("trajectory-active-truth", "data"),
        State("trajectory-active-observation-time", "data"),
        State("trajectory-models", "value"),
        State("trajectory-particles", "value"),
        State("trajectory-chains", "value"),
        State("trajectory-cores", "value"),
        State("trajectory-seed", "value"),
        State("trajectory-threshold", "value"),
        State("trajectory-correlation", "value"),
        State("trajectory-lambda-prior-bounds", "value"),
        State("trajectory-sigma-lambda-prior", "value"),
        State("trajectory-sigma-eta-prior", "value"),
        State("trajectory-beta-prior-sd", "value"),
        State("trajectory-n-quad", "value"),
        State({"type": "trajectory-condition-colour", "index": ALL}, "value"),
        State({"type": "trajectory-condition-colour", "index": ALL}, "id"),
        prevent_initial_call=True,
        background=True,
        interval=350,
        progress=[
            Output("trajectory-pymc-progress-bar", "value"),
            Output("trajectory-pymc-progress-label", "children"),
            Output("trajectory-pymc-progress-meta", "children"),
            Output("trajectory-chain-progress", "children"),
        ],
        progress_default=[
            0,
            "PyMC SMC sampler",
            "Start inference to see each chain's SMC stage and tempering value β.",
            [],
        ],
        running=[
            (Output("trajectory-pymc-progress", "className"), "barracuda-pymc-progress is-active", "barracuda-pymc-progress is-hidden"),
            (Output("trajectory-inference", "aria-busy"), "true", "false"),
            (Output("trajectory-inference-controls", "disabled"), True, False),
            (Output("trajectory-run", "className"), "barracuda-button primary full is-running", "barracuda-button primary full"),
        ],
    )
    def run_inference(set_progress, _clicks, records, truth, observation_time, models, particles, chains, cores, seed, threshold, correlation, prior_bounds, sigma_lambda_prior, sigma_eta_prior, beta_prior_sd, n_quad, colour_values, colour_ids):
        try:
            frame = _frame_from_records(records)
            selected_models = [str(model) for model in (models or [])]
            if not selected_models:
                raise ValueError("Select at least one trajectory model for inference.")
            bounds = list(prior_bounds or [])
            if len(bounds) != 2:
                raise ValueError("Choose both mean contact rate prior bounds.")
            settings = TrajectorySettings(
                draws=int(particles),
                chains=int(chains),
                cores=min(int(cores), int(chains)),
                seed=parse_optional_seed(seed),
                threshold=float(threshold),
                correlation_threshold=float(correlation),
                lambda_prior_bounds=(float(bounds[0]), float(bounds[1])),
                sigma_lambda_prior=float(sigma_lambda_prior),
                sigma_eta_prior=float(sigma_eta_prior),
                beta_prior_sd=float(beta_prior_sd),
                n_quad=int(n_quad),
            )
            condition_labels = list(dict.fromkeys(frame["condition"].astype(str)))
            supplied_colours = {
                str(component_id.get("index")): value
                for component_id, value in zip(colour_ids or [], colour_values or [])
                if isinstance(component_id, dict)
            }
            condition_colours = sanitize_condition_colours(condition_labels, supplied_colours)
            total_chains = int(settings.chains)
            chain_states: dict[int, tuple[int, float]] = {}

            def publish_progress(condition_index, total_conditions, condition_label, model_index, total_models, model_label):
                set_progress(
                    sampling_progress_payload(
                        condition_index=int(condition_index),
                        total_conditions=int(total_conditions),
                        condition_label=str(condition_label),
                        model_index=int(model_index),
                        total_models=int(total_models),
                        model_label=str(model_label),
                        chains=total_chains,
                        particles=int(settings.draws),
                        chain_states=chain_states,
                    )
                )

            def model_started(condition_index, total_conditions, condition_label, model_index, total_models, model_label):
                chain_states.clear()
                publish_progress(condition_index, total_conditions, condition_label, model_index, total_models, model_label)

            def sampler_progress(condition_index, total_conditions, condition_label, model_index, total_models, model_label, chain, stage, beta):
                chain_index = int(chain)
                if not 0 <= chain_index < total_chains:
                    return
                chain_states[chain_index] = (max(0, int(stage)), min(1.0, max(0.0, float(beta))))
                publish_progress(condition_index, total_conditions, condition_label, model_index, total_models, model_label)

            results = run_trajectory_conditions(
                frame,
                observation_time=float(observation_time),
                settings=settings,
                model_keys=selected_models,
                progress_callback=model_started,
                sampler_progress_callback=sampler_progress,
            )
            set_progress(sampling_complete_payload(chains=total_chains, particles=int(settings.draws), chain_states=chain_states))
            evidence = trajectory_evidence_frame(results)
            posterior_draws = trajectory_posterior_draws(results, max_draws=1000)
            archive = build_trajectory_archive(
                results,
                frame,
                float(observation_time),
                settings=settings,
                truth=truth,
            )
            encoded = base64.b64encode(archive).decode("ascii")
            download = html.A(
                "Download trajectories, posterior files and tables",
                href=f"data:application/zip;base64,{encoded}",
                download="barracuda_trajectory_inference.zip",
                className="barracuda-button primary full",
            )
            content, download_slot = render_trajectory_results(
                evidence=evidence,
                posterior_draws=posterior_draws,
                condition_colours=condition_colours,
                truth=truth,
                truth_model={
                    str(condition): str(values.get("true_model_key"))
                    for condition, values in truth.items()
                    if isinstance(values, Mapping) and values.get("true_model_key")
                }
                if truth
                else None,
                download=download,
                prefix="trajectory",
            )
        except Exception as exc:
            return html.Div(), html.Div(), note("Inference did not complete", str(exc), tone="amber")
        return content, download_slot, note("Inference complete", f"PyMC completed {len(condition_labels)} condition{'s' if len(condition_labels) != 1 else ''} across {len(selected_models)} candidate model{'s' if len(selected_models) != 1 else ''}.", tone="teal")

    @app.callback(
        Output({"type": "trajectory-model-panel", "index": ALL}, "style"),
        Input("trajectory-model-view", "value"),
        State({"type": "trajectory-model-panel", "index": ALL}, "id"),
        prevent_initial_call=True,
    )
    def choose_models_to_visualise(selected_models, panel_ids):
        return model_panel_styles(selected_models, panel_ids)

    @app.callback(
        Output({"type": "trajectory-posterior-marginal", "index": MATCH}, "figure"),
        Input({"type": "trajectory-posterior-parameter", "index": MATCH}, "value"),
        State({"type": "trajectory-posterior-parameter", "index": MATCH}, "id"),
        State("trajectory-posterior-data", "data"),
        prevent_initial_call=True,
    )
    def choose_marginal_parameter(parameter, component_id, payload):
        if not payload or not parameter or not isinstance(component_id, Mapping):
            return no_update
        model_key = str(component_id.get("index", ""))
        try:
            draws = pd.DataFrame(payload.get("records") or [])
            colours = {
                str(condition): str(colour)
                for condition, colour in (payload.get("condition_colours") or {}).items()
            }
            return posterior_marginal_figure(
                draws,
                str(parameter),
                colours,
                payload.get("truth"),
                model_key,
            )
        except Exception as exc:
            return _empty_figure(f"The marginal posterior could not be drawn: {exc}")
