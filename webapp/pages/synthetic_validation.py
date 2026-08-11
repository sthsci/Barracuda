"""Synthetic-data validation page and callbacks."""

from __future__ import annotations

import pandas as pd
from dash import Input, Output, State, dcc, html, no_update
import plotly.graph_objects as go

from webapp.analysis_ui import (
    MODEL_LABELS,
    csv_download_link,
    data_overview,
    field,
    inference_controls,
    model_selector,
    rate_distribution_figure,
    render_validation_results,
    settings_from_values,
    table_records,
)
from webapp.core.data import validate_count_frame
from webapp.core.inference import run_count_models
from webapp.core.simulation import (
    RATE_DISTRIBUTION_LABELS,
    paper_rate_distribution_for_model,
    simulate_event_counts,
)
from webapp.reporting import (
    joint_posterior_figure_from_draws,
    posterior_draws_from_store,
    posterior_parameters_for_models,
)
from webapp.progress_ui import (
    pymc_progress,
    sampling_complete_payload,
    sampling_progress_payload,
)
from webapp.ui import hero, note, schematic_figure


PATH = "/synthetic-validation"
TITLE = "Synthetic validation"


def layout() -> html.Div:
    return html.Div(
        [
            dcc.Store(id="synthetic-data"),
            dcc.Store(id="synthetic-truth"),
            dcc.Store(id="synthetic-time"),
            hero(
                "Event counts · Validation",
                "Synthetic data validation",
                "Choose a known population structure, generate event counts, then ask whether Bayesian inference recovers the parameters and ranks the generating model.",
                badge="Ground truth is visible",
            ),
            schematic_figure(
                "/assets/synthetic_validation_workflow.png",
                "Synthetic validation workflow: choose an event rate distribution, draw a rate for each NK cell, simulate tumour-cell events and counts per cell, infer μλ, σλ and φ₀, and compare the posterior with the known ground truth.",
                "The paper's validation workflow starts from known μλ, σλ and φ₀, generates cell-level event counts, and compares the inferred posterior with those known values.",
                variant="validation",
            ),
            html.P("The public demo accepts generated counts up to 100. If a setting exceeds that limit, reduce it and generate again. Choose priors that cover the truth you set.", className="orca-help"),
            html.Div(
                [
                    html.Span("Step A", className="orca-section-label"),
                    html.H2("Choose the ground truth"),
                    field(
                        "Generating model",
                        dcc.Dropdown(id="synthetic-ground-model", options=[{"label": label, "value": key} for key, label in MODEL_LABELS.items()], value="hetero3", clearable=False),
                    ),
                    html.Div(
                        [
                            field("Number of cells", dcc.Input(id="synthetic-n-cells", type="number", min=10, max=1000, step=10, value=100)),
                            field(
                                "Rate distribution set by the model",
                                dcc.Dropdown(
                                    id="synthetic-rate-distribution",
                                    options=[
                                        {"label": RATE_DISTRIBUTION_LABELS["fixed"], "value": "fixed"},
                                        {"label": RATE_DISTRIBUTION_LABELS["gamma"], "value": "gamma"},
                                    ],
                                    value="gamma",
                                    clearable=False,
                                    disabled=True,
                                ),
                                "The generating model sets this automatically, so the two controls cannot contradict each other.",
                            ),
                            field("Mean event rate among engaging cells, μλ", dcc.Input(id="synthetic-mu-lambda", type="number", min=0.01, max=100, step=0.25, value=4.0)),
                            field("Continuous cell-to-cell heterogeneity in event rates, σλ", dcc.Input(id="synthetic-sigma-lambda", type="number", min=0, max=50, step=0.25, value=3.0)),
                            field("Fraction of nonengaging cells, φ₀", dcc.Slider(id="synthetic-p-zero", min=0, max=0.95, step=0.05, value=0.2, marks={0: "0", 0.5: "0.5", 0.95: "0.95"}, tooltip={"placement": "bottom"})),
                        ],
                        className="orca-form-grid three",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span("Live model design", className="orca-section-label"),
                                    html.H3("Population distribution of cell-specific event rates λᵢ"),
                                    html.P(
                                        "This includes every cell. A zero-inflated model places probability φ₀ at λᵢ = 0, while the remaining population follows the rate structure named by the model.",
                                        className="orca-help",
                                    ),
                                ],
                                className="orca-rate-preview-copy",
                            ),
                            dcc.Graph(
                                id="synthetic-rate-distribution-preview",
                                figure=rate_distribution_figure("gamma", 4.0, 3.0, 0.2),
                                config={"displaylogo": False, "responsive": True},
                                className="orca-rate-preview-plot",
                            ),
                            html.Div(id="synthetic-rate-preview-note", className="orca-rate-preview-note"),
                        ],
                        className="orca-rate-preview",
                    ),
                    html.Details(
                        [
                            html.Summary("Alternative rate distributions · in development"),
                            html.P(
                                "Lognormal, truncated Normal and other positive rate families are not used to generate data in this release. Keeping them separate ensures that every public simulation matches one of the four models evaluated below.",
                                className="orca-help",
                            ),
                        ],
                        className="orca-details orca-development-details",
                    ),
                    html.Details(
                        [
                            html.Summary("Observation time T and reproducibility · T defaults to 1"),
                            html.Div(
                                [
                                    field(
                                        "Observation time T",
                                        dcc.Input(id="synthetic-observation-time", type="number", min=0.01, max=100, step=0.25, value=1.0),
                                        "Counts follow Nᵢ | λᵢ,T ~ Poisson(λᵢT). With T = 1, rates are events per complete observation window.",
                                    ),
                                    field(
                                        "Simulation seed (optional)",
                                        dcc.Input(id="synthetic-simulation-seed", type="text", value="", placeholder="Blank = a new dataset"),
                                        "Set a seed to reproduce exactly the same dataset.",
                                    ),
                                ],
                                className="orca-form-grid two",
                            ),
                        ],
                        className="orca-details orca-observation-details",
                    ),
                    html.Button("Generate synthetic data", id="synthetic-generate", n_clicks=0, className="orca-button primary full"),
                    html.Div(id="synthetic-generate-status", role="status", **{"aria-live": "polite"}),
                ],
                className="orca-workflow-panel",
            ),
            html.Div(id="synthetic-preview"),
            html.Div(
                [
                    html.Span("Step C", className="orca-section-label"),
                    html.H2("Run Bayesian inference"),
                    model_selector("synthetic"),
                    inference_controls("synthetic"),
                    html.P("Inference can take several minutes. Keep this page open until it finishes.", className="orca-help"),
                    html.Button("Run inference for selected models", id="synthetic-run", n_clicks=0, disabled=True, className="orca-button primary full"),
                    html.Div(id="synthetic-run-status", role="status", **{"aria-live": "polite"}),
                    pymc_progress("synthetic"),
                    dcc.Loading(
                        html.Div(
                            html.Div(
                                [
                                    html.Strong("Posterior and Bayes factor plots"),
                                    html.P(
                                        "Run inference for one or more models. Posterior distributions appear for every completed model, and comparative Bayes factors appear when at least two models are included.",
                                    ),
                                ],
                                className="orca-results-placeholder",
                            ),
                            id="synthetic-results",
                        ),
                        type="circle",
                        color="#304B3D",
                        className="orca-loading",
                    ),
                    html.Div(id="synthetic-download", className="orca-download-slot"),
                    html.Details(
                        [
                            html.Summary("Run the complete demonstration in Jupyter"),
                            html.P(
                                "The notebook contains the Gamma simulator, all four PyMC models, the joint posterior plot, the Bayes factor plot and the code for reopening downloaded InferenceData files.",
                                className="orca-help",
                            ),
                            html.A(
                                "Download the one-file demonstration notebook",
                                href="/assets/downloads/orca_synthetic_validation_demo.ipynb",
                                download="orca_synthetic_validation_demo.ipynb",
                                className="orca-button secondary download",
                            ),
                        ],
                        className="orca-details orca-notebook-details",
                    ),
                ],
                id="synthetic-inference-section",
                className="orca-workflow-panel is-hidden",
            ),
        ]
    )


def register_callbacks(app) -> None:
    @app.callback(
        Output("synthetic-sigma-lambda", "disabled"),
        Output("synthetic-p-zero", "disabled"),
        Output("synthetic-rate-distribution", "value"),
        Input("synthetic-ground-model", "value"),
    )
    def update_ground_controls(model_key: str):
        distributed = model_key in {"dis2p", "hetero3"}
        inflated = model_key in {"z2p", "hetero3"}
        return (
            not distributed,
            not inflated,
            paper_rate_distribution_for_model(model_key),
        )

    @app.callback(
        Output("synthetic-rate-distribution-preview", "figure"),
        Output("synthetic-rate-preview-note", "children"),
        Input("synthetic-ground-model", "value"),
        Input("synthetic-rate-distribution", "value"),
        Input("synthetic-mu-lambda", "value"),
        Input("synthetic-sigma-lambda", "value"),
        Input("synthetic-p-zero", "value"),
    )
    def update_rate_preview(model_key, rate_distribution, mu_lambda, sigma_lambda, p_zero):
        distributed = model_key in {"dis2p", "hetero3"}
        inflated = model_key in {"z2p", "hetero3"}
        effective_distribution = rate_distribution if distributed else "fixed"
        try:
            effective_sigma = float(sigma_lambda) if distributed else 0.0
            effective_phi = float(p_zero) if inflated else 0.0
            figure = rate_distribution_figure(
                effective_distribution,
                float(mu_lambda),
                effective_sigma,
                effective_phi,
            )
        except Exception as exc:
            fallback = rate_distribution_figure("fixed", 1.0, 0.0, 0.0)
            return fallback, note("Adjust the distribution parameters", str(exc), tone="amber")

        if not distributed:
            body = "Every engaging cell receives the same rate λ. σλ is fixed at zero for this population structure."
        elif effective_distribution == "gamma":
            body = (
                "For a randomly chosen cell, the population distribution is "
                "φ₀δ₀ + (1 − φ₀)Gamma(μλ, σλ). The shaded density integrates "
                f"to {1.0 - effective_phi:.2f}."
            )
        else:  # pragma: no cover - the public model mapping is fixed
            body = f"The selected model uses {RATE_DISTRIBUTION_LABELS[effective_distribution]}."
        if effective_phi > 0 and not distributed:
            body += f" The two bars have masses φ₀ = {effective_phi:.2f} and 1 − φ₀ = {1.0 - effective_phi:.2f}."
        return figure, html.P(body)

    @app.callback(
        Output("synthetic-posterior-figure", "figure"),
        Output("synthetic-posterior-figure", "style"),
        Output("synthetic-posterior-selection-summary", "children"),
        Input("synthetic-posterior-model-filter", "value"),
        State("synthetic-posterior-data", "data"),
        prevent_initial_call=True,
    )
    def update_posterior_model_view(selected_models, payload):
        selected_models = selected_models or []
        try:
            draws, selected, truth = posterior_draws_from_store(
                payload or {},
                selected_models,
            )
            parameters = posterior_parameters_for_models(selected)
            figure = joint_posterior_figure_from_draws(
                draws,
                selected,
                truth,
                parameters=parameters,
            )
        except (TypeError, ValueError) as exc:
            figure = go.Figure()
            figure.add_annotation(
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                text="Select at least one inference result",
                showarrow=False,
                font={"family": "Iowan Old Style, Georgia, serif", "size": 18},
            )
            figure.update_layout(
                template="none",
                height=430,
                paper_bgcolor="#FBF7ED",
                plot_bgcolor="#F3EDDF",
                margin={"l": 40, "r": 40, "t": 40, "b": 40},
            )
            figure.update_xaxes(visible=False)
            figure.update_yaxes(visible=False)
            return (
                figure,
                {"height": "430px"},
                str(exc).capitalize() + ".",
            )

        parameter_symbols = {
            "mu_lambda": "μλ",
            "sigma_lambda": "σλ",
            "p_zero": "φ₀",
        }
        symbols = [parameter_symbols[parameter] for parameter in parameters]
        if len(symbols) == 1:
            parameter_text = symbols[0]
        else:
            parameter_text = ", ".join(symbols[:-1]) + f" and {symbols[-1]}"
        return (
            figure,
            {"height": f"{int(figure.layout.height)}px"},
            f"{len(selected)} model{'s' if len(selected) != 1 else ''} shown · "
            f"{len(parameters)} parameter{'s' if len(parameters) != 1 else ''}: {parameter_text}.",
        )

    @app.callback(
        Output("synthetic-data", "data"),
        Output("synthetic-truth", "data"),
        Output("synthetic-time", "data"),
        Output("synthetic-preview", "children"),
        Output("synthetic-inference-section", "className"),
        Output("synthetic-run", "disabled"),
        Output("synthetic-generate-status", "children"),
        Input("synthetic-generate", "n_clicks"),
        State("synthetic-ground-model", "value"),
        State("synthetic-n-cells", "value"),
        State("synthetic-observation-time", "value"),
        State("synthetic-mu-lambda", "value"),
        State("synthetic-sigma-lambda", "value"),
        State("synthetic-p-zero", "value"),
        State("synthetic-simulation-seed", "value"),
        prevent_initial_call=True,
    )
    def generate_data(_clicks, model_key, n_cells, observation_time, mu_lambda, sigma_lambda, p_zero, seed_raw):
        try:
            from webapp.analysis_ui import parse_optional_seed

            frame, truth = simulate_event_counts(
                model_key=model_key,
                n_cells=int(n_cells),
                obs_time=float(observation_time),
                mu_lambda=float(mu_lambda),
                sigma_lambda=float(sigma_lambda),
                p_zero=float(p_zero),
                seed=parse_optional_seed(seed_raw),
            )
            frame = validate_count_frame(frame)
        except Exception as exc:
            return no_update, no_update, no_update, no_update, "orca-workflow-panel is-hidden", True, note("Could not generate data", str(exc), tone="amber")

        truth_frame = pd.DataFrame(
            {
                "Quantity": [
                    "Population structure",
                    "Engaging-cell rate distribution",
                    "Mean event rate among engaging cells, μλ",
                    "Continuous cell-to-cell heterogeneity in event rates, σλ",
                    "Fraction of nonengaging cells, φ₀",
                    "Observation time T",
                    "Seed used",
                ],
                "Ground truth": [
                    str(truth.get("model_label")),
                    str(truth.get("rate_distribution_label")),
                    f"{float(truth.get('mu_lambda', 0)):.4g}",
                    f"{float(truth.get('sigma_lambda', 0)):.4g}",
                    f"{float(truth.get('p_zero', 0)):.4g}",
                    f"{float(truth.get('observation_time', 1)):.4g}",
                    str(truth.get("seed")) if truth.get("seed") is not None else "Not fixed",
                ],
            }
        )
        from webapp.analysis_ui import data_table

        preview = html.Div(
            [
                html.Span("Step B", className="orca-section-label"),
                html.H2("Inspect the generated dataset"),
                data_table(truth_frame),
                data_overview(frame),
                csv_download_link(frame, "orca_synthetic_counts.csv", "Download this synthetic dataset"),
            ],
            className="orca-workflow-panel",
        )
        return table_records(frame), dict(truth), float(observation_time), preview, "orca-workflow-panel", False, note("Dataset generated", "The synthetic data passed the validation checks.", tone="teal")

    @app.callback(
        Output("synthetic-results", "children"),
        Output("synthetic-download", "children"),
        Output("synthetic-run-status", "children"),
        Input("synthetic-run", "n_clicks"),
        State("synthetic-data", "data"),
        State("synthetic-truth", "data"),
        State("synthetic-time", "data"),
        State("synthetic-models", "value"),
        State("synthetic-particles", "value"),
        State("synthetic-chains", "value"),
        State("synthetic-cores", "value"),
        State("synthetic-seed", "value"),
        State("synthetic-threshold", "value"),
        State("synthetic-correlation", "value"),
        State("synthetic-prior-bounds", "value"),
        State("synthetic-sigma-prior", "value"),
        prevent_initial_call=True,
        background=True,
        interval=350,
        progress=[
            Output("synthetic-pymc-progress-bar", "value"),
            Output("synthetic-pymc-progress-label", "children"),
            Output("synthetic-pymc-progress-meta", "children"),
            Output("synthetic-chain-progress", "children"),
        ],
        progress_default=[
            0,
            "PyMC SMC sampler",
            "Start inference to see each chain's SMC stage and tempering value β.",
            [],
        ],
        running=[
            (
                Output("synthetic-pymc-progress", "className"),
                "orca-pymc-progress is-active",
                "orca-pymc-progress is-hidden",
            ),
            (
                Output("synthetic-inference-section", "aria-busy"),
                "true",
                "false",
            ),
            (
                Output("synthetic-run", "className"),
                "orca-button primary full is-running",
                "orca-button primary full",
            ),
        ],
    )
    def run_inference(set_progress, _clicks, records, truth, observation_time, models, particles, chains, cores, seed, threshold, correlation, prior_bounds, sigma_prior):
        try:
            if not records:
                raise ValueError("Generate a dataset first.")
            if not models:
                raise ValueError("Select at least one model for inference.")
            frame = validate_count_frame(pd.DataFrame(records))
            settings = settings_from_values(particles, chains, cores, seed, threshold, correlation, prior_bounds, sigma_prior)
            selected_models = list(models)
            total_chains = int(settings.chains)
            chain_states: dict[int, tuple[int, float]] = {}

            def model_started(index: int, total: int, label: str) -> None:
                chain_states.clear()
                set_progress(
                    sampling_progress_payload(
                        model_index=index,
                        total_models=total,
                        model_label=label,
                        chains=total_chains,
                        particles=int(settings.draws),
                        chain_states=chain_states,
                    )
                )

            def sampler_progress(
                index: int,
                total: int,
                label: str,
                chain: int,
                stage: int,
                beta: float,
            ) -> None:
                chain_index = int(chain)
                if not 0 <= chain_index < total_chains:
                    return
                chain_states[chain_index] = (
                    max(0, int(stage)),
                    min(1.0, max(0.0, float(beta))),
                )
                set_progress(
                    sampling_progress_payload(
                        model_index=index,
                        total_models=total,
                        model_label=label,
                        chains=total_chains,
                        particles=int(settings.draws),
                        chain_states=chain_states,
                    )
                )

            results = run_count_models(
                frame,
                float(observation_time),
                settings=settings,
                model_keys=selected_models,
                progress_callback=model_started,
                sampler_progress_callback=sampler_progress,
            )
            set_progress(
                sampling_complete_payload(
                    chains=total_chains,
                    particles=int(settings.draws),
                    chain_states=chain_states,
                )
            )
            content, download = render_validation_results(
                results,
                data=frame,
                observation_time=float(observation_time),
                settings=settings,
                truth=truth,
                download_name="orca_synthetic_validation.zip",
            )
        except Exception as exc:
            return html.Div(), html.Div(), note("Inference did not complete", str(exc), tone="amber")
        return content, download, html.Div()
