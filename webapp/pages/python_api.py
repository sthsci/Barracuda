"""Reference page for the public ``barracuda`` Python package."""

from __future__ import annotations

from dash import html

from webapp.ui import note, page_header


PATH = "/python-api"
TITLE = "Python package API"
DOCS_URL = "https://sthsci.github.io/Barracuda/"
PYPI_URL = "https://pypi.org/project/cyto-barracuda/"
SOURCE_URL = "https://github.com/sthsci/Barracuda/tree/main/src/barracuda"


COUNT_EXAMPLE = """from barracuda import InferenceSettings, evidence_table, run_count_models

settings = InferenceSettings(draws=256, chains=1, cores=1, seed=42)
results = run_count_models(
    counts,  # pandas DataFrame: cell_id, count
    observation_time=1.0,
    settings=settings,
    model_keys=["homo", "z2p", "dis2p", "hetero3"],
)
print(evidence_table(results))"""

DONOR_EXAMPLE = """from barracuda import run_donor_aware_models

results = run_donor_aware_models(
    donor_counts,  # cell_id, donor_id, count
    observation_time=1.0,
    settings=settings,
    model_keys=["dis2p", "hetero3"],
)"""

TRAJECTORY_EXAMPLE = """from barracuda import (
    TrajectorySettings,
    run_trajectory_conditions,
    trajectory_evidence_frame,
)

results = run_trajectory_conditions(
    histories,  # cell_id, history; optional condition
    settings=TrajectorySettings(draws=256, chains=1, cores=1, seed=42),
)
print(trajectory_evidence_frame(results))"""


API_GROUPS = (
    (
        "Configuration and results",
        (
            ("__version__", "Installed package version."),
            ("MODEL_SPECS", "Candidate event-count model definitions keyed by model name."),
            ("TRAJECTORY_MODEL_SPECS", "Candidate trajectory model definitions keyed by model name."),
            ("InferenceSettings", "Validated SMC controls and priors for event-count inference."),
            ("InferenceResult", "One fitted event-count model and its evidence metadata."),
            ("TrajectorySettings", "Validated SMC controls and priors for trajectory inference."),
            ("TrajectorySimulationSpec", "Ground truth for one simulated trajectory condition."),
            ("TrajectoryResult", "One fitted trajectory model and its evidence metadata."),
        ),
    ),
    (
        "Event-count data",
        (
            ("sample_count_frame", "Return a small built-in donor-ignorant dataset."),
            ("sample_donor_frame", "Return a small built-in donor-aware dataset."),
            ("simulate_event_counts", "Simulate canonical count data and return its ground truth."),
            ("normalize_condition_frame", "Map an uploaded table to the condition-aware schema."),
            ("validate_count_frame", "Validate and canonicalise cell_id and count columns."),
            ("validate_donor_frame", "Validate and canonicalise donor-aware count columns."),
            ("validate_condition_frame", "Validate one to four independent conditions."),
        ),
    ),
    (
        "Event-count inference and output",
        (
            ("run_count_models", "Fit donor-ignorant candidate models with PyMC SMC."),
            ("run_donor_ignorant_models", "Descriptive alias for run_count_models."),
            ("run_donor_models", "Fit donor-aware hierarchical candidate models."),
            ("run_donor_aware_models", "Descriptive alias for run_donor_models."),
            ("run_condition_models", "Fit each condition independently with shared settings."),
            ("evidence_table", "Rank fitted models using their SMC marginal likelihoods."),
            ("summary_table", "Combine posterior means and credible intervals."),
            ("build_results_zip", "Bundle tables and posterior files for one analysis."),
            ("build_condition_results_zip", "Bundle results for a condition-wise analysis."),
        ),
    ),
    (
        "Trajectory data, inference and output",
        (
            ("simulate_trajectory_frame", "Simulate ordered contact histories for one or more conditions."),
            ("normalize_trajectory_frame", "Return canonical cell_id, condition and history data."),
            ("validate_trajectory_frame", "Validate and canonicalise trajectory data."),
            ("expanded_trajectory_frame", "Expand histories to one row per ordered contact."),
            ("run_trajectory_conditions", "Fit selected trajectory models within each condition."),
            ("trajectory_evidence_frame", "Rank trajectory models within each condition."),
            ("trajectory_summary_frame", "Summarise trajectory posterior parameters with HDIs."),
            ("trajectory_posterior_draws", "Extract paired posterior draws with public parameter names."),
            ("build_trajectory_archive", "Bundle trajectory data, diagnostics, draws and NetCDF files."),
        ),
    ),
)

DOCUMENTED_NAMES = tuple(
    name
    for _group_title, entries in API_GROUPS
    for name, _description in entries
)


def _code(source: str) -> html.Pre:
    return html.Pre(html.Code(source), className="barracuda-code-block")


def _api_group(title: str, entries: tuple[tuple[str, str], ...]) -> html.Details:
    return html.Details(
        [
            html.Summary(f"{title} · {len(entries)} names"),
            html.Div(
                [
                    html.Div(
                        [html.Code(name), html.P(description)],
                        className="barracuda-api-entry",
                    )
                    for name, description in entries
                ],
                className="barracuda-api-grid",
            ),
        ],
        open=True,
        className="barracuda-details barracuda-api-group",
    )


def layout() -> html.Div:
    return html.Div(
        [
            page_header(
                "Resources",
                "Python package API",
                "Use BARRACUDA's simulation, inference, and export functions directly from Python.",
                badge="PyPI: cyto-barracuda · Import: barracuda · Python 3.12",
                crumb="Python API",
            ),
            html.Nav(
                [
                    html.Span("On this page", className="barracuda-section-label"),
                    html.Ul(
                        [
                            html.Li(html.A([html.Span("01"), "Install"], href="#install")),
                            html.Li(html.A([html.Span("02"), "Event counts"], href="#event-counts-api")),
                            html.Li(html.A([html.Span("03"), "Donor-aware"], href="#donor-api")),
                            html.Li(html.A([html.Span("04"), "Trajectories"], href="#trajectory-api")),
                            html.Li(html.A([html.Span("05"), "Reference"], href="#api-reference")),
                        ],
                        className="barracuda-toc-list",
                    ),
                ],
                className="barracuda-toc",
                **{"aria-label": "Python API page contents"},
            ),
            html.Section(
                [
                    html.Span("Distribution and import name", className="barracuda-section-label"),
                    html.H2("Install from PyPI"),
                    html.P(
                        "The PyPI project is cyto-barracuda because the name barracuda was already registered. Import it as barracuda.",
                        className="barracuda-section-lead",
                    ),
                    _code("python -m pip install cyto-barracuda\n\npython -c \"import barracuda; print(barracuda.__version__)\""),
                    note(
                        "Start small",
                        "SMC inference is computationally expensive. Use low particle and chain counts to check a workflow, then select settings appropriate for the scientific analysis.",
                        tone="amber",
                    ),
                ],
                id="install",
                className="barracuda-workflow-panel barracuda-lesson-section",
            ),
            html.Section(
                [
                    html.Span("Donor-ignorant workflow", className="barracuda-section-label"),
                    html.H2("Event counts"),
                    html.P(
                        "Provide one non-negative integer count per cell. To analyse up to four conditions independently, add a condition column and call run_condition_models.",
                        className="barracuda-section-lead",
                    ),
                    _code(COUNT_EXAMPLE),
                ],
                id="event-counts-api",
                className="barracuda-workflow-panel barracuda-lesson-section",
            ),
            html.Section(
                [
                    html.Span("Hierarchical workflow", className="barracuda-section-label"),
                    html.H2("Donor-aware event counts"),
                    html.P(
                        "Include donor_id for every cell to separate within-donor heterogeneity from differences among donor means.",
                        className="barracuda-section-lead",
                    ),
                    _code(DONOR_EXAMPLE),
                ],
                id="donor-api",
                className="barracuda-workflow-panel barracuda-lesson-section",
            ),
            html.Section(
                [
                    html.Span("Ordered-contact workflow", className="barracuda-section-label"),
                    html.H2("Contact trajectories"),
                    html.P(
                        "Store each cell's ordered outcomes in history, for example 0,0,1,0. Use a blank history for a cell with no observed contacts.",
                        className="barracuda-section-lead",
                    ),
                    _code(TRAJECTORY_EXAMPLE),
                ],
                id="trajectory-api",
                className="barracuda-workflow-panel barracuda-lesson-section",
            ),
            html.Section(
                [
                    html.Span("Top-level imports", className="barracuda-section-label"),
                    html.H2("Public API reference"),
                    html.P(
                        "Import all supported public names directly from barracuda. Names beginning with an underscore are implementation details.",
                        className="barracuda-section-lead",
                    ),
                    *[_api_group(title, entries) for title, entries in API_GROUPS],
                    html.Div(
                        [
                            html.A(
                                "Open the full Python API documentation",
                                href=DOCS_URL,
                                target="_blank",
                                rel="noreferrer",
                                className="barracuda-button primary",
                            ),
                            html.A(
                                "View cyto-barracuda on PyPI",
                                href=PYPI_URL,
                                target="_blank",
                                rel="noreferrer",
                                className="barracuda-button secondary",
                            ),
                            html.A(
                                "Browse the Python source",
                                href=SOURCE_URL,
                                target="_blank",
                                rel="noreferrer",
                                className="barracuda-button secondary",
                            ),
                        ],
                        className="barracuda-bayes-actions",
                    ),
                ],
                id="api-reference",
                className="barracuda-workflow-panel barracuda-lesson-section",
            ),
        ]
    )


__all__ = [
    "API_GROUPS",
    "DOCS_URL",
    "DOCUMENTED_NAMES",
    "PATH",
    "PYPI_URL",
    "SOURCE_URL",
    "TITLE",
    "layout",
]
