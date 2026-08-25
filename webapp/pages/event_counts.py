"""Donor ignorant synthetic validation and real data analysis."""

from __future__ import annotations

from typing import Final

from dash import Input, Output, dcc, html

from webapp.pages import synthetic_validation
from webapp.pages.analysis_page import layout as analysis_layout
from webapp.pages.analysis_page import register_callbacks as register_analysis_callbacks
from webapp.ui import page_header


PATH = "/event-counts/donor-ignorant"
TITLE = "Donor ignorant analysis"

WORKFLOW_ID: Final[str] = "donor-ignorant-workflow"
SYNTHETIC_PANEL_ID: Final[str] = "donor-ignorant-synthetic-panel"
OWN_DATA_PANEL_ID: Final[str] = "donor-ignorant-own-data-panel"


def _without_hero(component: html.Div) -> list:
    """Return an existing workflow body without its repeated route hero."""

    children = component.children
    if not isinstance(children, (list, tuple)):
        return [children]
    return [
        child
        for child in children
        if getattr(child, "className", None) not in {"barracuda-hero", "barracuda-page-header"}
    ]


def _workflow_option(title: str, description: str) -> html.Div:
    return html.Div(
        [html.Strong(title), html.Small(description)],
        className="barracuda-model-option-copy",
    )


def _synthetic_body() -> html.Div:
    """Compose the established synthetic page without changing its IDs."""

    return html.Div(_without_hero(synthetic_validation.layout()))


def _own_data_body() -> html.Div:
    """Compose the condition-aware real data page with its original IDs."""

    page = analysis_layout(
        prefix="counts",
        donor_aware=False,
        kicker="Event counts · Donor ignorant · Real data",
        title="Event counts without donor labels",
        lead=(
            "Analyse as many as four experimental conditions. Upload a CSV, "
            "or enter counts directly in the browser."
        ),
        badge="Up to four experimental conditions · maximum 1,000 cells per condition",
    )
    return html.Div(_without_hero(page))


def layout() -> html.Div:
    return html.Div(
        [
            page_header(
                "Analyse",
                "Counts without donor labels",
                "Validate the method against a known synthetic truth, or analyse your own count data without donor labels.",
                badge="Synthetic validation • Real data analysis",
                crumb="Counts without donor labels",
            ),
            html.Section(
                [
                    html.Span("Start here", className="barracuda-section-label"),
                    html.H2("Which data do you want to use?"),
                    html.P(
                        "The choice changes the input workflow only. Provide a small dataset or generate one from a known truth; both routes run inference with the same four donor ignorant event count models.",
                        className="barracuda-section-lead",
                    ),
                    dcc.RadioItems(
                        id=WORKFLOW_ID,
                        options=[
                            {
                                "label": _workflow_option(
                                    "Synthetic data",
                                    "Set a known ground truth, generate counts and check parameter recovery and model evidence.",
                                ),
                                "value": "synthetic",
                            },
                            {
                                "label": _workflow_option(
                                    "My own data",
                                    "Provide a small dataset by uploading or entering one to four experimental conditions, then run inference for each condition independently.",
                                ),
                                "value": "own-data",
                            },
                        ],
                        value=None,
                        className="barracuda-model-checklist",
                        inputClassName="barracuda-check-input",
                        labelClassName="barracuda-model-option",
                    ),
                    html.P(
                        "Choose one option to continue.",
                        id="donor-ignorant-workflow-status",
                        className="barracuda-help",
                        role="status",
                        **{"aria-live": "polite"},
                    ),
                ],
                className="barracuda-workflow-panel",
            ),
            html.Div(
                _synthetic_body(),
                id=SYNTHETIC_PANEL_ID,
                className="is-hidden",
            ),
            html.Div(
                _own_data_body(),
                id=OWN_DATA_PANEL_ID,
                className="is-hidden",
            ),
        ]
    )


def register_callbacks(app) -> None:
    @app.callback(
        Output(SYNTHETIC_PANEL_ID, "className"),
        Output(OWN_DATA_PANEL_ID, "className"),
        Output("donor-ignorant-workflow-status", "children"),
        Input(WORKFLOW_ID, "value"),
    )
    def choose_workflow(workflow: str | None):
        if workflow == "synthetic":
            return "barracuda-merged-workflow", "is-hidden", "Synthetic validation selected."
        if workflow == "own-data":
            return "is-hidden", "barracuda-merged-workflow", "Own data analysis selected."
        return "is-hidden", "is-hidden", "Choose one option to continue."

    # Retain the established ``counts-*`` callback contract for the complete
    # condition-aware real data workflow.
    register_analysis_callbacks(app, prefix="counts", donor_aware=False)

    # The current registry normally registers the legacy synthetic route first.
    # This guard also keeps the merged page functional if that route later
    # remains only as a compatibility alias.
    if not any("synthetic-data.data" in key for key in app.callback_map):
        synthetic_validation.register_callbacks(app)
