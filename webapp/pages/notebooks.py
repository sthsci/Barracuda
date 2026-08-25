"""Google Colab teaching and analysis notebooks."""

from __future__ import annotations

from dash import dcc, html

from webapp.ui import page_header


PATH = "/notebooks"
TITLE = "Colab notebooks"
COLAB_ROOT = "https://colab.research.google.com/github/sthsci/Barracuda/blob/main"

NOTEBOOKS = (
    {
        "category": "Start here",
        "title": "Run the BARRACUDA web app",
        "path": "notebooks/00_run_the_barracuda_web_app.ipynb",
        "description": "Launch the interactive BARRACUDA web app in Google Colab.",
    },
    {
        "category": "Teaching",
        "title": "Bayesian inference 101",
        "path": "notebooks/01_bayesian_inference_101.ipynb",
        "description": "Learn priors, likelihoods, posteriors, SMC and Bayes factors with small examples.",
    },
    {
        "category": "Teaching",
        "title": "Event count model tutorial",
        "path": "notebooks/02_event_count_model_tutorial.ipynb",
        "description": "Explore the four population models used for per-cell event counts.",
    },
    {
        "category": "Teaching",
        "title": "Trajectory model tutorial",
        "path": "notebooks/03_trajectory_model_tutorial.ipynb",
        "description": "Explore how cell heterogeneity and contact history alter killing decisions.",
    },
    {
        "category": "Analysis",
        "title": "Event count analysis",
        "path": "notebooks/04_event_count_analysis.ipynb",
        "description": "Adapt the donor-ignorant count workflow to your own experimental conditions.",
    },
    {
        "category": "Analysis",
        "title": "Donor-aware analysis",
        "path": "notebooks/05_donor_aware_analysis.ipynb",
        "description": "Analyse count data while separating within-donor and between-donor variation.",
    },
    {
        "category": "Analysis",
        "title": "Trajectory analysis",
        "path": "notebooks/06_trajectory_analysis.ipynb",
        "description": "Analyse ordered contact histories and compare trajectory mechanisms.",
    },
)


def layout() -> html.Div:
    sections = []
    introductions = {
        "Start here": "Open the web app before choosing a focused notebook.",
        "Teaching": "Build intuition with guided examples and synthetic data.",
        "Analysis": "Use the reusable workflows as a starting point for your own data.",
    }
    for category in introductions:
        items = [item for item in NOTEBOOKS if item["category"] == category]
        sections.append(
            html.Section(
                [
                    html.Span("Notebook collection", className="barracuda-section-label"),
                    html.H2(category),
                    html.P(introductions[category], className="barracuda-section-lead"),
                    html.Div(
                        [
                            html.A(
                                [
                                    html.Span("Google Colab", className="barracuda-section-label"),
                                    html.H3(item["title"]),
                                    html.P(item["description"]),
                                    html.Span("Open notebook →", className="barracuda-workflow-choice-action"),
                                ],
                                href=f"{COLAB_ROOT}/{item['path']}",
                                target="_blank",
                                rel="noreferrer",
                                className="barracuda-workflow-choice",
                            )
                            for item in items
                        ],
                        className="barracuda-workflow-choice-grid",
                    ),
                ],
                className="barracuda-workflow-panel",
            )
        )
    return html.Div(
        [
            page_header(
                "Resources",
                "Learn and analyse in Google Colab",
                "Run BARRACUDA in Google Colab, follow guided examples, or adapt a workflow to approved data.",
                badge="Seven notebooks · Opens in a new tab",
                crumb="Google Colab notebooks",
            ),
            html.Div(
                [
                    html.Div([html.Strong("Prefer local Python?"), html.P("Install the package to use the simulation, inference, and export API directly.")]),
                    dcc.Link("Open the Python API", href="/python-api", className="barracuda-button secondary"),
                ],
                className="barracuda-resource-switcher",
            ),
            *sections,
        ]
    )


__all__ = ["COLAB_ROOT", "NOTEBOOKS", "PATH", "TITLE", "layout"]
