"""Trajectory model overview page."""

from __future__ import annotations

import pandas as pd
from dash import html

from webapp.analysis_ui import data_table
from webapp.ui import hero, markdown, note, step_card


PATH = "/trajectory"
TITLE = "Trajectory model"


def layout() -> html.Div:
    example = pd.DataFrame(
        {
            "cell_id": ["cell_001", "cell_001", "cell_001", "cell_002"],
            "contact_index": [1, 2, 3, 1],
            "outcome": [0, 1, 1, 0],
            "donor_id": ["donor_A", "donor_A", "donor_A", "donor_B"],
        }
    )
    return html.Div(
        [
            hero(
                "Trajectories",
                "Trajectory inference",
                "Ordered successful and unsuccessful contacts can separate stable cellular heterogeneity from the effects of previous interactions.",
                badge="Interface in development",
            ),
            note("Current availability", "The model and analysis code are available in the research repository. The interactive workflow is still being prepared.", tone="amber"),
            html.H2("What the trajectory model retains"),
            markdown("Event counts remember *how many* kills occurred. A trajectory also remembers *when* they occurred and what happened before each decision."),
            html.Div(
                [
                    markdown("$$x_{ij}\\sim\\mathrm{Bernoulli}(p_{ij})$$", class_name="orca-equation small", mathjax=True),
                    markdown("$$\\mathrm{logit}(p_{ij})=\\eta_i+\\beta_f f_{ij}+\\beta_s s_{ij}$$", class_name="orca-equation small", mathjax=True),
                ],
                className="orca-equation-row",
            ),
            markdown("Here, **ηᵢ** is cell *i*'s baseline killing propensity; **fᵢⱼ** and **sᵢⱼ** count previous failed and successful contacts. The coefficients **βf** and **βs** describe history effects."),
            html.Div(
                [
                    step_card("01", "Ordered input", "One row per contact, with cell ID, contact order and binary outcome."),
                    step_card("02", "Competing mechanisms", "Compare homogeneous and heterogeneous populations with or without history dependence."),
                    step_card("03", "Decision maps", "Summarise how prior successes and failures shift future killing probability."),
                ],
                className="orca-card-grid three",
            ),
            html.Div(
                [
                    html.Span("Input preview", className="orca-section-label"),
                    html.H2("Planned input format"),
                    data_table(example),
                    html.Button("Trajectory inference is not yet available", disabled=True, className="orca-button primary full"),
                    html.P("Use event count inference when you only need the number of contacts or kills per cell.", className="orca-help"),
                ],
                className="orca-workflow-panel",
            ),
        ]
    )
