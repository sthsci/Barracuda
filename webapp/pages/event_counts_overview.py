"""Entry page for the event count workflows."""

from __future__ import annotations

from dash import dcc, html

from webapp.analysis_ui import model_title
from webapp.core.inference import MODEL_SPECS
from webapp.ui import hero, schematic_figure


PATH = "/event-counts"
TITLE = "Event count analysis"


def _workflow_link(label: str, title: str, body: str, path: str) -> dcc.Link:
    return dcc.Link(
        [
            html.Span(label, className="barracuda-section-label"),
            html.H3(title),
            html.P(body),
            html.Span("Open analysis", className="barracuda-workflow-choice-action"),
        ],
        href=path,
        className="barracuda-workflow-choice",
    )


def layout() -> html.Div:
    constraints = {
        "hetero3": "μλ > 0 · σλ > 0 · φ₀ > 0",
        "z2p": "μλ > 0 · σλ = 0 · φ₀ > 0",
        "dis2p": "μλ > 0 · σλ > 0 · φ₀ = 0",
        "homo": "μλ > 0 · σλ = 0 · φ₀ = 0",
    }
    return html.Div(
        [
            hero(
                "Event counts",
                "Choose an event count workflow",
                "Use total contacts or kills per cell to compare population structures. Add donor labels when you also need to separate variation within and between donors.",
                badge="Counts per cell • Optional donor labels",
            ),
            html.Div(
                [
                    html.H2("Choose an analysis"),
                    html.P(
                        "Choose whether donor labels are part of the model. Each analysis then guides you through the appropriate data and inference workflow.",
                        className="barracuda-section-lead",
                    ),
                ],
                className="barracuda-section-intro",
            ),
            html.Div(
                [
                    _workflow_link(
                        "Without donor labels",
                        "Donor ignorant",
                        "Choose synthetic data or analyse one to four experimental conditions as independent populations.",
                        "/event-counts/donor-ignorant",
                    ),
                    _workflow_link(
                        "With donor labels",
                        "Donor aware",
                        "Analyse one to four conditions with a donor hierarchy, then separate within-donor variation from differences between donors.",
                        "/event-counts/donor-aware",
                    ),
                ],
                className="barracuda-workflow-choice-grid two",
            ),
            html.Section(
                [
                    html.Span("Models used in the paper", className="barracuda-section-label"),
                    html.H2("Four population structures"),
                    html.P(
                        "Each model gives a different explanation for variation in the counts observed across cells.",
                        className="barracuda-section-lead",
                    ),
                    schematic_figure(
                        "/assets/event_count_models.png",
                        "Four event rate models. Zero inflated Gamma combines a nonengaging mass at rate zero with continuously varying positive rates; zero inflated combines the nonengaging mass with one shared positive rate; Gamma has continuously varying positive rates and no nonengaging mass; homogeneous assigns one positive rate to every cell.",
                        "The paper compares the same four event count models throughout. μλ is the mean event rate among engaging cells, σλ measures continuous cell-to-cell heterogeneity in event rates, and φ₀ is the fraction of nonengaging cells.",
                        variant="models",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3(model_title(key)),
                                    html.P(MODEL_SPECS[key].description),
                                    html.Span(constraints[key]),
                                ],
                                className="barracuda-model-definition",
                            )
                            for key in ("hetero3", "z2p", "dis2p", "homo")
                        ],
                        className="barracuda-model-definition-grid",
                    ),
                    html.Details(
                        [
                            html.Summary("Paper notation and model equations"),
                            dcc.Markdown(
                                r"""
For cell $i$, the observed event count is $N_i$ and the common observation time is $T$:

$$N_i \mid \lambda_i,T \sim \operatorname{Poisson}(\lambda_iT).$$

For $\mathcal{M}_{\Gamma}$ and $\mathcal{M}_{\mathrm{ZI}\Gamma}$, event rates among engaging cells follow a Gamma distribution with shape $\alpha$ and rate $\beta$:

$$\lambda_i \sim \operatorname{Gamma}(\alpha,\beta),\qquad
\alpha=\frac{\mu_\lambda^2}{\sigma_\lambda^2},\qquad
\beta=\frac{\mu_\lambda}{\sigma_\lambda^2}.$$

$\mu_\lambda$ is the mean event rate among engaging cells. $\sigma_\lambda$ measures continuous cell-to-cell heterogeneity in their event rates. $\phi_0$ is the fraction of nonengaging cells assigned $\lambda_i=0$ in the zero inflated models.
""",
                                mathjax=True,
                                className="barracuda-model-equations",
                            ),
                        ],
                        className="barracuda-details barracuda-model-equation-details",
                    ),
                ],
                className="barracuda-model-reference",
            ),
        ]
    )
