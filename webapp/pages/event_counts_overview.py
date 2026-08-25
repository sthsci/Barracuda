"""Entry page for event-count workflows."""

from __future__ import annotations

from dash import dcc, html

from webapp.analysis_ui import model_title
from webapp.ui import page_header, schematic_figure


PATH = "/event-counts"
TITLE = "Event count analysis"


def _route(title: str, internal: str, body: str, schema: str, path: str) -> html.Article:
    return html.Article(
        [
            html.H3(title),
            html.Small(internal),
            html.P(body),
            html.Code(schema),
            dcc.Link("Open analysis →", href=path, className="barracuda-card-link"),
        ],
        className="barracuda-data-route",
    )


def layout() -> html.Div:
    rows = (
        ("homo", "One shared rate", "No", "Random count variation only", "μλ"),
        ("dis2p", "Varies continuously", "No", "Continuous cell-to-cell rate variation", "μλ, σλ"),
        ("z2p", "One shared positive rate", "Yes", "A nonengaging group plus engaging cells", "μλ, φ₀"),
        ("hetero3", "Varies continuously", "Yes", "Continuous rate variation plus nonengaging cells", "μλ, σλ, φ₀"),
    )
    return html.Div(
        [
            page_header(
                "Analyse",
                "Event count analysis",
                "Compare explanations for variation in total contacts or kills observed across individual cells.",
                crumb="Event counts",
            ),
            html.Section(
                [
                    html.H2("What information is present in your table?"),
                    html.P("Choose the route that matches your columns. Both routes support one to four experimental conditions.", className="barracuda-section-lead"),
                    html.Div(
                        [
                            _route(
                                "Counts without donor labels",
                                "Donor-ignorant",
                                "Use one total count per cell when donor identity is unavailable or outside the question.",
                                "cell_id · condition · count",
                                "/event-counts/donor-ignorant",
                            ),
                            _route(
                                "Counts grouped by donor",
                                "Donor-aware",
                                "Add donor identifiers to separate variation among cells within donors from differences between donors.",
                                "cell_id · donor_id · condition · count",
                                "/event-counts/donor-aware",
                            ),
                        ],
                        className="barracuda-data-route-grid",
                    ),
                ],
                className="barracuda-overview-section",
            ),
            html.Section(
                [
                    html.Span("Model reference", className="barracuda-eyebrow"),
                    html.H2("Four explanations for population structure"),
                    html.P("The models differ in whether engaging cells share a rate and whether a distinct nonengaging fraction is present.", className="barracuda-section-lead"),
                    html.Div(
                        [
                            html.Table(
                                [
                                    html.Thead(html.Tr([html.Th(label, scope="col") for label in ("Model", "Cell-specific rates", "Nonengaging fraction", "Biological interpretation", "Parameters")])),
                                    html.Tbody(
                                        [
                                            html.Tr(
                                                [
                                                    html.Th(model_title(key), scope="row"),
                                                    html.Td(rates),
                                                    html.Td(nonengaging),
                                                    html.Td(interpretation),
                                                    html.Td(html.Code(parameters)),
                                                ]
                                            )
                                            for key, rates, nonengaging, interpretation, parameters in rows
                                        ]
                                    ),
                                ],
                                className="barracuda-model-table",
                            )
                        ],
                        className="barracuda-table-scroll",
                        tabIndex=0,
                        role="region",
                        **{"aria-label": "Scrollable comparison of event count models"},
                    ),
                    schematic_figure(
                        "/assets/event_count_models_panel_a.png",
                        "Four event-rate models compare a shared positive rate, continuous positive-rate variation, a nonengaging fraction, and both continuous variation and a nonengaging fraction.",
                        "The same four event-count models are compared throughout BARRACUDA.",
                        variant="models",
                    ),
                    html.Details(
                        [
                            html.Summary("Model equations and paper notation"),
                            dcc.Markdown(
                                r"""
For cell $i$, the observed event count is $N_i$ and the common observation time is $T$:

$$N_i \mid \lambda_i,T \sim \operatorname{Poisson}(\lambda_iT).$$

For the continuously heterogeneous models:

$$\lambda_i \sim \operatorname{Gamma}(\alpha,\beta),\qquad
\alpha=\frac{\mu_\lambda^2}{\sigma_\lambda^2},\qquad
\beta=\frac{\mu_\lambda}{\sigma_\lambda^2}.$$

$\mu_\lambda$ is the mean rate among engaging cells, $\sigma_\lambda$ is continuous rate heterogeneity, and $\phi_0$ is the nonengaging fraction.
""",
                                mathjax=True,
                                className="barracuda-model-equations",
                            ),
                        ],
                        className="barracuda-details",
                    ),
                ],
                className="barracuda-overview-section",
            ),
        ]
    )
