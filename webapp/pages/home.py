"""Barracuda project home page."""

from __future__ import annotations

from dash import dcc, html

from webapp.ui import hero, step_card


PATH = "/"
TITLE = "Home"


def _subpage_link(title: str, body: str, path: str) -> dcc.Link:
    return dcc.Link(
        [html.Strong(title), html.Span(body)],
        href=path,
        className="barracuda-explore-link",
    )


def layout() -> html.Div:
    return html.Div(
        [
            hero(
                "Bayesian Analysis Resolving Randomness and Alternative Causes Underlying Differential Activity",
                "BARRACUDA",
                "A Bayesian framework for finding where variation in immune cell cytotoxicity comes from.",
                badge="Event counts • Donor structure • Contact trajectories",
            ),
            html.Div(
                [
                    html.P(
                        "Why do immune cells behave differently? BARRACUDA tests whether variation comes from chance, stable differences between cells, donor effects or previous interactions.",
                        className="barracuda-question",
                    ),
                    html.P(
                        "Use counts to separate randomness from population structure, then use ordered contact histories to test whether earlier encounters change later killing decisions.",
                        className="barracuda-home-summary",
                    ),
                ],
                className="barracuda-home-intro",
            ),
            html.Figure(
                [
                    html.A(
                        html.Img(
                            src="/assets/figure_abstract.png",
                            alt="Graphical abstract showing time lapse imaging of NK and tumour cells, conversion to a single cell contact history, and inference of population heterogeneity and changes across contacts.",
                            className="barracuda-graphic-abstract",
                        ),
                        href="/assets/figure_abstract.png",
                        target="_blank",
                        rel="noreferrer",
                        **{"aria-label": "Open the graphical abstract at full size"},
                    ),
                    html.Figcaption(
                        "Time lapse imaging provides counts and ordered histories that BARRACUDA uses to distinguish population heterogeneity from changes over time."
                    ),
                ],
                className="barracuda-graphic-figure",
            ),
            html.Div(
                [
                    html.H2("One question, three levels of information"),
                    html.P(
                        "BARRACUDA reads the same biology at increasing levels of detail.",
                        className="barracuda-section-lead",
                    ),
                ],
                className="barracuda-section-intro",
            ),
            html.Div(
                [
                    step_card("01", "Event counts", "Separate chance from stable differences between cells."),
                    step_card("02", "Donor structure", "Separate variation within donors from differences between donors."),
                    step_card("03", "Contact trajectories", "Test whether earlier contacts change later killing decisions."),
                ],
                className="barracuda-card-grid three barracuda-level-grid",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("New to Bayesian inference?", className="barracuda-section-label"),
                            html.P("Start with priors, likelihoods, posterior distributions and Bayes factors."),
                        ]
                    ),
                    dcc.Link("Open Bayesian inference 101", href="/bayesian-101", className="barracuda-button secondary"),
                ],
                className="barracuda-primer-callout",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Prefer a notebook?", className="barracuda-section-label"),
                            html.P("Run the teaching and analysis workflows directly in Google Colab."),
                        ]
                    ),
                    dcc.Link("Browse Colab notebooks", href="/notebooks", className="barracuda-button secondary"),
                ],
                className="barracuda-primer-callout",
            ),
            html.Div(
                [
                    html.H2("Explore BARRACUDA"),
                    html.P(
                        "Choose the data view that matches your question.",
                        className="barracuda-section-lead",
                    ),
                ],
                className="barracuda-section-intro",
            ),
            html.Div(
                [
                    html.Section(
                        [
                            html.Span("Count data", className="barracuda-section-label"),
                            html.H3(dcc.Link("Event counts", href="/event-counts")),
                            html.P(
                                "Analyse contacts or kills with or without a donor hierarchy. Donor ignorant analysis also includes synthetic validation.",
                                className="barracuda-explore-copy",
                            ),
                            html.Div(
                                [
                                    _subpage_link("Donor ignorant", "Choose synthetic data or analyse up to four conditions without donor labels.", "/event-counts/donor-ignorant"),
                                    _subpage_link("Donor aware", "Compare conditions while separating variation within and between donors.", "/event-counts/donor-aware"),
                                ],
                                className="barracuda-explore-links",
                            ),
                        ],
                        className="barracuda-explore-card",
                    ),
                    html.Section(
                        [
                            html.Span("Ordered data", className="barracuda-section-label"),
                            html.H3(dcc.Link("Contact trajectories", href="/trajectory")),
                            html.P(
                                "Use the order of successful and unsuccessful contacts to distinguish stable killing propensity from the effects of previous interactions.",
                                className="barracuda-explore-copy",
                            ),
                            html.Div(
                                [_subpage_link("Trajectory inference", "Choose synthetic data or upload ordered contact histories.", "/trajectory")],
                                className="barracuda-explore-links",
                            ),
                        ],
                        className="barracuda-explore-card",
                    ),
                ],
                className="barracuda-explore-grid",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Research context", className="barracuda-section-label"),
                            html.H2("Research team and code"),
                            html.P(
                                [
                                    "Elephes Sung",
                                    html.Sup("1"),
                                    ", Cathal Hosty",
                                    html.Sup("1"),
                                    ", Leanne Peiser",
                                    html.Sup("2"),
                                    ", Lara Stepan",
                                    html.Sup("2"),
                                    ", Daniel M Davis",
                                    html.Sup("1"),
                                    " and Ruben Perez-Carrasco",
                                    html.Sup("1"),
                                    ".",
                                ],
                                className="barracuda-research-team",
                            ),
                            html.H3("Affiliations", className="barracuda-affiliations-title"),
                            html.Ol(
                                [
                                    html.Li("Department of Life Sciences, Imperial College London, London SW7 2AZ, UK"),
                                    html.Li("Immuno-Oncology Cellular Therapy Thematic Research Center, Bristol Myers Squibb, Seattle, WA 98109, USA"),
                                ],
                                className="barracuda-affiliations",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.A(
                                "Open the GitHub repository",
                                href="https://github.com/sthsci/Barracuda",
                                target="_blank",
                                rel="noreferrer",
                                className="barracuda-button primary",
                            )
                        ],
                        className="barracuda-research-actions",
                    ),
                ],
                className="barracuda-research-band",
            ),
        ]
    )
