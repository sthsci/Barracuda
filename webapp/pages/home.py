"""BARRACUDA project home and analysis chooser."""

from __future__ import annotations

from dash import dcc, html

from webapp.ui import task_card


PATH = "/"
TITLE = "Home"


def layout() -> html.Div:
    return html.Div(
        [
            html.Section(
                [
                    html.Div(
                        [
                            html.Span("Bayesian analysis of single cell cytotoxicity", className="barracuda-eyebrow"),
                            html.H1("Identify where variation in killer cell behaviour comes from"),
                            html.P(
                                "BARRACUDA compares stochastic, heterogeneous, donor structured, and history dependent explanations of contact and killing data.",
                                className="barracuda-home-lead",
                            ),
                            html.Div(
                                [
                                    html.A("Choose an analysis", href="#choose-analysis", className="barracuda-button primary"),
                                    dcc.Link("Try synthetic validation", href="/event-counts/donor-ignorant#donor-ignorant-workflow", className="barracuda-button secondary"),
                                ],
                                className="barracuda-home-actions",
                            ),
                            dcc.Link("Learn the Bayesian framework →", href="/bayesian-101", className="barracuda-text-link"),
                        ],
                        className="barracuda-home-hero-copy",
                    ),
                    html.Figure(
                        [
                            html.Img(
                                src="/assets/figure_abstract_papercraft.png",
                                alt="Paper diorama showing microscopy observations becoming an ordered contact history and then two candidate biological mechanisms.",
                                width=1672,
                                height=941,
                            ),
                            html.Figcaption("From single-cell observations to biological explanations with quantified uncertainty."),
                        ]
                    ),
                ],
                className="barracuda-home-hero",
            ),
            html.Section(
                [
                    html.Span("Choose an analysis", className="barracuda-eyebrow"),
                    html.H2("Start from the information in your data"),
                    html.Div(
                        [
                            task_card(
                                "Counts without donor labels",
                                "You have one contact or kill count per cell.",
                                "cell_id, condition, count",
                                "Compare homogeneous, continuously heterogeneous, and nonengaging population structures.",
                                "/event-counts/donor-ignorant",
                                "Analyse event counts",
                                specimen="counts",
                            ),
                            task_card(
                                "Counts grouped by donor",
                                "Each cell also has a donor identifier.",
                                "cell_id, donor_id, condition, count",
                                "Separate within donor cellular variation from differences between donors.",
                                "/event-counts/donor-aware",
                                "Analyse donor grouped counts",
                                specimen="donors",
                            ),
                            task_card(
                                "Ordered contact histories",
                                "You know the order of lethal and nonlethal contacts for each cell.",
                                "cell_id, condition, history",
                                "Test whether previous contacts alter later killing decisions.",
                                "/trajectory",
                                "Analyse contact histories",
                                specimen="history",
                            ),
                        ],
                        className="barracuda-task-grid",
                    ),
                ],
                id="choose-analysis",
                className="barracuda-home-section",
            ),
            html.Section(
                [
                    html.Span("How it works", className="barracuda-eyebrow"),
                    html.H2("One auditable path from cells to evidence"),
                    html.Ol(
                        [
                            html.Li([html.Span("1"), html.Strong("Prepare cell-level data"), html.P("Validate a compact CSV or generate a synthetic example.")]),
                            html.Li([html.Span("2"), html.Strong("Compare mechanistic models"), html.P("Fit competing population explanations with sequential Monte Carlo.")]),
                            html.Li([html.Span("3"), html.Strong("Interpret uncertainty and evidence"), html.P("Read posterior intervals, marginal distributions and Bayes factors together.")]),
                        ],
                        className="barracuda-process",
                    ),
                ],
                className="barracuda-home-section",
            ),
            html.Section(
                html.Details(
                    [
                        html.Summary("Research context, team and correspondence"),
                        html.P(
                            "BARRACUDA stands for Bayesian Analysis Resolving Randomness and Alternative Causes Underlying Differential Activity. It tests whether immune-cell variation reflects chance, stable cellular differences, donor effects, or previous interactions."
                        ),
                        html.H3("Authors"),
                        html.P("Elephes Sung¹†, Cathal Hosty¹†, Leanne Peiser², Lara Stepan², Daniel M Davis¹*, and Ruben Perez-Carrasco¹*."),
                        html.P("¹ Department of Life Sciences, Imperial College London, London SW7 2AZ, UK. ² Bristol Myers Squibb, Seattle, WA, USA."),
                        html.P("† These authors contributed equally. * Correspondence: d.davis@imperial.ac.uk; r.perez-carrasco@imperial.ac.uk."),
                        html.Div(
                            [
                                html.A("GitHub repository", href="https://github.com/sthsci/Barracuda", target="_blank", rel="noreferrer", className="barracuda-button secondary"),
                                dcc.Link("Google Colab notebooks", href="/notebooks", className="barracuda-button secondary"),
                            ],
                            className="barracuda-home-actions",
                        ),
                        html.P("Research use only. Use synthetic or approved anonymous data.", className="barracuda-help"),
                    ],
                    className="barracuda-details barracuda-research-context",
                ),
                className="barracuda-home-section",
            ),
        ]
    )
