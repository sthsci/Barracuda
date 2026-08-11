"""Donor aware hierarchical analysis page."""

from __future__ import annotations

from dash import dcc, html

from webapp.pages.analysis_page import layout as analysis_layout
from webapp.pages.analysis_page import register_callbacks as register_analysis_callbacks


PATH = "/event-counts/donor-aware"
TITLE = "Donor aware analysis"


def layout() -> html.Div:
    page = analysis_layout(
        prefix="donor",
        donor_aware=True,
        kicker="Event counts · Donor aware",
        title="Donor aware condition analysis",
        lead="Fit one to four experimental conditions while allowing the mean event rate μλ,d, continuous cell-to-cell heterogeneity σλ,d and fraction of nonengaging cells φ₀,d to vary between donors.",
        badge="2 to 12 donors per condition · Section 2 hierarchy",
    )
    children = list(page.children)
    children.insert(
        3,
        html.Section(
            [
                html.Span("How the hierarchy is read", className="orca-section-label"),
                html.H2("Cells within donors, then donors within each condition"),
                html.P(
                    "Each experimental condition is fitted independently. Within a condition, donor parameters are estimated jointly around shared reference priors, while reported population parameters are cell-weighted moments of the donor mixture.",
                    className="orca-section-lead",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Strong("1 · Model evidence"),
                                html.P("Compare the four candidate count models with SMC marginal likelihoods and Bayes factors."),
                            ]
                        ),
                        html.Div(
                            [
                                html.Strong("2 · Sources of heterogeneity"),
                                html.P("Split population variance into continuous variation within donors and differences between donor means."),
                            ]
                        ),
                        html.Div(
                            [
                                html.Strong("3 · Posterior views"),
                                html.P("Inspect population posteriors by condition, all donors within a condition, and conditions within each donor."),
                            ]
                        ),
                        html.Div(
                            [
                                html.Strong("4 · Condition contrasts"),
                                html.P("Choose any two conditions and compare their independent posterior particle distributions."),
                            ]
                        ),
                    ],
                    className="orca-bf-flow",
                ),
                html.Details(
                    [
                        html.Summary("Variance decomposition and comparison rule"),
                        dcc.Markdown(
                            r"""
For active-donor weights $\widetilde w_d$, the population variance is decomposed for every posterior particle:

$$V_{\mathrm{within}}=\sum_d\widetilde w_d\sigma_{\lambda,d}^2,$$

$$V_{\mathrm{between}}=\sum_d\widetilde w_d(\mu_{\lambda,d}-\bar\mu_\lambda)^2,$$

$$\bar\sigma_\lambda=\sqrt{V_{\mathrm{within}}+V_{\mathrm{between}}}.$$

Fits for two experimental conditions are independent. A contrast therefore uses every possible particle pair when practical; for larger posteriors it uses a reproducible uniform sample of independent pairs. It does **not** subtract only the two posterior means.
""",
                            mathjax=True,
                            className="orca-model-equations",
                        ),
                    ],
                    className="orca-details",
                ),
            ],
            className="orca-workflow-panel orca-donor-method",
        ),
    )
    return html.Div(children)


def register_callbacks(app) -> None:
    register_analysis_callbacks(app, prefix="donor", donor_aware=True)
    from webapp.donor_interactive import register_donor_contrast_callbacks
    from webapp.donor_reporting import register_donor_reporting_callbacks

    register_donor_reporting_callbacks(app, prefix="donor")
    register_donor_contrast_callbacks(app, prefix="donor")
