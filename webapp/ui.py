"""Reusable Dash components for the Barracuda interface."""

from __future__ import annotations

from collections.abc import Iterable

from dash import dcc, html


def app_header(items: Iterable[tuple[str, str, str]], *, workspace_id: str) -> html.Header:
    """Render the shared desktop and native, keyboard-accessible mobile nav."""

    def links(suffix: str) -> list[html.A]:
        return [
            html.A(label, href=path, id=f"{item_id}{suffix}", className="barracuda-nav-link")
            for label, path, item_id in items
        ]

    return html.Header(
        html.Div(
            [
                html.A(
                    [
                        html.Img(
                            src="/assets/barracuda-abstract-consistent-posterior-mark.png",
                            alt="",
                            width=44,
                            height=44,
                            className="barracuda-mark",
                            **{"aria-hidden": "true"},
                        ),
                        html.Strong("BARRACUDA"),
                    ],
                    href="/",
                    className="barracuda-brand",
                    **{"aria-label": "BARRACUDA home"},
                ),
                html.Nav(links(""), className="barracuda-desktop-nav", **{"aria-label": "Primary navigation"}),
                html.A("Workspace", href="/workspace", id=workspace_id, className="barracuda-workspace-link"),
                html.Details(
                    [
                        html.Summary("Menu", **{"aria-label": "Open navigation menu"}),
                        html.Nav(
                            [*links("-mobile"), html.A("Workspace", href="/workspace", className="barracuda-nav-link")],
                            **{"aria-label": "Mobile navigation"},
                        ),
                    ],
                    className="barracuda-mobile-menu",
                ),
            ],
            className="barracuda-header-inner",
        ),
        className="barracuda-header",
    )


def breadcrumbs(items: Iterable[tuple[str, str | None]]) -> html.Nav:
    entries = list(items)
    children: list = []
    for index, (label, path) in enumerate(entries):
        if index:
            children.append(html.Span("/", **{"aria-hidden": "true"}))
        children.append(
            dcc.Link(label, href=path) if path else html.Span(label, **{"aria-current": "page"})
        )
    return html.Nav(children, className="barracuda-breadcrumbs", **{"aria-label": "Breadcrumb"})


def page_header(
    eyebrow: str,
    title: str,
    lead: str,
    *,
    crumb: str | None = None,
    badge: str | None = None,
    educational: bool = False,
) -> html.Header:
    children: list = []
    if crumb:
        root = {"Analyse": "/event-counts", "Learn": "/bayesian-101", "Resources": "/notebooks"}.get(eyebrow)
        children.append(breadcrumbs([(eyebrow, root), (crumb, None)]))
    children.extend([html.Span(eyebrow, className="barracuda-eyebrow"), html.H1(title), html.P(lead)])
    if badge:
        children.append(html.Span(badge, className="barracuda-status-badge"))
    return html.Header(
        children,
        className=f"barracuda-page-header{' educational' if educational else ''}",
    )


def analysis_stepper(prefix: str, current: int = 0) -> html.Ol:
    stages = ("Data", "Models", "Compute", "Results", "Export")
    return html.Ol(
        [
            html.Li(
                [
                    html.Span("✓" if index < current else str(index + 1), **{"aria-hidden": "true"}),
                    html.Strong(stage),
                    html.Small("Complete" if index < current else "Current" if index == current else "Not started"),
                ],
                className="is-complete" if index < current else "is-current" if index == current else "",
                **({"aria-current": "step"} if index == current else {}),
            )
            for index, stage in enumerate(stages)
        ],
        id=f"{prefix}-stepper",
        className="barracuda-stepper",
        **{"aria-label": "Analysis progress"},
    )


def task_card(
    title: str,
    use_when: str,
    schema: str,
    outcome: str,
    path: str,
    action: str,
) -> html.Article:
    return html.Article(
        [
            html.H3(title),
            html.P([html.Strong("Use when: "), use_when]),
            html.P([html.Strong("Required data: "), html.Code(schema)]),
            html.P(outcome),
            html.A(f"{action} →", href=path, className="barracuda-card-link", **{"aria-label": f"{action}: {title}"}),
        ],
        className="barracuda-task-card",
    )


def empty_state(title: str, body: str) -> html.Div:
    return html.Div([html.Strong(title), html.P(body)], className="barracuda-empty-state")


def advanced_settings(summary: str, children: list) -> html.Details:
    return html.Details([html.Summary(summary), *children], className="barracuda-details")


def interpretation_callout(title: str, body: str) -> html.Aside:
    return html.Aside([html.Strong(title), html.P(body)], className="barracuda-interpretation")


def hero(kicker: str, title: str, lead: str, badge: str | None = None) -> html.Section:
    children: list = [
        html.Div(kicker, className="barracuda-kicker"),
        html.H1(title),
        html.P(lead),
    ]
    if badge:
        children.append(html.Span(badge, className="barracuda-badge"))
    return html.Section(children, className="barracuda-hero")


def note(title: str, body: str, tone: str = "teal") -> html.Div:
    safe_tone = tone if tone in {"teal", "amber", "navy"} else "teal"
    return html.Div(
        [html.Strong(title), html.Span(body)],
        className=f"barracuda-note barracuda-note-{safe_tone}",
    )


def step_card(number: str, title: str, body: str) -> html.Div:
    return html.Div(
        [html.Span(number), html.H3(title), html.P(body)],
        className="barracuda-step",
    )


def route_card(
    number: str,
    title: str,
    body: str,
    path: str,
    link_label: str,
) -> html.Div:
    return html.Div(
        [
            html.Span(f"Section {number}", className="barracuda-route-label"),
            step_card(number, title, body),
            dcc.Link(f"{link_label} →", href=path, className="barracuda-card-link"),
        ],
        className="barracuda-route-card",
    )


def metric(label: str, value: str, *, accent: str = "sage") -> html.Div:
    return html.Div(
        [html.Span(label, className="barracuda-metric-label"), html.Strong(value)],
        className=f"barracuda-metric barracuda-metric-{accent}",
    )


def metrics(items: Iterable[tuple[str, str]]) -> html.Div:
    return html.Div(
        [metric(label, value) for label, value in items],
        className="barracuda-metrics",
    )


def section_intro(label: str, title: str, body: str | None = None) -> html.Div:
    children: list = [html.Span(label, className="barracuda-section-label"), html.H2(title)]
    if body:
        children.append(html.P(body, className="barracuda-section-lead"))
    return html.Div(children, className="barracuda-section-intro")


def schematic_figure(
    src: str,
    alt: str,
    caption: str,
    *,
    variant: str,
) -> html.Figure:
    """Render a wide paper schematic accessibly on desktop and phones."""

    return html.Figure(
        [
            html.Div(
                html.Img(
                    src=src,
                    alt=alt,
                    width=4016 if variant == "models" else 2520,
                    height=1560 if variant == "models" else 672,
                ),
                className="barracuda-schematic-scroll",
                tabIndex=0,
                role="region",
                **{"aria-label": "Scrollable research schematic"},
            ),
            html.Span(
                "Swipe or scroll horizontally to inspect the full schematic.",
                className="barracuda-schematic-scroll-hint",
            ),
            html.Figcaption(caption),
        ],
        className=f"barracuda-schematic barracuda-schematic-{variant}",
    )


def research_warning() -> html.Div:
    return note(
        "Exploratory research software",
        "Preview runs use fewer particles and are intended for learning and interface testing, not research conclusions.",
        tone="amber",
    )


def markdown(text: str, *, class_name: str = "barracuda-copy", mathjax: bool = False) -> dcc.Markdown:
    return dcc.Markdown(text, className=class_name, mathjax=mathjax, link_target="_blank")
