"""Reusable Dash components for the Orca interface."""

from __future__ import annotations

from collections.abc import Iterable

from dash import dcc, html


def hero(kicker: str, title: str, lead: str, badge: str | None = None) -> html.Section:
    children: list = [
        html.Div(kicker, className="orca-kicker"),
        html.H1(title),
        html.P(lead),
    ]
    if badge:
        children.append(html.Span(badge, className="orca-badge"))
    return html.Section(children, className="orca-hero")


def note(title: str, body: str, tone: str = "teal") -> html.Div:
    safe_tone = tone if tone in {"teal", "amber", "navy"} else "teal"
    return html.Div(
        [html.Strong(title), html.Span(body)],
        className=f"orca-note orca-note-{safe_tone}",
    )


def step_card(number: str, title: str, body: str) -> html.Div:
    return html.Div(
        [html.Span(number), html.H3(title), html.P(body)],
        className="orca-step",
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
            html.Span(f"Section {number}", className="orca-route-label"),
            step_card(number, title, body),
            dcc.Link(f"{link_label} →", href=path, className="orca-card-link"),
        ],
        className="orca-route-card",
    )


def metric(label: str, value: str, *, accent: str = "sage") -> html.Div:
    return html.Div(
        [html.Span(label, className="orca-metric-label"), html.Strong(value)],
        className=f"orca-metric orca-metric-{accent}",
    )


def metrics(items: Iterable[tuple[str, str]]) -> html.Div:
    return html.Div(
        [metric(label, value) for label, value in items],
        className="orca-metrics",
    )


def section_intro(label: str, title: str, body: str | None = None) -> html.Div:
    children: list = [html.Span(label, className="orca-section-label"), html.H2(title)]
    if body:
        children.append(html.P(body, className="orca-section-lead"))
    return html.Div(children, className="orca-section-intro")


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
                html.Img(src=src, alt=alt),
                className="orca-schematic-scroll",
                tabIndex=0,
                role="region",
                **{"aria-label": "Scrollable research schematic"},
            ),
            html.Span(
                "Swipe or scroll horizontally to inspect the full schematic.",
                className="orca-schematic-scroll-hint",
            ),
            html.Figcaption(caption),
        ],
        className=f"orca-schematic orca-schematic-{variant}",
    )


def research_warning() -> html.Div:
    return note(
        "Exploratory research software",
        "Preview runs use fewer particles and are intended for learning and interface testing, not research conclusions.",
        tone="amber",
    )


def markdown(text: str, *, class_name: str = "orca-copy", mathjax: bool = False) -> dcc.Markdown:
    return dcc.Markdown(text, className=class_name, mathjax=mathjax, link_target="_blank")
