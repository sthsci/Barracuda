from __future__ import annotations

from pathlib import Path

from dash.development.base_component import Component

from webapp.dashapp import create_app


ROOT = Path(__file__).resolve().parents[1]


def _walk(component):
    if isinstance(component, Component):
        yield component
        children = getattr(component, "children", None)
        if children is not None:
            yield from _walk(children)
    elif isinstance(component, (list, tuple)):
        for child in component:
            yield from _walk(child)


def test_compact_header_replaces_sidebar_and_global_figure_controls() -> None:
    app = create_app()
    components = list(_walk(app.layout))
    ids = {getattr(component, "id", None) for component in components}
    classes = {getattr(component, "className", None) for component in components}

    assert "barracuda-header" in classes
    assert "barracuda-sidebar" not in classes
    assert {"nav-home", "nav-event-counts", "nav-contact-histories", "nav-learn", "nav-resources"} <= ids
    assert "barracuda-figure-width" not in ids
    assert "barracuda-figure-height-scale" not in ids
    assert not (ROOT / "webapp" / "assets" / "figure_controls.js").exists()
    assert not any("figure-sizing" in key for key in app.callback_map)


def test_scientific_figures_remain_responsive_without_global_resizing_script() -> None:
    css = (ROOT / "webapp" / "assets" / "styles.css").read_text()

    assert ".barracuda-trajectory-encoding-legend" in css
    assert ".barracuda-analysis-workbench" in css
    assert "grid-template-columns: minmax(320px, 352px) minmax(0, 1fr)" in css
    assert "@media (max-width: 980px)" in css
