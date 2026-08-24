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


def test_global_figure_controls_are_persistent_and_registered() -> None:
    app = create_app()
    by_id = {
        component_id: component
        for component in _walk(app.layout)
        if isinstance((component_id := getattr(component, "id", None)), str)
    }

    width = by_id["barracuda-figure-width"]
    height = by_id["barracuda-figure-height-scale"]
    assert (width.min, width.max, width.value) == (60, 100, 100)
    assert (height.min, height.max, height.value) == (0.75, 1.75, 1.0)
    assert width.persistence is True and width.persistence_type == "local"
    assert height.persistence is True and height.persistence_type == "local"
    assert "barracuda-figure-sizing-applied.data" in app.callback_map
    assert any(
        "barracuda-figure-width.value" in key
        and "barracuda-figure-height-scale.value" in key
        for key in app.callback_map
    )


def test_global_figure_javascript_resizes_current_and_future_plotly_graphs() -> None:
    source = (ROOT / "webapp" / "assets" / "figure_controls.js").read_text()

    assert '.querySelectorAll(".js-plotly-plot")' in source
    assert "MutationObserver" in source
    assert "Plotly.Plots.resize" in source
    assert "(max-width: 820px)" in source
    assert "MIN_HEIGHT = 320" in source
    assert "MAX_HEIGHT = 1600" in source
    assert "barracudaBaseHeight" in source
    assert "barracudaAppliedHeight" in source
    assert '/px\\s*$/i.test(inlineHeight)' in source
    assert 'plot.style.height !== heightValue' in source
    assert "root.offsetParent === null" in source
    assert 'attributeFilter: ["style", "class"]' in source

    css = (ROOT / "webapp" / "assets" / "styles.css").read_text()
    assert ".barracuda-trajectory-encoding-legend" in css
    assert ".barracuda-figure-sized" in css
    assert (
        ".barracuda-trajectory-empirical-plot .svg-container { width: 100% !important; height: 100% !important; }"
        not in css
    )
