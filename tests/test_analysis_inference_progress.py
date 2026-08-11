from __future__ import annotations

from dash.development.base_component import Component

from webapp.dashapp import PAGES, create_app


def _walk(component):
    if isinstance(component, Component):
        yield component
        children = getattr(component, "children", None)
        if children is not None:
            yield from _walk(children)
    elif isinstance(component, (list, tuple)):
        for child in component:
            yield from _walk(child)


def _analysis_callback(app, prefix: str):
    return next(
        value
        for key, value in app.callback_map.items()
        if f"{prefix}-results.children" in key
    )


def _callback_spec(app, prefix: str):
    return next(
        spec
        for spec in app._callback_list
        if f"{prefix}-results.children" in spec.get("output", "")
    )


def test_uploaded_analysis_layouts_expose_live_pymc_progress() -> None:
    pages = {page.PATH: page for page in PAGES}
    for path, prefix in (
        ("/event-counts/donor-ignorant", "counts"),
        ("/event-counts/donor-aware", "donor"),
    ):
        components = list(_walk(pages[path].layout()))
        ids = {
            component_id
            for component in components
            if isinstance(
                (component_id := getattr(component, "id", None)),
                str,
            )
        }
        assert {
            f"{prefix}-pymc-progress",
            f"{prefix}-pymc-progress-bar",
            f"{prefix}-pymc-progress-label",
            f"{prefix}-pymc-progress-meta",
            f"{prefix}-chain-progress",
            f"{prefix}-inference-controls",
        } <= ids
        fieldset = next(
            component
            for component in components
            if getattr(component, "id", None)
            == f"{prefix}-inference-controls"
        )
        assert f"{prefix}-run" in {
            getattr(component, "id", None) for component in _walk(fieldset)
        }


def test_uploaded_analysis_inference_runs_in_background_with_live_progress() -> None:
    app = create_app()
    expected_suffixes = {
        "pymc-progress-bar.value",
        "pymc-progress-label.children",
        "pymc-progress-meta.children",
        "chain-progress.children",
    }

    for prefix in ("counts", "donor"):
        callback = _analysis_callback(app, prefix)
        background = callback["background"]
        assert background is not None
        assert background["interval"] <= 500
        assert {
            str(output).removeprefix(f"{prefix}-")
            for output in background["progress"]
        } == expected_suffixes
        assert len(background["progressDefault"]) == len(
            background["progress"]
        )


def test_uploaded_analysis_locks_controls_while_background_job_runs() -> None:
    app = create_app()
    for prefix in ("counts", "donor"):
        running = _callback_spec(app, prefix)["running"]
        running_on = running["running"]
        running_off = running["runningOff"]

        assert running_on[f"{prefix}-inference-controls.disabled"] is True
        assert running_off[f"{prefix}-inference-controls.disabled"] is False
        assert f"{prefix}-run.disabled" not in running_on
        assert f"{prefix}-run.disabled" not in running_off
        assert running_on[f"{prefix}-inference.aria-busy"] == "true"
        assert running_off[f"{prefix}-inference.aria-busy"] == "false"
        assert "is-active" in running_on[
            f"{prefix}-pymc-progress.className"
        ]
