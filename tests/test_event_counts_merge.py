from __future__ import annotations

from collections import Counter

from dash.development.base_component import Component

from webapp.dashapp import create_app
from webapp.core.conditions import APPLE_COLOUR_PRESETS, MAX_CONDITIONS
from webapp.pages import event_counts


def _walk(component):
    if isinstance(component, Component):
        yield component
        children = getattr(component, "children", None)
        if children is not None:
            yield from _walk(children)
    elif isinstance(component, (list, tuple)):
        for child in component:
            yield from _walk(child)


def test_donor_ignorant_route_composes_both_workflows_without_duplicate_ids() -> None:
    content = event_counts.layout()
    components = list(_walk(content))
    ids = [
        component_id
        for component in components
        if isinstance((component_id := getattr(component, "id", None)), str)
    ]

    assert not {
        component_id: count
        for component_id, count in Counter(ids).items()
        if count > 1
    }
    assert {
        "donor-ignorant-workflow",
        "donor-ignorant-synthetic-panel",
        "donor-ignorant-own-data-panel",
        "synthetic-generate",
        "synthetic-run",
        "counts-upload",
        "counts-table",
        "counts-run",
    } <= set(ids)

    workflow = next(
        component
        for component in components
        if getattr(component, "id", None) == "donor-ignorant-workflow"
    )
    assert workflow.value is None
    assert {option["value"] for option in workflow.options} == {
        "synthetic",
        "own-data",
    }


def test_own_data_workflow_preserves_counts_ids_and_condition_controls() -> None:
    components = list(_walk(event_counts.layout()))
    by_id = {
        component_id: component
        for component in components
        if isinstance((component_id := getattr(component, "id", None)), str)
    }

    assert MAX_CONDITIONS == 4
    assert len(APPLE_COLOUR_PRESETS) >= MAX_CONDITIONS
    assert {
        "counts-upload",
        "counts-table",
        "counts-observation-time",
        "counts-condition-colour-controls",
        "counts-condition-colours-section",
        "counts-run",
        "counts-results",
    } <= by_id.keys()
    fields = [definition["field"] for definition in by_id["counts-table"].columnDefs]
    assert fields == ["cell_id", "condition", "count"]


def test_merged_workflow_callbacks_are_registered_once() -> None:
    app = create_app()
    callback_keys = tuple(app.callback_map)

    assert sum(
        "donor-ignorant-synthetic-panel.className" in key
        for key in callback_keys
    ) == 1
    assert sum("counts-results.children" in key for key in callback_keys) == 1
    assert sum("synthetic-results.children" in key for key in callback_keys) == 1
