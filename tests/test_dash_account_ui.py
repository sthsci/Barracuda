from __future__ import annotations

from collections import Counter

from dash.development.base_component import Component

from webapp.dashapp import PAGE_BY_PATH, create_app
from webapp.pages import shared, workspace


def _walk(component):
    if isinstance(component, Component):
        yield component
        children = getattr(component, "children", None)
        if children is not None:
            yield from _walk(children)
    elif isinstance(component, (list, tuple)):
        for child in component:
            yield from _walk(child)


def test_optional_account_controls_do_not_replace_dash_routes() -> None:
    app = create_app()
    shell_ids = [
        component_id
        for component in _walk(app.layout)
        if isinstance((component_id := getattr(component, "id", None)), str)
    ]
    workspace_ids = [
        component_id
        for component in _walk(workspace.layout())
        if isinstance((component_id := getattr(component, "id", None)), str)
    ]

    assert not {component_id for component_id, count in Counter(shell_ids).items() if count > 1}
    assert not {component_id for component_id, count in Counter(workspace_ids).items() if count > 1}
    assert {
        "barracuda-account-session",
        "barracuda-counts-snapshot",
        "barracuda-donor-snapshot",
        "barracuda-trajectory-snapshot",
    } <= set(shell_ids)
    assert {
        "barracuda-account-panel",
        "barracuda-login-submit",
        "barracuda-register-submit",
        "barracuda-save-dataset",
        "barracuda-dataset-download",
        "barracuda-share-create",
    } <= set(shell_ids)
    assert "barracuda-account-panel" not in workspace_ids
    assert PAGE_BY_PATH["/workspace"] is workspace
    assert PAGE_BY_PATH["/shared"] is shared
    assert "frontend" not in app.title.lower()


def test_account_callbacks_are_registered_without_duplicate_outputs() -> None:
    app = create_app()
    callback_keys = tuple(app.callback_map)

    assert sum("barracuda-account-panel.className" in key for key in callback_keys) == 1
    assert sum("barracuda-account-session.data" in key for key in callback_keys) == 1
    assert sum("barracuda-save-status.children" in key for key in callback_keys) == 1
    assert sum("barracuda-share-status.children" in key for key in callback_keys) == 1
    assert sum("barracuda-shared-content.children" in key for key in callback_keys) == 1
