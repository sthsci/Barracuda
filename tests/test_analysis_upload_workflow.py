from __future__ import annotations

import base64
import re
from contextlib import contextmanager

import pytest
from dash._callback_context import context_value
from dash._utils import AttributeDict
from dash.development.base_component import Component

from webapp.dashapp import create_app
from webapp.pages.analysis_page import layout as analysis_layout


def _walk(component):
    if isinstance(component, Component):
        yield component
        children = getattr(component, "children", None)
        if children is not None:
            yield from _walk(children)
    elif isinstance(component, (list, tuple)):
        for child in component:
            yield from _walk(child)


def _text(component) -> str:
    parts: list[str] = []
    for item in _walk(component):
        children = getattr(item, "children", None)
        if isinstance(children, str):
            parts.append(children)
        elif isinstance(children, (int, float)):
            parts.append(str(children))
    return " ".join(parts)


def _callback(app, output_fragment: str):
    matches = [
        definition["callback"].__wrapped__
        for output, definition in app.callback_map.items()
        if output_fragment in output
    ]
    assert len(matches) == 1, output_fragment
    return matches[0]


@contextmanager
def _triggered(component_id: str, property_name: str, value):
    token = context_value.set(
        AttributeDict(
            triggered_inputs=[
                {
                    "prop_id": f"{component_id}.{property_name}",
                    "value": value,
                }
            ]
        )
    )
    try:
        yield
    finally:
        context_value.reset(token)


def _csv_contents(*, donor_aware: bool) -> str:
    if donor_aware:
        header = "cell_id,donor_id,condition,count"
        rows = [
            f"{condition.lower()}_{donor}_{index},{donor},{condition},{count}"
            for condition, offset in (("Control", 0), ("Treatment", 1))
            for donor in ("donor_A", "donor_B")
            for index, count in enumerate((0 + offset, 1 + offset, 2 + offset), 1)
        ]
    else:
        header = "cell_id,condition,count"
        rows = [
            f"{condition.lower()}_{index},{condition},{count}"
            for condition, offset in (("Control", 0), ("Treatment", 1))
            for index, count in enumerate(
                (
                    0 + offset,
                    1 + offset,
                    2 + offset,
                    1 + offset,
                    3 + offset,
                    0 + offset,
                ),
                1,
            )
        ]
    payload = ("\n".join([header, *rows]) + "\n").encode("utf-8")
    return "data:text/csv;base64," + base64.b64encode(payload).decode("ascii")


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.mark.parametrize(
    ("prefix", "donor_aware"),
    (("counts", False), ("donor", True)),
)
def test_csv_upload_is_validated_and_enables_inference(
    app,
    prefix: str,
    donor_aware: bool,
) -> None:
    """Exercise the same two-callback hand-off used by the browser."""

    upload = _callback(app, f"{prefix}-table.rowData")
    validate = _callback(app, f"{prefix}-valid-data.data")
    contents = _csv_contents(donor_aware=donor_aware)

    with _triggered(f"{prefix}-upload", "contents", contents):
        rows, _columns, grid_options, upload_class, _actions, upload_status = upload(
            "upload",
            contents,
            0,
            0,
            [],
            [],
        )

    assert rows
    assert "is-hidden" not in upload_class
    assert "CSV loaded" in _text(upload_status)
    assert grid_options["pagination"] is True

    (
        validation_status,
        overview,
        valid_records,
        inference_disabled,
        colour_controls,
        colour_section_class,
    ) = validate(rows, None, 1.0)

    assert "Dataset ready" in _text(validation_status)
    assert valid_records == rows
    assert inference_disabled is False
    assert "is-hidden" not in colour_section_class
    assert overview.__class__.__name__ == "Details"
    assert getattr(overview, "open", False) is not True
    assert any(
        component.__class__.__name__ == "Summary"
        for component in _walk(overview)
    )
    colour_labels = {
        component.id["index"]
        for component in _walk(colour_controls)
        if isinstance(getattr(component, "id", None), dict)
        and component.id.get("type") == f"{prefix}-condition-colour"
    }
    assert colour_labels == {"Control", "Treatment"}


@pytest.mark.parametrize(
    ("prefix", "donor_aware"),
    (
        ("counts", False),
        ("donor", True),
    ),
)
def test_actual_data_analysis_starts_with_upload_and_uses_inference_language(
    prefix: str,
    donor_aware: bool,
) -> None:
    content = analysis_layout(
        prefix=prefix,
        donor_aware=donor_aware,
        kicker="Test",
        title="Test analysis",
        lead="Test lead",
        badge="Test badge",
    )
    by_id = {
        component_id: component
        for component in _walk(content)
        if isinstance((component_id := getattr(component, "id", None)), str)
    }

    source = by_id[f"{prefix}-source"]
    assert source.value == "upload"
    assert {option["value"] for option in source.options} == {"upload", "edit", "example"}
    assert "Use an example" in {str(option["label"]) for option in source.options}
    assert "is-hidden" not in by_id[f"{prefix}-upload-panel"].className

    assert re.search(
        r"\brun\b.*\binference\b",
        _text(by_id[f"{prefix}-run"]),
        re.IGNORECASE,
    )
    assert not re.search(r"\bfit(?:s|ted|ting)?\b", _text(content), re.IGNORECASE)


@pytest.mark.parametrize("prefix", ("counts", "donor"))
def test_actual_data_preview_is_compact(app, prefix: str) -> None:
    upload = _callback(app, f"{prefix}-table.rowData")
    donor_aware = prefix == "donor"
    contents = _csv_contents(donor_aware=donor_aware)

    with _triggered(f"{prefix}-upload", "contents", contents):
        _rows, _columns, grid_options, *_rest = upload(
            "upload",
            contents,
            0,
            0,
            [],
            [],
        )

    assert grid_options["pagination"] is True
    assert grid_options["paginationPageSize"] <= 8
    assert grid_options["paginationPageSizeSelector"] is False


@pytest.mark.parametrize("prefix", ("counts", "donor"))
def test_edit_source_starts_with_a_neutral_template_instead_of_example_data(
    app,
    prefix: str,
) -> None:
    update_source = _callback(app, f"{prefix}-table.rowData")

    with _triggered(f"{prefix}-source", "value", "edit"):
        rows, _columns, _grid, upload_class, action_class, status = update_source(
            "edit",
            None,
            0,
            0,
            [],
            [],
        )

    assert len(rows) >= 5
    assert {row["condition"] for row in rows} == {"Condition 1"}
    assert {row["count"] for row in rows} == {0}
    assert "is-hidden" in upload_class
    assert "is-hidden" not in action_class
    assert "example" not in _text(status).lower()

    with _triggered(f"{prefix}-add-row", "n_clicks", 1):
        added_rows, *_rest = update_source("edit", None, 1, 0, rows, [])
    assert len(added_rows) == len(rows) + 1
    assert added_rows[-1]["condition"] == "Condition 1"


@pytest.mark.parametrize(("prefix", "donor_aware"), (("counts", False), ("donor", True)))
def test_example_source_loads_valid_fictional_conditions(app, prefix: str, donor_aware: bool) -> None:
    update_source = _callback(app, f"{prefix}-table.rowData")

    with _triggered(f"{prefix}-source", "value", "example"):
        rows, _columns, _grid, upload_class, action_class, status = update_source(
            "example", None, 0, 0, [], []
        )

    assert upload_class == "is-hidden"
    assert action_class.endswith("is-hidden")
    assert {row["condition"] for row in rows} == {"Control", "Treatment"}
    assert all(str(row["cell_id"]).startswith(("control_", "treatment_")) for row in rows)
    if donor_aware:
        assert len({row["donor_id"] for row in rows}) >= 2
    assert "Synthetic example loaded" in _text(status)


@pytest.mark.parametrize("prefix", ("counts", "donor"))
def test_each_analysis_pipeline_output_has_one_callback(app, prefix: str) -> None:
    registered_outputs = [definition["output"] for definition in app._callback_list]
    expected_once = (
        f"{prefix}-table.rowData",
        f"{prefix}-valid-data.data",
        f"{prefix}-run.disabled",
        f"{prefix}-results.children",
        f"{prefix}-run-status.children",
    )

    for output in expected_once:
        assert sum(output in registered for registered in registered_outputs) == 1
