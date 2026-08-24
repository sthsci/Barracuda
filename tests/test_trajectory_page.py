from __future__ import annotations

import base64
import inspect
import json
from collections.abc import Callable

import pandas as pd
import pytest
from dash.development.base_component import Component

from webapp.dashapp import create_app
from webapp.pages import trajectory


@pytest.fixture(scope="module")
def app():
    return create_app()


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


def _callback_definition(app, output_fragment: str) -> dict:
    matches = [
        definition
        for key, definition in app.callback_map.items()
        if output_fragment in key
    ]
    assert len(matches) == 1, output_fragment
    return matches[0]


def _callback(app, output_fragment: str) -> Callable:
    callback = _callback_definition(app, output_fragment)["callback"]
    return getattr(callback, "__wrapped__", callback)


def test_layout_has_unique_ids_release_terms_and_safe_defaults() -> None:
    content = trajectory.layout()
    components = list(_walk(content))
    component_ids = [
        component_id
        for component in components
        if (component_id := getattr(component, "id", None)) is not None
    ]
    canonical_ids = [
        json.dumps(component_id, sort_keys=True)
        if isinstance(component_id, dict)
        else str(component_id)
        for component_id in component_ids
    ]
    assert len(canonical_ids) == len(set(canonical_ids))

    by_id = {
        component_id: component
        for component in components
        if isinstance((component_id := getattr(component, "id", None)), str)
    }
    assert by_id["trajectory-workflow"].value is None
    assert by_id["trajectory-run"].disabled is True
    assert by_id["trajectory-observation-time"].value == 1.0
    assert by_id["trajectory-upload-observation-time"].value == 1.0
    assert by_id["trajectory-synthetic-panel"].className.endswith("is-hidden")
    assert by_id["trajectory-upload-panel"].className.endswith("is-hidden")
    assert by_id["trajectory-data-section"].className.endswith("is-hidden")
    assert by_id["trajectory-inference"].className.endswith("is-hidden")
    assert by_id["trajectory-empirical-figure"].style == {"height": "700px"}
    assert by_id["trajectory-figure-height"].value == 700
    assert by_id["trajectory-arrow-scale"].value == 1.0
    assert [option["value"] for option in by_id["trajectory-models"].options] == list(
        trajectory.MODEL_ORDER
    )

    page_text = _text(content)
    assert "Mean contact rate, μλ" in page_text
    assert "Contact-rate SD, σλ" in page_text
    assert "Previous non-lethal effect, βf" in page_text
    assert "Previous lethal effect, βs" in page_text
    assert "not a nonengaging fraction" in page_text
    assert "beta_x" not in page_text
    assert "beta_y" not in page_text
    assert "immune-cell" not in page_text.lower()


def test_all_trajectory_callbacks_register_with_matching_signatures(app) -> None:
    required_outputs = {
        "trajectory-synthetic-panel.className",
        "trajectory-sigma-eta.disabled",
        "trajectory-synthetic-data.data",
        "trajectory-upload-data.data",
        "trajectory-active-data.data",
        "trajectory-workload.children",
        "trajectory-results.children",
        '"type":"trajectory-model-panel"',
        '"type":"trajectory-posterior-marginal"',
    }
    registered = "\n".join(app.callback_map)
    assert all(output in registered for output in required_outputs)

    marginal = _callback_definition(
        app,
        '"type":"trajectory-posterior-marginal"',
    )
    assert [state["id"] for state in marginal["state"]] == [
        '{"index":["MATCH"],"type":"trajectory-posterior-parameter"}',
        "trajectory-posterior-data",
    ]
    assert list(inspect.signature(getattr(marginal["callback"], "__wrapped__")).parameters) == [
        "parameter",
        "component_id",
        "payload",
    ]

    inference = _callback_definition(app, "trajectory-results.children")
    assert list(inspect.signature(getattr(inference["callback"], "__wrapped__")).parameters) == [
        "set_progress",
        "_clicks",
        "records",
        "truth",
        "observation_time",
        "models",
        "particles",
        "chains",
        "cores",
        "seed",
        "threshold",
        "correlation",
        "prior_bounds",
        "sigma_lambda_prior",
        "sigma_eta_prior",
        "beta_prior_sd",
        "n_quad",
        "colour_values",
        "colour_ids",
    ]


def test_background_inference_exposes_native_pymc_progress_and_locks_controls(app) -> None:
    inference = _callback_definition(app, "trajectory-results.children")
    background = inference["background"]
    assert background is not None
    assert background["interval"] == 350
    assert {str(output) for output in background["progress"]} == {
        "trajectory-pymc-progress-bar.value",
        "trajectory-pymc-progress-label.children",
        "trajectory-pymc-progress-meta.children",
        "trajectory-chain-progress.children",
    }
    assert background["progressDefault"] == [
        0,
        "PyMC SMC sampler",
        "Start inference to see each chain's SMC stage and tempering value β.",
        [],
    ]

    callback_registration = next(
        definition
        for definition in app._callback_list
        if "trajectory-results.children" in definition["output"]
    )
    assert callback_registration["running"] == {
        "running": {
            "trajectory-pymc-progress.className": "barracuda-pymc-progress is-active",
            "trajectory-inference.aria-busy": "true",
            "trajectory-inference-controls.disabled": True,
            "trajectory-run.className": "barracuda-button primary full is-running",
        },
        "runningOff": {
            "trajectory-pymc-progress.className": "barracuda-pymc-progress is-hidden",
            "trajectory-inference.aria-busy": "false",
            "trajectory-inference-controls.disabled": False,
            "trajectory-run.className": "barracuda-button primary full",
        },
    }


@pytest.mark.parametrize(
    ("model_key", "expected"),
    [
        (
            "homogeneous_history_independent",
            (True, 0.0, True, 0.0, True, 0.0),
        ),
        (
            "homogeneous_history_dependent",
            (True, 0.0, False, 0.8, False, -0.8),
        ),
        (
            "heterogeneous_history_independent",
            (False, 0.75, True, 0.0, True, 0.0),
        ),
        (
            "heterogeneous_history_dependent",
            (False, 0.75, False, 0.8, False, -0.8),
        ),
    ],
)
def test_ground_truth_model_controls_follow_the_four_model_family(
    app,
    model_key: str,
    expected: tuple,
) -> None:
    callback = _callback(app, "trajectory-sigma-eta.disabled")
    assert callback(model_key) == expected


def test_csv_upload_preserves_leading_zeroes_and_zero_contact_cells(app) -> None:
    csv_text = (
        "cell_id,condition,history\n"
        "cell_1,Control,001\n"
        "cell_2,Control,\n"
    )
    encoded = base64.b64encode(csv_text.encode("utf-8")).decode("ascii")
    callback = _callback(app, "trajectory-upload-data.data")

    records, status = callback(f"data:text/csv;base64,{encoded}")

    frame = trajectory._frame_from_records(records)
    assert frame.to_dict("records") == [
        {"cell_id": "cell_1", "condition": "Control", "history": (0, 0, 1)},
        {"cell_id": "cell_2", "condition": "Control", "history": ()},
    ]
    assert "Trajectory CSV loaded" in _text(status)


def test_synthetic_truth_reaches_active_data_and_uploads_clear_it(app) -> None:
    generate = _callback(app, "trajectory-synthetic-data.data")
    records, truth, _status = generate(
        1,
        "heterogeneous_history_dependent",
        20,
        4.0,
        2.0,
        0.25,
        0.75,
        0.8,
        -0.8,
        1.5,
        "123",
    )
    assert len(records) == 20
    assert truth["Synthetic"]["true_model_key"] == (
        "heterogeneous_history_dependent"
    )
    assert truth["Synthetic"]["observation_time"] == 1.5
    assert truth["Synthetic"]["beta_f"] == 0.8
    assert truth["Synthetic"]["beta_s"] == -0.8

    activate = _callback(app, "trajectory-active-data.data")
    synthetic = activate("synthetic", records, None, 1.0, 700, 1.0, truth)
    assert synthetic[0] == records
    assert synthetic[1] == truth
    assert synthetic[2] == 1.5
    assert synthetic[3:5] == ("barracuda-workflow-panel", "barracuda-workflow-panel")
    assert synthetic[6].className == "barracuda-trajectory-encoding-legend"
    assert len(synthetic[7].data) > 0
    assert synthetic[8] == {"height": f"{int(synthetic[7].layout.height)}px"}
    assert synthetic[-1] is False

    uploaded = activate("upload", None, records, 2.0, 820, 0.8, truth)
    assert uploaded[1] is None
    assert uploaded[2] == 2.0
    assert uploaded[7].layout.height == 820
    assert uploaded[8] == {"height": "820px"}
    assert uploaded[-1] is False


def test_marginal_selector_uses_selected_parameter_colours_and_truth(app) -> None:
    model_key = "homogeneous_history_independent"
    posterior_records = [
        {
            "condition": "Control",
            "model_key": model_key,
            "chain": 0,
            "draw": draw,
            "mu_lambda": 3.5 + 0.025 * draw,
            "sigma_lambda": 1.0 + 0.005 * draw,
            "mu_eta": -1.0 + 0.002 * draw,
        }
        for draw in range(40)
    ]
    payload = {
        "records": posterior_records,
        "condition_colours": {"Control": "#007AFF"},
        "truth": {"Control": {"mu_lambda": 4.0}},
    }
    callback = _callback(app, '"type":"trajectory-posterior-marginal"')

    figure = callback(
        "mu_lambda",
        {"type": "trajectory-posterior-parameter", "index": model_key},
        payload,
    )

    histogram = next(trace for trace in figure.data if trace.type == "histogram")
    assert histogram.marker.color == "#007AFF"
    assert "μλ" in str(figure.layout.xaxis.title.text)
    assert any(float(shape.x0) == pytest.approx(4.0) for shape in figure.layout.shapes)


def test_one_model_inference_forwards_native_chain_updates_to_dash_progress(
    app,
    monkeypatch,
) -> None:
    model_key = "homogeneous_history_independent"
    progress_updates: list[tuple] = []
    observed: dict[str, object] = {}

    def fake_run(frame, **kwargs):
        observed.update(kwargs)
        kwargs["progress_callback"](1, 1, "Control", 1, 1, "Hom-HI")
        kwargs["sampler_progress_callback"](
            1,
            1,
            "Control",
            1,
            1,
            "Hom-HI",
            0,
            2,
            0.4,
        )
        kwargs["sampler_progress_callback"](
            1,
            1,
            "Control",
            1,
            1,
            "Hom-HI",
            1,
            3,
            0.6,
        )
        return {"Control": {model_key: object()}}

    evidence = pd.DataFrame(
        [
            {
                "condition": "Control",
                "model_key": model_key,
                "log_evidence": -1.0,
            }
        ]
    )
    draws = pd.DataFrame(
        [
            {
                "condition": "Control",
                "model_key": model_key,
                "chain": 0,
                "draw": draw,
                "mu_lambda": 4.0,
                "sigma_lambda": 1.0,
                "mu_eta": -1.0,
            }
            for draw in range(4)
        ]
    )
    monkeypatch.setattr(trajectory, "run_trajectory_conditions", fake_run)
    monkeypatch.setattr(trajectory, "trajectory_evidence_frame", lambda _results: evidence)
    monkeypatch.setattr(
        trajectory,
        "trajectory_posterior_draws",
        lambda _results, max_draws: draws,
    )
    monkeypatch.setattr(trajectory, "build_trajectory_archive", lambda *args, **kwargs: b"zip")
    monkeypatch.setattr(
        trajectory,
        "render_trajectory_results",
        lambda **kwargs: ("rendered results", "rendered download"),
    )
    callback = _callback(app, "trajectory-results.children")
    records = [
        {"cell_id": "cell_1", "condition": "Control", "history": "0,1"}
    ]

    content, download, _status = callback(
        progress_updates.append,
        1,
        records,
        None,
        1.0,
        [model_key],
        32,
        2,
        1,
        "123",
        0.5,
        0.01,
        [-1.0, 1.5],
        2.0,
        1.0,
        1.0,
        5,
        ["#007AFF"],
        [{"type": "trajectory-condition-colour", "index": "Control"}],
    )

    assert content == "rendered results"
    assert download == "rendered download"
    assert observed["model_keys"] == [model_key]
    assert observed["settings"].particles == 32
    assert observed["settings"].chains == 2
    assert [update[0] for update in progress_updates] == pytest.approx(
        [0.0, 0.2, 0.5, 1.0]
    )
    assert "Direct PyMC SMC updates" in progress_updates[1][2]
    assert "2 independent chains" in progress_updates[1][2]
    assert "32 particles per chain" in progress_updates[1][2]
    assert "Stage 2" in _text(progress_updates[1][3][0])
    assert progress_updates[-1][1] == "PyMC sampling complete"
