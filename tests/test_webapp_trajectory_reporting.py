from __future__ import annotations

import base64
import inspect
import math

from dash.development.base_component import Component
import numpy as np
import pandas as pd
import pytest

import webapp.trajectory_reporting as reporting
from webapp.trajectory_reporting import (
    BF3_LOG10,
    TRAJECTORY_MODEL_LABELS,
    empirical_state_arrow_figure,
    empirical_state_encoding_legend,
    empirical_state_summary,
    expanded_history_frame,
    joint_posterior_figure,
    model_panel_styles,
    posterior_marginal_figure,
    render_trajectory_results,
    trajectory_bayes_factor_figure,
    trajectory_posterior_payload,
)


def _walk(component):
    if isinstance(component, Component):
        yield component
        children = getattr(component, "children", None)
        if children is not None:
            yield from _walk(children)
    elif isinstance(component, (list, tuple)):
        for child in component:
            yield from _walk(child)


def _component_ids(component) -> list[object]:
    return [
        item.id
        for item in _walk(component)
        if getattr(item, "id", None) is not None
    ]


def _posterior_draws() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    models = [
        "homogeneous_history_independent",
        "heterogeneous_history_dependent",
    ]
    for condition_index, condition in enumerate(["Control", "Treatment"]):
        for model_index, model_key in enumerate(models):
            for draw in range(30):
                centred = (draw - 14.5) / 20.0
                values = {
                    "mu_lambda": 3.0 + condition_index + 0.15 * centred,
                    "sigma_lambda": 0.9 + 0.07 * centred,
                    "mu_eta": -1.0 + 0.10 * condition_index + 0.03 * centred,
                    "sigma_eta": 0.65 + 0.08 * centred,
                    "beta_f": centred,
                    # Deliberately deterministic so the test can prove that
                    # paired draws, rather than independently shuffled
                    # marginals, reach the lower triangle.
                    "beta_s": 2.0 * centred,
                }
                for parameter, value in values.items():
                    rows.append(
                        {
                            "condition": condition,
                            "model_key": model_key,
                            "chain": model_index,
                            "draw": draw,
                            "parameter": parameter,
                            "value": value,
                        }
                    )
    return pd.DataFrame(rows)


def _evidence() -> pd.DataFrame:
    values = {
        "Control": (-10.0, -11.0),
        "Treatment": (-14.0, -13.0),
    }
    models = [
        "homogeneous_history_independent",
        "heterogeneous_history_dependent",
    ]
    return pd.DataFrame(
        [
            {
                "condition": condition,
                "model_key": model_key,
                "log_evidence": log_evidence,
            }
            for condition, condition_values in values.items()
            for model_key, log_evidence in zip(models, condition_values)
        ]
    )


def test_histories_expand_to_precontact_states_in_order() -> None:
    frame = pd.DataFrame(
        {
            "cell_id": ["cell_1", "cell_2", "cell_3"],
            "condition": ["Control", "Control", "Treatment"],
            "history": ["[0, 1, 0]", "11", []],
        }
    )

    expanded = expanded_history_frame(frame)

    assert list(expanded.columns) == [
        "cell_id",
        "condition",
        "contact_index",
        "x_before",
        "y_before",
        "outcome",
    ]
    assert expanded.to_dict("records") == [
        {
            "cell_id": "cell_1",
            "condition": "Control",
            "contact_index": 1,
            "x_before": 0,
            "y_before": 0,
            "outcome": 0,
        },
        {
            "cell_id": "cell_1",
            "condition": "Control",
            "contact_index": 2,
            "x_before": 1,
            "y_before": 0,
            "outcome": 1,
        },
        {
            "cell_id": "cell_1",
            "condition": "Control",
            "contact_index": 3,
            "x_before": 1,
            "y_before": 1,
            "outcome": 0,
        },
        {
            "cell_id": "cell_2",
            "condition": "Control",
            "contact_index": 1,
            "x_before": 0,
            "y_before": 0,
            "outcome": 1,
        },
        {
            "cell_id": "cell_2",
            "condition": "Control",
            "contact_index": 2,
            "x_before": 0,
            "y_before": 1,
            "outcome": 1,
        },
    ]


def test_history_validation_rejects_ambiguous_cells_and_outcomes() -> None:
    duplicate = pd.DataFrame(
        {
            "cell_id": ["cell_1", "cell_1"],
            "condition": ["Control", "Control"],
            "history": [[0], [1]],
        }
    )
    invalid = pd.DataFrame(
        {"cell_id": ["cell_1"], "condition": ["Control"], "history": [[2]]}
    )

    with pytest.raises(ValueError, match="unique"):
        expanded_history_frame(duplicate)
    with pytest.raises(ValueError, match="only 0 and 1"):
        expanded_history_frame(invalid)


def test_state_summary_counts_cells_contacts_and_empirical_probability() -> None:
    frame = pd.DataFrame(
        {
            "cell_id": ["a", "b", "c", "d"],
            "condition": ["Control"] * 4,
            "history": [[0, 1], [1], [1], [1]],
        }
    )

    summary = empirical_state_summary(frame)
    origin = summary.loc[
        (summary["x_before"] == 0) & (summary["y_before"] == 0)
    ].iloc[0]
    after_failure = summary.loc[
        (summary["x_before"] == 1) & (summary["y_before"] == 0)
    ].iloc[0]

    assert int(origin.n_cells) == 4
    assert int(origin.n_contacts) == 4
    assert int(origin.n_lethal) == 3
    assert float(origin.empirical_lethal_probability) == pytest.approx(0.75)
    assert float(origin.log2_n_cells) == pytest.approx(math.log2(4.0))
    assert int(after_failure.n_cells) == 1
    assert float(after_failure.empirical_lethal_probability) == 1.0


def test_state_summary_accepts_the_core_expanded_column_names() -> None:
    expanded = pd.DataFrame(
        {
            "cell_id": ["a", "b"],
            "condition": ["Control", "Control"],
            "contact_index": [1, 1],
            "previous_nonlethal_contacts": [0, 0],
            "previous_lethal_contacts": [0, 0],
            "outcome": [0, 1],
        }
    )

    summary = empirical_state_summary(expanded)

    assert len(summary) == 1
    assert int(summary.iloc[0].n_cells) == 2
    assert float(summary.iloc[0].empirical_lethal_probability) == 0.5


def test_empirical_arrow_map_facets_and_uses_log_cell_count_length() -> None:
    frame = pd.DataFrame(
        {
            "cell_id": ["a", "b", "c", "d", "e"],
            "condition": ["Control"] * 4 + ["Treatment"],
            "history": [[0, 1], [1], [1], [1], [0, 0]],
        }
    )

    figure = empirical_state_arrow_figure(
        frame,
        {"Control": "#007AFF", "Treatment": "#FF3B30"},
    )

    titles = {str(annotation.text) for annotation in figure.layout.annotations}
    assert {"Control", "Treatment"}.issubset(titles)
    arrow_traces = [
        trace
        for trace in figure.data
        if trace.type == "scatter"
        and trace.mode == "lines+markers"
        and trace.x[0] is not None
    ]
    control_origin, control_after_failure = arrow_traces[:2]
    origin_length = math.hypot(
        float(control_origin.x[1]) - float(control_origin.x[0]),
        float(control_origin.y[1]) - float(control_origin.y[0]),
    )
    after_length = math.hypot(
        float(control_after_failure.x[1]) - float(control_after_failure.x[0]),
        float(control_after_failure.y[1]) - float(control_after_failure.y[0]),
    )
    assert origin_length > after_length
    assert figure.layout.yaxis.scaleanchor == "x"
    assert figure.layout.yaxis2.scaleanchor == "x2"
    assert not figure.layout.images
    legend = empirical_state_encoding_legend(frame)
    legend_source = str(legend.children[0].src)
    assert legend_source.startswith("data:image/svg+xml;base64,")
    legend_svg = base64.b64decode(legend_source.split(",", 1)[1]).decode()
    assert "Empirical killing probability" in legend_svg
    assert "arrow length increases with log₂ n" in legend_svg
    assert "horizontal = non-lethal" in legend_svg
    assert "vertical = lethal" in legend_svg

    larger = empirical_state_arrow_figure(frame, arrow_scale=1.4)
    larger_arrow = next(
        trace
        for trace in larger.data
        if trace.type == "scatter"
        and trace.mode == "lines+markers"
        and trace.x[0] is not None
    )
    larger_length = math.hypot(
        float(larger_arrow.x[1]) - float(larger_arrow.x[0]),
        float(larger_arrow.y[1]) - float(larger_arrow.y[0]),
    )
    assert larger_length > origin_length


def test_bayes_factor_axis_is_exact_continuous_and_marks_best_and_truth() -> None:
    figure = trajectory_bayes_factor_figure(
        _evidence(),
        truth_model={
            "Control": "homogeneous_history_independent",
            "Treatment": "heterogeneous_history_dependent",
        },
        condition_colours={"Control": "#007AFF", "Treatment": "#FF3B30"},
    )

    bars = next(trace for trace in figure.data if trace.type == "bar")
    values = np.asarray(bars.x, dtype=float)
    assert np.all(values <= 0.0)
    assert np.count_nonzero(np.isclose(values, 0.0)) == 2
    np.testing.assert_allclose(
        np.sort(np.unique(values)),
        [-1.0 / np.log(10.0), 0.0],
    )
    boundaries = {
        float(shape.x1)
        for shape in figure.layout.shapes
    }
    assert any(np.isclose(value, -BF3_LOG10) for value in boundaries)
    assert {-2.0, -1.0, 0.0}.issubset(boundaries)
    assert figure.layout.xaxis.type in (None, "linear")
    assert float(figure.layout.xaxis.range[0]) < -2.0
    assert float(figure.layout.xaxis.range[1]) > 0.0
    best = next(trace for trace in figure.data if trace.name == "Best model")
    np.testing.assert_allclose(np.asarray(best.x, dtype=float), 0.0)
    assert all("Best model" in str(label) for label in best.y)
    assert sum("Ground truth" in str(label) for label in bars.y) == 2
    assert set(bars.marker.color) == {"#007AFF", "#FF3B30"}


def test_positive_model_over_best_bayes_factor_is_rejected() -> None:
    evidence = pd.DataFrame(
        {
            "model_key": ["one"],
            "log10_bf_model_vs_best": [0.2],
        }
    )
    with pytest.raises(ValueError, match="cannot be positive"):
        trajectory_bayes_factor_figure(evidence)


def test_posterior_payload_preserves_draw_pairing_and_model_parameters() -> None:
    payload = trajectory_posterior_payload(_posterior_draws())

    assert payload["models"] == [
        "homogeneous_history_independent",
        "heterogeneous_history_dependent",
    ]
    assert payload["parameters"]["homogeneous_history_independent"] == [
        "mu_lambda",
        "sigma_lambda",
        "mu_eta",
    ]
    assert payload["parameters"]["heterogeneous_history_dependent"] == [
        "mu_lambda",
        "sigma_lambda",
        "mu_eta",
        "sigma_eta",
        "beta_f",
        "beta_s",
    ]
    records = pd.DataFrame(payload["records"])
    assert len(records) == 2 * 2 * 30
    dependent = records.loc[
        records["model_key"] == "heterogeneous_history_dependent"
    ]
    np.testing.assert_allclose(dependent["beta_s"], 2.0 * dependent["beta_f"])


def test_joint_and_marginal_posteriors_keep_colours_pairing_and_truth() -> None:
    draws = _posterior_draws()
    colours = {"Control": "#007AFF", "Treatment": "#FF3B30"}
    truth = {
        "Control": {"beta_f": 0.0, "beta_s": 0.0, "mu_lambda": 3.5},
        "Treatment": {"beta_f": 0.1, "beta_s": 0.2, "mu_lambda": 4.5},
    }
    marginal = posterior_marginal_figure(
        draws,
        "beta_f",
        colours,
        truth,
        "heterogeneous_history_dependent",
    )
    joint = joint_posterior_figure(
        draws,
        "heterogeneous_history_dependent",
        colours,
        truth,
        parameters=["beta_f", "beta_s"],
    )

    histograms = [trace for trace in marginal.data if trace.type == "histogram"]
    assert [trace.marker.color for trace in histograms] == [
        "#007AFF",
        "#FF3B30",
    ]
    assert any(np.isclose(float(shape.x0), 0.0) for shape in marginal.layout.shapes)
    assert histograms[0].xbins.to_plotly_json() == histograms[1].xbins.to_plotly_json()
    assert len(marginal.layout.shapes) == 4  # two HDIs and two truth lines
    contours = [
        trace for trace in joint.data if trace.type == "histogram2dcontour"
    ]
    assert len(contours) == 2
    for trace in contours:
        np.testing.assert_allclose(
            np.asarray(trace.y, dtype=float),
            2.0 * np.asarray(trace.x, dtype=float),
        )
    assert any(
        trace.type == "scatter" and "ground truth" in str(trace.name).lower()
        for trace in joint.data
    )
    assert int(joint.layout.height) >= 490


def test_result_renderer_puts_bayes_factors_first_and_exposes_callback_ids() -> None:
    content, downloads = render_trajectory_results(
        evidence=_evidence(),
        posterior_draws=_posterior_draws(),
        condition_colours={"Control": "#007AFF", "Treatment": "#FF3B30"},
        truth={
            "Control": {"mu_lambda": 3.5, "mu_eta": -0.8},
            "Treatment": {"mu_lambda": 4.5, "mu_eta": -0.6},
        },
        truth_model="heterogeneous_history_dependent",
        prefix="trajectory",
    )

    assert content.children[0].children[1].children == (
        "Bayes factors by experimental condition"
    )
    ids = _component_ids(content)
    assert "trajectory-bayes-factor-figure" in ids
    assert "trajectory-posterior-data" in ids
    assert "trajectory-model-view" in ids
    posterior_store = next(
        item for item in _walk(content) if getattr(item, "id", None) == "trajectory-posterior-data"
    )
    assert posterior_store.data["condition_colours"] == {
        "Control": "#007AFF",
        "Treatment": "#FF3B30",
    }
    expected_models = set(TRAJECTORY_MODEL_LABELS).intersection(
        set(_evidence()["model_key"])
    )
    assert {
        item["index"]
        for item in ids
        if isinstance(item, dict) and item["type"] == "trajectory-model-panel"
    } == expected_models
    heterogeneous_joint = next(
        item
        for item in _walk(content)
        if getattr(item, "id", None)
        == {
            "type": "trajectory-posterior-joint",
            "index": "heterogeneous_history_dependent",
        }
    )
    assert heterogeneous_joint.style["minWidth"] == "1320px"
    assert {
        item["index"]
        for item in ids
        if isinstance(item, dict)
        and item["type"] == "trajectory-posterior-parameter"
    } == expected_models
    links = [
        item
        for item in _walk(downloads)
        if getattr(item, "download", None) is not None
    ]
    assert {item.download for item in links} == {
        "barracuda_trajectory_model_evidence.csv",
        "barracuda_trajectory_posterior_samples.csv",
    }


def test_single_model_result_explains_why_no_bayes_factor_is_drawn() -> None:
    evidence = _evidence().loc[
        _evidence()["model_key"] == "homogeneous_history_independent"
    ]
    draws = _posterior_draws().loc[
        _posterior_draws()["model_key"] == "homogeneous_history_independent"
    ]
    content, _downloads = render_trajectory_results(
        evidence=evidence,
        posterior_draws=draws,
        condition_colours={"Control": "#007AFF", "Treatment": "#FF3B30"},
    )

    ids = _component_ids(content)
    assert "trajectory-bayes-factor-unavailable" in ids
    assert "trajectory-bayes-factor-figure" not in ids
    unavailable = next(
        item
        for item in _walk(content)
        if getattr(item, "id", None) == "trajectory-bayes-factor-unavailable"
    )
    assert "at least two models" in str(unavailable.children[0].children).lower()


def test_model_panel_styles_support_any_model_combination() -> None:
    ids = [
        {"type": "trajectory-model-panel", "index": "one"},
        {"type": "trajectory-model-panel", "index": "two"},
        {"type": "trajectory-model-panel", "index": "three"},
    ]
    assert model_panel_styles(["one", "three"], ids) == [
        {},
        {"display": "none"},
        {},
    ]


def test_reporting_source_has_no_scipy_kde_dependency() -> None:
    source = inspect.getsource(reporting).lower()
    assert "scipy" not in source
    assert "gaussian_kde" not in source
