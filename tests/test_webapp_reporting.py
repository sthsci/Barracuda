from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import arviz as az
from dash.development.base_component import Component
from multiprocess import Process, Queue
import numpy as np

from webapp.analysis_ui import rate_distribution_figure, render_validation_results
from webapp.core.data import sample_count_frame
from webapp.core.inference import InferenceResult, InferenceSettings, MODEL_SPECS
from webapp.reporting import (
    bayes_factor_figure,
    joint_posterior_figure,
    joint_posterior_figure_from_draws,
    posterior_draws_from_store,
    posterior_parameters_for_models,
    posterior_store_payload,
    validation_figure_artifacts,
)
from webapp.core.inference import posterior_draw_table
from webapp.dashapp import create_app


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
    return " ".join(
        str(children)
        for item in _walk(component)
        if isinstance((children := getattr(item, "children", None)), (str, int, float))
    )


def _results() -> dict[str, InferenceResult]:
    rng = np.random.default_rng(2026)
    base = rng.normal(size=160)
    posterior_by_model = {
        "homo": {"lambda": 4.0 + 0.30 * base},
        "z2p": {
            "lambda": 4.1 + 0.35 * base,
            "p_zero": np.clip(0.20 + 0.025 * base, 0.01, 0.99),
        },
        "dis2p": {
            "mu_lambda": 4.0 + 0.40 * base,
            "sigma_lambda": np.clip(3.0 + 0.25 * base, 0.01, None),
        },
        "hetero3": {
            "mu_lambda": 4.0 + 0.45 * base,
            "sigma_lambda": np.clip(3.0 + 0.30 * base, 0.01, None),
            "p_zero": np.clip(0.20 + 0.03 * base, 0.01, 0.99),
        },
    }
    evidence = {"homo": -14.0, "z2p": -12.0, "dis2p": -11.0, "hetero3": -10.0}
    results = {}
    for model_key, posterior in posterior_by_model.items():
        idata = az.from_dict(
            posterior={
                name: np.asarray(values, dtype=float).reshape(1, -1)
                for name, values in posterior.items()
            }
        )
        results[model_key] = InferenceResult(
            model_key=model_key,
            model_label=MODEL_SPECS[model_key].label,
            donor_aware=False,
            idata=idata,
            model=None,
            log_evidence=evidence[model_key],
            elapsed_seconds=0.1,
            n_cells=12,
            observation_time=1.0,
        )
    return results


def _render_static_artifacts_in_worker(queue: Queue) -> None:
    artifacts = validation_figure_artifacts(
        _results(),
        {"mu_lambda": 4.0, "sigma_lambda": 3.0, "p_zero": 0.2},
    )
    queue.put({name: len(content) for name, content in artifacts.items()})


def test_full_population_rate_preview_contains_mass_and_weighted_density() -> None:
    figure = rate_distribution_figure("gamma", 4.0, 3.0, 0.2)

    zero_mass = next(trace for trace in figure.data if trace.type == "bar")
    density = next(trace for trace in figure.data if trace.type == "scatter")
    assert float(zero_mass.x[0]) == 0.0
    assert float(zero_mass.y[0]) == 0.2
    np.testing.assert_allclose(
        np.trapz(np.asarray(density.y), np.asarray(density.x)),
        0.8,
        atol=0.01,
    )
    axis_titles = " ".join(
        str(getattr(getattr(axis, "title", None), "text", ""))
        for axis in (figure.layout.yaxis, figure.layout.yaxis2)
    )
    assert "engaging cells" not in axis_titles.lower()


def test_fixed_zero_inflated_preview_has_two_population_masses() -> None:
    figure = rate_distribution_figure("fixed", 4.0, 0.0, 0.2)
    bars = figure.data[0]

    assert set(np.asarray(bars.x, dtype=float)) == {0.0, 4.0}
    np.testing.assert_allclose(np.asarray(bars.y, dtype=float), [0.2, 0.8])
    assert float(np.sum(bars.y)) == 1.0


def test_joint_posterior_and_bayes_factor_figures_match_reference_convention() -> None:
    truth = {"mu_lambda": 4.0, "sigma_lambda": 3.0, "p_zero": 0.2}
    results = _results()
    joint = joint_posterior_figure(results, truth)
    bayes = bayes_factor_figure(results, truth)

    assert joint.layout.height >= 800
    assert any(trace.type == "histogram2dcontour" for trace in joint.data)
    assert any(
        trace.type == "scatter" and trace.name == "Ground truth"
        for trace in joint.data
    )
    assert len(joint.layout.shapes) >= 9

    values = np.asarray(bayes.data[0].x, dtype=float)
    raw_values = np.asarray(bayes.data[0].customdata, dtype=float)[:, 0]
    assert values.min() == 0.0
    np.testing.assert_allclose(raw_values.max(), 4.0 / np.log(10.0))
    np.testing.assert_allclose(values, raw_values)
    assert "BF(𝓜<sub>best</sub> / 𝓜)" in bayes.layout.xaxis.title.text
    np.testing.assert_allclose(
        np.asarray(bayes.layout.xaxis.tickvals, dtype=float)[:4],
        [0.0, np.log10(3.0), 1.0, 2.0],
    )
    assert list(bayes.layout.xaxis.ticktext)[:4] == ["0", "log₁₀ 3", "1", "2"]
    assert {trace.name for trace in bayes.data[1:]} == {
        "Anecdotal · BF 1–3",
        "Moderate · BF 3–10",
        "Strong · BF 10–100",
        "Extreme · BF ≥100",
    }
    best_rows = [
        index
        for index, label in enumerate(bayes.data[0].y)
        if "Best model" in str(label)
    ]
    assert len(best_rows) == 1
    assert "𝓜_ZIΓ" in str(bayes.data[0].y[best_rows[0]])
    assert str(bayes.data[0].text[best_rows[0]]) == "Best model · 0.00"
    assert len(bayes.layout.shapes) == 4
    spans = [(float(shape.x0), float(shape.x1)) for shape in bayes.layout.shapes]
    np.testing.assert_allclose(
        spans,
        [
            (0.0, np.log10(3.0)),
            (np.log10(3.0), 1.0),
            (1.0, 2.0),
            (2.0, 3.0),
        ],
    )


def test_large_bayes_factors_are_not_rescaled_or_clipped() -> None:
    results = _results()
    results["homo"] = replace(results["homo"], log_evidence=-100.0)
    figure = bayes_factor_figure(results)

    plotted = np.asarray(figure.data[0].x, dtype=float)
    raw = np.asarray(figure.data[0].customdata, dtype=float)[:, 0]
    np.testing.assert_allclose(plotted, raw)
    assert plotted.max() > 30.0
    assert float(figure.layout.xaxis.range[1]) > plotted.max()
    assert float(figure.layout.shapes[-1].x0) == 2.0
    assert float(figure.layout.shapes[-1].x1) == float(figure.layout.xaxis.range[1])
    large_ticks = np.asarray(figure.layout.xaxis.tickvals, dtype=float)
    assert large_ticks[0] == 0.0
    assert np.all(np.diff(large_ticks) == 10.0)
    assert not np.any(np.isclose(large_ticks, np.log10(3.0)))


def test_joint_posterior_model_subsets_adjust_the_parameter_grid() -> None:
    truth = {"mu_lambda": 4.0, "sigma_lambda": 3.0, "p_zero": 0.2}
    draws = posterior_draw_table(_results(), max_draws_per_model=80)
    cases = [
        (["homo"], ["mu_lambda"], 430),
        (["z2p"], ["mu_lambda", "p_zero"], 570),
        (["dis2p"], ["mu_lambda", "sigma_lambda"], 570),
        (
            ["z2p", "dis2p"],
            ["mu_lambda", "sigma_lambda", "p_zero"],
            855,
        ),
        (
            ["homo", "z2p", "dis2p", "hetero3"],
            ["mu_lambda", "sigma_lambda", "p_zero"],
            855,
        ),
    ]

    for model_keys, parameters, height in cases:
        assert posterior_parameters_for_models(model_keys) == parameters
        figure = joint_posterior_figure_from_draws(draws, model_keys, truth)
        assert figure.layout.height == height
        shown_models = {
            trace.legendgroup
            for trace in figure.data
            if trace.legendgroup and trace.legendgroup != "truth"
        }
        assert shown_models == set(model_keys)
        assert figure.layout.title.text == "Ground truth: " + ", ".join(
            {
                "mu_lambda": "μλ = 4",
                "sigma_lambda": "σλ = 3",
                "p_zero": "φ₀ = 0.2",
            }[parameter]
            for parameter in parameters
        )


def test_posterior_store_preserves_paired_draw_rows() -> None:
    results = _results()
    draws = posterior_draw_table(results, max_draws_per_model=80)
    payload = posterior_store_payload(
        draws,
        list(results),
        {"mu_lambda": 4.0, "sigma_lambda": 3.0, "p_zero": 0.2},
    )
    restored, selected, truth = posterior_draws_from_store(
        payload,
        ["z2p", "dis2p"],
    )

    assert selected == ["z2p", "dis2p"]
    assert truth == {"mu_lambda": 4.0, "sigma_lambda": 3.0, "p_zero": 0.2}
    for model_key in selected:
        parameters = posterior_parameters_for_models([model_key])
        original = draws.loc[draws["model_key"] == model_key, parameters]
        recovered = restored.loc[restored["model_key"] == model_key, parameters]
        np.testing.assert_allclose(
            recovered.to_numpy(dtype=float),
            original.to_numpy(dtype=float),
        )


def test_posterior_filter_callback_rebuilds_and_compacts_the_plot() -> None:
    results = _results()
    draws = posterior_draw_table(results, max_draws_per_model=80)
    payload = posterior_store_payload(
        draws,
        list(results),
        {"mu_lambda": 4.0, "sigma_lambda": 3.0, "p_zero": 0.2},
    )
    app = create_app()
    callback = next(
        value["callback"].__wrapped__
        for key, value in app.callback_map.items()
        if "synthetic-posterior-figure.figure" in key
    )

    figure, style, summary = callback(["z2p"], payload)
    assert figure.layout.height == 570
    assert style == {"height": "570px"}
    assert summary == "1 model shown · 2 parameters: μλ and φ₀."

    empty, empty_style, empty_summary = callback([], payload)
    assert empty.layout.height == 430
    assert empty_style == {"height": "430px"}
    assert empty_summary == "Select at least one inference result."


def test_validation_results_hide_tables_and_offer_plot_csv_and_idata_exports() -> None:
    results = _results()
    truth = {
        "model_key": "hetero3",
        "model_label": MODEL_SPECS["hetero3"].label,
        "mu_lambda": 4.0,
        "sigma_lambda": 3.0,
        "p_zero": 0.2,
        "observation_time": 1.0,
        "seed": 2026,
    }
    content, downloads = render_validation_results(
        results,
        data=sample_count_frame(),
        observation_time=1.0,
        settings=InferenceSettings(draws=16, chains=1, cores=1),
        truth=truth,
    )

    text = _text(content)
    assert "Joint posterior distributions" in text
    assert "Bayes factors" in text
    assert "Model comparison" not in text
    assert "Posterior summaries" not in text
    assert "Ground truth recovery check" not in text
    assert not any(item.__class__.__name__ == "AgGrid" for item in _walk(content))

    by_id = {
        component_id: component
        for component in _walk(content)
        if isinstance((component_id := getattr(component, "id", None)), str)
    }
    selector = by_id["synthetic-posterior-model-filter"]
    assert selector.value == list(results)
    assert [option["value"] for option in selector.options] == list(results)
    assert by_id["synthetic-posterior-data"].data["schema_version"] == 1
    assert by_id["synthetic-posterior-figure"].style["height"] == "855px"

    links = [item for item in _walk([content, downloads]) if item.__class__.__name__ == "A"]
    filenames = {getattr(link, "download", None) for link in links}
    assert {
        "barracuda_joint_posterior.png",
        "barracuda_joint_posterior.pdf",
        "barracuda_posterior_samples.csv",
        "barracuda_bayes_factors.png",
        "barracuda_bayes_factors.pdf",
        "barracuda_model_evidence.csv",
        "barracuda_posterior_summary.csv",
        "barracuda_ground_truth_recovery.csv",
        "barracuda_synthetic_validation.zip",
    } <= filenames
    assert ".nc file per model" in _text(downloads)


def test_static_exports_finish_inside_a_background_worker() -> None:
    queue = Queue()
    worker = Process(target=_render_static_artifacts_in_worker, args=(queue,))
    worker.start()
    worker.join(timeout=30)

    assert worker.exitcode == 0
    sizes = queue.get(timeout=2)
    assert set(sizes) == {
        "figures/joint_posterior.png",
        "figures/joint_posterior.pdf",
        "figures/bayes_factors.png",
        "figures/bayes_factors.pdf",
    }
    assert all(size > 1_000 for size in sizes.values())


def test_downloadable_notebook_is_clean_and_self_contained() -> None:
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "webapp"
        / "assets"
        / "downloads"
        / "barracuda_synthetic_validation_demo.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert notebook["metadata"]["kernelspec"]["name"] == "python3"
    assert all(not cell.get("outputs") for cell in notebook["cells"])
    assert "from simulator import" not in source
    assert "from inference import" not in source
    assert "find_section_root" not in source
    assert "pm.sample_smc" in source
    assert "progressbar=True" in source
    assert "to_netcdf" in source
    assert "joint_posterior.pdf" in source
    assert "bayes_factors.pdf" in source
