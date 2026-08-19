from __future__ import annotations

import base64
import time
from pathlib import Path

import diskcache
from dash.development.base_component import Component

from webapp.analysis_ui import read_uploaded_csv
from webapp.dashapp import OrcaDiskcacheManager, PAGE_BY_PATH, PAGES, create_app
from webapp.pages import bayes_101, notebooks


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


def test_dash_server_and_health_endpoint_load() -> None:
    app = create_app()
    client = app.server.test_client()

    response = client.get("/")
    health = client.get("/healthz")

    assert response.status_code == 200
    assert b"Orca" in response.data
    assert health.status_code == 200
    assert health.get_json() == {"application": "orca-dash", "status": "ok"}


def test_every_route_has_distinct_content_and_no_streamlit_dependency() -> None:
    expected = {
        "/": "One question, three levels of information",
        "/bayesian-101": "The update at the heart of Bayesian inference",
        "/notebooks": "Learn and analyse in Google Colab",
        "/event-counts": "Choose an analysis",
        "/event-counts/donor-ignorant": "Which data do you want to use?",
        "/event-counts/donor-aware": "Provide counts with donor labels",
        "/trajectory": "Which trajectory data do you want to use?",
    }
    assert {page.PATH for page in PAGES} == set(expected)
    for page in PAGES:
        content = page.layout()
        assert expected[page.PATH] in _text(content)
    assert PAGE_BY_PATH["/synthetic-validation"].PATH == "/event-counts/donor-ignorant"


def test_colab_notebook_hub_links_every_workflow_from_the_home_page() -> None:
    expected = {
        f"{notebooks.COLAB_ROOT}/{path}"
        for path in (
            "notebooks/00_run_the_orca_web_app.ipynb",
            "notebooks/01_bayesian_inference_101.ipynb",
            "notebooks/02_event_count_model_tutorial.ipynb",
            "notebooks/03_trajectory_model_tutorial.ipynb",
            "notebooks/04_event_count_analysis.ipynb",
            "notebooks/05_donor_aware_analysis.ipynb",
            "notebooks/06_trajectory_analysis.ipynb",
        )
    }
    links = [
        component
        for component in _walk(notebooks.layout())
        if component.__class__.__name__ == "A"
    ]

    assert {link.href for link in links} == expected
    assert all(link.target == "_blank" and link.rel == "noreferrer" for link in links)
    assert all(category in _text(notebooks.layout()) for category in ("Start here", "Teaching", "Analysis"))
    assert any(
        getattr(component, "href", None) == notebooks.PATH
        for component in _walk(PAGE_BY_PATH["/"].layout())
    )


def test_dash_layout_contains_upload_edit_inference_and_download_surfaces() -> None:
    ids: set[str] = set()
    for page in PAGES:
        for component in _walk(page.layout()):
            component_id = getattr(component, "id", None)
            if isinstance(component_id, str):
                ids.add(component_id)

    required = {
        "coin-probability",
        "coin-tosses",
        "coin-hdi-percent",
        "coin-frequency-figure",
        "coin-toss-scene",
        "likelihood-surface",
        "prior-surface",
        "unnormalised-posterior-surface",
        "posterior-surface",
        "mcmc-animation",
        "smc-animation",
        "synthetic-generate",
        "synthetic-run",
        "synthetic-rate-distribution",
        "synthetic-rate-distribution-preview",
        "synthetic-rate-preview-note",
        "synthetic-pymc-progress-bar",
        "synthetic-pymc-progress-label",
        "synthetic-pymc-progress-meta",
        "synthetic-chain-progress",
        "counts-upload",
        "counts-table",
        "counts-run",
        "counts-condition-colour-controls",
        "donor-ignorant-workflow",
        "donor-upload",
        "donor-table",
        "donor-run",
        "donor-condition-colour-controls",
        "trajectory-workflow",
        "trajectory-generate",
        "trajectory-upload",
        "trajectory-empirical-figure",
        "trajectory-condition-colour-controls",
        "trajectory-run",
        "trajectory-pymc-progress-bar",
        "trajectory-pymc-progress-label",
        "trajectory-pymc-progress-meta",
        "trajectory-chain-progress",
    }
    assert required <= ids


def test_event_count_schematics_are_on_their_intended_routes() -> None:
    layouts = {page.PATH: page.layout() for page in PAGES}
    images_by_route = {
        path: [
            component
            for component in _walk(content)
            if component.__class__.__name__ == "Img"
        ]
        for path, content in layouts.items()
    }

    overview_sources = {image.src for image in images_by_route["/event-counts"]}
    validation_sources = {
        image.src
        for image in images_by_route["/event-counts/donor-ignorant"]
    }
    assert overview_sources == {"/assets/event_count_models.png"}
    assert validation_sources == {"/assets/synthetic_validation_workflow.png"}
    assert all(
        getattr(image, "alt", "")
        for images in images_by_route.values()
        for image in images
    )
    assert "synthetic_validation_workflow.png" not in " ".join(overview_sources)
    assert "event_count_models.png" not in " ".join(validation_sources)

    asset_root = Path(__file__).resolve().parents[1] / "webapp" / "assets"
    assert (asset_root / "event_count_models.png").is_file()
    assert (asset_root / "synthetic_validation_workflow.png").is_file()


def test_synthetic_controls_use_paper_terms_and_fold_observation_time() -> None:
    synthetic = next(
        page for page in PAGES if page.PATH == "/event-counts/donor-ignorant"
    )
    content = synthetic.layout()
    by_id = {
        component_id: component
        for component in _walk(content)
        if isinstance((component_id := getattr(component, "id", None)), str)
    }

    assert by_id["synthetic-observation-time"].value == 1.0
    distribution = by_id["synthetic-rate-distribution"]
    assert distribution.value == "gamma"
    assert distribution.disabled is True
    assert {option["value"] for option in distribution.options} == {
        "fixed",
        "gamma",
    }
    details = [
        component
        for component in _walk(content)
        if component.__class__.__name__ == "Details"
    ]
    assert any(
        "synthetic-observation-time"
        in {
            getattr(child, "id", None)
            for child in _walk(detail)
        }
        for detail in details
    )
    page_text = _text(content)
    assert "A demonstration is not calibration" not in page_text
    assert "formal calibration" not in page_text
    assert "Fraction of nonengaging cells, φ₀" in page_text
    assert "Mean event rate among engaging cells, μλ" in page_text
    assert "Continuous cell-to-cell heterogeneity in event rates, σλ" in page_text
    assert "Alternative rate distributions · in development" in page_text
    assert "Lognormal, truncated Normal" in page_text
    assert any(
        "𝓜_ZIΓ" in str(option["label"])
        for option in by_id["synthetic-ground-model"].options
    )

    notebook_links = [
        component
        for component in _walk(content)
        if component.__class__.__name__ == "A"
        and getattr(component, "download", None)
        == "orca_synthetic_validation_demo.ipynb"
    ]
    assert len(notebook_links) == 1
    assert notebook_links[0].href == "/assets/downloads/orca_synthetic_validation_demo.ipynb"
    notebook_path = (
        Path(__file__).resolve().parents[1]
        / "webapp"
        / "assets"
        / "downloads"
        / "orca_synthetic_validation_demo.ipynb"
    )
    assert notebook_path.is_file()


def test_synthetic_inference_uses_a_background_callback_with_live_progress() -> None:
    app = create_app()
    callback = next(
        value
        for key, value in app.callback_map.items()
        if "synthetic-results.children" in key
    )
    background = callback["background"]

    assert background is not None
    assert background["interval"] == 350
    progress_outputs = {str(output) for output in background["progress"]}
    assert progress_outputs == {
        "synthetic-pymc-progress-bar.value",
        "synthetic-pymc-progress-label.children",
        "synthetic-pymc-progress-meta.children",
        "synthetic-chain-progress.children",
    }


def test_background_result_survives_restricted_macos_process_cleanup(
    monkeypatch,
    tmp_path,
) -> None:
    manager = OrcaDiskcacheManager(diskcache.Cache(str(tmp_path / "callback-cache")))

    def denied_cleanup(_manager, _job) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(
        "dash.DiskcacheManager.terminate_job",
        denied_cleanup,
    )

    assert manager.terminate_job(12345) is None


def test_background_manager_returns_a_completed_worker_result(tmp_path) -> None:
    manager = OrcaDiskcacheManager(diskcache.Cache(str(tmp_path / "callback-cache")))

    def finish_job(result_key, _progress_key, _args, _context) -> None:
        manager.handle.set(result_key, {"plots": "ready"})

    result_key = "synthetic-result"
    job = manager.call_job_fn(result_key, finish_job, [], {})
    deadline = time.monotonic() + 5
    while manager.job_running(job) and time.monotonic() < deadline:
        time.sleep(0.01)

    assert manager.get_result(result_key, job) == {"plots": "ready"}


def test_uploaded_count_workflows_fold_observation_time_at_one() -> None:
    for path, input_id in (
        ("/event-counts/donor-ignorant", "counts-observation-time"),
        ("/event-counts/donor-aware", "donor-observation-time"),
    ):
        page = next(page for page in PAGES if page.PATH == path)
        content = page.layout()
        controls = {
            component_id: component
            for component in _walk(content)
            if isinstance((component_id := getattr(component, "id", None)), str)
        }
        assert controls[input_id].value == 1.0
        assert any(
            input_id
            in {
                getattr(child, "id", None)
                for child in _walk(detail)
            }
            for detail in _walk(content)
            if detail.__class__.__name__ == "Details"
        )


def test_foundations_coin_uses_fixed_uniform_prior_and_known_extreme_truth() -> None:
    figure, values = bayes_101._coin_figure(1.0, 12, 0)
    metrics = dict(values)

    assert metrics["Tosses observed"] == "12 heads · 0 tails"
    assert metrics["Observed P(head)"] == "1.000"
    assert metrics["Posterior mean P(head)"] == "0.929"
    assert metrics["Posterior 95% HDI"] == "0.794–1.000"
    assert figure.data[0].name == "Uniform prior"


def test_foundations_page_has_linkable_sections_portrait_and_animated_samplers() -> None:
    content = bayes_101.layout()
    by_id = {
        component_id: component
        for component in _walk(content)
        if isinstance((component_id := getattr(component, "id", None)), str)
    }

    assert {"bayes-theorem", "coin-experiment", "computation", "bayes-factors", "thomas-bayes"} <= by_id.keys()
    assert by_id["mcmc-animation"].figure.frames
    assert by_id["smc-animation"].figure.frames
    images = [component for component in _walk(content) if component.__class__.__name__ == "Img"]
    assert any(image.src == "/assets/thomas_bayes.png" for image in images)


def test_dash_csv_upload_parser_enforces_the_utf8_csv_contract() -> None:
    payload = b"cell_id,count\ncell_001,1\ncell_002,0\n"
    contents = "data:text/csv;base64," + base64.b64encode(payload).decode("ascii")
    frame = read_uploaded_csv(contents)
    assert frame.to_dict("list") == {"cell_id": ["cell_001", "cell_002"], "count": [1, 0]}
