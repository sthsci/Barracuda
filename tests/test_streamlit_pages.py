from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_every_navigation_page_loads_without_an_exception() -> None:
    app = AppTest.from_file("streamlit_app.py")
    expected_pages = [
        ("webapp/pages/bayes_101.py", "The update at the heart of Bayesian inference"),
        ("webapp/pages/synthetic_validation.py", "A. Choose the ground truth"),
        ("webapp/pages/event_counts.py", "A. Provide a small dataset"),
        ("webapp/pages/donor_aware.py", "A. Provide donor-labelled counts"),
        ("webapp/pages/trajectory.py", "What the trajectory model retains"),
    ]

    app.run(timeout=30)
    for index, (page_path, expected_header) in enumerate(expected_pages):
        if index:
            app.switch_page(page_path).run(timeout=30)
        assert not app.exception
        assert expected_header in [header.value for header in app.header]


def test_synthetic_page_generates_a_default_dataset() -> None:
    app = AppTest.from_file("streamlit_app.py").run(timeout=30)
    app.switch_page("webapp/pages/synthetic_validation.py").run(timeout=30)

    generate = next(
        button
        for button in app.button
        if button.label == "Generate synthetic data"
    )
    simulation_seed = next(
        field
        for field in app.text_input
        if field.label == "Simulation seed (optional)"
    )
    simulation_seed.input("2026")
    generate.click().run(timeout=30)

    assert not app.exception
    assert "B. Inspect the generated dataset" in [
        header.value for header in app.header
    ]
    assert len(app.dataframe) >= 2
