from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_every_navigation_page_loads_without_an_exception() -> None:
    app = AppTest.from_file("streamlit_app.py")
    expected_pages = [
        ("webapp/pages/home.py", "One question, three levels of information"),
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


def test_foundations_page_uses_a_fixed_uniform_coin_prior() -> None:
    app = AppTest.from_file("streamlit_app.py").run(timeout=30)
    app.switch_page("webapp/pages/bayes_101.py").run(timeout=30)

    slider_labels = [slider.label for slider in app.slider]
    assert "True probability of heads" in slider_labels
    assert "Number of tosses" in slider_labels
    assert "Prior α" not in slider_labels
    assert "Prior β" not in slider_labels
    assert any(button.label == "Toss again" for button in app.button)
    assert any("Uniform(0, 1)" in markdown.value for markdown in app.markdown)
    assert not app.exception

    probability_slider = next(
        slider for slider in app.slider if slider.label == "True probability of heads"
    )
    toss_slider = next(
        slider for slider in app.slider if slider.label == "Number of tosses"
    )
    probability_slider.set_value(1.0)
    toss_slider.set_value(12)
    app.run(timeout=30)

    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Tosses observed"] == "12 heads · 0 tails"
    assert metrics["Observed P(head)"] == "1.000"
    assert metrics["Posterior mean P(head)"] == "0.929"
    assert metrics["Posterior 95% interval"] == "0.753–0.998"
    assert not app.exception
