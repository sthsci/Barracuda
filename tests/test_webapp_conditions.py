from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import pytest

from webapp.condition_reporting import condition_bayes_factor_figure
from webapp.core import condition_inference
from webapp.core.condition_inference import run_condition_models
from webapp.core.conditions import (
    APPLE_COLOUR_PRESETS,
    normalize_condition_frame,
    sample_condition_frame,
    sanitize_condition_colours,
    split_condition_frame,
    validate_condition_frame,
)
from webapp.core.inference import InferenceResult, InferenceSettings
from webapp.reporting import BF3_LOG10


def test_legacy_count_table_becomes_one_condition() -> None:
    raw = pd.DataFrame(
        {
            "cell_id": [f"cell_{index}" for index in range(5)],
            "count": [0, 1, 2, 1, 3],
        }
    )
    mapped, message = normalize_condition_frame(raw, donor_aware=False)
    valid = validate_condition_frame(mapped, donor_aware=False)

    assert list(valid.columns) == ["cell_id", "condition", "count"]
    assert valid["condition"].unique().tolist() == ["Group 1"]
    assert "Group 1" in message


def test_condition_table_supports_four_groups_but_not_five() -> None:
    frames = []
    for condition in ("A", "B", "C", "D"):
        frame = pd.DataFrame(
            {
                "cell_id": [f"{condition}_{index}" for index in range(5)],
                "condition": condition,
                "count": [0, 1, 2, 1, 3],
            }
        )
        frames.append(frame)
    valid = validate_condition_frame(pd.concat(frames), donor_aware=False)
    assert list(split_condition_frame(valid, donor_aware=False)) == ["A", "B", "C", "D"]

    fifth = frames[0].copy()
    fifth["cell_id"] = "E_" + fifth["cell_id"]
    fifth["condition"] = "E"
    with pytest.raises(ValueError, match="at most 4"):
        validate_condition_frame(pd.concat([*frames, fifth]), donor_aware=False)


def test_donor_condition_example_is_valid_in_every_group() -> None:
    frame = sample_condition_frame(donor_aware=True)
    groups = split_condition_frame(frame, donor_aware=True)

    assert list(groups) == ["Control", "Treatment"]
    assert all(group["donor_id"].nunique() == 3 for group in groups.values())
    assert all(len(group) == 12 for group in groups.values())


def test_condition_colours_have_presets_and_validate_custom_hex() -> None:
    labels = ["Control", "Treatment"]
    colours = sanitize_condition_colours(
        labels,
        {"Control": "#123abc", "Treatment": "not-a-colour"},
    )

    assert len(APPLE_COLOUR_PRESETS) >= 8
    assert colours["Control"] == "#123ABC"
    assert colours["Treatment"].startswith("#")


def test_condition_sampler_progress_keeps_condition_model_and_pymc_chain(
    monkeypatch,
) -> None:
    model_events: list[tuple] = []
    sampler_events: list[tuple] = []

    def fake_runner(
        _frame,
        _observation_time,
        *,
        settings,
        model_keys,
        progress_callback,
        sampler_progress_callback,
    ):
        assert settings.chains == 2
        assert model_keys == ["homo", "z2p"]
        progress_callback(1, 2, "model one")
        sampler_progress_callback(1, 2, "model one", 1, 3, 0.75)
        return {}

    monkeypatch.setattr(condition_inference, "run_count_models", fake_runner)
    results = run_condition_models(
        sample_condition_frame(donor_aware=False),
        1.0,
        settings=InferenceSettings(draws=8, chains=2, cores=1),
        model_keys=["homo", "z2p"],
        donor_aware=False,
        progress_callback=lambda *event: model_events.append(event),
        sampler_progress_callback=lambda *event: sampler_events.append(event),
    )

    assert list(results) == ["Control", "Treatment"]
    assert model_events == [
        (1, 2, "Control", 1, 2, "model one"),
        (2, 2, "Treatment", 1, 2, "model one"),
    ]
    assert sampler_events == [
        (1, 2, "Control", 1, 2, "model one", 1, 3, 0.75),
        (2, 2, "Treatment", 1, 2, "model one", 1, 3, 0.75),
    ]


def _result(key: str, log_evidence: float) -> InferenceResult:
    if key == "homo":
        posterior = {"lambda": np.ones((1, 20))}
    else:
        posterior = {
            "mu_lambda": np.ones((1, 20)),
            "sigma_lambda": np.ones((1, 20)),
        }
    return InferenceResult(
        model_key=key,
        model_label=key,
        donor_aware=False,
        idata=az.from_dict(posterior=posterior),
        model=None,
        log_evidence=log_evidence,
        elapsed_seconds=0.1,
        n_cells=20,
        observation_time=1.0,
    )


def test_multi_condition_bayes_factor_uses_true_threshold_positions() -> None:
    results = {
        "Control": {
            "homo": _result("homo", 0.0),
            "dis2p": _result("dis2p", -np.log(10.0) * 2.5),
        },
        "Treatment": {
            "homo": _result("homo", -np.log(10.0)),
            "dis2p": _result("dis2p", 0.0),
        },
    }
    figure = condition_bayes_factor_figure(results)
    rectangles = [shape for shape in figure.layout.shapes if shape.type == "rect"]

    assert [float(shape.x0) for shape in rectangles[:4]] == pytest.approx(
        [0.0, BF3_LOG10, 1.0, 2.0]
    )
    bar = next(trace for trace in figure.data if trace.type == "bar")
    assert sum("Best model" in label for label in bar.y) == 2
    assert max(map(float, bar.x)) == pytest.approx(2.5)
