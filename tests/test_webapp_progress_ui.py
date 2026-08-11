from __future__ import annotations

import pytest
from dash.development.base_component import Component

from webapp.progress_ui import (
    pymc_progress,
    sampling_complete_payload,
    sampling_progress_payload,
)


def _walk(component):
    if isinstance(component, Component):
        yield component
        children = getattr(component, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                yield from _walk(child)
        elif children is not None:
            yield from _walk(children)


def test_shared_pymc_progress_has_stable_accessible_ids() -> None:
    component = pymc_progress("counts")
    by_id = {
        child.id: child
        for child in _walk(component)
        if getattr(child, "id", None) is not None
    }

    assert set(by_id) == {
        "counts-pymc-progress",
        "counts-pymc-progress-bar",
        "counts-pymc-progress-label",
        "counts-pymc-progress-meta",
        "counts-chain-progress",
    }
    assert by_id["counts-pymc-progress"].role == "region"
    assert by_id["counts-pymc-progress-bar"].max == 1
    assert by_id["counts-chain-progress"].role == "list"


def test_progress_uses_mean_beta_across_all_configured_chains() -> None:
    overall, label, meta, rows = sampling_progress_payload(
        condition_index=2,
        total_conditions=3,
        condition_label="Treatment",
        model_index=2,
        total_models=4,
        model_label="M_ZI",
        chains=2,
        particles=256,
        chain_states={0: (3, 1.0), 1: (2, 0.0)},
    )

    assert overall == pytest.approx((5 + 0.5) / 12)
    assert label == "Condition 2 of 3: Treatment · Model 2 of 4: M_ZI"
    assert "2 independent chains" in meta
    assert "256 particles per chain" in meta
    assert len(rows) == 2
    assert rows[0].children[0].children == "Chain 1"
    assert rows[0].children[2].children == "Stage 3 · β = 1.000"
    assert rows[1].children[0].children == "Chain 2"


def test_missing_chain_update_is_rendered_as_waiting_at_zero() -> None:
    overall, _label, _meta, rows = sampling_progress_payload(
        model_index=1,
        total_models=1,
        model_label="Model",
        chains=2,
        particles=64,
        chain_states={0: (4, 0.8)},
    )

    assert overall == pytest.approx(0.4)
    assert rows[1].children[2].children == "Stage 0 · β = 0.000"


def test_complete_payload_preserves_stage_and_finishes_all_chains() -> None:
    value, label, meta, rows = sampling_complete_payload(
        chains=2,
        particles=128,
        chain_states={0: (4, 0.8), 1: (5, 1.0)},
    )

    assert value == 1.0
    assert label == "PyMC sampling complete"
    assert "Preparing posterior and Bayes factor plots" in meta
    assert rows[0].children[2].children == "Stage 4 · β = 1.000"
    assert rows[1].children[2].children == "Stage 5 · β = 1.000"
