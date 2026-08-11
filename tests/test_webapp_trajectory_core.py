from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from zipfile import ZipFile

import arviz as az
import numpy as np
import pandas as pd
import pytest

from webapp.core import trajectory
from webapp.core.trajectory import (
    TrajectoryResult,
    TrajectorySettings,
    build_trajectory_archive,
    expanded_trajectory_frame,
    normalize_trajectory_frame,
    read_trajectory_csv,
    run_trajectory_conditions,
    simulate_trajectory_frame,
    trajectory_evidence_frame,
    trajectory_posterior_draws,
    truth_model_key,
)


def test_compact_csv_preserves_leading_zeroes_and_empty_histories() -> None:
    raw = read_trajectory_csv(
        "cell_id,condition,history\ncell_1,Control,001\ncell_2,Control,\n"
    )
    normalized = normalize_trajectory_frame(raw)

    assert normalized.to_dict("records") == [
        {"cell_id": "cell_1", "condition": "Control", "history": (0, 0, 1)},
        {"cell_id": "cell_2", "condition": "Control", "history": ()},
    ]


def test_paper_wide_table_and_event_level_table_normalize_identically() -> None:
    wide = pd.DataFrame(
        {
            "Cell": ["a", "b"],
            "condition": ["A", "A"],
            "1": [0, 1],
            "2": [1, ""],
        }
    )
    long = pd.DataFrame(
        {
            "cell_id": ["a", "a", "b"],
            "condition": ["A", "A", "A"],
            "contact_index": [2, 1, 1],
            "outcome": [1, 0, 1],
        }
    )

    pd.testing.assert_frame_equal(
        normalize_trajectory_frame(wide),
        normalize_trajectory_frame(long),
    )


def test_event_level_input_preserves_a_declared_zero_contact_cell() -> None:
    raw = pd.DataFrame(
        {
            "cell_id": ["a", "b"],
            "contact_index": [1, ""],
            "outcome": [0, ""],
        }
    )
    normalized = normalize_trajectory_frame(raw)

    assert normalized["history"].tolist() == [(0,), ()]


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            pd.DataFrame(
                {"cell_id": ["a", "a"], "contact_index": [1, 1], "outcome": [0, 1]}
            ),
            "duplicate contact indices",
        ),
        (
            pd.DataFrame(
                {"cell_id": ["a", "a"], "contact_index": [1, 3], "outcome": [0, 1]}
            ),
            "must be consecutive",
        ),
        (
            pd.DataFrame({"cell_id": ["a"], "history": ["012"]}),
            "must be 0 or 1",
        ),
        (
            pd.DataFrame({"cell_id": ["a", "a"], "history": ["0", "1"]}),
            "appear once",
        ),
    ],
)
def test_invalid_order_binary_values_and_duplicate_cells_are_rejected(
    raw: pd.DataFrame,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_trajectory_frame(raw)


def test_internal_gap_in_wide_history_is_rejected() -> None:
    raw = pd.DataFrame({"Cell": ["a"], "1": [0], "2": [""], "3": [1]})

    with pytest.raises(ValueError, match="after a blank"):
        normalize_trajectory_frame(raw)


def test_at_most_four_conditions_and_each_condition_needs_an_event() -> None:
    valid = pd.DataFrame(
        {
            "cell_id": [f"cell_{index}" for index in range(4)],
            "condition": list("ABCD"),
            "history": ["0", "1", "0", "1"],
        }
    )
    assert normalize_trajectory_frame(valid)["condition"].tolist() == list("ABCD")

    fifth = pd.concat(
        [
            valid,
            pd.DataFrame({"cell_id": ["cell_5"], "condition": ["E"], "history": ["0"]}),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="at most 4"):
        normalize_trajectory_frame(fifth)

    empty_group = pd.DataFrame(
        {
            "cell_id": ["a", "b"],
            "condition": ["A", "B"],
            "history": ["0", ""],
        }
    )
    with pytest.raises(ValueError, match="condition 'B' has no contact outcomes"):
        normalize_trajectory_frame(empty_group)


def test_expanded_contacts_use_public_history_names() -> None:
    expanded = expanded_trajectory_frame(
        pd.DataFrame({"cell_id": ["a"], "history": ["010"]})
    )

    assert expanded[
        ["previous_nonlethal_contacts", "previous_lethal_contacts", "outcome"]
    ].to_dict("records") == [
        {"previous_nonlethal_contacts": 0, "previous_lethal_contacts": 0, "outcome": 0},
        {"previous_nonlethal_contacts": 1, "previous_lethal_contacts": 0, "outcome": 1},
        {"previous_nonlethal_contacts": 1, "previous_lethal_contacts": 1, "outcome": 0},
    ]


@pytest.mark.parametrize(
    ("sigma_eta", "beta_f", "beta_s", "expected"),
    [
        (0.0, 0.0, 0.0, "homogeneous_history_independent"),
        (0.0, 0.4, 0.0, "homogeneous_history_dependent"),
        (0.5, 0.0, 0.0, "heterogeneous_history_independent"),
        (0.5, 0.4, -0.4, "heterogeneous_history_dependent"),
    ],
)
def test_ground_truth_model_classification(
    sigma_eta: float,
    beta_f: float,
    beta_s: float,
    expected: str,
) -> None:
    assert truth_model_key(sigma_eta, beta_f, beta_s) == expected


def test_synthetic_conditions_are_reproducible_and_return_public_truth() -> None:
    first, truths = simulate_trajectory_frame(n_cells=20, seed=42)
    second, second_truths = simulate_trajectory_frame(n_cells=20, seed=42)

    pd.testing.assert_frame_equal(first, second)
    assert truths == second_truths
    assert truths["Synthetic"]["beta_f"] == 0.8
    assert truths["Synthetic"]["beta_s"] == -0.8
    assert "beta_x" not in truths["Synthetic"]
    assert "beta_y" not in truths["Synthetic"]
    assert truths["Synthetic"]["true_model_key"] == "heterogeneous_history_dependent"


def test_synthetic_preflight_rejects_an_excessive_expected_event_count() -> None:
    with pytest.raises(ValueError, match="expected contacts"):
        simulate_trajectory_frame(
            n_cells=1_000,
            mu_lambda=50.0,
            observation_time=100.0,
        )


def test_web_smc_settings_apply_conservative_caps() -> None:
    with pytest.raises(ValueError, match="draws must be at most"):
        TrajectorySettings(draws=2_001)
    with pytest.raises(ValueError, match="chains must be at most"):
        TrajectorySettings(chains=5)


def _idata(offset: float = 0.0) -> az.InferenceData:
    base = np.arange(8, dtype=float).reshape(2, 4) + offset
    return az.from_dict(
        posterior={
            "mu_lambda": base,
            "sigma_lambda": base + 10,
            "mu_eta": base + 20,
            "sigma_eta": base + 30,
            "beta_x": base + 40,
            "beta_y": base + 50,
        }
    )


def _result(
    condition: str,
    model_key: str,
    evidence: float,
    *,
    offset: float = 0.0,
) -> TrajectoryResult:
    return TrajectoryResult(
        condition=condition,
        model_key=model_key,
        model_label=trajectory.TRAJECTORY_MODEL_SPECS[model_key].label,
        idata=_idata(offset),
        log_evidence=evidence,
        elapsed_seconds=0.5,
        n_cells=2,
        n_events=3,
        observation_time=1.0,
    )


def test_run_conditions_forwards_model_and_native_chain_progress(monkeypatch) -> None:
    calls: list[dict] = []
    model_events: list[tuple] = []
    sampler_events: list[tuple] = []

    class FakeBackend:
        ModelSpec = SimpleNamespace

        @staticmethod
        def prepare_data(group):
            return SimpleNamespace(n_cells=len(group), z=np.array([0, 1]))

        @staticmethod
        def build_model(data, spec, **kwargs):
            return (data, spec, kwargs)

        @staticmethod
        def sample_smc(model, **kwargs):
            calls.append(kwargs)
            return _idata()

        @staticmethod
        def log_evidence(_idata_value):
            return -2.0

    def fake_progress_runner(callback, operation):
        callback(0, 3, 0.75)
        return operation()

    monkeypatch.setattr(trajectory, "_load_trajectory_backend", lambda: FakeBackend)
    monkeypatch.setattr(trajectory, "_run_with_native_smc_progress", fake_progress_runner)
    frame = pd.DataFrame(
        {
            "cell_id": ["a", "b"],
            "condition": ["Control", "Treatment"],
            "history": ["0", "1"],
        }
    )
    results = run_trajectory_conditions(
        frame,
        settings=TrajectorySettings(draws=8, chains=1, cores=1, seed=10, n_quad=5),
        model_keys=["hom_hi"],
        progress_callback=lambda *event: model_events.append(event),
        sampler_progress_callback=lambda *event: sampler_events.append(event),
    )

    assert list(results) == ["Control", "Treatment"]
    assert model_events[0][:5] == (1, 2, "Control", 1, 1)
    assert sampler_events == [
        (1, 2, "Control", 1, 1, trajectory.TRAJECTORY_MODEL_SPECS["homogeneous_history_independent"].label, 0, 3, 0.75),
        (2, 2, "Treatment", 1, 1, trajectory.TRAJECTORY_MODEL_SPECS["homogeneous_history_independent"].label, 0, 3, 0.75),
    ]
    assert [call["random_seed"] for call in calls] == [10, 104_739]
    assert all(call["progressbar"] is True for call in calls)


def test_evidence_is_on_the_raw_log10_scale_and_marks_best() -> None:
    results = {
        "A": {
            "homogeneous_history_independent": _result(
                "A", "homogeneous_history_independent", 0.0
            ),
            "heterogeneous_history_dependent": _result(
                "A", "heterogeneous_history_dependent", -np.log(10.0) * 2.5
            ),
        }
    }
    evidence = trajectory_evidence_frame(results)

    assert evidence.loc[0, "is_best"]
    assert evidence.loc[1, "log10_BF_model_vs_best"] == pytest.approx(-2.5)
    assert evidence.loc[1, "log10_BF_best_vs_model"] == pytest.approx(2.5)


def test_posterior_draws_remain_paired_and_rename_beta_parameters() -> None:
    result = _result("A", "heterogeneous_history_dependent", 0.0)
    draws = trajectory_posterior_draws(
        {"A": {result.model_key: result}},
        model_keys=[result.model_key],
        max_draws=None,
    )

    assert draws["beta_f"].tolist() == (draws["mu_lambda"] + 40).tolist()
    assert draws["beta_s"].tolist() == (draws["mu_lambda"] + 50).tolist()
    assert "beta_x" not in draws and "beta_y" not in draws
    assert draws[["chain", "draw"]].to_records(index=False).tolist() == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
    ]


def test_archive_contains_reusable_tables_configuration_and_netcdf(monkeypatch) -> None:
    frame = pd.DataFrame(
        {"cell_id": ["a", "b"], "condition": ["A", "A"], "history": ["01", "1"]}
    )
    result = _result("A", "heterogeneous_history_dependent", -1.0)
    results = {"A": {result.model_key: result}}
    monkeypatch.setattr(trajectory, "_idata_to_netcdf_bytes", lambda _idata_value: b"netcdf")

    payload = build_trajectory_archive(
        results,
        frame,
        1.0,
        TrajectorySettings(draws=8, chains=1, cores=1, n_quad=5),
        truth={"A": {"beta_f": 0.8, "beta_s": -0.8}},
    )
    with ZipFile(BytesIO(payload)) as archive:
        names = set(archive.namelist())
        config = archive.read("configuration.json").decode()
        normalized = archive.read("normalized_trajectories.csv").decode()

    assert {
        "normalized_trajectories.csv",
        "expanded_contacts.csv",
        "model_evidence.csv",
        "posterior_summary.csv",
        "posterior_draws.csv",
        "configuration.json",
        "ground_truth.csv",
        "conditions/a/posterior_heterogeneous_history_dependent.nc",
        "README.txt",
    }.issubset(names)
    assert '"analysis": "donor_ignorant_contact_trajectory"' in config
    assert "a,A,01" in normalized
