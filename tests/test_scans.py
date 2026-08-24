from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from barracuda import scans, validation


def test_count_ground_truth_grid_is_one_at_a_time_and_deduplicates_baseline():
    grid = scans.plan_count_ground_truth_grid(
        sigma_lambda_values=[0.0, 3.0, 6.0],
        p_zero_values=[0.0, 0.2, 0.4],
    )

    assert len(grid) == 5
    assert grid["point_id"].is_unique
    assert set(grid["true_model"]) == {"z2p", "dis2p", "hetero3"}
    assert grid["true_model"].str.lower().eq(grid["true_model"]).all()
    baseline = grid.loc[grid["is_baseline"]].iloc[0]
    assert baseline["sweep_membership"] == "p_zero,sigma_lambda"
    assert baseline["reference_model"] == "hetero3"
    assert not ((grid["sigma_lambda"] == 6.0) & (grid["p_zero"] == 0.4)).any()


def test_count_scan_uses_nested_prefixes_and_explicit_bf_direction(monkeypatch):
    captured_prefixes = []
    progress = []

    def fake_fit(frame, observation_time, *, settings, model_keys):
        captured_prefixes.append(frame["cell_id"].tolist())
        n_cells = len(frame)
        return {
            "homo": SimpleNamespace(model_key="homo", log_evidence=float(n_cells)),
            "hetero3": SimpleNamespace(
                model_key="hetero3", log_evidence=float(n_cells - 2)
            ),
        }

    monkeypatch.setattr(scans, "fit_event_count_models", fake_fit)
    result = scans.run_count_bf_scan(
        [12, 10, 12],
        scenarios=[validation.COUNT_SCENARIOS[3]],
        model_keys=["homo", "hetero3"],
        progress_callback=lambda current, total, message: progress.append(
            (current, total, message)
        ),
    )

    assert captured_prefixes[0] == captured_prefixes[1][:10]
    assert list(result["n_cells"].drop_duplicates()) == [10, 12]
    assert set(result["model_key"]) == {"homo", "hetero3"}
    assert result["model_key"].str.lower().eq(result["model_key"]).all()
    hetero = result.loc[result["model_key"] == "hetero3"]
    assert np.allclose(hetero["log_bf_model_vs_true"], -2.0)
    assert np.allclose(hetero["log10_bf_model_vs_true"], -2.0 / np.log(10.0))
    assert set(result.loc[result["is_best"], "model_key"]) == {"homo"}
    assert [(item[0], item[1]) for item in progress] == [(1, 2), (2, 2)]


def test_count_scan_seeds_are_reproducible_and_replicates_are_independent(monkeypatch):
    def fake_fit(frame, observation_time, *, settings, model_keys):
        return {
            key: SimpleNamespace(model_key=key, log_evidence=-float(index))
            for index, key in enumerate(model_keys)
        }

    monkeypatch.setattr(scans, "fit_event_count_models", fake_fit)
    kwargs = dict(
        sample_sizes=[10],
        scenarios=[validation.COUNT_SCENARIOS[3]],
        replicates=2,
        model_keys=["homo"],
        base_seed=77,
    )
    first = scans.run_count_bf_scan(**kwargs)
    second = scans.run_count_bf_scan(**kwargs)

    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("replicate")["simulation_seed"].first().nunique() == 2
    assert first.groupby("replicate")["inference_seed"].first().nunique() == 2


def test_trajectory_scan_uses_nested_cell_prefixes_without_pymc(monkeypatch):
    captured_prefixes = []
    scenario = validation.TRAJECTORY_SCENARIOS[3]
    other_model = "heterogeneous_history_dependent"

    def fake_fit(frame, observation_time, *, settings, model_keys):
        captured_prefixes.append(frame["cell_id"].tolist())
        condition = frame["condition"].iloc[0]
        n_cells = len(frame)
        return {
            condition: {
                scenario.true_model: SimpleNamespace(
                    model_key=scenario.true_model,
                    log_evidence=float(n_cells),
                ),
                other_model: SimpleNamespace(
                    model_key=other_model,
                    log_evidence=float(n_cells + 1),
                ),
            }
        }

    monkeypatch.setattr(scans, "fit_trajectory_models", fake_fit)
    result = scans.run_trajectory_bf_scan(
        [3, 5],
        scenarios=[scenario],
        model_keys=[scenario.true_model, other_model],
    )

    assert captured_prefixes[0] == captured_prefixes[1][:3]
    true_rows = result.loc[result["model_key"] == scenario.true_model]
    assert np.allclose(true_rows["log_bf_model_vs_true"], 0.0)
    assert np.allclose(true_rows["log_bf_model_vs_best"], -1.0)
    assert set(result["best_model"]) == {other_model}
    assert set(result["workflow"]) == {"trajectory"}
    assert result["mu_eta"].iloc[0] == pytest.approx(np.log(0.25 / 0.75))


def test_scan_schema_rejects_reversed_bf_and_uppercase_model_keys(monkeypatch):
    def fake_fit(frame, observation_time, *, settings, model_keys):
        return {
            "homo": SimpleNamespace(model_key="homo", log_evidence=2.0),
            "hetero3": SimpleNamespace(model_key="hetero3", log_evidence=1.0),
        }

    monkeypatch.setattr(scans, "fit_event_count_models", fake_fit)
    valid = scans.run_count_bf_scan(
        [10],
        scenarios=[validation.COUNT_SCENARIOS[3]],
        model_keys=["homo", "hetero3"],
    )

    reversed_direction = valid.copy()
    reversed_direction["log_bf_model_vs_true"] *= -1
    with pytest.raises(ValueError, match="wrong direction"):
        scans.validate_bf_scan_schema(reversed_direction)

    uppercase = valid.copy()
    uppercase.loc[uppercase["model_key"] == "homo", "model_key"] = "HOMO"
    with pytest.raises(ValueError, match="lowercase canonical"):
        scans.validate_bf_scan_schema(uppercase)


def test_scan_schema_rejects_missing_model_and_inconsistent_metadata(monkeypatch):
    def fake_fit(frame, observation_time, *, settings, model_keys):
        return {
            "homo": SimpleNamespace(model_key="homo", log_evidence=2.0),
            "hetero3": SimpleNamespace(model_key="hetero3", log_evidence=1.0),
        }

    monkeypatch.setattr(scans, "fit_event_count_models", fake_fit)
    valid = scans.run_count_bf_scan(
        [10, 12],
        scenarios=[validation.COUNT_SCENARIOS[3]],
        model_keys=["homo", "hetero3"],
    )

    missing_model = valid.loc[
        ~((valid["n_cells"] == 12) & (valid["model_key"] == "hetero3"))
    ].copy()
    with pytest.raises(ValueError, match="candidate model coverage"):
        scans.validate_bf_scan_schema(missing_model)

    changed_truth = valid.copy()
    changed_truth.loc[changed_truth["n_cells"] == 12, "mu_lambda"] = 5.0
    with pytest.raises(ValueError, match="mu_lambda must be constant"):
        scans.validate_bf_scan_schema(changed_truth)

    changed_seed = valid.copy()
    changed_seed.loc[changed_seed["n_cells"] == 12, "simulation_seed"] += 1
    with pytest.raises(ValueError, match="simulation_seed must be constant"):
        scans.validate_bf_scan_schema(changed_seed)


def test_scan_summary_aggregates_replicates_and_selection_rates(monkeypatch):
    def fake_fit(frame, observation_time, *, settings, model_keys):
        # Stable per-prefix evidence; separate simulation seeds still establish
        # independent replicated datasets in the scan metadata.
        return {
            "homo": SimpleNamespace(model_key="homo", log_evidence=3.0),
            "hetero3": SimpleNamespace(model_key="hetero3", log_evidence=1.0),
        }

    monkeypatch.setattr(scans, "fit_event_count_models", fake_fit)
    scan = scans.run_count_bf_scan(
        [10, 12],
        scenarios=[validation.COUNT_SCENARIOS[3]],
        replicates=3,
        model_keys=["homo", "hetero3"],
    )
    summary = scans.summarize_bf_scan(scan, interval=(0.1, 0.9))

    assert set(summary["n_replicates"]) == {3}
    homo = summary.loc[summary["model_key"] == "homo"]
    hetero = summary.loc[summary["model_key"] == "hetero3"]
    assert np.allclose(homo["selection_rate"], 1.0)
    assert np.allclose(hetero["selection_rate"], 0.0)
    assert np.allclose(hetero["mean_log_bf_model_vs_true"], -2.0)
    assert set(summary["interval_lower_probability"]) == {0.1}
    assert set(summary["interval_upper_probability"]) == {0.9}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sigma_lambda_values": [-1.0]}, "non-negative"),
        ({"p_zero_values": [1.0]}, "0 <= value < 1"),
        ({"reference_model": "other"}, "unknown reference_model"),
    ],
)
def test_ground_truth_grid_rejects_noncanonical_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        scans.plan_count_ground_truth_grid(**kwargs)
