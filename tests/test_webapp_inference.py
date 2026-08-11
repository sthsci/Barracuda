from __future__ import annotations

from io import BytesIO
import json
import math
from types import SimpleNamespace
from zipfile import ZipFile

import arviz as az
import numpy as np
import pandas as pd
import pytest

from webapp.core.data import sample_count_frame, sample_donor_frame
from webapp.core import inference as facade
from webapp.core.inference import (
    InferenceResult,
    InferenceSettings,
    build_results_zip,
    evidence_table,
    run_count_models,
    run_donor_models,
    summary_table,
)


def _fake_idata(evidence: float = -10.0):
    idata = az.from_dict(
        posterior={
            "lambda": np.array([[1.0, 1.2, 1.1, 0.9]]),
            "p_zero": np.array([[0.1, 0.2, 0.15, 0.12]]),
            "mu_lambda": np.array([[1.0, 1.2, 1.1, 0.9]]),
            "sigma_lambda": np.array([[0.5, 0.6, 0.4, 0.5]]),
            "mu_lambda_population": np.array([[1.0, 1.2, 1.1, 0.9]]),
            "sigma_lambda_population": np.array([[0.5, 0.6, 0.4, 0.5]]),
            "phi_0_population": np.array([[0.1, 0.2, 0.15, 0.12]]),
            "mu_lambda_donor": np.array(
                [
                    [
                        [0.8, 1.0, 1.2],
                        [0.9, 1.1, 1.3],
                        [0.7, 1.2, 1.1],
                        [0.8, 1.0, 1.4],
                    ]
                ]
            ),
            "sigma_lambda_donor": np.array(
                [
                    [
                        [0.3, 0.5, 0.7],
                        [0.4, 0.6, 0.8],
                        [0.2, 0.5, 0.6],
                        [0.3, 0.4, 0.7],
                    ]
                ]
            ),
            "phi_0_donor": np.array(
                [
                    [
                        [0.05, 0.10, 0.20],
                        [0.08, 0.12, 0.22],
                        [0.04, 0.11, 0.18],
                        [0.06, 0.09, 0.21],
                    ]
                ]
            ),
        },
        coords={"donor": [0, 1, 2]},
        dims={
            "mu_lambda_donor": ["donor"],
            "sigma_lambda_donor": ["donor"],
            "phi_0_donor": ["donor"],
        },
    )
    idata.attrs["fake_evidence"] = evidence
    return idata


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def _call(self, name: str, *args, **kwargs):
        evidence = -float(len(self.calls) + 1)
        self.calls.append((name, args, kwargs))
        return {"idata": _fake_idata(evidence), "model": f"fake-{name}"}

    def inference_homo(self, *args, **kwargs):
        return self._call("inference_homo", *args, **kwargs)

    def inference_Z2P(self, *args, **kwargs):
        return self._call("inference_Z2P", *args, **kwargs)

    def inference_Dis2P(self, *args, **kwargs):
        return self._call("inference_Dis2P", *args, **kwargs)

    def inference_hetero3(self, *args, **kwargs):
        return self._call("inference_hetero3", *args, **kwargs)

    @staticmethod
    def smc_log_evidence(idata) -> float:
        return float(idata.attrs["fake_evidence"])


@pytest.fixture
def fake_backend(monkeypatch) -> FakeBackend:
    backend = FakeBackend()
    monkeypatch.setattr(facade, "_load_backend", lambda donor_aware: backend)
    return backend


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"draws": 0}, "draws"),
        ({"chains": 0}, "chains"),
        ({"cores": 0}, "cores"),
        ({"seed": -1}, "seed"),
        ({"threshold": 0}, "threshold"),
        ({"lambda_prior_bounds": (1, 1)}, "strictly increasing"),
        ({"p_prior_bounds": (1, 0)}, "positive"),
        ({"std_prior_factor": 0}, "greater than zero"),
        ({"donor_deviation_prior": (0.2, 0.2)}, "three positive"),
    ],
)
def test_settings_reject_invalid_controls(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        InferenceSettings(**kwargs)


def test_count_models_run_sequentially_with_none_seed_and_progress(fake_backend) -> None:
    events: list[tuple[int, int, str]] = []
    settings = InferenceSettings(draws=12, chains=1, cores=1, seed=None)
    results = run_count_models(
        sample_count_frame(),
        2.5,
        settings,
        ["homo", "hetero3"],
        lambda index, total, label: events.append((index, total, label)),
    )

    assert list(results) == ["homo", "hetero3"]
    assert [call[0] for call in fake_backend.calls] == [
        "inference_homo",
        "inference_hetero3",
    ]
    assert fake_backend.calls[0][1][1] == 2.5
    assert fake_backend.calls[0][2]["random_seed"] is None
    assert fake_backend.calls[1][2]["p_prior_bounds"] == (1.0, 1.0)
    assert fake_backend.calls[1][2]["std_prior_factor"] == 3.0
    assert events == [
        (1, 2, "𝓜_homo · Homogeneous Poisson"),
        (2, 2, "𝓜_ZIΓ · Zero inflated heterogeneous Gamma Poisson"),
    ]
    assert all(result.model is None for result in results.values())


def test_model_specs_use_the_paper_symbols_names_and_donor_prior_defaults() -> None:
    assert {key: spec.notation for key, spec in facade.MODEL_SPECS.items()} == {
        "homo": "𝓜_homo",
        "z2p": "𝓜_ZI",
        "dis2p": "𝓜_Γ",
        "hetero3": "𝓜_ZIΓ",
    }
    assert facade.MODEL_SPECS["dis2p"].label.endswith(
        "Heterogeneous Gamma Poisson"
    )
    assert facade.MODEL_SPECS["hetero3"].label.endswith(
        "Zero inflated heterogeneous Gamma Poisson"
    )
    assert facade.InferenceSettings().donor_deviation_prior == (0.3, 0.3, 1.0)
    assert facade.InferenceSettings().lambda_prior_bounds == (-1.5, 1.5)
    assert facade.InferenceSettings().std_prior_factor == 3.0


def test_donor_wrapper_encodes_all_donors_and_uses_relative_prior_shapes(fake_backend) -> None:
    settings = InferenceSettings(
        draws=8,
        donor_deviation_prior=(0.11, 0.22, 0.33),
    )
    results = run_donor_models(
        sample_donor_frame(),
        1.0,
        settings,
        ["z2p", "dis2p", "hetero3"],
    )

    assert list(results) == ["z2p", "dis2p", "hetero3"]
    first_args = fake_backend.calls[0][1]
    assert np.array_equal(np.unique(first_args[1]), np.array([0, 1, 2]))
    assert fake_backend.calls[0][2]["donor_num"] == 3
    assert fake_backend.calls[0][2]["deviation_prior"] == (0.11, 0.33)
    assert fake_backend.calls[1][2]["deviation_prior"] == (0.11, 0.22)
    assert fake_backend.calls[2][2]["deviation_prior"] == (0.11, 0.22, 0.33)
    assert results["z2p"].donor_labels == ("donor_A", "donor_B", "donor_C")


def _result(
    key: str,
    evidence: float,
    *,
    donor_aware: bool = False,
) -> InferenceResult:
    labels = ("D1", "D2", "D3") if donor_aware else ()
    return InferenceResult(
        model_key=key,
        model_label=facade.MODEL_SPECS[key].label,
        donor_aware=donor_aware,
        idata=_fake_idata(evidence),
        model=None,
        log_evidence=evidence,
        elapsed_seconds=0.25,
        n_cells=12,
        observation_time=1.0,
        donor_labels=labels,
    )


def test_evidence_table_reports_log10_bayes_factors_against_best() -> None:
    results = {"homo": _result("homo", -12), "hetero3": _result("hetero3", -10)}
    table = evidence_table(results)

    assert table.loc[0, "model_key"] == "hetero3"
    assert bool(table.loc[0, "is_best"])
    expected = 2 / math.log(10)
    assert table.loc[1, "log10_BF_best_vs_model"] == pytest.approx(expected)
    assert table.loc[1, "log10_BF_model_vs_best"] == pytest.approx(-expected)


def test_summary_table_selects_scientific_parameters_and_labels_donors() -> None:
    count_summary = summary_table({"hetero3": _result("hetero3", -10)})
    assert set(count_summary["parameter"]) == {
        "mu_lambda",
        "sigma_lambda",
        "p_zero",
    }

    donor_summary = summary_table(
        {"hetero3": _result("hetero3", -10, donor_aware=True)}
    )
    assert "mu_lambda_donor[D1]" in set(donor_summary["parameter"])
    assert "phi_0_population" in set(donor_summary["parameter"])


def test_build_results_zip_contains_valid_compact_in_memory_reports() -> None:
    results = {"homo": _result("homo", -12), "hetero3": _result("hetero3", -10)}
    settings = InferenceSettings(draws=16, seed=None)
    payload = build_results_zip(
        results,
        sample_count_frame(),
        1.0,
        settings,
        truth={"model_key": "hetero3", "seed": None},
    )

    with ZipFile(BytesIO(payload)) as archive:
        assert set(archive.namelist()) == {
            "input_data.csv",
            "model_evidence.csv",
            "posterior_summary.csv",
            "posterior_samples.csv",
            "posterior_homo_smc.nc",
            "posterior_hetero3_smc.nc",
            "run_metadata.json",
            "ground_truth.json",
            "README.md",
        }
        evidence = pd.read_csv(archive.open("model_evidence.csv"))
        posterior_samples = pd.read_csv(archive.open("posterior_samples.csv"))
        metadata = json.load(archive.open("run_metadata.json"))
        truth = json.load(archive.open("ground_truth.json"))
        readme = archive.read("README.md").decode("utf-8")

    assert evidence.iloc[0]["model_key"] == "hetero3"
    assert metadata["settings"]["seed"] is None
    assert metadata["n_cells"] == 12
    assert truth == {"model_key": "hetero3", "seed": None}
    assert {"mu_lambda", "sigma_lambda", "p_zero"} <= set(posterior_samples)
    assert "az.from_netcdf" in readme


def test_posterior_draw_table_preserves_parameter_pairing() -> None:
    idata = az.from_dict(
        posterior={
            "mu_lambda": np.array([[1.0, 2.0, 3.0, 4.0]]),
            "sigma_lambda": np.array([[10.0, 20.0, 30.0, 40.0]]),
            "p_zero": np.array([[0.1, 0.2, 0.3, 0.4]]),
        }
    )
    result = InferenceResult(
        model_key="hetero3",
        model_label=facade.MODEL_SPECS["hetero3"].label,
        donor_aware=False,
        idata=idata,
        model=None,
        log_evidence=-1.0,
        elapsed_seconds=0.1,
        n_cells=4,
        observation_time=1.0,
    )

    draws = facade.posterior_draw_table(
        {"hetero3": result}, max_draws_per_model=2
    )

    assert draws[["mu_lambda", "sigma_lambda", "p_zero"]].to_dict("records") == [
        {"mu_lambda": 1.0, "sigma_lambda": 10.0, "p_zero": 0.1},
        {"mu_lambda": 4.0, "sigma_lambda": 40.0, "p_zero": 0.4},
    ]


def test_report_rejects_mixed_donor_modes() -> None:
    results = {
        "homo": _result("homo", -12),
        "hetero3": _result("hetero3", -10, donor_aware=True),
    }
    with pytest.raises(ValueError, match="cannot mix"):
        build_results_zip(
            results,
            sample_count_frame(),
            1.0,
            InferenceSettings(),
        )
