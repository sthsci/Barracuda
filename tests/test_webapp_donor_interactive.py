from __future__ import annotations

import arviz as az
import numpy as np

from webapp.donor_interactive import (
    contrast_from_payload,
    contrast_summary,
    donor_contrast_payload,
    donor_contrast_section,
)
from webapp.core.inference import InferenceResult


def _hetero_result(condition_shift: float) -> InferenceResult:
    mu = np.asarray(
        [[[1.0 + condition_shift, 2.0 + condition_shift], [2.0 + condition_shift, 4.0 + condition_shift]]]
    )
    sigma = np.asarray(
        [[[0.5 + condition_shift, 1.0 + condition_shift], [1.0 + condition_shift, 2.0 + condition_shift]]]
    )
    phi = np.asarray(
        [[[0.1 + 0.01 * condition_shift, 0.2 + 0.01 * condition_shift], [0.2 + 0.01 * condition_shift, 0.3 + 0.01 * condition_shift]]]
    )
    idata = az.from_dict(
        posterior={
            "mu_lambda_donor": mu,
            "sigma_lambda_donor": sigma,
            "phi_0_donor": phi,
            "mu_lambda_population": mu.mean(axis=2),
            "sigma_lambda_population": sigma.mean(axis=2),
            "phi_0_population": phi.mean(axis=2),
        },
        coords={"donor": [0, 1]},
        dims={
            "mu_lambda_donor": ["donor"],
            "sigma_lambda_donor": ["donor"],
            "phi_0_donor": ["donor"],
        },
    )
    return InferenceResult(
        model_key="hetero3",
        model_label="hetero3",
        donor_aware=True,
        idata=idata,
        model=None,
        log_evidence=-10.0,
        elapsed_seconds=0.1,
        n_cells=20,
        observation_time=1.0,
        donor_labels=("D1", "D2"),
    )


def _results():
    return {
        "Control": {"hetero3": _hetero_result(0.0)},
        "Treatment": {"hetero3": _hetero_result(1.0)},
    }


def test_donor_contrast_payload_preserves_all_model_parameters() -> None:
    payload = donor_contrast_payload(_results())
    stored = payload["models"]["hetero3"]["conditions"]["Control"]

    assert stored["parameters"] == [
        "mu_lambda_donor",
        "sigma_lambda_donor",
        "phi_0_donor",
    ]
    assert stored["donors"] == ["D1", "D2"]
    assert len(stored["draws"]["D1"]) == 2


def test_donor_contrast_uses_every_independent_pair_and_includes_phi() -> None:
    payload = donor_contrast_payload(_results())
    frame, metadata = contrast_from_payload(
        payload,
        model_key="hetero3",
        comparison="Treatment",
        reference="Control",
        scale="absolute",
    )

    assert set(frame.columns) == {
        "donor_id",
        "delta_mu_lambda",
        "delta_sigma_lambda",
        "delta_phi_0",
    }
    assert len(frame) == 8  # two donors × (two comparison × two reference draws)
    assert all(row["exact_cartesian"] for row in metadata["donors"])
    assert {row["possible_pairs"] for row in metadata["donors"]} == {4}

    summary = contrast_summary(frame)
    assert len(summary) == 6
    assert any("nonengaging fraction" in value for value in summary["Parameter"])


def test_donor_contrast_section_explains_particle_rule_and_has_controls() -> None:
    section = donor_contrast_section(_results(), prefix="donor")
    text = str(section)

    assert "every Cartesian particle pair" in text
    assert "difference between two posterior means" in text
    assert "donor-contrast-model" in text
    assert "donor-contrast-reference" in text
    assert "donor-contrast-comparison" in text


def test_donor_contrast_section_explains_when_two_conditions_are_required() -> None:
    section = donor_contrast_section(
        {"Control": {"hetero3": _hetero_result(0.0)}},
        prefix="donor",
    )
    text = str(section)

    assert "Two conditions are required" in text
    assert "donor-contrast-model" not in text
