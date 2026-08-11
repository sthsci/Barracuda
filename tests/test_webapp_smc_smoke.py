from __future__ import annotations

import os

import pytest

from webapp.core.data import sample_count_frame, sample_donor_frame
from webapp.core.inference import (
    InferenceSettings,
    run_count_models,
    run_donor_models,
)


@pytest.mark.smc
@pytest.mark.skipif(
    os.environ.get("ORCA_RUN_SMC_SMOKE") != "1",
    reason="set ORCA_RUN_SMC_SMOKE=1 to run the optional PyMC SMC smoke test",
)
def test_real_homogeneous_smc_smoke() -> None:
    """Opt-in integration helper; deliberately excluded from routine tests."""

    results = run_count_models(
        sample_count_frame().head(6),
        1.0,
        InferenceSettings(draws=16, chains=1, cores=1, seed=2026),
        model_keys=["homo"],
    )
    assert "homo" in results
    assert results["homo"].log_evidence == pytest.approx(
        results["homo"].log_evidence
    )


@pytest.mark.smc
@pytest.mark.skipif(
    os.environ.get("ORCA_RUN_SMC_SMOKE") != "1",
    reason="set ORCA_RUN_SMC_SMOKE=1 to run the optional PyMC SMC smoke test",
)
def test_real_donor_aware_homogeneous_smc_smoke() -> None:
    """Exercise the donor-relative backend used by the Dash application."""

    results = run_donor_models(
        sample_donor_frame(),
        1.0,
        InferenceSettings(
            draws=16,
            chains=1,
            cores=1,
            seed=2026,
            lambda_prior_bounds=(-1.5, 1.5),
        ),
        model_keys=["homo"],
    )
    assert "homo" in results
    assert results["homo"].donor_labels == (
        "donor_A",
        "donor_B",
        "donor_C",
    )
    assert results["homo"].log_evidence == pytest.approx(
        results["homo"].log_evidence
    )
