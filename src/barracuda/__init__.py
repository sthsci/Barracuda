"""Public Python interface for the BARRACUDA Bayesian modelling framework."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("barracuda")
except PackageNotFoundError:  # Source checkout without installed metadata.
    __version__ = "0.1.0"

from .event_counts import (  # noqa: E402
    MODEL_SPECS,
    InferenceResult,
    InferenceSettings,
    build_condition_results_zip,
    build_results_zip,
    evidence_table,
    normalize_condition_frame,
    run_condition_models,
    run_count_models,
    run_donor_aware_models,
    run_donor_ignorant_models,
    run_donor_models,
    sample_count_frame,
    sample_donor_frame,
    simulate_event_counts,
    summary_table,
    validate_condition_frame,
    validate_count_frame,
    validate_donor_frame,
)
from .trajectories import (  # noqa: E402
    TRAJECTORY_MODEL_SPECS,
    TrajectoryResult,
    TrajectorySettings,
    TrajectorySimulationSpec,
    build_trajectory_archive,
    expanded_trajectory_frame,
    normalize_trajectory_frame,
    run_trajectory_conditions,
    simulate_trajectory_frame,
    trajectory_evidence_frame,
    trajectory_posterior_draws,
    trajectory_summary_frame,
    validate_trajectory_frame,
)

__all__ = [
    "MODEL_SPECS",
    "TRAJECTORY_MODEL_SPECS",
    "InferenceResult",
    "InferenceSettings",
    "TrajectoryResult",
    "TrajectorySettings",
    "TrajectorySimulationSpec",
    "__version__",
    "build_condition_results_zip",
    "build_results_zip",
    "build_trajectory_archive",
    "evidence_table",
    "expanded_trajectory_frame",
    "normalize_condition_frame",
    "normalize_trajectory_frame",
    "run_condition_models",
    "run_count_models",
    "run_donor_aware_models",
    "run_donor_ignorant_models",
    "run_donor_models",
    "run_trajectory_conditions",
    "sample_count_frame",
    "sample_donor_frame",
    "simulate_event_counts",
    "simulate_trajectory_frame",
    "summary_table",
    "trajectory_evidence_frame",
    "trajectory_posterior_draws",
    "trajectory_summary_frame",
    "validate_condition_frame",
    "validate_count_frame",
    "validate_donor_frame",
    "validate_trajectory_frame",
]
