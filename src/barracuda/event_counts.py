"""Simulation and inference for donor ignorant and donor aware event counts."""

try:
    from ._core.condition_inference import (
        ConditionResults,
        build_condition_results_zip,
        run_condition_models,
    )
    from ._core.conditions import (
        APPLE_COLOUR_PRESETS,
        CONDITION_COLUMN,
        MAX_CONDITIONS,
        default_condition_colours,
        normalize_condition_frame,
        sanitize_condition_colours,
        split_condition_frame,
        validate_condition_frame,
    )
    from ._core.data import (
        COUNT_COLUMNS,
        DONOR_COLUMNS,
        sample_count_frame,
        sample_donor_frame,
        validate_count_frame,
        validate_donor_frame,
        validate_observation_time,
    )
    from ._core.inference import (
        MODEL_SPECS,
        InferenceResult,
        InferenceSettings,
        ModelSpec,
        build_results_zip,
        evidence_table,
        posterior_draw_table,
        run_count_models,
        run_donor_models,
        summary_table,
    )
    from ._core.simulation import (
        MODEL_LABELS,
        PAPER_RATE_DISTRIBUTIONS,
        paper_rate_distribution_for_model,
        rate_distribution_curve,
        simulate_event_counts,
    )
except ModuleNotFoundError as exc:
    if not exc.name or not exc.name.startswith("barracuda._core"):
        raise
    from webapp.core.condition_inference import (
        ConditionResults,
        build_condition_results_zip,
        run_condition_models,
    )
    from webapp.core.conditions import (
        APPLE_COLOUR_PRESETS,
        CONDITION_COLUMN,
        MAX_CONDITIONS,
        default_condition_colours,
        normalize_condition_frame,
        sanitize_condition_colours,
        split_condition_frame,
        validate_condition_frame,
    )
    from webapp.core.data import (
        COUNT_COLUMNS,
        DONOR_COLUMNS,
        sample_count_frame,
        sample_donor_frame,
        validate_count_frame,
        validate_donor_frame,
        validate_observation_time,
    )
    from webapp.core.inference import (
        MODEL_SPECS,
        InferenceResult,
        InferenceSettings,
        ModelSpec,
        build_results_zip,
        evidence_table,
        posterior_draw_table,
        run_count_models,
        run_donor_models,
        summary_table,
    )
    from webapp.core.simulation import (
        MODEL_LABELS,
        PAPER_RATE_DISTRIBUTIONS,
        paper_rate_distribution_for_model,
        rate_distribution_curve,
        simulate_event_counts,
    )

# Descriptive aliases for readers who use the manuscript terminology.
run_donor_ignorant_models = run_count_models
run_donor_aware_models = run_donor_models

__all__ = [
    "APPLE_COLOUR_PRESETS",
    "CONDITION_COLUMN",
    "COUNT_COLUMNS",
    "DONOR_COLUMNS",
    "MAX_CONDITIONS",
    "MODEL_LABELS",
    "MODEL_SPECS",
    "PAPER_RATE_DISTRIBUTIONS",
    "ConditionResults",
    "InferenceResult",
    "InferenceSettings",
    "ModelSpec",
    "build_condition_results_zip",
    "build_results_zip",
    "default_condition_colours",
    "evidence_table",
    "normalize_condition_frame",
    "paper_rate_distribution_for_model",
    "posterior_draw_table",
    "rate_distribution_curve",
    "run_condition_models",
    "run_count_models",
    "run_donor_aware_models",
    "run_donor_ignorant_models",
    "run_donor_models",
    "sample_count_frame",
    "sample_donor_frame",
    "sanitize_condition_colours",
    "simulate_event_counts",
    "split_condition_frame",
    "summary_table",
    "validate_condition_frame",
    "validate_count_frame",
    "validate_donor_frame",
    "validate_observation_time",
]
