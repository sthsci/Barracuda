"""Strict, bounded configuration schemas for queued analysis jobs."""

from __future__ import annotations

import math
from typing import Any

from rest_framework import serializers

from .models import AnalysisJob


COMMON_KEYS = frozenset(
    {
        "models",
        "particles",
        "chains",
        "cores",
        "seed",
        "observation_time",
        "threshold",
        "correlation_threshold",
        "n_quad",
        "hdi_probability",
    }
)

MODEL_KEYS = {
    AnalysisJob.AnalysisType.EVENT_COUNT_DONOR_IGNORANT: frozenset(
        {"homo", "z2p", "dis2p", "hetero3"}
    ),
    AnalysisJob.AnalysisType.EVENT_COUNT_DONOR_AWARE: frozenset(
        {"homo", "z2p", "dis2p", "hetero3"}
    ),
    AnalysisJob.AnalysisType.TRAJECTORY_DONOR_IGNORANT: frozenset(
        {
            "homogeneous_history_independent",
            "homogeneous_history_dependent",
            "heterogeneous_history_independent",
            "heterogeneous_history_dependent",
        }
    ),
}


def _integer(value: Any, key: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise serializers.ValidationError({key: "Use an integer."})
    if not lower <= value <= upper:
        raise serializers.ValidationError({key: f"Choose a value from {lower} to {upper}."})
    return value


def _number(value: Any, key: str, lower: float, upper: float, *, lower_open=False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise serializers.ValidationError({key: "Use a number."})
    converted = float(value)
    if not math.isfinite(converted):
        raise serializers.ValidationError({key: "Use a finite number."})
    valid_lower = converted > lower if lower_open else converted >= lower
    if not valid_lower or converted > upper:
        comparison = "greater than" if lower_open else "at least"
        raise serializers.ValidationError(
            {key: f"Use a value {comparison} {lower} and at most {upper}."}
        )
    return converted


def validate_analysis_configuration(analysis_type: str, value: Any) -> dict[str, Any]:
    """Validate untrusted JSON before it can reach an analysis engine."""

    if analysis_type not in MODEL_KEYS:
        raise serializers.ValidationError({"analysis_type": "Unsupported analysis type."})
    if not isinstance(value, dict):
        raise serializers.ValidationError({"configuration": "Use a JSON object."})
    unknown = sorted(set(value) - COMMON_KEYS)
    if unknown:
        raise serializers.ValidationError(
            {"configuration": f"Unsupported setting(s): {', '.join(unknown)}."}
        )
    output = dict(value)
    if "models" in output:
        models = output["models"]
        if not isinstance(models, list) or not 1 <= len(models) <= 4:
            raise serializers.ValidationError({"models": "Choose one to four models."})
        if any(not isinstance(model, str) for model in models):
            raise serializers.ValidationError({"models": "Every model key must be text."})
        if len(models) != len(set(models)):
            raise serializers.ValidationError({"models": "Model keys must be unique."})
        unsupported = sorted(set(models) - MODEL_KEYS[analysis_type])
        if unsupported:
            raise serializers.ValidationError(
                {"models": f"Unsupported model(s) for {analysis_type}: {', '.join(unsupported)}."}
            )
    if "particles" in output:
        output["particles"] = _integer(output["particles"], "particles", 32, 2_000)
    if "chains" in output:
        output["chains"] = _integer(output["chains"], "chains", 1, 4)
    if "cores" in output:
        output["cores"] = _integer(output["cores"], "cores", 1, 4)
    if "chains" in output and "cores" in output and output["cores"] > output["chains"]:
        raise serializers.ValidationError({"cores": "Cores cannot exceed chains."})
    if "seed" in output and output["seed"] is not None:
        output["seed"] = _integer(output["seed"], "seed", 0, 2**32 - 1)
    if "observation_time" in output:
        output["observation_time"] = _number(
            output["observation_time"], "observation_time", 0.0, 100.0, lower_open=True
        )
    if "threshold" in output:
        output["threshold"] = _number(
            output["threshold"], "threshold", 0.0, 1.0, lower_open=True
        )
    if "correlation_threshold" in output:
        output["correlation_threshold"] = _number(
            output["correlation_threshold"], "correlation_threshold", 0.0, 1.0
        )
    if "n_quad" in output:
        output["n_quad"] = _integer(output["n_quad"], "n_quad", 5, 80)
    if "hdi_probability" in output:
        output["hdi_probability"] = _number(
            output["hdi_probability"], "hdi_probability", 0.5, 0.999
        )
    return output
