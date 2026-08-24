"""Strict orchestration boundary for an installed :mod:`barracuda` wheel.

This module deliberately contains no model, likelihood or simulation code. It
validates the job contract, asks the wheel to normalize the uploaded bytes
again, invokes the selected public adapter, and verifies the returned artifact
manifest before an outer worker uploads anything.

The expected wheel API is ``barracuda.worker_api.get_adapter(analysis_type)``.
The returned adapter exposes::

    normalize_csv(payload: bytes) -> {"data": object, "summary": Mapping}
    run(data, *, configuration, progress, output_dir) -> Mapping

``run`` returns a JSON-safe ``summary`` and an optional ``artifacts`` sequence
containing ``role``, relative ``path`` and ``media_type``. The orchestrator,
not the scientific wheel, owns object-store credentials and persistence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module, metadata
from io import BytesIO
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import Any, Protocol


SCHEMA_VERSION = 1
MAX_INPUT_BYTES = 5 * 1024 * 1024
MAX_OUTPUT_BYTES = 512 * 1024 * 1024
MAX_COMPACT_RESULT_BYTES = 256 * 1024
UINT32_MAX = 2**32 - 1

COUNT_MODELS = ("homo", "z2p", "dis2p", "hetero3")
TRAJECTORY_MODELS = (
    "homogeneous_history_independent",
    "homogeneous_history_dependent",
    "heterogeneous_history_independent",
    "heterogeneous_history_dependent",
)
ANALYSIS_TYPES = {
    "event_count_donor_ignorant": COUNT_MODELS,
    "event_count_donor_aware": COUNT_MODELS,
    "trajectory_donor_ignorant": TRAJECTORY_MODELS,
}

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "models",
    "observation_time",
    "hdi_probability",
    "sampler",
    "priors",
}
_SERVICE_FIELDS = {
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
_SAMPLER_FIELDS = {
    "particles",
    "chains",
    "cores",
    "seed",
    "threshold",
    "correlation_threshold",
}
_COUNT_PRIOR_FIELDS = {
    "lambda_prior_bounds",
    "p_prior_bounds",
    "std_prior_factor",
    "donor_deviation_prior",
}
_TRAJECTORY_PRIOR_FIELDS = {
    "lambda_prior_bounds",
    "sigma_lambda_prior",
    "p0_prior",
    "sigma_eta_prior",
    "beta_prior_sd",
    "n_quad",
}
_ROLE_PATTERN = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")
_MEDIA_TYPE_PATTERN = re.compile(
    r"[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*\Z"
)


class ContractError(ValueError):
    """Raised when a request or wheel response violates the worker contract."""


class RuntimeAdapter(Protocol):
    def normalize_csv(self, payload: bytes) -> Mapping[str, Any]: ...

    def run(
        self,
        data: object,
        *,
        configuration: Mapping[str, Any],
        progress: Callable[[Mapping[str, Any] | float], None],
        output_dir: Path,
    ) -> Mapping[str, Any]: ...


class RuntimeRegistry(Protocol):
    def get_adapter(self, analysis_type: str) -> RuntimeAdapter: ...


@dataclass(frozen=True)
class ValidatedRequest:
    analysis_type: str
    configuration: dict[str, Any]


def _public_package_version() -> str:
    """Return the installed public package version without importing PyMC."""

    try:
        return metadata.version("barracuda")
    except metadata.PackageNotFoundError:
        return "unknown"


def _safe_display_label(value: object) -> str:
    """Keep user-provided condition names safe for the progress transport."""

    text = " ".join(str(value).split())
    return text[:80] or "Condition"


def _frame_records(frame: object, *, maximum: int) -> tuple[list[dict[str, Any]], bool]:
    """Convert a pandas result table to bounded, ordinary JSON records.

    Pandas' JSON encoder intentionally converts NumPy scalars and timestamps to
    JSON primitives.  This avoids leaking live scientific objects into Celery
    or Redis while keeping the API payload small enough for a dashboard.
    """

    if maximum <= 0:
        raise ValueError("maximum must be positive")
    try:
        rows = int(len(frame))  # type: ignore[arg-type]
        limited = frame.head(maximum)  # type: ignore[union-attr]
        encoded = limited.to_json(orient="records")  # type: ignore[union-attr]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ContractError("barracuda returned an invalid result table") from exc
    decoded = json.loads(encoded)
    if not isinstance(decoded, list) or not all(isinstance(row, dict) for row in decoded):
        raise ContractError("barracuda returned an invalid result table")
    return decoded, rows > maximum


def _write_table(output_dir: Path, relative: str, frame: object) -> None:
    path = output_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        text = frame.to_csv(index=False, lineterminator="\n")  # type: ignore[union-attr]
    except (AttributeError, TypeError, ValueError) as exc:
        raise ContractError("barracuda returned an invalid result table") from exc
    path.write_text(text, encoding="utf-8")


def _progress_fraction(
    condition_index: int,
    total_conditions: int,
    model_index: int,
    total_models: int,
    beta: float,
    *,
    chain_index: int = 0,
    total_chains: int = 1,
) -> float:
    """Derive display progress from native SMC beta without inventing stages."""

    if total_conditions <= 0 or total_models <= 0:
        return 0.0
    completed_units = (condition_index - 1) * total_models + (model_index - 1)
    within_model = (
        min(max(0, int(chain_index)), total_chains - 1)
        + min(1.0, max(0.0, beta))
    ) / max(1, total_chains)
    return min(
        1.0,
        max(0.0, (completed_units + within_model) / (total_conditions * total_models)),
    )


class PublicBarracudaRuntime:
    """Registry backed solely by the stable :mod:`barracuda` public modules.

    The platform owns request validation, compact result shaping and artifact
    checking.  The scientific package remains the sole owner of CSV
    normalisation, model construction, SMC and scientific archive generation.
    Optional module injection makes this boundary fast to test without PyMC.
    """

    def __init__(self, *, event_counts: object | None = None, trajectories: object | None = None) -> None:
        self._event_counts = event_counts
        self._trajectories = trajectories

    @property
    def event_counts(self) -> object:
        if self._event_counts is None:
            self._event_counts = import_module("barracuda.event_counts")
        return self._event_counts

    @property
    def trajectories(self) -> object:
        if self._trajectories is None:
            self._trajectories = import_module("barracuda.trajectories")
        return self._trajectories

    def get_adapter(self, analysis_type: str) -> "RuntimeAdapter":
        if analysis_type == "event_count_donor_ignorant":
            return _CountAdapter(self.event_counts, donor_aware=False)
        if analysis_type == "event_count_donor_aware":
            return _CountAdapter(self.event_counts, donor_aware=True)
        if analysis_type == "trajectory_donor_ignorant":
            return _TrajectoryAdapter(self.trajectories)
        raise KeyError(analysis_type)


class _CountAdapter:
    """Public API adapter for the two event-count variants."""

    def __init__(self, module: object, *, donor_aware: bool) -> None:
        self.module = module
        self.donor_aware = donor_aware

    def normalize_csv(self, payload: bytes) -> Mapping[str, Any]:
        try:
            import pandas as pd

            raw = pd.read_csv(BytesIO(payload))
            normalized, message = self.module.normalize_condition_frame(  # type: ignore[union-attr]
                raw,
                donor_aware=self.donor_aware,
            )
            canonical = self.module.validate_condition_frame(  # type: ignore[union-attr]
                normalized,
                donor_aware=self.donor_aware,
            )
            labels = canonical["condition"].drop_duplicates().tolist()
        except Exception as exc:
            raise ContractError("the count CSV does not satisfy the barracuda schema") from exc
        return {
            "data": canonical,
            "summary": {
                "rows": int(len(canonical)),
                "conditions": int(len(labels)),
                "normalization_message": _safe_display_label(message),
                "donor_aware": self.donor_aware,
            },
        }

    def run(
        self,
        data: object,
        *,
        configuration: Mapping[str, Any],
        progress: Callable[[Mapping[str, Any] | float], None],
        output_dir: Path,
    ) -> Mapping[str, Any]:
        sampler = configuration["sampler"]
        priors = configuration["priors"]
        try:
            settings = self.module.InferenceSettings(  # type: ignore[union-attr]
                draws=sampler["particles"],
                chains=sampler["chains"],
                cores=sampler["cores"],
                seed=sampler["seed"],
                threshold=sampler["threshold"],
                correlation_threshold=sampler["correlation_threshold"],
                lambda_prior_bounds=tuple(priors["lambda_prior_bounds"]),
                p_prior_bounds=tuple(priors["p_prior_bounds"]),
                std_prior_factor=priors["std_prior_factor"],
                donor_deviation_prior=tuple(priors.get("donor_deviation_prior", (0.3, 0.3, 1.0))),
            )
            models = list(configuration["models"])

            def model_started(
                condition_index: int,
                total_conditions: int,
                condition_label: str,
                model_index: int,
                total_models: int,
                _model_label: str,
            ) -> None:
                key = models[model_index - 1] if model_index <= len(models) else "unknown"
                progress(
                    {
                        "phase": "sampling",
                        "fraction": _progress_fraction(condition_index, total_conditions, model_index, total_models, 0.0),
                        "condition": {"index": condition_index, "total": total_conditions, "label": _safe_display_label(condition_label)},
                        "model": {"index": model_index, "total": total_models, "key": key},
                    }
                )

            def native_progress(
                condition_index: int,
                total_conditions: int,
                condition_label: str,
                model_index: int,
                total_models: int,
                _model_label: str,
                chain: int,
                stage: int,
                beta: float,
            ) -> None:
                key = models[model_index - 1] if model_index <= len(models) else "unknown"
                fraction = _progress_fraction(condition_index, total_conditions, model_index, total_models, float(beta))
                progress(
                    {
                        "phase": "sampling",
                        "fraction": fraction,
                        "condition": {"index": condition_index, "total": total_conditions, "label": _safe_display_label(condition_label)},
                        "model": {"index": model_index, "total": total_models, "key": key},
                        "chain": {"index": int(chain), "total": int(sampler["chains"]), "stage": int(stage), "beta": float(beta)},
                    }
                )

            results = self.module.run_condition_models(  # type: ignore[union-attr]
                data,
                configuration["observation_time"],
                settings=settings,
                model_keys=models,
                donor_aware=self.donor_aware,
                progress_callback=model_started,
                sampler_progress_callback=native_progress,
            )
            evidence = _combine_condition_tables(results, self.module.evidence_table)  # type: ignore[union-attr]
            posterior_summary = _combine_condition_tables(results, self.module.summary_table)  # type: ignore[union-attr]
            posterior_draws = _combine_condition_tables(results, self.module.posterior_draw_table)  # type: ignore[union-attr]
            archive = self.module.build_condition_results_zip(  # type: ignore[union-attr]
                results,
                data,
                configuration["observation_time"],
                settings,
                donor_aware=self.donor_aware,
            )
        except ContractError:
            raise
        except Exception as exc:
            raise RuntimeError("barracuda event-count inference failed") from exc

        archive_path = output_dir / "results/barracuda-results.zip"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(bytes(archive))
        _write_table(output_dir, "results/model-evidence.csv", evidence)
        _write_table(output_dir, "results/posterior-summary.csv", posterior_summary)
        _write_table(output_dir, "results/posterior-draws.csv", posterior_draws)
        evidence_records, evidence_truncated = _frame_records(evidence, maximum=128)
        summary_records, summary_truncated = _frame_records(posterior_summary, maximum=512)
        # The complete draw table is downloadable as an artifact.  Keep the
        # database-bound preview small enough for a responsive web plot.
        draw_records, draws_truncated = _frame_records(posterior_draws, maximum=400)
        return {
            "summary": {
                "scientific_runtime": {"package": "barracuda", "version": _public_package_version()},
                "donor_aware": self.donor_aware,
                "evidence": evidence_records,
                "posterior_summary": summary_records,
                "posterior_draws": draw_records,
                "truncated": {
                    "evidence": evidence_truncated,
                    "posterior_summary": summary_truncated,
                    "posterior_draws": draws_truncated,
                },
            },
            "artifacts": _standard_artifacts(),
        }


class _TrajectoryAdapter:
    """Public API adapter for donor-ignorant contact trajectories."""

    def __init__(self, module: object) -> None:
        self.module = module

    def normalize_csv(self, payload: bytes) -> Mapping[str, Any]:
        try:
            raw = self.module.read_trajectory_csv(payload)  # type: ignore[union-attr]
            canonical = self.module.normalize_trajectory_frame(raw)  # type: ignore[union-attr]
            canonical = self.module.validate_trajectory_frame(canonical)  # type: ignore[union-attr]
            labels = canonical["condition"].drop_duplicates().tolist()
            events = sum(len(history) for history in canonical["history"])
        except Exception as exc:
            raise ContractError("the trajectory CSV does not satisfy the barracuda schema") from exc
        return {
            "data": canonical,
            "summary": {"rows": int(len(canonical)), "conditions": int(len(labels)), "events": int(events)},
        }

    def run(
        self,
        data: object,
        *,
        configuration: Mapping[str, Any],
        progress: Callable[[Mapping[str, Any] | float], None],
        output_dir: Path,
    ) -> Mapping[str, Any]:
        sampler = configuration["sampler"]
        priors = configuration["priors"]
        try:
            settings = self.module.TrajectorySettings(  # type: ignore[union-attr]
                draws=sampler["particles"], chains=sampler["chains"], cores=sampler["cores"], seed=sampler["seed"],
                threshold=sampler["threshold"], correlation_threshold=sampler["correlation_threshold"],
                lambda_prior_bounds=tuple(priors["lambda_prior_bounds"]), sigma_lambda_prior=priors["sigma_lambda_prior"],
                p0_prior=tuple(priors["p0_prior"]), sigma_eta_prior=priors["sigma_eta_prior"],
                beta_prior_sd=priors["beta_prior_sd"], n_quad=priors["n_quad"], prior_draws=0,
            )
            models = list(configuration["models"])

            def model_started(condition_index: int, total_conditions: int, condition_label: str, model_index: int, total_models: int, _model_label: str) -> None:
                key = models[model_index - 1] if model_index <= len(models) else "unknown"
                progress({"phase": "sampling", "fraction": _progress_fraction(condition_index, total_conditions, model_index, total_models, 0.0), "condition": {"index": condition_index, "total": total_conditions, "label": _safe_display_label(condition_label)}, "model": {"index": model_index, "total": total_models, "key": key}})

            def native_progress(condition_index: int, total_conditions: int, condition_label: str, model_index: int, total_models: int, _model_label: str, chain: int, stage: int, beta: float) -> None:
                key = models[model_index - 1] if model_index <= len(models) else "unknown"
                progress({"phase": "sampling", "fraction": _progress_fraction(condition_index, total_conditions, model_index, total_models, float(beta)), "condition": {"index": condition_index, "total": total_conditions, "label": _safe_display_label(condition_label)}, "model": {"index": model_index, "total": total_models, "key": key}, "chain": {"index": int(chain), "total": int(sampler["chains"]), "stage": int(stage), "beta": float(beta)}})

            results = self.module.run_trajectory_conditions(  # type: ignore[union-attr]
                data, configuration["observation_time"], settings=settings, model_keys=models,
                progress_callback=model_started, sampler_progress_callback=native_progress,
            )
            evidence = self.module.trajectory_evidence_frame(results)  # type: ignore[union-attr]
            posterior_summary = self.module.trajectory_summary_frame(results)  # type: ignore[union-attr]
            posterior_draws = self.module.trajectory_posterior_draws(results, max_draws=2_000, seed=sampler["seed"])  # type: ignore[union-attr]
            archive = self.module.build_trajectory_archive(  # type: ignore[union-attr]
                results, data, configuration["observation_time"], settings,
            )
        except ContractError:
            raise
        except Exception as exc:
            raise RuntimeError("barracuda trajectory inference failed") from exc

        archive_path = output_dir / "results/barracuda-results.zip"
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path.write_bytes(bytes(archive))
        _write_table(output_dir, "results/model-evidence.csv", evidence)
        _write_table(output_dir, "results/posterior-summary.csv", posterior_summary)
        _write_table(output_dir, "results/posterior-draws.csv", posterior_draws)
        evidence_records, evidence_truncated = _frame_records(evidence, maximum=128)
        summary_records, summary_truncated = _frame_records(posterior_summary, maximum=512)
        draw_records, draws_truncated = _frame_records(posterior_draws, maximum=400)
        return {
            "summary": {
                "scientific_runtime": {"package": "barracuda", "version": _public_package_version()},
                "evidence": evidence_records,
                "posterior_summary": summary_records,
                "posterior_draws": draw_records,
                "truncated": {"evidence": evidence_truncated, "posterior_summary": summary_truncated, "posterior_draws": draws_truncated},
            },
            "artifacts": _standard_artifacts(),
        }


def _combine_condition_tables(results: Mapping[str, Any], builder: Callable[[Any], object]) -> object:
    """Prefix each public count table with its condition without changing science."""

    try:
        import pandas as pd

        frames = []
        for condition, condition_results in results.items():
            table = builder(condition_results).copy()
            table.insert(0, "condition", _safe_display_label(condition))
            frames.append(table)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    except Exception as exc:
        raise ContractError("barracuda returned an invalid condition result table") from exc


def _standard_artifacts() -> list[dict[str, str]]:
    return [
        {"role": "results-archive", "path": "results/barracuda-results.zip", "media_type": "application/zip"},
        {"role": "model-evidence", "path": "results/model-evidence.csv", "media_type": "text/csv"},
        {"role": "posterior-summary", "path": "results/posterior-summary.csv", "media_type": "text/csv"},
        {"role": "posterior-draws", "path": "results/posterior-draws.csv", "media_type": "text/csv"},
    ]


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(map(str, mapping)) - allowed)
    if unknown:
        raise ContractError(f"{context} contains unknown fields: {', '.join(unknown)}")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ContractError(f"{name} keys must be strings")
    return value


def _integer(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ContractError(f"{name} must be between {minimum} and {maximum}")
    return value


def _finite(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ContractError(f"{name} must be a finite number")
    if minimum is not None:
        invalid = number <= minimum if strict_minimum else number < minimum
        if invalid:
            comparison = "greater than" if strict_minimum else "at least"
            raise ContractError(f"{name} must be {comparison} {minimum}")
    if maximum is not None and number > maximum:
        raise ContractError(f"{name} must be at most {maximum}")
    return number


def _number_pair(
    value: object,
    name: str,
    *,
    positive: bool = False,
    increasing: bool = False,
) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != 2
    ):
        raise ContractError(f"{name} must contain exactly two numbers")
    pair = [
        _finite(
            item,
            f"{name}[{index}]",
            minimum=0.0 if positive else None,
            strict_minimum=positive,
        )
        for index, item in enumerate(value)
    ]
    if increasing and pair[0] >= pair[1]:
        raise ContractError(f"{name} must be strictly increasing")
    return pair


def _positive_sequence(value: object, name: str, length: int) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != length
    ):
        raise ContractError(f"{name} must contain exactly {length} numbers")
    return [
        _finite(item, f"{name}[{index}]", minimum=0.0, strict_minimum=True)
        for index, item in enumerate(value)
    ]


def validate_configuration(
    analysis_type: str,
    configuration: Mapping[str, Any],
) -> ValidatedRequest:
    """Validate and fully materialize a tagged scientific request."""

    if analysis_type not in ANALYSIS_TYPES:
        raise ContractError(
            "analysis_type must be one of: " + ", ".join(ANALYSIS_TYPES)
        )
    config = _mapping(configuration, "configuration")
    # The HTTP API intentionally exposes a small, flat configuration object.
    # Keep the worker's persisted contract nested and explicit, while accepting
    # that public form at this boundary.  This is also useful for callers that
    # submit a canonical worker request directly.
    is_flat = bool(set(config) & _SERVICE_FIELDS) and not bool(
        set(config) & {"sampler", "priors", "schema_version"}
    )
    if is_flat:
        _reject_unknown(config, _SERVICE_FIELDS, "configuration")
        flat = config
        config = {
            "schema_version": SCHEMA_VERSION,
            "observation_time": flat.get("observation_time", 1.0),
            "hdi_probability": flat.get("hdi_probability", 0.95),
            "sampler": {
                key: flat[key]
                for key in (
                    "particles",
                    "chains",
                    "cores",
                    "seed",
                    "threshold",
                    "correlation_threshold",
                )
                if key in flat
            },
            "priors": (
                {"n_quad": flat["n_quad"]} if "n_quad" in flat else {}
            ),
        }
        if "models" in flat:
            config["models"] = flat["models"]
    else:
        _reject_unknown(config, _TOP_LEVEL_FIELDS, "configuration")
    version = config.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ContractError(f"unsupported schema_version {version!r}")

    allowed_models = ANALYSIS_TYPES[analysis_type]
    raw_models = config.get("models", list(allowed_models))
    if (
        not isinstance(raw_models, Sequence)
        or isinstance(raw_models, (str, bytes, bytearray))
        or not raw_models
    ):
        raise ContractError("models must be a non-empty list")
    models = [str(model) for model in raw_models]
    if len(models) != len(set(models)):
        raise ContractError("models must not contain duplicates")
    invalid_models = [model for model in models if model not in allowed_models]
    if invalid_models:
        raise ContractError(
            f"models are invalid for {analysis_type}: {', '.join(invalid_models)}"
        )

    observation_time = _finite(
        config.get("observation_time", 1.0),
        "observation_time",
        minimum=0.0,
        maximum=10_000.0,
        strict_minimum=True,
    )
    sampler = _mapping(config.get("sampler", {}), "sampler")
    _reject_unknown(sampler, _SAMPLER_FIELDS, "sampler")
    particles = _integer(
        sampler.get("particles", 64),
        "sampler.particles",
        minimum=32,
        maximum=1_000,
    )
    chains = _integer(
        sampler.get("chains", 1),
        "sampler.chains",
        minimum=1,
        maximum=4,
    )
    cores = _integer(
        sampler.get("cores", 1),
        "sampler.cores",
        minimum=1,
        maximum=4,
    )
    if cores > chains:
        raise ContractError("sampler.cores cannot exceed sampler.chains")
    raw_seed = sampler.get("seed")
    seed = (
        None
        if raw_seed is None
        else _integer(raw_seed, "sampler.seed", minimum=0, maximum=UINT32_MAX)
    )
    threshold = _finite(
        sampler.get("threshold", 0.5),
        "sampler.threshold",
        minimum=0.0,
        maximum=1.0,
        strict_minimum=True,
    )
    correlation = _finite(
        sampler.get("correlation_threshold", 0.01),
        "sampler.correlation_threshold",
        minimum=0.0,
        maximum=1.0,
    )

    priors = _mapping(config.get("priors", {}), "priors")
    if analysis_type.startswith("event_count"):
        _reject_unknown(priors, _COUNT_PRIOR_FIELDS, "priors")
        canonical_priors: dict[str, Any] = {
            "lambda_prior_bounds": _number_pair(
                priors.get("lambda_prior_bounds", [-1.5, 1.5]),
                "priors.lambda_prior_bounds",
                increasing=True,
            ),
            "p_prior_bounds": _number_pair(
                priors.get("p_prior_bounds", [1.0, 1.0]),
                "priors.p_prior_bounds",
                positive=True,
            ),
            "std_prior_factor": _finite(
                priors.get("std_prior_factor", 3.0),
                "priors.std_prior_factor",
                minimum=0.0,
                strict_minimum=True,
            ),
        }
        donor_prior = priors.get("donor_deviation_prior")
        if analysis_type == "event_count_donor_aware":
            canonical_priors["donor_deviation_prior"] = _positive_sequence(
                donor_prior if donor_prior is not None else [0.3, 0.3, 1.0],
                "priors.donor_deviation_prior",
                3,
            )
        elif donor_prior is not None:
            raise ContractError(
                "priors.donor_deviation_prior is only valid for donor-aware counts"
            )
    else:
        _reject_unknown(priors, _TRAJECTORY_PRIOR_FIELDS, "priors")
        canonical_priors = {
            "lambda_prior_bounds": _number_pair(
                priors.get("lambda_prior_bounds", [-1.0, 1.5]),
                "priors.lambda_prior_bounds",
                increasing=True,
            ),
            "sigma_lambda_prior": _finite(
                priors.get("sigma_lambda_prior", 2.0),
                "priors.sigma_lambda_prior",
                minimum=0.0,
                strict_minimum=True,
            ),
            "p0_prior": _number_pair(
                priors.get("p0_prior", [1.0, 1.0]),
                "priors.p0_prior",
                positive=True,
            ),
            "sigma_eta_prior": _finite(
                priors.get("sigma_eta_prior", 1.0),
                "priors.sigma_eta_prior",
                minimum=0.0,
                strict_minimum=True,
            ),
            "beta_prior_sd": _finite(
                priors.get("beta_prior_sd", 1.0),
                "priors.beta_prior_sd",
                minimum=0.0,
                strict_minimum=True,
            ),
            "n_quad": _integer(
                priors.get("n_quad", 20),
                "priors.n_quad",
                minimum=5,
                maximum=80,
            ),
        }

    return ValidatedRequest(
        analysis_type=analysis_type,
        configuration={
            "schema_version": SCHEMA_VERSION,
            "models": models,
            "observation_time": observation_time,
            "hdi_probability": _finite(
                config.get("hdi_probability", 0.95),
                "hdi_probability",
                minimum=0.5,
                maximum=0.999,
            ),
            "sampler": {
                "particles": particles,
                "chains": chains,
                "cores": cores,
                "seed": seed,
                "threshold": threshold,
                "correlation_threshold": correlation,
            },
            "priors": canonical_priors,
        },
    )


def _read_input(dataset_file: Any) -> bytes:
    if not hasattr(dataset_file, "read"):
        raise ContractError("dataset_file must be a readable binary stream")
    try:
        dataset_file.seek(0)
    except (AttributeError, OSError):
        pass
    payload = dataset_file.read(MAX_INPUT_BYTES + 1)
    if not isinstance(payload, (bytes, bytearray)):
        raise ContractError("dataset_file must return bytes")
    payload = bytes(payload)
    if not payload:
        raise ContractError("the input CSV is empty")
    if len(payload) > MAX_INPUT_BYTES:
        raise ContractError(f"the input CSV exceeds {MAX_INPUT_BYTES:,} bytes")
    return payload


def _json_safe(value: object, context: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{context} must be finite JSON data") from exc
    if len(encoded.encode("utf-8")) > MAX_COMPACT_RESULT_BYTES:
        raise ContractError(f"{context} exceeds the compact result limit")
    return json.loads(encoded)


def _load_runtime() -> RuntimeRegistry:
    try:
        return PublicBarracudaRuntime()
    except ImportError as exc:
        raise RuntimeError(
            "Install the pinned barracuda wheel exposing event_counts and trajectories"
        ) from exc


def _normalization_parts(value: object) -> tuple[object, dict[str, Any]]:
    result = _mapping(value, "barracuda normalization result")
    if set(result) != {"data", "summary"}:
        raise ContractError(
            "barracuda normalization result must contain exactly data and summary"
        )
    summary = _mapping(result["summary"], "normalization summary")
    return result["data"], _json_safe(summary, "normalization summary")


def _progress_forwarder(
    sink: Callable[[Mapping[str, Any]], None],
    *,
    allowed_models: Sequence[str],
) -> Callable[[Mapping[str, Any] | float], None]:
    sequence = 0

    def emit(value: Mapping[str, Any] | float) -> None:
        nonlocal sequence
        sequence += 1
        if isinstance(value, Mapping):
            payload = _validate_progress_event(value, allowed_models=allowed_models)
        else:
            payload = {
                "phase": "sampling",
                "fraction": _finite(
                    value,
                    "progress fraction",
                    minimum=0.0,
                    maximum=1.0,
                ),
            }
        sink({**payload, "schema_version": SCHEMA_VERSION, "sequence": sequence})

    return emit


def _progress_part(
    value: object,
    *,
    name: str,
    fields: set[str],
    index_minimum: int,
    include_label: bool = False,
    include_chain: bool = False,
) -> dict[str, Any]:
    item = _mapping(value, name)
    _reject_unknown(item, fields, name)
    index = _integer(item.get("index"), f"{name}.index", minimum=index_minimum, maximum=10_000)
    total = _integer(item.get("total"), f"{name}.total", minimum=1, maximum=10_000)
    if index >= total and index_minimum == 0:
        raise ContractError(f"{name}.index must be less than {name}.total")
    if index > total and index_minimum == 1:
        raise ContractError(f"{name}.index must not exceed {name}.total")
    normalized: dict[str, Any] = {"index": index, "total": total}
    if include_label:
        label = item.get("label")
        if not isinstance(label, str) or not label or len(label) > 80:
            raise ContractError(f"{name}.label is invalid")
        normalized["label"] = _safe_display_label(label)
    else:
        key = item.get("key")
        if not isinstance(key, str) or key not in _progress_allowed_models.get():
            raise ContractError(f"{name}.key is invalid")
        normalized["key"] = key
    if include_chain:
        normalized["stage"] = _integer(item.get("stage"), f"{name}.stage", minimum=0, maximum=100_000)
        normalized["beta"] = _finite(item.get("beta"), f"{name}.beta", minimum=0.0, maximum=1.0)
    return normalized


from contextvars import ContextVar

_progress_allowed_models: ContextVar[tuple[str, ...]] = ContextVar(
    "progress_allowed_models", default=()
)


def _validate_progress_event(
    value: Mapping[str, Any], *, allowed_models: Sequence[str]
) -> dict[str, Any]:
    token = _progress_allowed_models.set(tuple(allowed_models))
    try:
        return _validate_progress_event_for_models(value)
    finally:
        _progress_allowed_models.reset(token)


def _validate_progress_event_for_models(value: Mapping[str, Any]) -> dict[str, Any]:
    item = _mapping(value, "progress event")
    _reject_unknown(item, {"phase", "fraction", "condition", "model", "chain", "beta"}, "progress event")
    phase = item.get("phase")
    if phase not in {"sampling", "packaging"}:
        raise ContractError("progress event phase is invalid")
    normalized: dict[str, Any] = {"phase": phase}
    if "fraction" in item:
        normalized["fraction"] = _finite(item["fraction"], "progress event fraction", minimum=0.0, maximum=1.0)
    if "condition" in item:
        normalized["condition"] = _progress_part(
            item["condition"], name="progress event condition", fields={"index", "total", "label"}, index_minimum=1, include_label=True,
        )
    if "model" in item:
        normalized["model"] = _progress_part(
            item["model"], name="progress event model", fields={"index", "total", "key"}, index_minimum=1,
        )
    if "chain" in item:
        normalized["chain"] = _progress_part(
            item["chain"], name="progress event chain", fields={"index", "total", "stage", "beta"}, index_minimum=0, include_chain=True,
        )
    # Permit the compact beta-only progress shape emitted by older adapters.
    # New public adapters always include the richer ``chain`` object.
    if "beta" in item:
        normalized["beta"] = _finite(item["beta"], "progress event beta", minimum=0.0, maximum=1.0)
    return normalized


def _artifact_manifest(
    output_dir: Path,
    declarations: object,
) -> list[dict[str, Any]]:
    if declarations is None:
        return []
    if not isinstance(declarations, Sequence) or isinstance(
        declarations, (str, bytes, bytearray)
    ):
        raise ContractError("artifacts must be a list")
    root = output_dir.resolve()
    manifest: list[dict[str, Any]] = []
    total_bytes = 0
    seen_paths: set[str] = set()
    for index, raw in enumerate(declarations):
        item = _mapping(raw, f"artifacts[{index}]")
        _reject_unknown(item, {"role", "path", "media_type"}, f"artifacts[{index}]")
        role = item.get("role")
        relative = item.get("path")
        media_type = item.get("media_type")
        if not isinstance(role, str) or not _ROLE_PATTERN.fullmatch(role):
            raise ContractError(f"artifacts[{index}].role is invalid")
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise ContractError(f"artifacts[{index}].path is invalid")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ContractError(f"artifacts[{index}].path must stay below output_dir")
        normalized_relative = relative_path.as_posix()
        if normalized_relative in seen_paths:
            raise ContractError("artifact paths must be unique")
        seen_paths.add(normalized_relative)
        if not isinstance(media_type, str) or not _MEDIA_TYPE_PATTERN.fullmatch(
            media_type.lower()
        ):
            raise ContractError(f"artifacts[{index}].media_type is invalid")
        path = output_dir / relative_path
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"artifact does not name a regular file: {relative}")
        resolved = path.resolve()
        if root not in resolved.parents:
            raise ContractError(f"artifact resolves outside output_dir: {relative}")
        byte_size = resolved.stat().st_size
        total_bytes += byte_size
        if total_bytes > MAX_OUTPUT_BYTES:
            raise ContractError("declared artifacts exceed the total output limit")
        digest = sha256()
        with resolved.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        manifest.append(
            {
                "role": role,
                "path": normalized_relative,
                "media_type": media_type.lower(),
                "bytes": byte_size,
                "sha256": digest.hexdigest(),
            }
        )
    return manifest


def run_request(
    *,
    analysis_type: str,
    dataset_file: Any,
    configuration: Mapping[str, Any],
    progress: Callable[[Mapping[str, Any]], None],
    output_dir: str | Path,
    runtime: RuntimeRegistry | None = None,
) -> dict[str, Any]:
    """Run one validated request through the public wheel adapter.

    ``output_dir`` must be a worker-created directory dedicated to this
    attempt. Returned artifact paths remain relative to it; the caller must
    upload and then remove the directory.
    """

    validated = validate_configuration(analysis_type, configuration)
    payload = _read_input(dataset_file)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.is_symlink() or not destination.is_dir():
        raise ContractError("output_dir must be a real directory")

    registry = runtime or _load_runtime()
    try:
        adapter = registry.get_adapter(analysis_type)
    except (AttributeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"barracuda has no adapter for {analysis_type}") from exc
    normalized, normalization_summary = _normalization_parts(
        adapter.normalize_csv(payload)
    )
    result = _mapping(
        adapter.run(
            normalized,
            configuration=validated.configuration,
            progress=_progress_forwarder(
                progress,
                allowed_models=validated.configuration["models"],
            ),
            output_dir=destination,
        ),
        "barracuda run result",
    )
    _reject_unknown(result, {"summary", "artifacts"}, "barracuda run result")
    summary = _json_safe(
        _mapping(result.get("summary", {}), "run summary"),
        "run summary",
    )
    artifacts = _artifact_manifest(destination, result.get("artifacts", []))
    artifact_payloads: list[dict[str, Any]] = []
    for item in artifacts:
        payload_path = destination / item["path"]
        artifact_payloads.append(
            {
                "role": item["role"],
                "filename": Path(item["path"]).name,
                "content_type": item["media_type"],
                "payload": payload_path.read_bytes(),
                # Results are safe to share only through an explicitly-created
                # read-only project link; raw uploads remain private.
                "shareable": True,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_type": analysis_type,
        "input_sha256": sha256(payload).hexdigest(),
        "configuration": validated.configuration,
        "normalization": normalization_summary,
        "summary": summary,
        "artifacts": artifacts,
        # This private field is consumed by the Django execution boundary and
        # stripped before the compact result is stored in PostgreSQL.
        "_artifacts": artifact_payloads,
    }


def run_analysis(
    *,
    analysis_type: str,
    dataset_file: Any,
    configuration: Mapping[str, Any],
    progress: Callable[[Mapping[str, Any]], None],
) -> Mapping[str, Any]:
    """Django worker callable returning compact data and verified artifact bytes."""

    latest_fraction = 0.0

    def emit(event: Mapping[str, Any]) -> None:
        nonlocal latest_fraction
        fraction = event.get("fraction")
        if isinstance(fraction, (int, float)) and math.isfinite(float(fraction)):
            latest_fraction = max(latest_fraction, min(0.99, float(fraction)))
        # Preserve the public native SMC detail (condition, model, chain,
        # stage and beta) while preventing a later callback from moving the
        # dashboard backwards.
        progress({**dict(event), "fraction": latest_fraction})

    with TemporaryDirectory(prefix="barracuda-staged-") as temporary:
        manifest = run_request(
            analysis_type=analysis_type,
            dataset_file=dataset_file,
            configuration=configuration,
            progress=emit,
            output_dir=temporary,
        )
    output = {
        "schema_version": manifest["schema_version"],
        "analysis_type": manifest["analysis_type"],
        "input_sha256": manifest["input_sha256"],
        "configuration": manifest["configuration"],
        "normalization": manifest["normalization"],
        "summary": manifest["summary"],
        "artifact_count": len(manifest["artifacts"]),
        "artifact_roles": [item["role"] for item in manifest["artifacts"]],
        "_artifacts": manifest["_artifacts"],
    }
    return output


__all__ = [
    "ANALYSIS_TYPES",
    "ContractError",
    "SCHEMA_VERSION",
    "ValidatedRequest",
    "run_analysis",
    "run_request",
    "validate_configuration",
]
