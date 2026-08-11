"""UI neutral adapters around the paper's event count inference modules."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module, metadata
from io import BytesIO
import json
import math
from pathlib import Path
import platform
import re
from tempfile import TemporaryDirectory
import time
from types import ModuleType
from typing import Any, Final
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd

from .data import (
    validate_count_frame,
    validate_donor_frame,
    validate_observation_time,
)


ProgressCallback = Callable[[int, int, str], None]
SamplerProgressCallback = Callable[
    [int, int, str, int, int, float],
    None,
]


@dataclass(frozen=True)
class ModelSpec:
    """Display and backend metadata for one model in the comparison set."""

    key: str
    notation: str
    label: str
    short_label: str
    backend_function: str
    description: str
    count_parameters: tuple[str, ...]
    donor_parameters: tuple[str, ...]


MODEL_SPECS: Final[dict[str, ModelSpec]] = {
    "homo": ModelSpec(
        key="homo",
        notation="𝓜_homo",
        label="𝓜_homo · Homogeneous Poisson",
        short_label="𝓜_homo",
        backend_function="inference_homo",
        description="Every cell shares λ. Differences between counts arise from Poisson sampling variation.",
        count_parameters=("lambda",),
        donor_parameters=("mu_lambda_population", "mu_lambda_donor"),
    ),
    "z2p": ModelSpec(
        key="z2p",
        notation="𝓜_ZI",
        label="𝓜_ZI · Zero inflated Poisson",
        short_label="𝓜_ZI",
        backend_function="inference_Z2P",
        description="A fraction φ₀ is nonengaging. The remaining cells share one event rate λ.",
        count_parameters=("lambda", "p_zero"),
        donor_parameters=(
            "mu_lambda_population",
            "phi_0_population",
            "mu_lambda_donor",
            "phi_0_donor",
        ),
    ),
    "dis2p": ModelSpec(
        key="dis2p",
        notation="𝓜_Γ",
        label="𝓜_Γ · Heterogeneous Gamma Poisson",
        short_label="𝓜_Γ",
        backend_function="inference_Dis2P",
        description="Continuous cell-to-cell heterogeneity: event rates λᵢ among engaging cells follow Gamma(μλ, σλ), with φ₀ = 0.",
        count_parameters=("mu_lambda", "sigma_lambda"),
        donor_parameters=(
            "mu_lambda_population",
            "sigma_lambda_population",
            "mu_lambda_donor",
            "sigma_lambda_donor",
        ),
    ),
    "hetero3": ModelSpec(
        key="hetero3",
        notation="𝓜_ZIΓ",
        label="𝓜_ZIΓ · Zero inflated heterogeneous Gamma Poisson",
        short_label="𝓜_ZIΓ",
        backend_function="inference_hetero3",
        description="Continuous cell-to-cell heterogeneity plus a fraction φ₀ of nonengaging cells; positive rates follow Gamma(μλ, σλ).",
        count_parameters=("mu_lambda", "sigma_lambda", "p_zero"),
        donor_parameters=(
            "mu_lambda_population",
            "sigma_lambda_population",
            "phi_0_population",
            "mu_lambda_donor",
            "sigma_lambda_donor",
            "phi_0_donor",
        ),
    ),
}


def _positive_int(value: int, name: str, *, allow_none: bool = False) -> int | None:
    if allow_none and value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return converted


def _finite(value: float, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass(frozen=True)
class InferenceSettings:
    """Validated PyMC SMC controls and prior settings used by the demo."""

    draws: int = 256
    chains: int = 1
    cores: int | None = 1
    seed: int | None = None
    threshold: float = 0.5
    correlation_threshold: float = 0.01
    lambda_prior_bounds: tuple[float, float] = (-1.5, 1.5)
    p_prior_bounds: tuple[float, float] = (1.0, 1.0)
    std_prior_factor: float = 3.0
    donor_deviation_prior: tuple[float, float, float] = (0.3, 0.3, 1.0)

    def __post_init__(self) -> None:
        _positive_int(self.draws, "draws")
        _positive_int(self.chains, "chains")
        _positive_int(self.cores, "cores", allow_none=True)

        if self.seed is not None:
            if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
                self.seed,
                (int, np.integer),
            ):
                raise ValueError("seed must be an integer or None")
            if self.seed < 0 or self.seed > np.iinfo(np.uint32).max:
                raise ValueError("seed must be between 0 and 2**32 - 1")

        threshold = _finite(self.threshold, "threshold")
        correlation = _finite(
            self.correlation_threshold,
            "correlation_threshold",
        )
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be greater than zero and at most one")
        if not 0 <= correlation <= 1:
            raise ValueError("correlation_threshold must be between zero and one")

        if len(self.lambda_prior_bounds) != 2:
            raise ValueError("lambda_prior_bounds must contain lower and upper bounds")
        lower, upper = map(
            lambda value: _finite(value, "lambda_prior_bounds"),
            self.lambda_prior_bounds,
        )
        if lower >= upper:
            raise ValueError("lambda_prior_bounds must be strictly increasing")

        if len(self.p_prior_bounds) != 2 or any(
            _finite(value, "p_prior_bounds") <= 0
            for value in self.p_prior_bounds
        ):
            raise ValueError("p_prior_bounds must contain two positive values")
        if _finite(self.std_prior_factor, "std_prior_factor") <= 0:
            raise ValueError("std_prior_factor must be greater than zero")
        if len(self.donor_deviation_prior) != 3 or any(
            _finite(value, "donor_deviation_prior") <= 0
            for value in self.donor_deviation_prior
        ):
            raise ValueError(
                "donor_deviation_prior must contain three positive scales"
            )


@dataclass(frozen=True)
class InferenceResult:
    """One fitted model plus compact metadata needed by the web interface."""

    model_key: str
    model_label: str
    donor_aware: bool
    idata: Any
    model: Any
    log_evidence: float
    elapsed_seconds: float
    n_cells: int
    observation_time: float
    donor_labels: tuple[str, ...] = ()


_MODEL_ALIASES: Final[dict[str, str]] = {
    "homo": "homo",
    "homogeneous": "homo",
    "poisson": "homo",
    "z2p": "z2p",
    "zero_inflated": "z2p",
    "zero-inflated": "z2p",
    "dis2p": "dis2p",
    "distributed": "dis2p",
    "gamma_poisson": "dis2p",
    "hetero3": "hetero3",
    "full": "hetero3",
}


def _selected_specs(model_keys: Sequence[str] | None) -> list[ModelSpec]:
    if model_keys is None:
        return list(MODEL_SPECS.values())
    if isinstance(model_keys, str):
        model_keys = [model_keys]
    if not model_keys:
        raise ValueError("model_keys must contain at least one model")

    selected: list[ModelSpec] = []
    seen: set[str] = set()
    for requested in model_keys:
        normalized = str(requested).strip().lower().replace(" ", "_")
        try:
            key = _MODEL_ALIASES[normalized]
        except KeyError as exc:
            raise ValueError(
                f"unknown model key {requested!r}; choose from: "
                + ", ".join(MODEL_SPECS)
            ) from exc
        if key in seen:
            raise ValueError(f"model_keys contains the duplicate model {key!r}")
        seen.add(key)
        selected.append(MODEL_SPECS[key])
    return selected


def _load_backend(donor_aware: bool) -> ModuleType:
    module_name = (
        "bayesorca._backends.donor.inference_donor_relative"
        if donor_aware
        else "bayesorca._backends.event_counts.inference"
    )
    return import_module(module_name)


def _common_backend_kwargs(settings: InferenceSettings) -> dict[str, Any]:
    return {
        "draws": int(settings.draws),
        "chains": int(settings.chains),
        "cores": None if settings.cores is None else int(settings.cores),
        "lambda_prior_bounds": tuple(map(float, settings.lambda_prior_bounds)),
        "random_seed": None if settings.seed is None else int(settings.seed),
        "threshold": float(settings.threshold),
        "correlation_threshold": float(settings.correlation_threshold),
    }


def _model_backend_kwargs(
    spec: ModelSpec,
    settings: InferenceSettings,
    *,
    donor_aware: bool,
) -> dict[str, Any]:
    kwargs = _common_backend_kwargs(settings)
    if spec.key in {"z2p", "hetero3"}:
        kwargs["p_prior_bounds"] = tuple(map(float, settings.p_prior_bounds))
    if spec.key in {"dis2p", "hetero3"}:
        kwargs["std_prior_factor"] = float(settings.std_prior_factor)
    if donor_aware:
        mean_scale, sigma_scale, zero_scale = map(
            float,
            settings.donor_deviation_prior,
        )
        if spec.key == "homo":
            kwargs["deviation_prior"] = mean_scale
        elif spec.key == "z2p":
            kwargs["deviation_prior"] = (mean_scale, zero_scale)
        elif spec.key == "dis2p":
            kwargs["deviation_prior"] = (mean_scale, sigma_scale)
        else:
            kwargs["deviation_prior"] = (
                mean_scale,
                sigma_scale,
                zero_scale,
            )
    return kwargs


def _fit_models(
    *,
    frame: pd.DataFrame,
    observation_time: float,
    settings: InferenceSettings,
    specs: Sequence[ModelSpec],
    donor_aware: bool,
    progress_callback: ProgressCallback | None,
    sampler_progress_callback: SamplerProgressCallback | None,
) -> dict[str, InferenceResult]:
    backend = _load_backend(donor_aware)
    counts = frame["count"].to_numpy(dtype=np.int64)
    donor_index: np.ndarray | None = None
    donor_labels: tuple[str, ...] = ()
    if donor_aware:
        donor_index, uniques = pd.factorize(frame["donor_id"], sort=True)
        donor_index = donor_index.astype(np.int64)
        donor_labels = tuple(map(str, uniques.tolist()))

    results: dict[str, InferenceResult] = {}
    total = len(specs)
    for index, spec in enumerate(specs, start=1):
        if progress_callback is not None:
            progress_callback(index, total, spec.label)
        inference_function = getattr(backend, spec.backend_function)
        kwargs = _model_backend_kwargs(
            spec,
            settings,
            donor_aware=donor_aware,
        )
        if sampler_progress_callback is not None:
            kwargs["progress_callback"] = (
                lambda chain, stage, beta, *, _index=index, _total=total, _label=spec.label: sampler_progress_callback(
                    _index,
                    _total,
                    _label,
                    int(chain),
                    int(stage),
                    float(beta),
                )
            )
        positional: tuple[Any, ...]
        if donor_aware:
            kwargs["donor_num"] = len(donor_labels)
            positional = (counts, donor_index, observation_time)
        else:
            positional = (counts, observation_time)

        started = time.perf_counter()
        backend_result = inference_function(*positional, **kwargs)
        elapsed = time.perf_counter() - started
        if not isinstance(backend_result, Mapping) or "idata" not in backend_result:
            raise RuntimeError(
                f"{spec.backend_function} returned no 'idata' inference result"
            )
        idata = backend_result["idata"]
        log_evidence = float(backend.smc_log_evidence(idata))
        if not math.isfinite(log_evidence):
            raise RuntimeError(f"{spec.label} returned non-finite log evidence")
        results[spec.key] = InferenceResult(
            model_key=spec.key,
            model_label=spec.label,
            donor_aware=donor_aware,
            idata=idata,
            # The graph can be large and the interface only needs InferenceData.
            # Do not retain it in per-session state after sampling completes.
            model=None,
            log_evidence=log_evidence,
            elapsed_seconds=float(elapsed),
            n_cells=len(frame),
            observation_time=observation_time,
            donor_labels=donor_labels,
        )
    return results


def run_count_models(
    frame: pd.DataFrame,
    observation_time: float,
    settings: InferenceSettings | None = None,
    model_keys: Sequence[str] | None = None,
    progress_callback: ProgressCallback | None = None,
    sampler_progress_callback: SamplerProgressCallback | None = None,
) -> dict[str, InferenceResult]:
    """Validate and fit donor ignorant models sequentially with SMC."""

    validated = validate_count_frame(frame)
    duration = validate_observation_time(observation_time)
    controls = settings if settings is not None else InferenceSettings()
    if not isinstance(controls, InferenceSettings):
        raise TypeError("settings must be an InferenceSettings instance or None")
    return _fit_models(
        frame=validated,
        observation_time=duration,
        settings=controls,
        specs=_selected_specs(model_keys),
        donor_aware=False,
        progress_callback=progress_callback,
        sampler_progress_callback=sampler_progress_callback,
    )


def run_donor_models(
    frame: pd.DataFrame,
    observation_time: float,
    settings: InferenceSettings | None = None,
    model_keys: Sequence[str] | None = None,
    progress_callback: ProgressCallback | None = None,
    sampler_progress_callback: SamplerProgressCallback | None = None,
) -> dict[str, InferenceResult]:
    """Validate and fit the canonical donor-relative models sequentially."""

    validated = validate_donor_frame(frame)
    duration = validate_observation_time(observation_time)
    controls = settings if settings is not None else InferenceSettings()
    if not isinstance(controls, InferenceSettings):
        raise TypeError("settings must be an InferenceSettings instance or None")
    return _fit_models(
        frame=validated,
        observation_time=duration,
        settings=controls,
        specs=_selected_specs(model_keys),
        donor_aware=True,
        progress_callback=progress_callback,
        sampler_progress_callback=sampler_progress_callback,
    )


_EVIDENCE_COLUMNS: Final[list[str]] = [
    "model_key",
    "model",
    "log_evidence",
    "delta_log_evidence_vs_best",
    "log10_BF_model_vs_best",
    "log10_BF_best_vs_model",
    "is_best",
    "elapsed_seconds",
]


def evidence_table(results: Mapping[str, InferenceResult]) -> pd.DataFrame:
    """Build a ranked evidence table; Bayes factors are relative to the best fit."""

    if not results:
        return pd.DataFrame(columns=_EVIDENCE_COLUMNS)
    rows: list[dict[str, Any]] = []
    for result in results.values():
        if not isinstance(result, InferenceResult):
            raise TypeError("results values must be InferenceResult instances")
        rows.append(
            {
                "model_key": result.model_key,
                "model": result.model_label,
                "log_evidence": float(result.log_evidence),
                "elapsed_seconds": float(result.elapsed_seconds),
            }
        )
    table = pd.DataFrame(rows)
    best = float(table["log_evidence"].max())
    table["delta_log_evidence_vs_best"] = table["log_evidence"] - best
    table["log10_BF_model_vs_best"] = (
        table["delta_log_evidence_vs_best"] / np.log(10.0)
    )
    table["log10_BF_best_vs_model"] = -table["log10_BF_model_vs_best"]
    table["is_best"] = np.isclose(
        table["log_evidence"].to_numpy(dtype=float),
        best,
    )
    return (
        table.loc[:, _EVIDENCE_COLUMNS]
        .sort_values(
            ["log_evidence", "model_key"],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )


_SUMMARY_PREFIX: Final[list[str]] = ["model_key", "model", "parameter"]


def _donor_parameter_label(parameter: str, labels: tuple[str, ...]) -> str:
    match = re.fullmatch(r"(.+)_donor\[(\d+)\]", parameter)
    if match is None:
        return parameter
    index = int(match.group(2))
    if index >= len(labels):
        return parameter
    return f"{match.group(1)}_donor[{labels[index]}]"


def summary_table(
    results: Mapping[str, InferenceResult],
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    """Combine model-specific posterior means and HDIs into one table."""

    probability = _finite(hdi_prob, "hdi_prob")
    if not 0 < probability < 1:
        raise ValueError("hdi_prob must be between zero and one")
    if not results:
        return pd.DataFrame(columns=_SUMMARY_PREFIX)

    import arviz as az

    tables: list[pd.DataFrame] = []
    for result in results.values():
        if not isinstance(result, InferenceResult):
            raise TypeError("results values must be InferenceResult instances")
        spec = MODEL_SPECS[result.model_key]
        requested = (
            spec.donor_parameters if result.donor_aware else spec.count_parameters
        )
        posterior = getattr(result.idata, "posterior", None)
        available_names = set(getattr(posterior, "data_vars", {}))
        available = [name for name in requested if name in available_names]
        if not available:
            raise RuntimeError(
                f"no reportable posterior variables found for {result.model_label}"
            )
        table = az.summary(
            result.idata,
            var_names=available,
            kind="stats",
            hdi_prob=probability,
            round_to=None,
        ).rename_axis("parameter").reset_index()
        if result.donor_labels:
            table["parameter"] = table["parameter"].map(
                lambda value: _donor_parameter_label(
                    str(value),
                    result.donor_labels,
                )
            )
        table.insert(0, "model", result.model_label)
        table.insert(0, "model_key", result.model_key)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


_COMMON_PARAMETER_SOURCES: Final[dict[str, dict[str, str]]] = {
    "homo": {"mu_lambda": "lambda"},
    "z2p": {"mu_lambda": "lambda", "p_zero": "p_zero"},
    "dis2p": {"mu_lambda": "mu_lambda", "sigma_lambda": "sigma_lambda"},
    "hetero3": {
        "mu_lambda": "mu_lambda",
        "sigma_lambda": "sigma_lambda",
        "p_zero": "p_zero",
    },
}


def posterior_draw_table(
    results: Mapping[str, InferenceResult],
    *,
    max_draws_per_model: int | None = None,
) -> pd.DataFrame:
    """Return paired posterior draws on the common event-count parameter axes.

    Rows, rather than individual parameter vectors, are subsampled so the joint
    dependence between parameters is preserved.
    """

    if max_draws_per_model is not None:
        max_draws_per_model = int(max_draws_per_model)
        if max_draws_per_model <= 0:
            raise ValueError("max_draws_per_model must be positive or None")

    tables: list[pd.DataFrame] = []
    for model_key, result in results.items():
        if result.donor_aware:
            continue
        mapping = _COMMON_PARAMETER_SOURCES[model_key]
        vectors: dict[str, np.ndarray] = {}
        for public_name, source_name in mapping.items():
            if source_name not in result.idata.posterior.data_vars:
                continue
            values = result.idata.posterior[source_name]
            extra_dims = [dim for dim in values.dims if dim not in {"chain", "draw"}]
            if extra_dims:
                continue
            vectors[public_name] = np.asarray(values, dtype=float).reshape(-1)
        if not vectors:
            continue
        row_count = min(len(values) for values in vectors.values())
        frame = pd.DataFrame(
            {name: values[:row_count] for name, values in vectors.items()}
        )
        frame.insert(0, "posterior_draw", np.arange(row_count, dtype=int))
        frame.insert(0, "model", result.model_label)
        frame.insert(0, "model_key", model_key)
        if max_draws_per_model is not None and len(frame) > max_draws_per_model:
            indices = np.linspace(
                0,
                len(frame) - 1,
                max_draws_per_model,
                dtype=int,
            )
            frame = frame.iloc[indices].reset_index(drop=True)
        tables.append(frame)

    columns = [
        "model_key",
        "model",
        "posterior_draw",
        "mu_lambda",
        "sigma_lambda",
        "p_zero",
    ]
    if not tables:
        return pd.DataFrame(columns=columns)
    return pd.concat(tables, ignore_index=True).reindex(columns=columns)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Serialize a report table to UTF-8 CSV entirely in memory."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _zip_write(archive: ZipFile, name: str, content: bytes | str) -> None:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    payload = content.encode("utf-8") if isinstance(content, str) else content
    archive.writestr(info, payload)


def build_results_zip(
    results: Mapping[str, InferenceResult],
    data: pd.DataFrame,
    observation_time: float,
    settings: InferenceSettings,
    *,
    truth: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, bytes] | None = None,
) -> bytes:
    """Create a reproducible bundle with tables and raw posterior files."""

    if not results:
        raise ValueError("at least one inference result is required")
    if not isinstance(settings, InferenceSettings):
        raise TypeError("settings must be an InferenceSettings instance")
    result_values = list(results.values())
    if not all(isinstance(result, InferenceResult) for result in result_values):
        raise TypeError("results values must be InferenceResult instances")
    modes = {result.donor_aware for result in result_values}
    if len(modes) != 1:
        raise ValueError("a report cannot mix donor aware and donor ignorant results")
    donor_aware = modes.pop()
    validated_data = (
        validate_donor_frame(data) if donor_aware else validate_count_frame(data)
    )
    duration = validate_observation_time(observation_time)

    evidence = evidence_table(results)
    summaries = summary_table(results)
    posterior_draws = posterior_draw_table(results)
    metadata_payload = {
        "observation_time": duration,
        "donor_aware": donor_aware,
        "n_cells": len(validated_data),
        "models": [result.model_key for result in result_values],
        "settings": asdict(settings),
        "software": {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "pymc": _package_version("pymc"),
            "arviz": _package_version("arviz"),
        },
    }
    metadata_json = json.dumps(
        metadata_payload,
        indent=2,
        sort_keys=True,
        default=_json_default,
    ) + "\n"

    buffer = BytesIO()
    with ZipFile(buffer, mode="w") as archive:
        _zip_write(archive, "input_data.csv", csv_bytes(validated_data))
        _zip_write(archive, "model_evidence.csv", csv_bytes(evidence))
        _zip_write(archive, "posterior_summary.csv", csv_bytes(summaries))
        _zip_write(archive, "posterior_samples.csv", csv_bytes(posterior_draws))
        _zip_write(archive, "run_metadata.json", metadata_json)
        if truth is not None:
            truth_json = json.dumps(
                dict(truth),
                indent=2,
                sort_keys=True,
                default=_json_default,
            ) + "\n"
            _zip_write(archive, "ground_truth.json", truth_json)
        with TemporaryDirectory(prefix="orca-idata-") as temporary_directory:
            temporary_root = Path(temporary_directory)
            for result in result_values:
                filename = f"posterior_{result.model_key}_smc.nc"
                path = temporary_root / filename
                result.idata.to_netcdf(str(path))
                _zip_write(archive, filename, path.read_bytes())
        for name, content in (artifacts or {}).items():
            safe_name = str(name).replace("\\", "/").lstrip("/")
            if not safe_name or ".." in Path(safe_name).parts:
                raise ValueError(f"unsafe artifact name: {name!r}")
            _zip_write(archive, safe_name, bytes(content))
        _zip_write(
            archive,
            "README.md",
            "# Orca Bayesian event count results\n"
            "\n"
            "The `posterior_<model>_smc.nc` files are ArviZ InferenceData "
            "objects produced by PyMC Sequential Monte Carlo. The CSV files "
            "contain the input counts, SMC evidence, posterior summaries and "
            "paired posterior draws used by the joint plot.\n"
            "\n"
            "```python\n"
            "import arviz as az\n"
            "import pandas as pd\n"
            "\n"
            "idata = az.from_netcdf('posterior_hetero3_smc.nc')\n"
            "print(az.summary(idata, var_names=['mu_lambda', 'sigma_lambda', 'p_zero']))\n"
            "draws = idata.posterior[['mu_lambda', 'sigma_lambda', 'p_zero']].to_dataframe()\n"
            "az.plot_pair(idata, var_names=['mu_lambda', 'sigma_lambda', 'p_zero'], kind='kde', marginals=True)\n"
            "evidence = pd.read_csv('model_evidence.csv')\n"
            "```\n"
            "\n"
            "The Bayes factor column `log10_BF_best_vs_model` compares the "
            "largest SMC log evidence with each candidate model. Posterior "
            "intervals in `posterior_summary.csv` are 95% HDIs. Prior settings "
            "and software versions are recorded in `run_metadata.json`.\n",
        )
    return buffer.getvalue()
