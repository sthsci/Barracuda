"""UI-neutral adapters for donor-ignorant contact trajectory inference.

The research implementation stores one ordered binary contact history per
cell.  This module keeps that representation canonical at the web boundary,
while accepting the compact, wide, and event-level CSV layouts users are
likely to have.  Public labels use ``beta_f`` and ``beta_s``; the research
backend continues to receive its native ``beta_x`` and ``beta_y`` names.
"""

from __future__ import annotations

from ast import literal_eval
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from io import BytesIO, StringIO
import json
import math
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import time
from typing import Any, Final
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd


CANONICAL_COLUMNS: Final[tuple[str, ...]] = (
    "cell_id",
    "condition",
    "history",
)
DEFAULT_CONDITION: Final[str] = "Condition 1"
MAX_CONDITIONS: Final[int] = 4
MAX_CELLS: Final[int] = 1_000
MAX_EVENTS: Final[int] = 20_000
MAX_EVENTS_PER_CELL: Final[int] = 250
MAX_CSV_DRAWS_PER_RESULT: Final[int] = 10_000
MAX_SMC_DRAWS: Final[int] = 2_000
MAX_SMC_CHAINS: Final[int] = 4
MAX_SMC_CORES: Final[int] = 4
MAX_QUADRATURE_NODES: Final[int] = 80
UINT32_MODULUS: Final[int] = 2**32

PUBLIC_PARAMETERS: Final[tuple[str, ...]] = (
    "mu_lambda",
    "sigma_lambda",
    "mu_eta",
    "sigma_eta",
    "beta_f",
    "beta_s",
)
PUBLIC_TO_BACKEND_PARAMETER: Final[dict[str, str]] = {
    "mu_lambda": "mu_lambda",
    "sigma_lambda": "sigma_lambda",
    "mu_eta": "mu_eta",
    "sigma_eta": "sigma_eta",
    "beta_f": "beta_x",
    "beta_s": "beta_y",
}
BACKEND_TO_PUBLIC_PARAMETER: Final[dict[str, str]] = {
    backend: public for public, backend in PUBLIC_TO_BACKEND_PARAMETER.items()
}


@dataclass(frozen=True)
class TrajectoryModelSpec:
    """One of the four trajectory decision mechanisms in the paper."""

    key: str
    notation: str
    label: str
    short_label: str
    heterogeneous: bool
    history_dependent: bool
    parameters: tuple[str, ...]


TRAJECTORY_MODEL_SPECS: Final[dict[str, TrajectoryModelSpec]] = {
    "homogeneous_history_independent": TrajectoryModelSpec(
        key="homogeneous_history_independent",
        notation="𝓜_Hom-HI",
        label="𝓜_Hom-HI · Homogeneous, history independent",
        short_label="Hom-HI",
        heterogeneous=False,
        history_dependent=False,
        parameters=("mu_lambda", "sigma_lambda", "mu_eta"),
    ),
    "homogeneous_history_dependent": TrajectoryModelSpec(
        key="homogeneous_history_dependent",
        notation="𝓜_Hom-HD",
        label="𝓜_Hom-HD · Homogeneous, history dependent",
        short_label="Hom-HD",
        heterogeneous=False,
        history_dependent=True,
        parameters=("mu_lambda", "sigma_lambda", "mu_eta", "beta_f", "beta_s"),
    ),
    "heterogeneous_history_independent": TrajectoryModelSpec(
        key="heterogeneous_history_independent",
        notation="𝓜_Het-HI",
        label="𝓜_Het-HI · Heterogeneous, history independent",
        short_label="Het-HI",
        heterogeneous=True,
        history_dependent=False,
        parameters=("mu_lambda", "sigma_lambda", "mu_eta", "sigma_eta"),
    ),
    "heterogeneous_history_dependent": TrajectoryModelSpec(
        key="heterogeneous_history_dependent",
        notation="𝓜_Het-HD",
        label="𝓜_Het-HD · Heterogeneous, history dependent",
        short_label="Het-HD",
        heterogeneous=True,
        history_dependent=True,
        parameters=PUBLIC_PARAMETERS,
    ),
}

_MODEL_ALIASES: Final[dict[str, str]] = {
    "hom_hi": "homogeneous_history_independent",
    "hom-hi": "homogeneous_history_independent",
    "homogeneous_history_independent": "homogeneous_history_independent",
    "hom_hd": "homogeneous_history_dependent",
    "hom-hd": "homogeneous_history_dependent",
    "homogeneous_history_dependent": "homogeneous_history_dependent",
    "het_hi": "heterogeneous_history_independent",
    "het-hi": "heterogeneous_history_independent",
    "heterogeneous_history_independent": "heterogeneous_history_independent",
    "het_hd": "heterogeneous_history_dependent",
    "het-hd": "heterogeneous_history_dependent",
    "heterogeneous_history_dependent": "heterogeneous_history_dependent",
}


def _positive_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    converted = int(value)
    if converted < 1:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and converted > maximum:
        raise ValueError(f"{name} must be at most {maximum:,}")
    return converted


def _nonnegative_int(value: Any, name: str, *, maximum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    converted = int(value)
    if converted < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    if maximum is not None and converted > maximum:
        raise ValueError(f"{name} must be at most {maximum:,}")
    return converted


def _finite(value: Any, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


@dataclass(frozen=True)
class TrajectorySettings:
    """Validated SMC and prior controls for a web trajectory run."""

    draws: int = 256
    chains: int = 1
    cores: int | None = 1
    seed: int | None = 2026
    threshold: float = 0.5
    correlation_threshold: float = 0.01
    lambda_prior_bounds: tuple[float, float] = (-5.0, 2.0)
    sigma_lambda_prior: float = 1.0
    p0_prior: tuple[float, float] = (1.0, 1.0)
    sigma_eta_prior: float = 1.0
    beta_prior_sd: float = 1.0
    n_quad: int = 20
    prior_draws: int = 0

    def __post_init__(self) -> None:
        _positive_int(self.draws, "draws", maximum=MAX_SMC_DRAWS)
        _positive_int(self.chains, "chains", maximum=MAX_SMC_CHAINS)
        if self.cores is not None:
            _positive_int(self.cores, "cores", maximum=MAX_SMC_CORES)
        if self.seed is not None:
            _nonnegative_int(self.seed, "seed", maximum=UINT32_MODULUS - 1)
        threshold = _finite(self.threshold, "threshold")
        correlation = _finite(self.correlation_threshold, "correlation_threshold")
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be greater than zero and at most one")
        if not 0 <= correlation <= 1:
            raise ValueError("correlation_threshold must be between zero and one")
        if len(self.lambda_prior_bounds) != 2:
            raise ValueError("lambda_prior_bounds must contain two values")
        lower, upper = map(
            lambda value: _finite(value, "lambda_prior_bounds"),
            self.lambda_prior_bounds,
        )
        if lower >= upper:
            raise ValueError("lambda_prior_bounds must be strictly increasing")
        if _finite(self.sigma_lambda_prior, "sigma_lambda_prior") <= 0:
            raise ValueError("sigma_lambda_prior must be greater than zero")
        if len(self.p0_prior) != 2 or any(
            _finite(value, "p0_prior") <= 0 for value in self.p0_prior
        ):
            raise ValueError("p0_prior must contain two positive values")
        if _finite(self.sigma_eta_prior, "sigma_eta_prior") <= 0:
            raise ValueError("sigma_eta_prior must be greater than zero")
        if _finite(self.beta_prior_sd, "beta_prior_sd") <= 0:
            raise ValueError("beta_prior_sd must be greater than zero")
        _positive_int(self.n_quad, "n_quad", maximum=MAX_QUADRATURE_NODES)
        _nonnegative_int(self.prior_draws, "prior_draws", maximum=MAX_SMC_DRAWS)

    @property
    def particles(self) -> int:
        """User-facing alias for the PyMC ``draws`` argument."""

        return int(self.draws)


@dataclass(frozen=True)
class TrajectorySimulationSpec:
    """Ground truth for one independently simulated condition."""

    condition: str = "Synthetic"
    n_cells: int = 100
    mu_lambda: float = 4.0
    sigma_lambda: float = 2.0
    p0: float = 0.25
    sigma_eta: float = 0.75
    beta_f: float = 0.8
    beta_s: float = -0.8
    observation_time: float = 1.0
    seed: int | None = 2026

    def __post_init__(self) -> None:
        _clean_label(self.condition, "condition")
        _positive_int(self.n_cells, "n_cells", maximum=MAX_CELLS)
        if _finite(self.mu_lambda, "mu_lambda") <= 0:
            raise ValueError("mu_lambda must be greater than zero")
        if _finite(self.sigma_lambda, "sigma_lambda") < 0:
            raise ValueError("sigma_lambda must be non-negative")
        probability = _finite(self.p0, "p0")
        if not 0 < probability < 1:
            raise ValueError("p0 must be strictly between zero and one")
        if _finite(self.sigma_eta, "sigma_eta") < 0:
            raise ValueError("sigma_eta must be non-negative")
        _finite(self.beta_f, "beta_f")
        _finite(self.beta_s, "beta_s")
        if _finite(self.observation_time, "observation_time") <= 0:
            raise ValueError("observation_time must be greater than zero")
        if self.seed is not None:
            _nonnegative_int(self.seed, "seed", maximum=UINT32_MODULUS - 1)


@dataclass(frozen=True)
class TrajectoryResult:
    """One fitted model for one experimental condition."""

    condition: str
    model_key: str
    model_label: str
    idata: Any
    log_evidence: float
    elapsed_seconds: float
    n_cells: int
    n_events: int
    observation_time: float


TrajectoryResults = dict[str, dict[str, TrajectoryResult]]
TrajectoryProgressCallback = Callable[[int, int, str, int, int, str], None]
TrajectorySamplerProgressCallback = Callable[
    [int, int, str, int, int, str, int, int, float], None
]


_COLUMN_ALIASES: Final[dict[str, set[str]]] = {
    "cell_id": {"cell", "cell_id", "cellid", "nk_cell", "nk_cell_id"},
    "condition": {
        "condition",
        "condition_label",
        "experimental_condition",
        "experimental_group",
        "group",
        "treatment",
    },
    "history": {
        "history",
        "history_string",
        "contact_history",
        "trajectory",
    },
    "contact_index": {
        "contact_index",
        "contact_order",
        "event_index",
        "event_id",
        "order",
    },
    "outcome": {
        "outcome",
        "z",
        "kill",
        "lethal",
        "is_lethal",
    },
}


def _column_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().lower() in {"nan", "none", "null"}
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _clean_label(value: Any, name: str) -> str:
    if _is_missing(value):
        raise ValueError(f"{name} must not be blank")
    text = str(value).strip()
    if len(text) > 100:
        raise ValueError(f"{name} must contain at most 100 characters")
    return text


def _parse_binary(value: Any, *, context: str) -> int:
    if _is_missing(value):
        raise ValueError(f"{context} is missing")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be 0 or 1") from exc
    if not np.isfinite(numeric) or not numeric.is_integer() or int(numeric) not in (0, 1):
        raise ValueError(f"{context} must be 0 or 1")
    return int(numeric)


def _parse_history(value: Any, *, context: str) -> tuple[int, ...]:
    if _is_missing(value):
        return ()
    parsed = value
    if isinstance(value, str):
        text = value.strip()
        compact = "".join(text.split())
        if compact and set(compact).issubset({"0", "1"}):
            parsed = list(compact)
        else:
            try:
                parsed = literal_eval(text)
            except (SyntaxError, ValueError):
                stripped = text.strip("[]()")
                if not stripped:
                    return ()
                parsed = [part for part in re.split(r"[\s,;]+", stripped) if part]
    if isinstance(parsed, str) or np.isscalar(parsed):
        values = [parsed]
    else:
        try:
            values = list(parsed)
        except TypeError as exc:
            raise ValueError(f"{context} must be an ordered sequence of 0 and 1") from exc
    history = tuple(
        _parse_binary(item, context=f"{context} outcome")
        for item in values
        if not _is_missing(item)
    )
    if len(history) > MAX_EVENTS_PER_CELL:
        raise ValueError(
            f"{context} contains {len(history):,} contacts; the limit is "
            f"{MAX_EVENTS_PER_CELL:,} per cell"
        )
    return history


def read_trajectory_csv(source: bytes | str) -> pd.DataFrame:
    """Read CSV text without losing leading zeroes in compact histories."""

    if isinstance(source, bytes):
        stream: BytesIO | StringIO = BytesIO(source)
    elif isinstance(source, str):
        stream = StringIO(source)
    else:
        raise TypeError("source must be CSV text or bytes")
    try:
        return pd.read_csv(stream, dtype=str, keep_default_na=False)
    except Exception as exc:
        raise ValueError(f"Could not read trajectory CSV: {exc}") from exc


def _as_frame(frame: Any) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    if isinstance(frame, (bytes, str)):
        return read_trajectory_csv(frame)
    if isinstance(frame, Sequence) and not isinstance(frame, (str, bytes)):
        return pd.DataFrame(frame)
    raise TypeError("trajectory data must be a DataFrame, records, or CSV text")


def _resolve_columns(frame: pd.DataFrame) -> dict[str, Any]:
    found: dict[str, Any] = {}
    for column in frame.columns:
        token = _column_token(column)
        for canonical, aliases in _COLUMN_ALIASES.items():
            if token not in aliases:
                continue
            if canonical in found:
                raise ValueError(f"Multiple columns match {canonical!r}")
            found[canonical] = column
    return found


def _base_identity_frame(raw: pd.DataFrame, columns: Mapping[str, Any]) -> pd.DataFrame:
    if "cell_id" in columns:
        cell_ids = [
            _clean_label(value, f"cell_id in row {row + 1}")
            for row, value in enumerate(raw[columns["cell_id"]])
        ]
    else:
        cell_ids = [f"cell_{index + 1:04d}" for index in range(len(raw))]
    if "condition" in columns:
        conditions = [
            DEFAULT_CONDITION if _is_missing(value) else _clean_label(value, "condition")
            for value in raw[columns["condition"]]
        ]
    else:
        conditions = [DEFAULT_CONDITION] * len(raw)
    return pd.DataFrame({"cell_id": cell_ids, "condition": conditions}, index=raw.index)


def _normalise_compact(raw: pd.DataFrame, columns: Mapping[str, Any]) -> pd.DataFrame:
    identity = _base_identity_frame(raw, columns)
    identity["history"] = [
        _parse_history(value, context=f"history in row {row + 1}")
        for row, value in enumerate(raw[columns["history"]])
    ]
    return identity.reset_index(drop=True)


def _normalise_wide(raw: pd.DataFrame, columns: Mapping[str, Any]) -> pd.DataFrame:
    wide_columns: list[tuple[int, Any]] = []
    excluded = set(columns.values())
    for column in raw.columns:
        if column in excluded:
            continue
        text = str(column).strip()
        if re.fullmatch(r"\d+", text):
            wide_columns.append((int(text), column))
    if not wide_columns:
        raise ValueError(
            "Trajectory data need a history column, contact_index/outcome columns, "
            "or numbered contact columns such as 1, 2, 3"
        )
    order_values = [order for order, _ in wide_columns]
    if len(order_values) != len(set(order_values)):
        raise ValueError("Numbered contact columns must have unique indices")
    wide_columns.sort(key=lambda pair: pair[0])
    identity = _base_identity_frame(raw, columns)
    histories: list[tuple[int, ...]] = []
    for row_number, (_, row) in enumerate(raw.iterrows(), start=1):
        values: list[int] = []
        reached_end = False
        for _, column in wide_columns:
            value = row[column]
            if _is_missing(value):
                reached_end = True
                continue
            if reached_end:
                raise ValueError(
                    f"row {row_number} has a contact outcome after a blank numbered column"
                )
            values.append(_parse_binary(value, context=f"outcome in row {row_number}"))
        histories.append(_parse_history(values, context=f"history in row {row_number}"))
    identity["history"] = histories
    return identity.reset_index(drop=True)


def _normalise_long(raw: pd.DataFrame, columns: Mapping[str, Any]) -> pd.DataFrame:
    if "cell_id" not in columns:
        raise ValueError("Event-level trajectory data require a cell_id column")
    identity = _base_identity_frame(raw, columns)
    work = identity.copy()
    work["contact_index"] = raw[columns["contact_index"]].to_numpy()
    work["outcome"] = raw[columns["outcome"]].to_numpy()
    records: list[dict[str, Any]] = []
    for (condition, cell_id), group in work.groupby(
        ["condition", "cell_id"], sort=False, dropna=False
    ):
        blank = group["contact_index"].map(_is_missing) & group["outcome"].map(_is_missing)
        if bool(blank.any()):
            if len(group) != 1 or not bool(blank.all()):
                raise ValueError(
                    f"{condition} / {cell_id} mixes an empty trajectory row with contacts"
                )
            history: tuple[int, ...] = ()
        else:
            if bool(group["contact_index"].map(_is_missing).any()) or bool(
                group["outcome"].map(_is_missing).any()
            ):
                raise ValueError(f"{condition} / {cell_id} has an incomplete contact row")
            indices: list[int] = []
            outcomes: list[int] = []
            for row_number, row in group.iterrows():
                try:
                    numeric_index = float(row["contact_index"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"contact_index for {condition} / {cell_id} must be an integer"
                    ) from exc
                if (
                    not np.isfinite(numeric_index)
                    or not numeric_index.is_integer()
                    or numeric_index < 0
                ):
                    raise ValueError(
                        f"contact_index for {condition} / {cell_id} must be a non-negative integer"
                    )
                indices.append(int(numeric_index))
                outcomes.append(
                    _parse_binary(
                        row["outcome"],
                        context=f"outcome for {condition} / {cell_id} at row {row_number + 1}",
                    )
                )
            if len(indices) != len(set(indices)):
                raise ValueError(f"{condition} / {cell_id} has duplicate contact indices")
            paired = sorted(zip(indices, outcomes), key=lambda pair: pair[0])
            start = paired[0][0]
            expected = list(range(start, start + len(paired)))
            if start not in (0, 1) or [index for index, _ in paired] != expected:
                raise ValueError(
                    f"contact indices for {condition} / {cell_id} must be consecutive "
                    "and start at 0 or 1"
                )
            history = _parse_history(
                [outcome for _, outcome in paired],
                context=f"history for {condition} / {cell_id}",
            )
        records.append({"cell_id": cell_id, "condition": condition, "history": history})
    return pd.DataFrame(records, columns=CANONICAL_COLUMNS)


def _validate_canonical(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("trajectory data must contain at least one cell")
    if len(frame) > MAX_CELLS:
        raise ValueError(f"trajectory data may contain at most {MAX_CELLS:,} cells")
    duplicates = frame.duplicated(["condition", "cell_id"], keep=False)
    if bool(duplicates.any()):
        first = frame.loc[duplicates, ["condition", "cell_id"]].iloc[0]
        raise ValueError(
            "Each cell must appear once per condition; duplicate: "
            f"{first['condition']} / {first['cell_id']}"
        )
    conditions = frame["condition"].drop_duplicates().tolist()
    if len(conditions) > MAX_CONDITIONS:
        raise ValueError(f"trajectory data may contain at most {MAX_CONDITIONS} conditions")
    total_events = int(sum(len(history) for history in frame["history"]))
    if total_events == 0:
        raise ValueError("trajectory data must contain at least one contact outcome")
    if total_events > MAX_EVENTS:
        raise ValueError(f"trajectory data may contain at most {MAX_EVENTS:,} contact outcomes")
    for condition, group in frame.groupby("condition", sort=False):
        if sum(len(history) for history in group["history"]) == 0:
            raise ValueError(f"condition {condition!r} has no contact outcomes")
    return frame.loc[:, CANONICAL_COLUMNS].reset_index(drop=True)


def normalize_trajectory_frame(frame: Any) -> pd.DataFrame:
    """Return canonical ``cell_id, condition, history`` trajectory data."""

    raw = _as_frame(frame)
    if raw.empty:
        raise ValueError("trajectory data must contain at least one row")
    columns = _resolve_columns(raw)
    if "history" in columns:
        canonical = _normalise_compact(raw, columns)
    elif "contact_index" in columns or "outcome" in columns:
        if not {"contact_index", "outcome"}.issubset(columns):
            raise ValueError(
                "Event-level data require both contact_index and outcome columns"
            )
        canonical = _normalise_long(raw, columns)
    else:
        canonical = _normalise_wide(raw, columns)
    return _validate_canonical(canonical)


def validate_trajectory_frame(frame: Any) -> pd.DataFrame:
    """Validate and canonicalise trajectory data (public explicit alias)."""

    return normalize_trajectory_frame(frame)


def expanded_trajectory_frame(frame: Any) -> pd.DataFrame:
    """Expand cell histories into one ordered row per observed contact."""

    canonical = validate_trajectory_frame(frame)
    rows: list[dict[str, Any]] = []
    for record in canonical.itertuples(index=False):
        failed = 0
        successful = 0
        for contact_index, outcome in enumerate(record.history, start=1):
            rows.append(
                {
                    "cell_id": record.cell_id,
                    "condition": record.condition,
                    "contact_index": contact_index,
                    "previous_nonlethal_contacts": failed,
                    "previous_lethal_contacts": successful,
                    "outcome": int(outcome),
                }
            )
            if int(outcome) == 1:
                successful += 1
            else:
                failed += 1
    return pd.DataFrame(
        rows,
        columns=[
            "cell_id",
            "condition",
            "contact_index",
            "previous_nonlethal_contacts",
            "previous_lethal_contacts",
            "outcome",
        ],
    )


def truth_model_key(
    sigma_eta: float,
    beta_f: float,
    beta_s: float,
    *,
    tolerance: float = 1e-12,
) -> str:
    """Classify a synthetic truth into the four trajectory mechanisms."""

    sigma = _finite(sigma_eta, "sigma_eta")
    if sigma < 0:
        raise ValueError("sigma_eta must be non-negative")
    beta_failed = _finite(beta_f, "beta_f")
    beta_successful = _finite(beta_s, "beta_s")
    tol = _finite(tolerance, "tolerance")
    if tol < 0:
        raise ValueError("tolerance must be non-negative")
    heterogeneous = not math.isclose(sigma, 0.0, abs_tol=tol)
    history_dependent = not (
        math.isclose(beta_failed, 0.0, abs_tol=tol)
        and math.isclose(beta_successful, 0.0, abs_tol=tol)
    )
    prefix = "heterogeneous" if heterogeneous else "homogeneous"
    suffix = "history_dependent" if history_dependent else "history_independent"
    return f"{prefix}_{suffix}"


def _coerce_simulation_specs(
    specs: Any,
    *,
    condition: str,
    n_cells: int,
    mu_lambda: float,
    sigma_lambda: float,
    p0: float,
    sigma_eta: float,
    beta_f: float,
    beta_s: float,
    observation_time: float,
    seed: int | None,
) -> list[TrajectorySimulationSpec]:
    if specs is None:
        return [
            TrajectorySimulationSpec(
                condition=condition,
                n_cells=n_cells,
                mu_lambda=mu_lambda,
                sigma_lambda=sigma_lambda,
                p0=p0,
                sigma_eta=sigma_eta,
                beta_f=beta_f,
                beta_s=beta_s,
                observation_time=observation_time,
                seed=seed,
            )
        ]
    if isinstance(specs, TrajectorySimulationSpec):
        return [specs]
    if isinstance(specs, Mapping):
        if "condition" in specs or "n_cells" in specs:
            return [TrajectorySimulationSpec(**dict(specs))]
        return [
            TrajectorySimulationSpec(condition=str(label), **dict(values))
            for label, values in specs.items()
        ]
    if isinstance(specs, Sequence) and not isinstance(specs, (str, bytes)):
        return [
            item
            if isinstance(item, TrajectorySimulationSpec)
            else TrajectorySimulationSpec(**dict(item))
            for item in specs
        ]
    raise TypeError("specs must be simulation settings or a sequence of settings")


def simulate_trajectory_frame(
    specs: Any = None,
    *,
    condition: str = "Synthetic",
    n_cells: int = 100,
    mu_lambda: float = 4.0,
    sigma_lambda: float = 2.0,
    p0: float = 0.25,
    sigma_eta: float = 0.75,
    beta_f: float = 0.8,
    beta_s: float = -0.8,
    observation_time: float = 1.0,
    seed: int | None = 2026,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Simulate one or more independent conditions and return their truths."""

    configurations = _coerce_simulation_specs(
        specs,
        condition=condition,
        n_cells=n_cells,
        mu_lambda=mu_lambda,
        sigma_lambda=sigma_lambda,
        p0=p0,
        sigma_eta=sigma_eta,
        beta_f=beta_f,
        beta_s=beta_s,
        observation_time=observation_time,
        seed=seed,
    )
    if not configurations:
        raise ValueError("at least one synthetic condition is required")
    labels = [config.condition for config in configurations]
    if len(labels) != len(set(labels)):
        raise ValueError("synthetic condition names must be unique")
    if len(labels) > MAX_CONDITIONS:
        raise ValueError(f"at most {MAX_CONDITIONS} synthetic conditions are supported")
    if sum(config.n_cells for config in configurations) > MAX_CELLS:
        raise ValueError(f"synthetic data may contain at most {MAX_CELLS:,} cells")
    expected_events = sum(
        config.n_cells * config.mu_lambda * config.observation_time
        for config in configurations
    )
    if expected_events > MAX_EVENTS:
        raise ValueError(
            "synthetic settings imply "
            f"{expected_events:,.0f} expected contacts; reduce the number of cells, "
            f"mean contact rate, or observation time to stay within {MAX_EVENTS:,}"
        )

    try:
        simulator = import_module("barracuda._backends.trajectories.simulator")
    except ModuleNotFoundError as exc:
        if not exc.name or not exc.name.startswith("barracuda"):
            raise
        simulator = import_module("section_3.src.simulator")
    frames: list[pd.DataFrame] = []
    truths: dict[str, dict[str, Any]] = {}
    for config in configurations:
        backend_params = simulator.ContactKillParams(
            n_cells=int(config.n_cells),
            mean_lambda=float(config.mu_lambda),
            sigma_lambda=float(config.sigma_lambda),
            p0=float(config.p0),
            sigma_eta=float(config.sigma_eta),
            beta_x=float(config.beta_f),
            beta_y=float(config.beta_s),
            duration=float(config.observation_time),
        )
        simulated = simulator.simulate_contact_kill(
            backend_params,
            seed=config.seed,
            return_latent=False,
        )
        frames.append(
            pd.DataFrame(
                {
                    "cell_id": [f"cell_{index + 1:04d}" for index in range(len(simulated))],
                    "condition": config.condition,
                    "history": simulated["history"].map(tuple),
                }
            )
        )
        mu_eta = float(np.log(config.p0 / (1.0 - config.p0)))
        truths[config.condition] = {
            "mu_lambda": float(config.mu_lambda),
            "sigma_lambda": float(config.sigma_lambda),
            "p0": float(config.p0),
            "mu_eta": mu_eta,
            "sigma_eta": float(config.sigma_eta),
            "beta_f": float(config.beta_f),
            "beta_s": float(config.beta_s),
            "observation_time": float(config.observation_time),
            "true_model_key": truth_model_key(
                config.sigma_eta,
                config.beta_f,
                config.beta_s,
            ),
        }
    canonical = validate_trajectory_frame(pd.concat(frames, ignore_index=True))
    return canonical, truths


def _selected_specs(model_keys: Sequence[str] | str | None) -> list[TrajectoryModelSpec]:
    if model_keys is None:
        return list(TRAJECTORY_MODEL_SPECS.values())
    if isinstance(model_keys, str):
        model_keys = [model_keys]
    if not model_keys:
        raise ValueError("model_keys must contain at least one model")
    selected: list[TrajectoryModelSpec] = []
    seen: set[str] = set()
    for requested in model_keys:
        token = str(requested).strip().lower().replace(" ", "_")
        try:
            key = _MODEL_ALIASES[token]
        except KeyError as exc:
            raise ValueError(f"unknown trajectory model {requested!r}") from exc
        if key in seen:
            raise ValueError(f"model_keys contains duplicate model {key!r}")
        seen.add(key)
        selected.append(TRAJECTORY_MODEL_SPECS[key])
    return selected


def _condition_model_seed(
    seed: int | None,
    condition_index: int,
    model_index: int,
) -> int | None:
    if seed is None:
        return None
    return int(
        (int(seed) + 104_729 * int(condition_index) + 1_009 * int(model_index))
        % UINT32_MODULUS
    )


def _load_trajectory_backend():
    try:
        return import_module("barracuda._backends.trajectories.inference")
    except ModuleNotFoundError as exc:
        if not exc.name or not exc.name.startswith("barracuda"):
            raise
        return import_module("section_3.src.inference")


def _run_with_native_smc_progress(callback, operation):
    try:
        progress = import_module(
            "barracuda._backends.event_counts.smc_progress"
        )
    except ModuleNotFoundError as exc:
        if not exc.name or not exc.name.startswith("barracuda"):
            raise
        progress = import_module("section_1.src.smc_progress")
    return progress.run_with_smc_progress(callback, operation)


def run_trajectory_conditions(
    frame: Any,
    observation_time: float = 1.0,
    *,
    settings: TrajectorySettings | None = None,
    model_keys: Sequence[str] | str | None = None,
    progress_callback: TrajectoryProgressCallback | None = None,
    sampler_progress_callback: TrajectorySamplerProgressCallback | None = None,
) -> TrajectoryResults:
    """Run selected trajectory models independently within each condition."""

    canonical = validate_trajectory_frame(frame)
    duration = _finite(observation_time, "observation_time")
    if duration <= 0:
        raise ValueError("observation_time must be greater than zero")
    controls = settings if settings is not None else TrajectorySettings()
    if not isinstance(controls, TrajectorySettings):
        raise TypeError("settings must be a TrajectorySettings instance or None")
    specs = _selected_specs(model_keys)
    backend = _load_trajectory_backend()
    groups = list(canonical.groupby("condition", sort=False))
    total_conditions = len(groups)
    total_models = len(specs)
    results: TrajectoryResults = {}
    for condition_index, (condition, group) in enumerate(groups, start=1):
        history_data = backend.prepare_data(group[["history"]])
        condition_results: dict[str, TrajectoryResult] = {}
        for model_index, spec in enumerate(specs, start=1):
            if progress_callback is not None:
                progress_callback(
                    condition_index,
                    total_conditions,
                    str(condition),
                    model_index,
                    total_models,
                    spec.label,
                )
            backend_spec = backend.ModelSpec(
                name=spec.key,
                heterogeneous=spec.heterogeneous,
                history_dependent=spec.history_dependent,
            )
            model = backend.build_model(
                history_data,
                backend_spec,
                duration=duration,
                lambda_prior_bounds=tuple(map(float, controls.lambda_prior_bounds)),
                sigma_lambda_prior=float(controls.sigma_lambda_prior),
                p0_prior=tuple(map(float, controls.p0_prior)),
                sigma_eta_prior=float(controls.sigma_eta_prior),
                beta_prior_sd=float(controls.beta_prior_sd),
                n_quad=int(controls.n_quad),
            )
            seed = _condition_model_seed(
                controls.seed,
                condition_index - 1,
                model_index - 1,
            )

            def native_progress(
                chain: int,
                stage: int,
                beta: float,
                *,
                _condition_index: int = condition_index,
                _condition: str = str(condition),
                _model_index: int = model_index,
                _spec: TrajectoryModelSpec = spec,
            ) -> None:
                if sampler_progress_callback is not None:
                    sampler_progress_callback(
                        _condition_index,
                        total_conditions,
                        _condition,
                        _model_index,
                        total_models,
                        _spec.label,
                        int(chain),
                        int(stage),
                        float(beta),
                    )

            started = time.perf_counter()
            idata = _run_with_native_smc_progress(
                native_progress if sampler_progress_callback is not None else None,
                lambda: backend.sample_smc(
                    model,
                    draws=int(controls.draws),
                    chains=int(controls.chains),
                    cores=None if controls.cores is None else int(controls.cores),
                    random_seed=seed,
                    prior_draws=int(controls.prior_draws),
                    threshold=float(controls.threshold),
                    correlation_threshold=float(controls.correlation_threshold),
                    progressbar=True,
                    retry_sequential=True,
                ),
            )
            elapsed = time.perf_counter() - started
            log_evidence = float(backend.log_evidence(idata))
            if not np.isfinite(log_evidence):
                raise RuntimeError(f"{spec.label} returned non-finite log evidence")
            condition_results[spec.key] = TrajectoryResult(
                condition=str(condition),
                model_key=spec.key,
                model_label=spec.label,
                idata=idata,
                log_evidence=log_evidence,
                elapsed_seconds=float(elapsed),
                n_cells=int(history_data.n_cells),
                n_events=int(history_data.z.size),
                observation_time=duration,
            )
        results[str(condition)] = condition_results
    return results


def trajectory_evidence_frame(results: Mapping[str, Mapping[str, TrajectoryResult]]) -> pd.DataFrame:
    """Rank models within every condition using raw SMC marginal likelihoods."""

    columns = [
        "condition",
        "model_key",
        "model",
        "short_model",
        "log_evidence",
        "best_log_evidence",
        "delta_log_evidence_vs_best",
        "log10_BF_model_vs_best",
        "log10_BF_best_vs_model",
        "is_best",
        "elapsed_seconds",
    ]
    rows: list[dict[str, Any]] = []
    for condition, condition_results in results.items():
        if not condition_results:
            continue
        typed_results: list[TrajectoryResult] = []
        for result in condition_results.values():
            if not isinstance(result, TrajectoryResult):
                raise TypeError("results must contain TrajectoryResult values")
            typed_results.append(result)
        values = [float(result.log_evidence) for result in typed_results]
        if not all(np.isfinite(values)):
            raise ValueError(f"condition {condition!r} contains non-finite evidence")
        best = max(values)
        for result in typed_results:
            delta = float(result.log_evidence - best)
            rows.append(
                {
                    "condition": str(condition),
                    "model_key": result.model_key,
                    "model": result.model_label,
                    "short_model": TRAJECTORY_MODEL_SPECS[result.model_key].short_label,
                    "log_evidence": float(result.log_evidence),
                    "best_log_evidence": float(best),
                    "delta_log_evidence_vs_best": delta,
                    "log10_BF_model_vs_best": delta / np.log(10.0),
                    "log10_BF_best_vs_model": -delta / np.log(10.0),
                    "is_best": bool(np.isclose(result.log_evidence, best)),
                    "elapsed_seconds": float(result.elapsed_seconds),
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows, columns=columns)
    condition_order = {condition: index for index, condition in enumerate(results)}
    frame["_condition_order"] = frame["condition"].map(condition_order)
    return (
        frame.sort_values(
            ["_condition_order", "log_evidence"],
            ascending=[True, False],
            kind="stable",
        )
        .drop(columns="_condition_order")
        .reset_index(drop=True)
    )


def _public_parameters(parameters: Sequence[str] | None) -> list[str]:
    if parameters is None:
        return list(PUBLIC_PARAMETERS)
    selected: list[str] = []
    for requested in parameters:
        text = str(requested)
        public = BACKEND_TO_PUBLIC_PARAMETER.get(text, text)
        if public not in PUBLIC_TO_BACKEND_PARAMETER:
            raise ValueError(f"unknown trajectory parameter {requested!r}")
        if public not in selected:
            selected.append(public)
    if not selected:
        raise ValueError("parameters must contain at least one parameter")
    return selected


def trajectory_posterior_draws(
    results: Mapping[str, Mapping[str, TrajectoryResult]],
    *,
    model_keys: Sequence[str] | str | None = None,
    parameters: Sequence[str] | None = None,
    max_draws: int | None = 6_000,
    seed: int | None = 17,
) -> pd.DataFrame:
    """Extract paired scalar posterior draws with public parameter names."""

    selected_specs = _selected_specs(model_keys)
    selected_models = {spec.key for spec in selected_specs}
    requested_parameters = _public_parameters(parameters)
    if max_draws is not None:
        _positive_int(max_draws, "max_draws")
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    for condition, condition_results in results.items():
        for model_key, result in condition_results.items():
            if model_key not in selected_models:
                continue
            if not isinstance(result, TrajectoryResult):
                raise TypeError("results must contain TrajectoryResult values")
            spec = TRAJECTORY_MODEL_SPECS[model_key]
            public_parameters = [
                parameter
                for parameter in requested_parameters
                if parameter in spec.parameters
                and PUBLIC_TO_BACKEND_PARAMETER[parameter] in result.idata.posterior
            ]
            if not public_parameters:
                continue
            arrays: dict[str, np.ndarray] = {}
            size: int | None = None
            chain_count: int | None = None
            draw_count: int | None = None
            for public in public_parameters:
                backend_name = PUBLIC_TO_BACKEND_PARAMETER[public]
                data_array = result.idata.posterior[backend_name]
                if "chain" not in data_array.dims or "draw" not in data_array.dims:
                    raise ValueError(f"posterior parameter {backend_name!r} is not scalar by chain/draw")
                extra_dims = [dim for dim in data_array.dims if dim not in {"chain", "draw"}]
                if any(int(data_array.sizes[dim]) != 1 for dim in extra_dims):
                    raise ValueError(f"posterior parameter {backend_name!r} is not scalar")
                ordered = data_array.transpose("chain", "draw", *extra_dims)
                values = np.asarray(ordered, dtype=float).reshape(-1)
                if size is None:
                    size = values.size
                    chain_count = int(data_array.sizes["chain"])
                    draw_count = int(data_array.sizes["draw"])
                elif values.size != size:
                    raise ValueError("posterior parameters do not share paired draws")
                arrays[public] = values
            assert size is not None and chain_count is not None and draw_count is not None
            finite = np.ones(size, dtype=bool)
            for values in arrays.values():
                finite &= np.isfinite(values)
            indices = np.flatnonzero(finite)
            if max_draws is not None and indices.size > int(max_draws):
                indices = np.sort(rng.choice(indices, size=int(max_draws), replace=False))
            chain_indices = np.repeat(np.arange(chain_count), draw_count)[indices]
            draw_indices = np.tile(np.arange(draw_count), chain_count)[indices]
            frame = pd.DataFrame(
                {
                    "condition": str(condition),
                    "model_key": model_key,
                    "model": result.model_label,
                    "chain": chain_indices,
                    "draw": draw_indices,
                    **{public: values[indices] for public, values in arrays.items()},
                }
            )
            frames.append(frame)
    columns = ["condition", "model_key", "model", "chain", "draw", *requested_parameters]
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).reindex(columns=columns)


def trajectory_summary_frame(
    results: Mapping[str, Mapping[str, TrajectoryResult]],
    *,
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    """Summarise all semantically estimated parameters using HDIs."""

    probability = _finite(hdi_prob, "hdi_prob")
    if not 0 < probability < 1:
        raise ValueError("hdi_prob must be between zero and one")
    import arviz as az

    rows: list[dict[str, Any]] = []
    draws = trajectory_posterior_draws(
        results,
        parameters=PUBLIC_PARAMETERS,
        max_draws=None,
    )
    if draws.empty:
        return pd.DataFrame(
            columns=[
                "condition",
                "model_key",
                "model",
                "parameter",
                "mean",
                "sd",
                "median",
                "hdi_lower",
                "hdi_upper",
                "hdi_probability",
                "n_draws",
            ]
        )
    for (condition, model_key, model), group in draws.groupby(
        ["condition", "model_key", "model"], sort=False
    ):
        spec = TRAJECTORY_MODEL_SPECS[str(model_key)]
        for parameter in spec.parameters:
            values = pd.to_numeric(group[parameter], errors="coerce").dropna().to_numpy(float)
            if not values.size:
                continue
            lower, upper = np.asarray(az.hdi(values, hdi_prob=probability), dtype=float)
            rows.append(
                {
                    "condition": condition,
                    "model_key": model_key,
                    "model": model,
                    "parameter": parameter,
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "median": float(np.median(values)),
                    "hdi_lower": float(lower),
                    "hdi_upper": float(upper),
                    "hdi_probability": probability,
                    "n_draws": int(values.size),
                }
            )
    return pd.DataFrame(rows)


def _safe_slug(label: str, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", str(label).strip().lower()).strip("_")
    base = base or "condition"
    candidate = base
    suffix = 2
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _write_bytes(archive: ZipFile, name: str, content: bytes) -> None:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, content)


def _idata_to_netcdf_bytes(idata: Any) -> bytes:
    import arviz as az

    with TemporaryDirectory(prefix="barracuda-trajectory-") as directory:
        path = Path(directory) / "posterior.nc"
        az.to_netcdf(idata, str(path))
        return path.read_bytes()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def build_trajectory_archive(
    results: Mapping[str, Mapping[str, TrajectoryResult]],
    frame: Any,
    observation_time: float,
    settings: TrajectorySettings,
    *,
    truth: Mapping[str, Mapping[str, Any]] | None = None,
    hdi_prob: float = 0.95,
    max_csv_draws_per_result: int = MAX_CSV_DRAWS_PER_RESULT,
) -> bytes:
    """Build a portable ZIP containing data, diagnostics, draws, and NetCDFs."""

    canonical = validate_trajectory_frame(frame)
    duration = _finite(observation_time, "observation_time")
    if duration <= 0:
        raise ValueError("observation_time must be greater than zero")
    if not isinstance(settings, TrajectorySettings):
        raise TypeError("settings must be a TrajectorySettings instance")
    _positive_int(max_csv_draws_per_result, "max_csv_draws_per_result")
    data_conditions = canonical["condition"].drop_duplicates().tolist()
    if list(results) != data_conditions:
        raise ValueError("result conditions must match the normalized input data")

    normalized_csv = canonical.copy()
    normalized_csv["history"] = normalized_csv["history"].map(
        lambda history: "".join(map(str, history))
    )
    expanded = expanded_trajectory_frame(canonical)
    evidence = trajectory_evidence_frame(results)
    summary = trajectory_summary_frame(results, hdi_prob=hdi_prob)
    draws = trajectory_posterior_draws(
        results,
        max_draws=int(max_csv_draws_per_result),
        seed=settings.seed,
    )
    config = {
        "schema_version": 1,
        "analysis": "donor_ignorant_contact_trajectory",
        "observation_time": duration,
        "settings": asdict(settings),
        "conditions": data_conditions,
        "models_by_condition": {
            condition: list(condition_results) for condition, condition_results in results.items()
        },
        "n_cells": int(len(canonical)),
        "n_events": int(len(expanded)),
        "posterior_csv_draw_cap_per_result": int(max_csv_draws_per_result),
        "ground_truth": truth,
    }

    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        _write_bytes(archive, "normalized_trajectories.csv", normalized_csv.to_csv(index=False).encode())
        _write_bytes(archive, "expanded_contacts.csv", expanded.to_csv(index=False).encode())
        _write_bytes(archive, "model_evidence.csv", evidence.to_csv(index=False).encode())
        _write_bytes(archive, "posterior_summary.csv", summary.to_csv(index=False).encode())
        _write_bytes(archive, "posterior_draws.csv", draws.to_csv(index=False).encode())
        _write_bytes(
            archive,
            "configuration.json",
            json.dumps(config, indent=2, sort_keys=True, default=_json_default).encode(),
        )
        if truth:
            truth_rows = [
                {"condition": condition, **dict(values)}
                for condition, values in truth.items()
            ]
            _write_bytes(
                archive,
                "ground_truth.csv",
                pd.DataFrame(truth_rows).to_csv(index=False).encode(),
            )
        used: set[str] = set()
        for condition, condition_results in results.items():
            condition_slug = _safe_slug(condition, used)
            for model_key, result in condition_results.items():
                _write_bytes(
                    archive,
                    f"conditions/{condition_slug}/posterior_{model_key}.nc",
                    _idata_to_netcdf_bytes(result.idata),
                )
        _write_bytes(
            archive,
            "README.txt",
            (
                "Barracuda donor-ignorant contact trajectory analysis\n\n"
                "normalized_trajectories.csv contains one ordered binary history per cell.\n"
                "expanded_contacts.csv contains one row per contact.\n"
                "model_evidence.csv reports SMC marginal likelihoods and Bayes factors.\n"
                "posterior_draws.csv uses beta_f and beta_s public labels and preserves paired draws.\n"
                "Each condition folder contains ArviZ-compatible NetCDF inference data.\n"
            ).encode(),
        )
    return buffer.getvalue()


__all__ = [
    "BACKEND_TO_PUBLIC_PARAMETER",
    "CANONICAL_COLUMNS",
    "MAX_CELLS",
    "MAX_CONDITIONS",
    "MAX_EVENTS",
    "MAX_QUADRATURE_NODES",
    "MAX_SMC_CHAINS",
    "MAX_SMC_CORES",
    "MAX_SMC_DRAWS",
    "PUBLIC_PARAMETERS",
    "PUBLIC_TO_BACKEND_PARAMETER",
    "TRAJECTORY_MODEL_SPECS",
    "TrajectoryModelSpec",
    "TrajectoryResult",
    "TrajectoryResults",
    "TrajectorySettings",
    "TrajectorySimulationSpec",
    "build_trajectory_archive",
    "expanded_trajectory_frame",
    "normalize_trajectory_frame",
    "read_trajectory_csv",
    "run_trajectory_conditions",
    "simulate_trajectory_frame",
    "trajectory_evidence_frame",
    "trajectory_posterior_draws",
    "trajectory_summary_frame",
    "truth_model_key",
    "validate_trajectory_frame",
]
