"""Canonical tabular schemas and validation for event count data."""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


COUNT_COLUMNS: Final[tuple[str, str]] = ("cell_id", "count")
DONOR_COLUMNS: Final[tuple[str, str, str]] = (
    "cell_id",
    "donor_id",
    "count",
)
MIN_CELLS: Final[int] = 5
MAX_CELLS: Final[int] = 1_000
MAX_EVENT_COUNT: Final[int] = 100
MIN_DONORS: Final[int] = 2
MAX_DONORS: Final[int] = 12
MIN_CELLS_PER_DONOR: Final[int] = 3


def _require_exact_columns(
    frame: pd.DataFrame,
    expected: tuple[str, ...],
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("data must be provided as a pandas DataFrame")
    if frame.empty:
        raise ValueError("data must contain at least one cell")

    actual = list(frame.columns)
    missing = [column for column in expected if column not in actual]
    extra = [column for column in actual if column not in expected]
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing columns: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected columns: {', '.join(map(str, extra))}")
        raise ValueError(
            f"expected exactly {', '.join(expected)} ({'; '.join(details)})"
        )
    return frame.loc[:, list(expected)].copy()


def _clean_identifier(series: pd.Series, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{name} must be present for every row")
    cleaned = series.astype(str).str.strip()
    if cleaned.eq("").any():
        raise ValueError(f"{name} must be present for every row")
    return cleaned


def _clean_counts(series: pd.Series) -> pd.Series:
    contains_boolean = series.map(
        lambda value: isinstance(value, (bool, np.bool_)),
        na_action="ignore",
    ).eq(True).any()
    if pd.api.types.is_bool_dtype(series.dtype) or contains_boolean:
        raise ValueError("count must contain integers, not booleans")
    try:
        numeric = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("count must contain numeric integer values") from exc

    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    if not np.all(np.isfinite(values)):
        raise ValueError("count must contain only finite values")
    if np.any(values < 0):
        raise ValueError("count must be greater than or equal to zero")
    if np.any(values != np.floor(values)):
        raise ValueError("count must contain whole numbers; fractions are invalid")
    if np.any(values > np.iinfo(np.int64).max):
        raise ValueError("count is too large to store as a 64-bit integer")
    return pd.Series(values.astype(np.int64), index=series.index, name="count")


def _validate_unique_cells(frame: pd.DataFrame) -> None:
    duplicated = frame["cell_id"].duplicated(keep=False)
    if duplicated.any():
        duplicate = frame.loc[duplicated, "cell_id"].iloc[0]
        raise ValueError(f"cell_id values must be unique (duplicate: {duplicate})")


def _validate_demo_scope(frame: pd.DataFrame) -> None:
    """Apply bounded public-demo constraints before model construction."""

    if len(frame) < MIN_CELLS:
        raise ValueError(f"data must contain at least {MIN_CELLS} cells")
    if len(frame) > MAX_CELLS:
        raise ValueError(f"data may contain at most {MAX_CELLS:,} cells")
    if int(frame["count"].max()) > MAX_EVENT_COUNT:
        raise ValueError(
            f"count may not exceed {MAX_EVENT_COUNT} in this public demo"
        )
    if not frame["count"].gt(0).any():
        raise ValueError("data must contain at least one positive event count")


def validate_count_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized frame in the canonical ``cell_id,count`` schema.

    Counts must be finite, non-negative integers. Cell identifiers must be
    present and unique. The caller's frame is never modified.
    """

    validated = _require_exact_columns(frame, COUNT_COLUMNS)
    validated["cell_id"] = _clean_identifier(validated["cell_id"], "cell_id")
    validated["count"] = _clean_counts(validated["count"])
    _validate_unique_cells(validated)
    _validate_demo_scope(validated)
    return validated.reset_index(drop=True)


def validate_donor_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized canonical donor aware event count frame.

    In addition to the count-data rules, every row must have a donor label.
    For categorical donor columns, every declared category must be represented;
    this catches spreadsheets that specify donors but omit all cells for one.
    """

    validated = _require_exact_columns(frame, DONOR_COLUMNS)

    declared_donors: set[str] | None = None
    original_donor = validated["donor_id"]
    if isinstance(original_donor.dtype, pd.CategoricalDtype):
        declared_donors = {
            str(value).strip() for value in original_donor.cat.categories
        }

    validated["cell_id"] = _clean_identifier(validated["cell_id"], "cell_id")
    validated["donor_id"] = _clean_identifier(
        validated["donor_id"],
        "donor_id",
    )
    validated["count"] = _clean_counts(validated["count"])
    _validate_unique_cells(validated)

    if declared_donors is not None:
        observed = set(validated["donor_id"])
        missing = sorted(declared_donors - observed)
        if missing:
            raise ValueError(
                "every declared donor must have at least one cell; missing: "
                + ", ".join(missing)
            )
    if validated.groupby("donor_id", sort=False, observed=True).size().le(0).any():
        raise ValueError("every donor must have at least one cell")
    _validate_demo_scope(validated)

    donor_sizes = validated.groupby(
        "donor_id",
        sort=True,
        observed=True,
    ).size()
    donor_count = len(donor_sizes)
    if not MIN_DONORS <= donor_count <= MAX_DONORS:
        raise ValueError(
            f"donor aware data must contain {MIN_DONORS} to {MAX_DONORS} donors"
        )
    sparse = donor_sizes[donor_sizes < MIN_CELLS_PER_DONOR]
    if not sparse.empty:
        labels = ", ".join(map(str, sparse.index.tolist()))
        raise ValueError(
            f"each donor must have at least {MIN_CELLS_PER_DONOR} cells; "
            f"too few for: {labels}"
        )
    return validated.reset_index(drop=True)


def validate_observation_time(observation_time: float) -> float:
    """Validate the one global observation time used for every input cell."""

    try:
        value = float(observation_time)
    except (TypeError, ValueError) as exc:
        raise ValueError("observation_time must be a finite number greater than zero") from exc
    if not np.isfinite(value) or value <= 0:
        raise ValueError("observation_time must be a finite number greater than zero")
    return value


def sample_count_frame() -> pd.DataFrame:
    """Small built in donor ignorant dataset suitable for the demo editor."""

    return pd.DataFrame(
        {
            "cell_id": [f"cell_{index:03d}" for index in range(1, 13)],
            "count": [0, 1, 2, 1, 4, 0, 3, 2, 5, 1, 0, 3],
        }
    )


def sample_donor_frame() -> pd.DataFrame:
    """Small built-in dataset with three represented donor groups."""

    return pd.DataFrame(
        {
            "cell_id": [f"cell_{index:03d}" for index in range(1, 13)],
            "donor_id": [
                "donor_A",
                "donor_A",
                "donor_A",
                "donor_A",
                "donor_B",
                "donor_B",
                "donor_B",
                "donor_B",
                "donor_C",
                "donor_C",
                "donor_C",
                "donor_C",
            ],
            "count": [0, 1, 2, 1, 4, 0, 3, 2, 5, 1, 0, 3],
        }
    )
