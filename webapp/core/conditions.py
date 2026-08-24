"""Experimental-condition schemas shared by the count analysis pages.

The inference backends intentionally remain condition agnostic: every
experimental condition is fitted independently with the same model and prior
settings.  This module validates the outer table and then exposes one canonical
count table per condition.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import pandas as pd

from .data import (
    sample_count_frame,
    sample_donor_frame,
    validate_count_frame,
    validate_donor_frame,
)


CONDITION_COLUMN: Final[str] = "condition"
MAX_CONDITIONS: Final[int] = 4

# Apple system colours are familiar defaults on macOS.  Users can replace
# every value with the native colour input before running an analysis.
APPLE_COLOUR_PRESETS: Final[tuple[tuple[str, str], ...]] = (
    ("Blue", "#007AFF"),
    ("Purple", "#AF52DE"),
    ("Pink", "#FF2D55"),
    ("Red", "#FF3B30"),
    ("Orange", "#FF9500"),
    ("Yellow", "#FFCC00"),
    ("Green", "#34C759"),
    ("Teal", "#00C7BE"),
)


def condition_columns(*, donor_aware: bool) -> tuple[str, ...]:
    if donor_aware:
        return ("cell_id", "donor_id", CONDITION_COLUMN, "count")
    return ("cell_id", CONDITION_COLUMN, "count")


def default_condition_colours(labels: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Assign stable, high-contrast defaults in the supplied label order."""

    palette = [colour for _name, colour in APPLE_COLOUR_PRESETS]
    return {
        str(label): palette[index % len(palette)]
        for index, label in enumerate(labels)
    }


def _clean_condition_labels(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise ValueError("experimental condition must be present for every row")
    cleaned = series.astype(str).str.strip()
    if cleaned.eq("").any():
        raise ValueError("experimental condition must be present for every row")
    return cleaned


def validate_condition_frame(
    frame: pd.DataFrame,
    *,
    donor_aware: bool,
) -> pd.DataFrame:
    """Validate one to four independently fitted experimental conditions.

    Cell identifiers need only be unique *within* a condition.  This supports
    common tables in which each source file numbers cells from one again.
    Donor-aware validation is also applied independently to every condition,
    so each fitted hierarchy contains two to twelve represented donors.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("data must be provided as a pandas DataFrame")
    expected = list(condition_columns(donor_aware=donor_aware))
    missing = [column for column in expected if column not in frame.columns]
    extra = [column for column in frame.columns if column not in expected]
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing columns: " + ", ".join(missing))
        if extra:
            details.append("unexpected columns: " + ", ".join(map(str, extra)))
        raise ValueError(
            f"expected exactly {', '.join(expected)} ({'; '.join(details)})"
        )
    if frame.empty:
        raise ValueError("data must contain at least one cell")

    working = frame.loc[:, expected].copy()
    working[CONDITION_COLUMN] = _clean_condition_labels(
        working[CONDITION_COLUMN]
    )
    labels = list(dict.fromkeys(working[CONDITION_COLUMN].tolist()))
    if len(labels) > MAX_CONDITIONS:
        raise ValueError(
            f"this release supports at most {MAX_CONDITIONS} experimental conditions"
        )

    validator = validate_donor_frame if donor_aware else validate_count_frame
    base_columns = (
        ["cell_id", "donor_id", "count"]
        if donor_aware
        else ["cell_id", "count"]
    )
    validated_groups: list[pd.DataFrame] = []
    for label in labels:
        group = working.loc[
            working[CONDITION_COLUMN] == label,
            base_columns,
        ]
        try:
            validated = validator(group)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"condition {label!r}: {exc}") from exc
        validated.insert(
            2 if donor_aware else 1,
            CONDITION_COLUMN,
            label,
        )
        validated_groups.append(validated)
    return pd.concat(validated_groups, ignore_index=True).loc[:, expected]


def split_condition_frame(
    frame: pd.DataFrame,
    *,
    donor_aware: bool,
) -> dict[str, pd.DataFrame]:
    """Return canonical backend frames in stable experimental-group order."""

    validated = validate_condition_frame(frame, donor_aware=donor_aware)
    base_columns = (
        ["cell_id", "donor_id", "count"]
        if donor_aware
        else ["cell_id", "count"]
    )
    return {
        str(label): group.loc[:, base_columns].reset_index(drop=True)
        for label, group in validated.groupby(
            CONDITION_COLUMN,
            sort=False,
            observed=True,
        )
    }


def normalize_condition_frame(
    raw: pd.DataFrame,
    *,
    donor_aware: bool,
) -> tuple[pd.DataFrame, str]:
    """Map a user CSV to the public condition-aware schema.

    Named canonical columns are preferred.  A legacy table without a
    condition column is accepted and placed in ``Group 1``.
    """

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("The uploaded CSV is empty.")

    aliases = {
        "cell": "cell_id",
        "cellid": "cell_id",
        "cell_id": "cell_id",
        "donor": "donor_id",
        "donorid": "donor_id",
        "donor_id": "donor_id",
        "group": CONDITION_COLUMN,
        "condition": CONDITION_COLUMN,
        "treatment": CONDITION_COLUMN,
        "experimental_group": CONDITION_COLUMN,
        "experimental_condition": CONDITION_COLUMN,
        "count": "count",
        "counts": "count",
        "event_count": "count",
    }
    renamed: dict[object, str] = {}
    for column in raw.columns:
        normalized = (
            str(column)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        if normalized in aliases and aliases[normalized] not in renamed.values():
            renamed[column] = aliases[normalized]
    named = raw.rename(columns=renamed)
    base_required = ["cell_id", "donor_id", "count"] if donor_aware else ["cell_id", "count"]
    if all(column in named.columns for column in base_required):
        mapped = named.loc[:, base_required].copy()
        if CONDITION_COLUMN in named.columns:
            insert_at = 2 if donor_aware else 1
            mapped.insert(insert_at, CONDITION_COLUMN, named[CONDITION_COLUMN])
            message = "Recognised the Barracuda cell, condition and count columns."
        else:
            insert_at = 2 if donor_aware else 1
            mapped.insert(insert_at, CONDITION_COLUMN, "Group 1")
            message = "No condition column was found, so all rows were assigned to Group 1."
        return mapped.loc[:, condition_columns(donor_aware=donor_aware)], message

    columns = list(raw.columns)
    if donor_aware:
        if len(columns) >= 4:
            mapped = pd.DataFrame(
                {
                    "cell_id": raw.iloc[:, 0],
                    "donor_id": raw.iloc[:, 1],
                    CONDITION_COLUMN: raw.iloc[:, 2],
                    "count": raw.iloc[:, 3],
                }
            )
            message = "Mapped the first four columns to cell, donor, condition and count."
        elif len(columns) >= 3:
            mapped = pd.DataFrame(
                {
                    "cell_id": raw.iloc[:, 0],
                    "donor_id": raw.iloc[:, 1],
                    CONDITION_COLUMN: "Group 1",
                    "count": raw.iloc[:, 2],
                }
            )
            message = "Mapped the first three columns and assigned all rows to Group 1."
        else:
            raise ValueError(
                "The CSV needs cell_id, donor_id and count columns, with an optional condition column."
            )
    else:
        if len(columns) >= 3:
            mapped = pd.DataFrame(
                {
                    "cell_id": raw.iloc[:, 0],
                    CONDITION_COLUMN: raw.iloc[:, 1],
                    "count": raw.iloc[:, 2],
                }
            )
            message = "Mapped the first three columns to cell, condition and count."
        elif len(columns) >= 2:
            mapped = pd.DataFrame(
                {
                    "cell_id": raw.iloc[:, 0],
                    CONDITION_COLUMN: "Group 1",
                    "count": raw.iloc[:, 1],
                }
            )
            message = "Mapped the first two columns and assigned all rows to Group 1."
        else:
            raise ValueError(
                "The CSV needs cell_id and count columns, with an optional condition column."
            )
    return mapped.loc[:, condition_columns(donor_aware=donor_aware)], message


def sample_condition_frame(*, donor_aware: bool) -> pd.DataFrame:
    """Return a small two-condition table for the editor and example mode."""

    base = sample_donor_frame() if donor_aware else sample_count_frame()
    control = base.copy()
    control["cell_id"] = "control_" + control["cell_id"].astype(str)
    control.insert(2 if donor_aware else 1, CONDITION_COLUMN, "Control")

    treatment = base.copy()
    treatment["cell_id"] = "treatment_" + treatment["cell_id"].astype(str)
    treatment["count"] = (
        treatment["count"].astype(int)
        + ([0, 1, 0, 1] * ((len(treatment) + 3) // 4))[: len(treatment)]
    )
    treatment.insert(2 if donor_aware else 1, CONDITION_COLUMN, "Treatment")
    return validate_condition_frame(
        pd.concat([control, treatment], ignore_index=True),
        donor_aware=donor_aware,
    )


def sanitize_condition_colours(
    labels: list[str] | tuple[str, ...],
    values: Mapping[str, object] | None,
) -> dict[str, str]:
    """Return valid ``#RRGGBB`` colours, falling back to stable defaults."""

    defaults = default_condition_colours(labels)
    supplied = values or {}
    result: dict[str, str] = {}
    for label in labels:
        value = str(supplied.get(label, "")).strip()
        if (
            len(value) == 7
            and value.startswith("#")
            and all(character in "0123456789abcdefABCDEF" for character in value[1:])
        ):
            result[label] = value.upper()
        else:
            result[label] = defaults[label]
    return result
