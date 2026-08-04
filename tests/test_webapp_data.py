from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from webapp.core.data import (
    sample_count_frame,
    sample_donor_frame,
    validate_count_frame,
    validate_donor_frame,
    validate_observation_time,
)


def test_sample_frames_follow_canonical_schemas() -> None:
    counts = validate_count_frame(sample_count_frame())
    donors = validate_donor_frame(sample_donor_frame())

    assert list(counts) == ["cell_id", "count"]
    assert list(donors) == ["cell_id", "donor_id", "count"]
    assert counts["count"].dtype == np.dtype("int64")
    assert donors["donor_id"].nunique() == 3


def test_validation_normalizes_identifiers_and_numeric_strings_without_mutation() -> None:
    source = pd.DataFrame(
        {
            "cell_id": [" a ", "b", "c", "d", "e"],
            "count": ["0", "2", "1", "0", "3"],
        }
    )
    result = validate_count_frame(source)

    assert result.to_dict("list") == {
        "cell_id": ["a", "b", "c", "d", "e"],
        "count": [0, 2, 1, 0, 3],
    }
    assert source.loc[0, "cell_id"] == " a "


@pytest.mark.parametrize(
    "bad_count, message",
    [
        (-1, "greater than or equal"),
        (1.5, "fractions"),
        (np.inf, "finite"),
        (np.nan, "finite"),
        ("not-a-number", "numeric"),
        (True, "booleans"),
    ],
)
def test_count_validation_rejects_invalid_counts(bad_count, message: str) -> None:
    frame = pd.DataFrame({"cell_id": ["cell_1"], "count": [bad_count]})
    with pytest.raises(ValueError, match=message):
        validate_count_frame(frame)


def test_validation_rejects_empty_duplicate_and_noncanonical_frames() -> None:
    with pytest.raises(ValueError, match="at least one"):
        validate_count_frame(pd.DataFrame(columns=["cell_id", "count"]))
    with pytest.raises(ValueError, match="unique"):
        validate_count_frame(
            pd.DataFrame({"cell_id": ["same", "same"], "count": [0, 1]})
        )
    with pytest.raises(ValueError, match="unexpected columns"):
        validate_count_frame(
            pd.DataFrame(
                {"cell_id": ["cell_1"], "count": [1], "time": [2.0]}
            )
        )


def test_donor_validation_requires_every_label_and_declared_category() -> None:
    missing_label = pd.DataFrame(
        {"cell_id": ["a", "b"], "donor_id": ["D1", None], "count": [0, 1]}
    )
    with pytest.raises(ValueError, match="every row"):
        validate_donor_frame(missing_label)

    categorical = pd.DataFrame(
        {
            "cell_id": ["a", "b"],
            "donor_id": pd.Categorical(["D1", "D1"], categories=["D1", "D2"]),
            "count": [0, 1],
        }
    )
    with pytest.raises(ValueError, match="missing: D2"):
        validate_donor_frame(categorical)


def test_demo_scope_rejects_too_few_too_many_all_zero_and_large_counts() -> None:
    with pytest.raises(ValueError, match="at least 5"):
        validate_count_frame(
            pd.DataFrame(
                {"cell_id": [f"c{i}" for i in range(4)], "count": [0, 1, 0, 1]}
            )
        )

    with pytest.raises(ValueError, match="at most 1,000"):
        validate_count_frame(
            pd.DataFrame(
                {
                    "cell_id": [f"c{i}" for i in range(1_001)],
                    "count": [1] * 1_001,
                }
            )
        )

    with pytest.raises(ValueError, match="positive event"):
        validate_count_frame(
            pd.DataFrame(
                {"cell_id": [f"c{i}" for i in range(5)], "count": [0] * 5}
            )
        )

    with pytest.raises(ValueError, match="may not exceed 100"):
        validate_count_frame(
            pd.DataFrame(
                {"cell_id": [f"c{i}" for i in range(5)], "count": [101, 1, 1, 1, 1]}
            )
        )


def test_donor_scope_requires_multiple_well_represented_donors() -> None:
    one_donor = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(5)],
            "donor_id": ["D1"] * 5,
            "count": [0, 1, 2, 1, 0],
        }
    )
    with pytest.raises(ValueError, match="2 to 12 donors"):
        validate_donor_frame(one_donor)

    sparse_donor = pd.DataFrame(
        {
            "cell_id": [f"c{i}" for i in range(6)],
            "donor_id": ["D1", "D1", "D2", "D2", "D2", "D2"],
            "count": [0, 1, 0, 1, 2, 1],
        }
    )
    with pytest.raises(ValueError, match="at least 3 cells"):
        validate_donor_frame(sparse_donor)


@pytest.mark.parametrize("value", [0, -1, np.inf, np.nan, "none"])
def test_observation_time_must_be_globally_positive_and_finite(value) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        validate_observation_time(value)


def test_observation_time_is_normalized_to_float() -> None:
    assert validate_observation_time("2.5") == 2.5
