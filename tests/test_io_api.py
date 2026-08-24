from __future__ import annotations

from io import BytesIO
import json
from zipfile import ZipFile

import arviz as az
import numpy as np
import pandas as pd
import pytest

from barracuda import (
    build_scan_archive,
    canonical_json,
    configuration_fingerprint,
    dataframe_checksum,
    load_inference_data,
    load_scan_bundle,
    save_inference_data,
    save_scan_bundle,
)


def test_configuration_fingerprints_ignore_mapping_order():
    first = {"draws": 256, "models": ["homo", "dis2p"]}
    second = {"models": ["homo", "dis2p"], "draws": np.int64(256)}
    assert canonical_json(first) == canonical_json(second)
    assert configuration_fingerprint(first) == configuration_fingerprint(second)


def test_dataframe_checksum_covers_row_and_column_order():
    frame = pd.DataFrame({"cell_id": ["a", "b"], "count": [0, 2]})
    assert dataframe_checksum(frame) == dataframe_checksum(frame.copy())
    assert dataframe_checksum(frame) != dataframe_checksum(frame.iloc[::-1])
    assert dataframe_checksum(frame) != dataframe_checksum(frame[["count", "cell_id"]])
    same_csv_different_schema = frame.astype({"count": "string"})
    assert frame.to_csv(index=False) == same_csv_different_schema.to_csv(index=False)
    assert dataframe_checksum(frame) != dataframe_checksum(same_csv_different_schema)


def test_scan_bundle_round_trip_and_configuration_guard(tmp_path):
    table = pd.DataFrame(
        {
            "scenario": ["S"],
            "n_cells": [20],
            "model_key": ["homo"],
            "log_evidence": [-4.5],
        }
    )
    config = {"sample_sizes": [20], "replicates": 1}
    saved = save_scan_bundle(table, tmp_path / "scan", configuration=config)
    assert saved.manifest["n_rows"] == 1
    loaded = load_scan_bundle(
        tmp_path / "scan",
        expected_configuration=config,
    )
    pd.testing.assert_frame_equal(loaded.table, table)

    with pytest.raises(ValueError, match="does not match"):
        load_scan_bundle(
            tmp_path / "scan",
            expected_configuration={"sample_sizes": [30], "replicates": 1},
        )
    with pytest.raises(FileExistsError):
        save_scan_bundle(table, tmp_path / "scan", configuration=config)


def test_scan_bundle_detects_modified_results(tmp_path):
    table = pd.DataFrame({"model_key": ["homo"], "log_evidence": [-2.0]})
    root = tmp_path / "scan"
    save_scan_bundle(table, root, configuration={"seed": 1})
    (root / "scan_results.csv").write_text("model_key,log_evidence\nhomo,-3\n")
    with pytest.raises(ValueError, match="checksum"):
        load_scan_bundle(root)


def test_inference_data_atomic_round_trip(tmp_path):
    idata = az.from_dict(posterior={"theta": np.arange(12).reshape(2, 6)})
    path = save_inference_data(idata, tmp_path / "posterior.nc")
    loaded = load_inference_data(path)
    np.testing.assert_array_equal(loaded.posterior["theta"], idata.posterior["theta"])
    with pytest.raises(FileExistsError):
        save_inference_data(idata, path)
    save_inference_data(idata, path, overwrite=True)


def test_scan_archive_is_deterministic_and_rejects_unsafe_artifacts():
    table = pd.DataFrame({"model_key": ["homo"], "log_evidence": [-2.0]})
    first = build_scan_archive(table, configuration={"seed": 9})
    second = build_scan_archive(table, configuration={"seed": 9})
    assert first == second
    with ZipFile(BytesIO(first)) as archive:
        assert set(archive.namelist()) == {
            "README.txt",
            "scan_manifest.json",
            "scan_results.csv",
        }
        manifest = json.loads(archive.read("scan_manifest.json"))
        assert manifest["configuration"] == {"seed": 9}
    with pytest.raises(ValueError, match="unsafe"):
        build_scan_archive(
            table,
            configuration={"seed": 9},
            artifacts={"../secret.txt": b"no"},
        )


def test_scan_archive_sorts_artifacts_and_rejects_normalized_duplicates():
    table = pd.DataFrame({"model_key": ["homo"], "log_evidence": [-2.0]})
    first = build_scan_archive(
        table,
        configuration={"seed": 9},
        artifacts={"z.txt": b"z", "nested/a.txt": b"a"},
    )
    second = build_scan_archive(
        table,
        configuration={"seed": 9},
        artifacts={"nested/a.txt": b"a", "z.txt": b"z"},
    )
    assert first == second
    with pytest.raises(ValueError, match="unique after path normalization"):
        build_scan_archive(
            table,
            configuration={"seed": 9},
            artifacts={"nested\\a.txt": b"first", "nested/a.txt": b"second"},
        )
