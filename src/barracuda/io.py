"""Reproducible persistence helpers for inference and validation outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Final
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import pandas as pd


SCAN_SCHEMA_VERSION: Final[int] = 1


@dataclass(frozen=True)
class ScanBundle:
    """A validated scan table and the manifest that describes it."""

    table: pd.DataFrame
    manifest: dict[str, Any]
    directory: Path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def canonical_json(payload: Any) -> str:
    """Serialize configuration data deterministically for hashing."""

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    )


def configuration_fingerprint(payload: Any) -> str:
    """Return a SHA-256 fingerprint for JSON-compatible configuration data."""

    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def dataframe_checksum(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
) -> str:
    """Hash a table's schema, row order, values, and index-independent CSV.

    A selected column order may be supplied when only the scientific input
    fields (rather than incidental report columns) should be covered.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if columns is None:
        selected = frame
    else:
        missing = [str(column) for column in columns if column not in frame]
        if missing:
            raise ValueError("frame is missing checksum columns: " + ", ".join(missing))
        selected = frame.loc[:, list(columns)]
    csv_payload = selected.to_csv(
        index=False,
        lineterminator="\n",
        float_format="%.17g",
    )
    schema = [
        {
            "name": str(column),
            "name_type": f"{type(column).__module__}.{type(column).__qualname__}",
            "dtype": repr(selected[column].dtype),
        }
        for column in selected.columns
    ]
    payload = (
        canonical_json({"columns": schema}).encode("utf-8")
        + b"\n"
        + csv_payload.encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def _atomic_bytes(path: Path, content: bytes, *, overwrite: bool) -> Path:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {destination}")
    with NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                # Linking a complete same-directory temporary file publishes
                # it atomically and fails if a concurrent writer won the name.
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise FileExistsError(f"refusing to overwrite {destination}") from exc
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def save_inference_data(
    idata: Any,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically save an ArviZ ``InferenceData`` object as NetCDF."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {destination}")
    with NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.stem}.",
        suffix=destination.suffix or ".nc",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        import arviz as az

        az.to_netcdf(idata, str(temporary))
        if overwrite:
            os.replace(temporary, destination)
        else:
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise FileExistsError(f"refusing to overwrite {destination}") from exc
            temporary.unlink()
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_inference_data(path: str | Path):
    """Load an ArviZ ``InferenceData`` NetCDF written by BARRACUDA."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    import arviz as az

    return az.from_netcdf(str(source))


def save_scan_bundle(
    table: pd.DataFrame,
    directory: str | Path,
    *,
    configuration: Mapping[str, Any],
    overwrite: bool = False,
) -> ScanBundle:
    """Persist a scan CSV and fingerprinted manifest with atomic file writes.

    Existing files are never silently reused.  Load them with
    :func:`load_scan_bundle` and an ``expected_configuration`` to verify a
    resume request before deciding whether additional inference is required.
    """

    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame")
    if not isinstance(configuration, Mapping):
        raise TypeError("configuration must be a mapping")
    root = Path(directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    table_path = root / "scan_results.csv"
    manifest_path = root / "scan_manifest.json"
    conflicts = [path for path in (table_path, manifest_path) if path.exists()]
    if conflicts and not overwrite:
        raise FileExistsError(
            "refusing to overwrite existing scan bundle: "
            + ", ".join(str(path) for path in conflicts)
        )
    csv = table.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    config = dict(configuration)
    manifest = {
        "schema_version": SCAN_SCHEMA_VERSION,
        "configuration": config,
        "configuration_fingerprint": configuration_fingerprint(config),
        "results_checksum": hashlib.sha256(csv.encode("utf-8")).hexdigest(),
        "n_rows": int(len(table)),
        "columns": [str(column) for column in table.columns],
    }
    _atomic_bytes(table_path, csv.encode("utf-8"), overwrite=overwrite)
    _atomic_bytes(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n").encode(
            "utf-8"
        ),
        overwrite=overwrite,
    )
    return ScanBundle(table=table.copy(), manifest=manifest, directory=root)


def load_scan_bundle(
    directory: str | Path,
    *,
    expected_configuration: Mapping[str, Any] | None = None,
    verify: bool = True,
) -> ScanBundle:
    """Load and optionally verify a saved scan bundle."""

    root = Path(directory).expanduser().resolve()
    table_path = root / "scan_results.csv"
    manifest_path = root / "scan_manifest.json"
    if not table_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"incomplete scan bundle at {root}; expected scan_results.csv and scan_manifest.json"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if verify:
        if manifest.get("schema_version") != SCAN_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported scan schema version {manifest.get('schema_version')!r}"
            )
        raw = table_path.read_bytes()
        actual_checksum = hashlib.sha256(raw).hexdigest()
        if actual_checksum != manifest.get("results_checksum"):
            raise ValueError("scan result checksum does not match the manifest")
        actual_fingerprint = configuration_fingerprint(manifest.get("configuration", {}))
        if actual_fingerprint != manifest.get("configuration_fingerprint"):
            raise ValueError("scan configuration fingerprint does not match the manifest")
    if expected_configuration is not None:
        expected = configuration_fingerprint(dict(expected_configuration))
        observed = manifest.get("configuration_fingerprint")
        if expected != observed:
            raise ValueError(
                "saved scan configuration does not match the requested configuration"
            )
    table = pd.read_csv(table_path)
    if verify:
        if len(table) != int(manifest.get("n_rows", -1)):
            raise ValueError("scan row count does not match the manifest")
        if list(map(str, table.columns)) != list(manifest.get("columns", [])):
            raise ValueError("scan columns do not match the manifest")
    return ScanBundle(table=table, manifest=manifest, directory=root)


def _zip_write(archive: ZipFile, name: str, content: bytes) -> None:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    archive.writestr(info, content)


def build_scan_archive(
    table: pd.DataFrame,
    *,
    configuration: Mapping[str, Any],
    recovery: pd.DataFrame | None = None,
    artifacts: Mapping[str, bytes] | None = None,
) -> bytes:
    """Build a deterministic portable ZIP for a Bayes-factor scan."""

    if not isinstance(table, pd.DataFrame):
        raise TypeError("table must be a pandas DataFrame")
    if recovery is not None and not isinstance(recovery, pd.DataFrame):
        raise TypeError("recovery must be a pandas DataFrame or None")
    csv = table.to_csv(index=False, lineterminator="\n", float_format="%.17g")
    config = dict(configuration)
    manifest = {
        "schema_version": SCAN_SCHEMA_VERSION,
        "configuration": config,
        "configuration_fingerprint": configuration_fingerprint(config),
        "results_checksum": hashlib.sha256(csv.encode("utf-8")).hexdigest(),
        "n_rows": int(len(table)),
        "columns": [str(column) for column in table.columns],
    }
    buffer = BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        _zip_write(archive, "scan_results.csv", csv.encode("utf-8"))
        _zip_write(
            archive,
            "scan_manifest.json",
            (json.dumps(manifest, indent=2, sort_keys=True, default=_json_default) + "\n").encode(
                "utf-8"
            ),
        )
        if recovery is not None:
            _zip_write(
                archive,
                "parameter_recovery.csv",
                recovery.to_csv(index=False, lineterminator="\n").encode("utf-8"),
            )
        normalized_artifacts: list[tuple[str, bytes]] = []
        for raw_name, content in (artifacts or {}).items():
            name = str(raw_name).replace("\\", "/").lstrip("/")
            if not name or ".." in Path(name).parts:
                raise ValueError(f"unsafe artifact name: {raw_name!r}")
            normalized_artifacts.append((name, bytes(content)))
        normalized_names = [name for name, _content in normalized_artifacts]
        if len(normalized_names) != len(set(normalized_names)):
            raise ValueError("artifact names must be unique after path normalization")
        for name, content in sorted(normalized_artifacts, key=lambda item: item[0]):
            _zip_write(archive, f"artifacts/{name}", bytes(content))
        _zip_write(
            archive,
            "README.txt",
            (
                "barracuda Bayes-factor scan\n\n"
                "scan_results.csv is the tidy long-form scan table.\n"
                "scan_manifest.json records the exact configuration and checksums.\n"
                "Adjacent sample sizes may be cumulative prefixes; consult the configuration.\n"
            ).encode("utf-8"),
        )
    return buffer.getvalue()


__all__ = [
    "SCAN_SCHEMA_VERSION",
    "ScanBundle",
    "build_scan_archive",
    "canonical_json",
    "configuration_fingerprint",
    "dataframe_checksum",
    "load_inference_data",
    "load_scan_bundle",
    "save_inference_data",
    "save_scan_bundle",
]
