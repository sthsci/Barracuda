"""Run the Orca count models independently across experimental conditions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from io import BytesIO
import re
from typing import Final
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd

from .conditions import split_condition_frame, validate_condition_frame
from .inference import (
    InferenceResult,
    InferenceSettings,
    build_results_zip,
    run_count_models,
    run_donor_models,
)


ConditionResults = dict[str, dict[str, InferenceResult]]
ConditionProgressCallback = Callable[[int, int, str, int, int, str], None]
ConditionSamplerProgressCallback = Callable[
    [int, int, str, int, int, str, int, int, float],
    None,
]
UINT32_MODULUS: Final[int] = 2**32


def _condition_seed(seed: int | None, index: int) -> int | None:
    if seed is None:
        return None
    return int((int(seed) + 104_729 * index) % UINT32_MODULUS)


def run_condition_models(
    frame: pd.DataFrame,
    observation_time: float,
    *,
    settings: InferenceSettings,
    model_keys: Sequence[str] | None,
    donor_aware: bool,
    progress_callback: ConditionProgressCallback | None = None,
    sampler_progress_callback: ConditionSamplerProgressCallback | None = None,
) -> ConditionResults:
    """Fit each condition separately with identical settings and model set.

    A fixed user seed remains reproducible, while a deterministic offset gives
    every independently fitted condition its own random stream.
    """

    if not isinstance(settings, InferenceSettings):
        raise TypeError("settings must be an InferenceSettings instance")
    groups = split_condition_frame(frame, donor_aware=donor_aware)
    runner = run_donor_models if donor_aware else run_count_models
    total_conditions = len(groups)
    output: ConditionResults = {}
    for condition_index, (condition, condition_frame) in enumerate(
        groups.items(),
        start=1,
    ):
        condition_settings = replace(
            settings,
            seed=_condition_seed(settings.seed, condition_index - 1),
        )

        def model_started(
            model_index: int,
            total_models: int,
            label: str,
            *,
            _condition_index: int = condition_index,
            _condition: str = condition,
        ) -> None:
            if progress_callback is not None:
                progress_callback(
                    _condition_index,
                    total_conditions,
                    _condition,
                    model_index,
                    total_models,
                    label,
                )

        def sampler_progress(
            model_index: int,
            total_models: int,
            label: str,
            chain: int,
            stage: int,
            beta: float,
            *,
            _condition_index: int = condition_index,
            _condition: str = condition,
        ) -> None:
            if sampler_progress_callback is not None:
                sampler_progress_callback(
                    _condition_index,
                    total_conditions,
                    _condition,
                    model_index,
                    total_models,
                    label,
                    int(chain),
                    int(stage),
                    float(beta),
                )

        output[condition] = runner(
            condition_frame,
            observation_time,
            settings=condition_settings,
            model_keys=model_keys,
            progress_callback=model_started,
            sampler_progress_callback=sampler_progress,
        )
    return output


def _safe_slug(label: str, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
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


def build_condition_results_zip(
    results: Mapping[str, Mapping[str, InferenceResult]],
    data: pd.DataFrame,
    observation_time: float,
    settings: InferenceSettings,
    *,
    donor_aware: bool,
) -> bytes:
    """Bundle one complete Orca result archive per experimental condition."""

    if not results:
        raise ValueError("at least one experimental condition is required")
    validated = validate_condition_frame(data, donor_aware=donor_aware)
    groups = split_condition_frame(validated, donor_aware=donor_aware)
    if list(results) != list(groups):
        raise ValueError("result conditions do not match the validated input data")

    buffer = BytesIO()
    used: set[str] = set()
    manifest_rows: list[dict[str, object]] = []
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        for index, (condition, group_results) in enumerate(results.items()):
            slug = _safe_slug(condition, used)
            condition_settings = replace(
                settings,
                seed=_condition_seed(settings.seed, index),
            )
            condition_archive = build_results_zip(
                group_results,
                groups[condition],
                observation_time,
                condition_settings,
            )
            _write_bytes(
                archive,
                f"conditions/{slug}/orca_results.zip",
                condition_archive,
            )
            manifest_rows.append(
                {
                    "condition": condition,
                    "folder": slug,
                    "cells": len(groups[condition]),
                    "models": ";".join(group_results),
                }
            )
        _write_bytes(
            archive,
            "condition_manifest.csv",
            pd.DataFrame(manifest_rows).to_csv(index=False).encode("utf-8"),
        )
        _write_bytes(
            archive,
            "README.txt",
            (
                "Orca multi-condition analysis\n\n"
                "Inference was run independently for each experimental condition with the "
                "same model and prior settings. Open the nested orca_results.zip "
                "inside each condition folder for evidence tables, posterior "
                "summaries and ArviZ NetCDF files.\n"
            ).encode("utf-8"),
        )
    return buffer.getvalue()
