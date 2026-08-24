"""Task-facing analysis engine boundary.

The API never imports research modules directly. Deployments select an engine
class through ``BARRACUDA_ANALYSIS_ENGINE``. Local development uses the deterministic
mock engine; a future ``barracuda`` package can expose a runner callable through
``BarracudaAnalysisEngine`` without changing HTTP or persistence code.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from hashlib import sha256
import json
import logging
from pathlib import Path
import re
from typing import Any, Protocol

from django.conf import settings
from django.core.files.base import ContentFile
from django.db import transaction
from django.utils import timezone
from django.utils.module_loading import import_string

from .models import AnalysisArtifact, AnalysisJob
from .storage import open_dataset


ProgressCallback = Callable[[float | Mapping[str, Any]], None]
logger = logging.getLogger(__name__)


class AnalysisEngine(Protocol):
    def execute(
        self,
        *,
        analysis_type: str,
        dataset,
        configuration: Mapping[str, Any],
        progress: ProgressCallback,
    ) -> Mapping[str, Any]: ...


class MockAnalysisEngine:
    """Deterministic local engine used by API tests and UI integration."""

    def execute(self, *, analysis_type, dataset, configuration, progress):
        progress({"fraction": 0.25, "phase": "validation", "message": "Validating data"})
        progress({"fraction": 0.75, "phase": "sampling", "message": "Mock sampling"})
        return {
            "engine": "mock",
            "analysis_type": analysis_type,
            "dataset_id": str(dataset.pk),
            "dataset_sha256": dataset.sha256,
            "configuration": dict(configuration),
        }


class BarracudaAnalysisEngine:
    """Adapter for an installed ``barracuda`` service callable.

    ``BARRACUDA_RUNNER`` must resolve to a callable accepting keyword
    arguments ``analysis_type``, ``dataset_file``, ``configuration`` and
    ``progress``. The API supplies a binary stream, so local and object storage
    behave identically.
    """

    def execute(self, *, analysis_type, dataset, configuration, progress):
        runner_path = getattr(settings, "BARRACUDA_RUNNER", "")
        if not runner_path:
            raise RuntimeError("BARRACUDA_RUNNER is not configured")
        runner = import_string(runner_path)
        with open_dataset(dataset, "rb") as dataset_file:
            output = runner(
                analysis_type=analysis_type,
                dataset_file=dataset_file,
                configuration=dict(configuration),
                progress=progress,
            )
        if not isinstance(output, Mapping):
            raise TypeError("The barracuda runner must return a mapping")
        return dict(output)


def get_analysis_engine() -> AnalysisEngine:
    engine_class = import_string(settings.BARRACUDA_ANALYSIS_ENGINE)
    return engine_class()


_PROGRESS_KEYS = frozenset(
    {
        "phase",
        "message",
        "condition_index",
        "condition_total",
        "condition",
        "model_index",
        "model_total",
        "model",
        "chain",
        "stage",
        "beta",
    }
)
_ROLE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _set_progress(job_id, value: float | Mapping[str, Any]) -> None:
    detail: dict[str, Any] = {}
    fraction: Any = value
    if isinstance(value, Mapping):
        fraction = value.get("fraction", value.get("progress", 0.0))
        detail = {key: value[key] for key in _PROGRESS_KEYS if key in value}
        try:
            encoded = json.dumps(detail, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError):
            detail = {"phase": "sampling", "message": "PyMC sampling is running"}
        else:
            if len(encoded.encode("utf-8")) > 8_192:
                detail = {"phase": "sampling", "message": "PyMC sampling is running"}
    bounded = min(0.99, max(0.0, float(fraction)))
    AnalysisJob.objects.filter(
        pk=job_id,
        status=AnalysisJob.Status.RUNNING,
    ).update(progress=bounded, progress_detail=detail)


def _validated_result_and_artifacts(
    output: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result = dict(output)
    raw_artifacts = result.pop("_artifacts", [])
    if not isinstance(raw_artifacts, Sequence) or isinstance(raw_artifacts, (str, bytes)):
        raise TypeError("_artifacts must be a list")
    artifacts: list[dict[str, Any]] = []
    total_bytes = 0
    roles: set[str] = set()
    for raw in raw_artifacts:
        if not isinstance(raw, Mapping):
            raise TypeError("Every artifact must be a mapping")
        role = str(raw.get("role", ""))
        if not _ROLE_PATTERN.fullmatch(role) or role in roles:
            raise ValueError("Artifact roles must be unique safe identifiers")
        payload = raw.get("payload")
        if not isinstance(payload, bytes):
            raise TypeError("Artifact payloads must be bytes")
        filename = Path(str(raw.get("filename", "artifact.bin"))).name[:180]
        if not filename or any(ord(character) < 32 for character in filename):
            raise ValueError("Artifact filenames must be safe")
        content_type = str(raw.get("content_type", "application/octet-stream"))[:100]
        if "\n" in content_type or "\r" in content_type:
            raise ValueError("Artifact content types must be safe")
        total_bytes += len(payload)
        if len(payload) > settings.BARRACUDA_MAX_ARTIFACT_BYTES:
            raise ValueError("An analysis artifact exceeds the configured size limit")
        if total_bytes > settings.BARRACUDA_MAX_ARTIFACT_BYTES:
            raise ValueError("Analysis artifacts exceed the configured total size limit")
        roles.add(role)
        artifacts.append(
            {
                "role": role,
                "payload": payload,
                "filename": filename,
                "content_type": content_type,
                "shareable": bool(raw.get("shareable", False)),
            }
        )
    try:
        result_json = json.dumps(result, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise TypeError("Analysis result must be finite JSON data") from exc
    if len(result_json.encode("utf-8")) > settings.BARRACUDA_MAX_RESULT_JSON_BYTES:
        raise ValueError("Analysis result JSON exceeds the configured size limit")
    return result, artifacts


def _persist_artifacts(job: AnalysisJob, artifacts: Sequence[Mapping[str, Any]]) -> None:
    created: list[AnalysisArtifact] = []
    try:
        for item in artifacts:
            payload = item["payload"]
            artifact = AnalysisArtifact(
                job=job,
                role=item["role"],
                filename=item["filename"],
                content_type=item["content_type"],
                byte_size=len(payload),
                sha256=sha256(payload).hexdigest(),
                shareable=item["shareable"],
            )
            artifact.file.save(
                item["filename"],
                ContentFile(payload, name=item["filename"]),
                save=True,
            )
            created.append(artifact)
    except Exception:
        for artifact in created:
            artifact.delete()
        raise


def execute_analysis_job(job_id: str) -> dict[str, Any]:
    """Execute one queued record with transactional state transitions."""

    with transaction.atomic():
        job = (
            AnalysisJob.objects.select_for_update()
            .select_related("dataset", "project")
            .get(pk=job_id)
        )
        if job.status == AnalysisJob.Status.CANCELLED:
            return {"status": "cancelled"}
        if job.status != AnalysisJob.Status.QUEUED:
            return {"status": job.status}
        job.status = AnalysisJob.Status.RUNNING
        job.started_at = timezone.now()
        job.progress = 0.0
        job.progress_detail = {"phase": "queued", "message": "Starting analysis"}
        job.error_code = ""
        job.error_message = ""
        job.save(
            update_fields=(
                "status",
                "started_at",
                "progress",
                "progress_detail",
                "error_code",
                "error_message",
            )
        )

    try:
        output = get_analysis_engine().execute(
            analysis_type=job.analysis_type,
            dataset=job.dataset,
            configuration=job.configuration,
            progress=lambda value: _set_progress(job.pk, value),
        )
        if not isinstance(output, Mapping):
            raise TypeError("Analysis engines must return a mapping")
        result, artifacts = _validated_result_and_artifacts(output)
        _persist_artifacts(job, artifacts)
        with transaction.atomic():
            locked = AnalysisJob.objects.select_for_update().get(pk=job.pk)
            if locked.status == AnalysisJob.Status.CANCELLED:
                return {"status": "cancelled"}
            locked.status = AnalysisJob.Status.SUCCEEDED
            locked.progress = 1.0
            locked.progress_detail = {
                "phase": "complete",
                "message": "Inference and result preparation completed",
            }
            locked.result = result
            locked.completed_at = timezone.now()
            locked.save(
                update_fields=(
                    "status",
                    "progress",
                    "progress_detail",
                    "result",
                    "completed_at",
                )
            )
        return {"status": "succeeded", "result": result}
    except Exception as exc:
        logger.exception("Analysis job %s failed", job.pk)
        AnalysisArtifact.objects.filter(job_id=job.pk).delete()
        with transaction.atomic():
            locked = AnalysisJob.objects.select_for_update().get(pk=job.pk)
            locked.status = AnalysisJob.Status.FAILED
            locked.error_code = "analysis_execution_failed"
            locked.error_message = "Analysis execution failed. Contact support with the job ID."
            locked.progress_detail = {
                "phase": "failed",
                "message": "Inference did not complete",
            }
            locked.completed_at = timezone.now()
            locked.save(
                update_fields=(
                    "status",
                    "error_code",
                    "error_message",
                    "progress_detail",
                    "completed_at",
                )
            )
        raise
