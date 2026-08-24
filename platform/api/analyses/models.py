from __future__ import annotations

import uuid
from pathlib import Path

from django.conf import settings
from django.core.validators import MinValueValidator
from django.db import models
from django.db.models import Q

from .storage import artifact_storage, dataset_storage, dataset_upload_to


class GuestSession(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    token_digest = models.CharField(max_length=64, unique=True, editable=False)
    token_hint = models.CharField(max_length=16, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    last_seen_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(db_index=True)
    revoked_at = models.DateTimeField(null=True, blank=True, db_index=True)
    claimed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="claimed_guest_sessions",
    )

    class Meta:
        ordering = ("-created_at",)
        indexes = [models.Index(fields=("expires_at", "revoked_at"))]

    def __str__(self) -> str:
        return f"Guest {self.token_hint}"


class AnalysisProject(models.Model):
    class Visibility(models.TextChoices):
        PRIVATE = "private", "Private"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=120)
    description = models.TextField(blank=True, max_length=2_000)
    owner_user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="analysis_projects",
    )
    owner_guest = models.ForeignKey(
        GuestSession,
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name="projects",
    )
    visibility = models.CharField(
        max_length=16,
        choices=Visibility.choices,
        default=Visibility.PRIVATE,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    expires_at = models.DateTimeField(null=True, blank=True, db_index=True)

    class Meta:
        ordering = ("-updated_at",)
        constraints = [
            models.CheckConstraint(
                condition=(
                    (Q(owner_user__isnull=False) & Q(owner_guest__isnull=True))
                    | (Q(owner_user__isnull=True) & Q(owner_guest__isnull=False))
                ),
                name="project_has_exactly_one_owner",
            )
        ]

    @property
    def is_guest_owned(self) -> bool:
        return self.owner_guest_id is not None

    def __str__(self) -> str:
        return self.name


class Dataset(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        AnalysisProject,
        on_delete=models.CASCADE,
        related_name="datasets",
    )
    file = models.FileField(storage=dataset_storage, upload_to=dataset_upload_to)
    original_name = models.CharField(max_length=255)
    content_type = models.CharField(max_length=100, default="text/csv")
    byte_size = models.PositiveBigIntegerField()
    sha256 = models.CharField(max_length=64, db_index=True)
    row_count = models.PositiveIntegerField(validators=[MinValueValidator(1)])
    column_count = models.PositiveSmallIntegerField(validators=[MinValueValidator(1)])
    columns = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("-created_at",)
        indexes = [models.Index(fields=("project", "created_at"))]

    def __str__(self) -> str:
        return self.original_name


class AnalysisJob(models.Model):
    class AnalysisType(models.TextChoices):
        EVENT_COUNT_DONOR_IGNORANT = (
            "event_count_donor_ignorant",
            "Event count, donor ignorant",
        )
        EVENT_COUNT_DONOR_AWARE = (
            "event_count_donor_aware",
            "Event count, donor aware",
        )
        TRAJECTORY_DONOR_IGNORANT = (
            "trajectory_donor_ignorant",
            "Trajectory, donor ignorant",
        )

    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        RUNNING = "running", "Running"
        SUCCEEDED = "succeeded", "Succeeded"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        AnalysisProject,
        on_delete=models.CASCADE,
        related_name="jobs",
    )
    dataset = models.ForeignKey(
        Dataset,
        on_delete=models.CASCADE,
        related_name="jobs",
    )
    analysis_type = models.CharField(max_length=40, choices=AnalysisType.choices)
    configuration = models.JSONField(default=dict)
    status = models.CharField(
        max_length=16,
        choices=Status.choices,
        default=Status.QUEUED,
        db_index=True,
    )
    progress = models.FloatField(default=0.0)
    progress_detail = models.JSONField(default=dict, blank=True)
    result = models.JSONField(null=True, blank=True)
    error_code = models.CharField(max_length=80, blank=True)
    error_message = models.TextField(blank=True, max_length=2_000)
    idempotency_key = models.CharField(max_length=128, null=True, blank=True)
    task_id = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ("-created_at",)
        indexes = [
            models.Index(fields=("project", "created_at")),
            models.Index(fields=("status", "created_at")),
        ]
        constraints = [
            models.UniqueConstraint(
                fields=("project", "idempotency_key"),
                condition=Q(idempotency_key__isnull=False),
                name="unique_job_idempotency_key_per_project",
            ),
            models.CheckConstraint(
                condition=Q(progress__gte=0.0) & Q(progress__lte=1.0),
                name="job_progress_between_zero_and_one",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.analysis_type}: {self.status}"


def artifact_upload_to(instance, filename: str) -> str:
    suffix = Path(filename).suffix.lower() or ".bin"
    return f"projects/{instance.job.project_id}/jobs/{instance.job_id}/artifacts/{instance.id}{suffix}"


class AnalysisArtifact(models.Model):
    """One verified downloadable product from a completed analysis job."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(
        AnalysisJob,
        on_delete=models.CASCADE,
        related_name="artifacts",
    )
    role = models.SlugField(max_length=64)
    file = models.FileField(storage=artifact_storage, upload_to=artifact_upload_to)
    filename = models.CharField(max_length=180)
    content_type = models.CharField(max_length=100, default="application/octet-stream")
    byte_size = models.PositiveBigIntegerField()
    sha256 = models.CharField(max_length=64)
    shareable = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ("created_at",)
        constraints = [
            models.UniqueConstraint(fields=("job", "role"), name="unique_artifact_role_per_job")
        ]

    def __str__(self) -> str:
        return f"{self.role} for {self.job_id}"


class ProjectShareLink(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    project = models.ForeignKey(
        AnalysisProject,
        on_delete=models.CASCADE,
        related_name="share_links",
    )
    token_digest = models.CharField(max_length=64, unique=True, editable=False)
    token_hint = models.CharField(max_length=16, editable=False)
    allow_dataset_download = models.BooleanField(default=False)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="created_project_share_links",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(db_index=True)
    revoked_at = models.DateTimeField(null=True, blank=True, db_index=True)

    class Meta:
        ordering = ("-created_at",)
        indexes = [models.Index(fields=("expires_at", "revoked_at"))]

    def __str__(self) -> str:
        return f"Share {self.token_hint} for {self.project_id}"
