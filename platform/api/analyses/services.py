from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256
from io import StringIO
import secrets
from typing import BinaryIO

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.files.base import ContentFile
from django.db import transaction
from django.utils import timezone
from rest_framework import exceptions

from .authentication import authenticate_guest_token, token_digest
from .models import AnalysisProject, Dataset, GuestSession, ProjectShareLink
from .storage import save_dataset_file


@dataclass(frozen=True)
class InspectedCsv:
    payload: bytes
    original_name: str
    content_type: str
    sha256: str
    row_count: int
    columns: tuple[str, ...]


def create_guest_session() -> tuple[GuestSession, str]:
    raw_token = f"barracuda_g_{secrets.token_urlsafe(32)}"
    now = timezone.now()
    guest = GuestSession.objects.create(
        token_digest=token_digest(raw_token),
        token_hint=raw_token[-8:],
        expires_at=now + timedelta(hours=settings.BARRACUDA_GUEST_RETENTION_HOURS),
    )
    return guest, raw_token


@transaction.atomic
def claim_guest_session(raw_token: str, user) -> int:
    if not getattr(user, "is_authenticated", False):
        raise exceptions.NotAuthenticated("An authenticated account is required.")
    guest = authenticate_guest_token(raw_token, touch=False)
    guest = GuestSession.objects.select_for_update().get(pk=guest.pk)
    now = timezone.now()
    if guest.revoked_at is not None or guest.expires_at <= now:
        raise exceptions.AuthenticationFailed("Guest session has expired.")
    count = AnalysisProject.objects.filter(owner_guest=guest).update(
        owner_guest=None,
        owner_user=user,
        expires_at=None,
    )
    guest.claimed_by = user
    guest.revoked_at = now
    guest.save(update_fields=("claimed_by", "revoked_at"))
    return count


def inspect_csv_upload(upload) -> InspectedCsv:
    max_bytes = settings.BARRACUDA_MAX_DATASET_BYTES
    try:
        upload.seek(0)
    except (AttributeError, OSError):
        pass
    payload = upload.read(max_bytes + 1)
    if not isinstance(payload, bytes):
        payload = bytes(payload)
    if not payload:
        raise exceptions.ValidationError({"file": "The CSV file is empty."})
    if len(payload) > max_bytes:
        raise exceptions.ValidationError(
            {"file": f"CSV files may contain at most {max_bytes:,} bytes."}
        )
    if b"\x00" in payload:
        raise exceptions.ValidationError({"file": "The CSV contains invalid NUL bytes."})
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise exceptions.ValidationError({"file": "The CSV must use UTF-8 encoding."}) from exc

    reader = csv.reader(StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
    except (StopIteration, csv.Error) as exc:
        raise exceptions.ValidationError({"file": "The CSV needs a header row."}) from exc
    columns = tuple(column.strip() for column in header)
    if not columns or any(not column for column in columns):
        raise exceptions.ValidationError({"file": "Every CSV column needs a name."})
    if len(columns) != len(set(columns)):
        raise exceptions.ValidationError({"file": "CSV column names must be unique."})
    if len(columns) > settings.BARRACUDA_MAX_DATASET_COLUMNS:
        raise exceptions.ValidationError(
            {"file": f"The CSV may contain at most {settings.BARRACUDA_MAX_DATASET_COLUMNS} columns."}
        )

    row_count = 0
    try:
        for row_count, row in enumerate(reader, start=1):
            if row_count > settings.BARRACUDA_MAX_DATASET_ROWS:
                raise exceptions.ValidationError(
                    {"file": f"The CSV may contain at most {settings.BARRACUDA_MAX_DATASET_ROWS:,} data rows."}
                )
            if len(row) != len(columns):
                raise exceptions.ValidationError(
                    {"file": f"CSV row {row_count + 1} has {len(row)} values; expected {len(columns)}."}
                )
    except csv.Error as exc:
        raise exceptions.ValidationError({"file": f"Malformed CSV: {exc}"}) from exc
    if row_count == 0:
        raise exceptions.ValidationError({"file": "The CSV needs at least one data row."})

    filename = str(getattr(upload, "name", "dataset.csv")).replace("\\", "/").split("/")[-1]
    if any(ord(character) < 32 for character in filename):
        raise exceptions.ValidationError({"file": "The filename contains invalid characters."})
    if not filename.lower().endswith(".csv"):
        raise exceptions.ValidationError({"file": "Upload a file with a .csv extension."})
    return InspectedCsv(
        payload=payload,
        original_name=filename[:255],
        content_type="text/csv",
        sha256=sha256(payload).hexdigest(),
        row_count=row_count,
        columns=columns,
    )


@transaction.atomic
def create_dataset(*, project: AnalysisProject, inspected: InspectedCsv) -> Dataset:
    dataset = Dataset(
        project=project,
        original_name=inspected.original_name,
        content_type=inspected.content_type,
        byte_size=len(inspected.payload),
        sha256=inspected.sha256,
        row_count=inspected.row_count,
        column_count=len(inspected.columns),
        columns=list(inspected.columns),
    )
    try:
        save_dataset_file(dataset, ContentFile(inspected.payload, name=inspected.original_name))
    except Exception:
        if dataset.file and dataset.file.name:
            dataset.file.storage.delete(dataset.file.name)
        raise
    return dataset


def create_share_link(
    *,
    project: AnalysisProject,
    creator,
    expires_in_hours: int | None,
    allow_dataset_download: bool,
) -> tuple[ProjectShareLink, str]:
    hours = expires_in_hours or settings.BARRACUDA_DEFAULT_SHARE_LINK_HOURS
    if hours < 1 or hours > settings.BARRACUDA_MAX_SHARE_LINK_HOURS:
        raise exceptions.ValidationError(
            {"expires_in_hours": f"Choose between 1 and {settings.BARRACUDA_MAX_SHARE_LINK_HOURS} hours."}
        )
    now = timezone.now()
    expires_at = now + timedelta(hours=hours)
    if project.expires_at is not None:
        if project.expires_at <= now:
            raise exceptions.ValidationError({"project_id": "This guest project has expired."})
        expires_at = min(expires_at, project.expires_at)
    raw_token = f"barracuda_s_{secrets.token_urlsafe(32)}"
    link = ProjectShareLink.objects.create(
        project=project,
        token_digest=token_digest(raw_token),
        token_hint=raw_token[-8:],
        allow_dataset_download=(
            bool(allow_dataset_download) if project.owner_user_id is not None else False
        ),
        created_by=(creator if isinstance(creator, get_user_model()) else None),
        expires_at=expires_at,
    )
    return link, raw_token


def resolve_share_link(raw_token: str) -> ProjectShareLink:
    if not raw_token or len(raw_token) > 200:
        raise exceptions.NotFound("Share link not found.")
    try:
        link = ProjectShareLink.objects.select_related("project").get(
            token_digest=token_digest(raw_token)
        )
    except ProjectShareLink.DoesNotExist as exc:
        raise exceptions.NotFound("Share link not found.") from exc
    now = timezone.now()
    if link.revoked_at is not None or link.expires_at <= now:
        raise exceptions.NotFound("Share link not found.")
    if link.project.expires_at is not None and link.project.expires_at <= now:
        raise exceptions.NotFound("Share link not found.")
    return link
