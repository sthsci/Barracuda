"""Storage indirection for private uploaded datasets."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from django.core.files.base import File
from django.core.files.storage import Storage, storages
from django.db.models.fields.files import FieldFile


class AliasStorage(Storage):
    """Resolve a Django storage alias for every operation.

    ``FileField`` evaluates a plain storage callable when Django imports the
    model.  That pins the resulting backend and makes test overrides (and a
    few deployment-time settings reloads) silently ineffective.  This small
    proxy keeps the field stable while resolving the configured alias at the
    point of use.
    """

    def __init__(self, alias: str) -> None:
        self.alias = alias

    @property
    def backend(self) -> Storage:
        return storages[self.alias]

    def _open(self, name, mode="rb"):
        return self.backend.open(name, mode)

    def _save(self, name, content):
        return self.backend.save(name, content)

    def delete(self, name):
        return self.backend.delete(name)

    def exists(self, name):
        return self.backend.exists(name)

    def listdir(self, path):
        return self.backend.listdir(path)

    def size(self, name):
        return self.backend.size(name)

    def url(self, name):
        return self.backend.url(name)

    def path(self, name):
        return self.backend.path(name)

    def get_accessed_time(self, name):
        return self.backend.get_accessed_time(name)

    def get_created_time(self, name):
        return self.backend.get_created_time(name)

    def get_modified_time(self, name):
        return self.backend.get_modified_time(name)

    def deconstruct(self):
        return (f"{self.__class__.__module__}.{self.__class__.__qualname__}", [self.alias], {})


dataset_storage = AliasStorage("datasets")
artifact_storage = AliasStorage("artifacts")


def dataset_upload_to(instance, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix != ".csv":
        suffix = ".csv"
    return f"projects/{instance.project_id}/datasets/{instance.id}{suffix}"


def open_dataset(dataset, mode: str = "rb") -> BinaryIO:
    """Open a dataset without assuming that storage exposes a local path."""

    return dataset.file.storage.open(dataset.file.name, mode)


def save_dataset_file(dataset, content: File) -> str:
    """Persist an upload through the configured Django storage backend."""

    dataset.file.save(dataset.original_name, content, save=False)
    dataset.save()
    return dataset.file.name


def delete_field_file(field_file: FieldFile) -> None:
    if field_file and field_file.name:
        field_file.storage.delete(field_file.name)
