from __future__ import annotations

from django.apps import AppConfig


class AnalysesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "analyses"

    def ready(self) -> None:
        from . import signals  # noqa: F401
