from __future__ import annotations

from django.db.models import QuerySet

from .authentication import GuestPrincipal
from .models import AnalysisProject


def projects_owned_by(principal) -> QuerySet[AnalysisProject]:
    if isinstance(principal, GuestPrincipal):
        return AnalysisProject.objects.filter(owner_guest=principal.guest_session)
    if getattr(principal, "is_authenticated", False):
        return AnalysisProject.objects.filter(owner_user=principal)
    return AnalysisProject.objects.none()


def owner_fields(principal) -> dict[str, object]:
    if isinstance(principal, GuestPrincipal):
        return {
            "owner_guest": principal.guest_session,
            "owner_user": None,
            "expires_at": principal.guest_session.expires_at,
        }
    if getattr(principal, "is_authenticated", False):
        return {"owner_user": principal, "owner_guest": None, "expires_at": None}
    raise PermissionError("An authenticated user or guest is required")


def owns_project(principal, project: AnalysisProject) -> bool:
    if isinstance(principal, GuestPrincipal):
        return project.owner_guest_id == principal.guest_session.pk
    return bool(
        getattr(principal, "is_authenticated", False)
        and project.owner_user_id == principal.pk
    )
