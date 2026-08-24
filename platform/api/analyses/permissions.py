from __future__ import annotations

from rest_framework.permissions import BasePermission

from .authentication import GuestPrincipal


class IsUserOrGuest(BasePermission):
    message = "Authentication or a valid guest session is required."

    def has_permission(self, request, view) -> bool:
        user = getattr(request, "user", None)
        return bool(
            isinstance(user, GuestPrincipal)
            or (user is not None and getattr(user, "is_authenticated", False))
        )
