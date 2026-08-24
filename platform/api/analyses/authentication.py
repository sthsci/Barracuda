from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hmac
from hashlib import sha256

from django.conf import settings
from django.utils import timezone
from rest_framework import authentication, exceptions

from .models import GuestSession


GUEST_HEADER = "HTTP_X_BARRACUDA_GUEST_TOKEN"


def token_digest(token: str) -> str:
    return hmac.new(
        settings.BARRACUDA_CAPABILITY_HMAC_KEY.encode("utf-8"),
        token.encode("utf-8"),
        sha256,
    ).hexdigest()


@dataclass(frozen=True)
class GuestPrincipal:
    guest_session: GuestSession

    @property
    def pk(self):
        return self.guest_session.pk

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def is_anonymous(self) -> bool:
        return False

    @property
    def username(self) -> str:
        return f"guest-{self.guest_session.token_hint}"

    def __str__(self) -> str:
        return self.username


def authenticate_guest_token(raw_token: str, *, touch: bool = True) -> GuestSession:
    if not raw_token or len(raw_token) > 200:
        raise exceptions.AuthenticationFailed("Invalid guest token.")
    try:
        guest = GuestSession.objects.get(token_digest=token_digest(raw_token))
    except GuestSession.DoesNotExist as exc:
        raise exceptions.AuthenticationFailed("Invalid guest token.") from exc
    now = timezone.now()
    if guest.revoked_at is not None or guest.expires_at <= now:
        raise exceptions.AuthenticationFailed("Guest session has expired.")
    if touch and guest.last_seen_at < now - timedelta(minutes=5):
        GuestSession.objects.filter(pk=guest.pk).update(last_seen_at=now)
        guest.last_seen_at = now
    return guest


class GuestSessionAuthentication(authentication.BaseAuthentication):
    """Authenticate a time-limited guest through ``X-Barracuda-Guest-Token``."""

    def authenticate(self, request):
        raw_token = request.META.get(GUEST_HEADER, "").strip()
        if not raw_token:
            return None
        guest = authenticate_guest_token(raw_token)
        return GuestPrincipal(guest), guest

    def authenticate_header(self, request) -> str:
        return "X-Barracuda-Guest-Token"
