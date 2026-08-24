"""Settings for the standalone Barracuda REST API.

The defaults are deliberately local-development defaults. Production must set
``BARRACUDA_API_DEBUG=0``, a strong ``BARRACUDA_API_SECRET_KEY``, allowed hosts, a
PostgreSQL ``DATABASE_URL``, and an appropriate dataset storage backend.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


BASE_DIR = Path(__file__).resolve().parent.parent


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}")
    return value


def database_from_url(raw_url: str) -> dict[str, object]:
    parsed = urlparse(raw_url)
    scheme = parsed.scheme.lower()
    if scheme in {"sqlite", "sqlite3"}:
        if parsed.netloc and parsed.netloc not in {"", "localhost"}:
            raise RuntimeError("SQLite DATABASE_URL must not contain a remote host")
        raw_path = unquote(parsed.path)
        if raw_path in {"", "/"}:
            name = BASE_DIR / "var" / "barracuda-api.sqlite3"
        elif raw_path == "/:memory:":
            name = ":memory:"
        elif raw_path.startswith("//"):
            name = Path(raw_path[1:])
        else:
            name = BASE_DIR / raw_path.lstrip("/")
        return {"ENGINE": "django.db.backends.sqlite3", "NAME": str(name)}
    if scheme in {"postgres", "postgresql"}:
        query = parse_qs(parsed.query)
        options: dict[str, str] = {}
        if "sslmode" in query:
            options["sslmode"] = query["sslmode"][-1]
        return {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": unquote(parsed.path.lstrip("/")),
            "USER": unquote(parsed.username or ""),
            "PASSWORD": unquote(parsed.password or ""),
            "HOST": parsed.hostname or "localhost",
            "PORT": parsed.port or 5432,
            "CONN_MAX_AGE": env_int("BARRACUDA_DB_CONN_MAX_AGE", 60, minimum=0),
            "CONN_HEALTH_CHECKS": True,
            "OPTIONS": options,
        }
    raise RuntimeError("DATABASE_URL must use sqlite, postgres, or postgresql")


DEBUG = env_bool("BARRACUDA_API_DEBUG", True)
SECRET_KEY = os.getenv("BARRACUDA_API_SECRET_KEY", "")
if not SECRET_KEY:
    if not DEBUG:
        raise RuntimeError("BARRACUDA_API_SECRET_KEY is required when DEBUG is false")
    SECRET_KEY = "insecure-local-development-only"

BARRACUDA_CAPABILITY_HMAC_KEY = os.getenv("BARRACUDA_CAPABILITY_HMAC_KEY", "")
if not BARRACUDA_CAPABILITY_HMAC_KEY:
    if not DEBUG:
        raise RuntimeError("BARRACUDA_CAPABILITY_HMAC_KEY is required when DEBUG is false")
    BARRACUDA_CAPABILITY_HMAC_KEY = "insecure-local-capability-pepper-only"

ALLOWED_HOSTS = [
    host.strip()
    for host in os.getenv("BARRACUDA_API_ALLOWED_HOSTS", "localhost,127.0.0.1,testserver").split(",")
    if host.strip()
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "corsheaders",
    "rest_framework",
    "rest_framework.authtoken",
    "analyses.apps.AnalysesConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "barracuda_api.urls"
WSGI_APPLICATION = "barracuda_api.wsgi.application"
ASGI_APPLICATION = "barracuda_api.asgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ]
        },
    }
]

DATABASES = {
    "default": database_from_url(
        os.getenv("DATABASE_URL", "sqlite:///var/barracuda-api.sqlite3")
    )
}
if DATABASES["default"]["ENGINE"] == "django.db.backends.sqlite3":
    database_name = DATABASES["default"]["NAME"]
    if database_name != ":memory:":
        Path(str(database_name)).parent.mkdir(parents=True, exist_ok=True)

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-gb"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "var" / "static"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

DATASET_STORAGE_ROOT = Path(
    os.getenv("BARRACUDA_DATASET_STORAGE_ROOT", str(BASE_DIR / "var" / "datasets"))
)
dataset_backend = os.getenv(
    "BARRACUDA_DATASET_STORAGE_BACKEND", "django.core.files.storage.FileSystemStorage"
)
dataset_options_raw = os.getenv("BARRACUDA_DATASET_STORAGE_OPTIONS", "")
if dataset_options_raw:
    try:
        dataset_options = json.loads(dataset_options_raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("BARRACUDA_DATASET_STORAGE_OPTIONS must be valid JSON") from exc
    if not isinstance(dataset_options, dict):
        raise RuntimeError("BARRACUDA_DATASET_STORAGE_OPTIONS must be a JSON object")
else:
    dataset_options = (
        {"location": str(DATASET_STORAGE_ROOT)}
        if dataset_backend == "django.core.files.storage.FileSystemStorage"
        else {}
    )

STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
        "OPTIONS": {"location": str(BASE_DIR / "var" / "media")},
    },
    "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    "datasets": {"BACKEND": dataset_backend, "OPTIONS": dataset_options},
    "artifacts": {"BACKEND": dataset_backend, "OPTIONS": dataset_options},
}

BARRACUDA_GUEST_RETENTION_HOURS = env_int("BARRACUDA_GUEST_RETENTION_HOURS", 24)
BARRACUDA_DEFAULT_SHARE_LINK_HOURS = env_int("BARRACUDA_DEFAULT_SHARE_LINK_HOURS", 72)
BARRACUDA_MAX_SHARE_LINK_HOURS = env_int("BARRACUDA_MAX_SHARE_LINK_HOURS", 720)
BARRACUDA_MAX_DATASET_BYTES = env_int("BARRACUDA_MAX_DATASET_BYTES", 5 * 1024 * 1024)
BARRACUDA_MAX_DATASET_ROWS = env_int("BARRACUDA_MAX_DATASET_ROWS", 50_000)
BARRACUDA_MAX_DATASET_COLUMNS = env_int("BARRACUDA_MAX_DATASET_COLUMNS", 100)
BARRACUDA_MAX_ARTIFACT_BYTES = env_int("BARRACUDA_MAX_ARTIFACT_BYTES", 100 * 1024 * 1024)
BARRACUDA_MAX_RESULT_JSON_BYTES = env_int("BARRACUDA_MAX_RESULT_JSON_BYTES", 2 * 1024 * 1024)

CORS_ALLOWED_ORIGINS = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "BARRACUDA_CORS_ALLOWED_ORIGINS", "http://127.0.0.1:8501,http://localhost:8501"
    ).split(",")
    if origin.strip()
]
CORS_ALLOW_CREDENTIALS = True
CORS_URLS_REGEX = r"^/api/.*$"

CELERY_BROKER_URL = os.getenv(
    "BARRACUDA_CELERY_BROKER_URL", "redis://localhost:6379/0"
)
CELERY_RESULT_BACKEND = os.getenv(
    "BARRACUDA_CELERY_RESULT_BACKEND", "redis://localhost:6379/1"
)
CELERY_TASK_ALWAYS_EAGER = env_bool("BARRACUDA_TASK_ALWAYS_EAGER", DEBUG)
CELERY_TASK_EAGER_PROPAGATES = env_bool("BARRACUDA_TASK_EAGER_PROPAGATES", DEBUG)
CELERY_TASK_TRACK_STARTED = True
CELERY_TASK_TIME_LIMIT = env_int("BARRACUDA_TASK_TIME_LIMIT_SECONDS", 3600)
CELERY_TASK_SOFT_TIME_LIMIT = env_int("BARRACUDA_TASK_SOFT_TIME_LIMIT_SECONDS", 3300)
CELERY_TASK_ACKS_LATE = True
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
BARRACUDA_ANALYSIS_ENGINE = os.getenv(
    "BARRACUDA_ANALYSIS_ENGINE", "analyses.execution.MockAnalysisEngine"
)
BARRACUDA_RUNNER = os.getenv("BARRACUDA_RUNNER", "")

DATA_UPLOAD_MAX_MEMORY_SIZE = BARRACUDA_MAX_DATASET_BYTES + 64 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = min(BARRACUDA_MAX_DATASET_BYTES, 2 * 1024 * 1024)

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.TokenAuthentication",
        "rest_framework.authentication.SessionAuthentication",
        "analyses.authentication.GuestSessionAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": ["analyses.permissions.IsUserOrGuest"],
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES": [
        "rest_framework.parsers.JSONParser",
        "rest_framework.parsers.FormParser",
        "rest_framework.parsers.MultiPartParser",
    ],
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
        "rest_framework.throttling.UserRateThrottle",
        "rest_framework.throttling.ScopedRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "anon": "60/hour",
        "user": "1000/hour",
        "guest_session_create": "20/hour",
        "dataset_upload": "30/hour",
        "job_create": "60/hour",
        "job_read": "3600/hour",
        "api_read": "2000/hour",
    },
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 50,
    "EXCEPTION_HANDLER": "analyses.exceptions.api_exception_handler",
}

SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = "DENY"
SESSION_COOKIE_HTTPONLY = True
CSRF_COOKIE_HTTPONLY = True
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG
SECURE_SSL_REDIRECT = env_bool("BARRACUDA_API_SECURE_SSL_REDIRECT", not DEBUG)
if env_bool("BARRACUDA_API_BEHIND_HTTPS_PROXY", False):
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
