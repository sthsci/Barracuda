from __future__ import annotations

from django.contrib.auth import logout
from django.db import IntegrityError, connection, transaction
from django.db.models import Count
from django.http import FileResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from uuid import UUID
from rest_framework import mixins, status, viewsets
from rest_framework.authentication import SessionAuthentication, TokenAuthentication
from rest_framework.authtoken.models import Token
from rest_framework.decorators import action
from rest_framework.exceptions import APIException, ValidationError
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle
from rest_framework.views import APIView

from .access import owner_fields, projects_owned_by
from .authentication import GUEST_HEADER
from .models import (
    AnalysisArtifact,
    AnalysisJob,
    AnalysisProject,
    Dataset,
    ProjectShareLink,
)
from .permissions import IsUserOrGuest
from .serializers import (
    AnalysisJobCreateSerializer,
    AnalysisJobSerializer,
    AnalysisArtifactSerializer,
    DatasetSerializer,
    DatasetUploadSerializer,
    LoginSerializer,
    ProjectSerializer,
    RegisterSerializer,
    ShareLinkCreateSerializer,
    ShareLinkSerializer,
    SharedProjectSerializer,
    validate_idempotency_key,
)
from .services import (
    claim_guest_session,
    create_guest_session,
    create_share_link,
    resolve_share_link,
)
from .storage import open_dataset
from .tasks import dispatch_analysis_job


def no_store(response):
    """Prevent capability and account tokens from entering browser caches/referrers."""

    response["Cache-Control"] = "no-store, private"
    response["Pragma"] = "no-cache"
    response["Referrer-Policy"] = "no-referrer"
    return response


def optional_uuid_query(request, name: str):
    """Return a validated optional UUID query parameter or a DRF 400."""

    raw = request.query_params.get(name)
    if not raw:
        return None
    try:
        return UUID(raw)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValidationError({name: "Enter a valid UUID."}) from exc


class HealthView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]
    throttle_classes = []

    def get(self, request):
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        except Exception:
            return Response(
                {"status": "unavailable", "database": "unavailable"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return Response({"status": "ok", "database": "ok"})


class GuestSessionCreateView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]
    throttle_classes = [ScopedRateThrottle]
    throttle_scope = "guest_session_create"

    def post(self, request):
        guest, raw_token = create_guest_session()
        return no_store(Response(
            {
                "id": str(guest.pk),
                "guest_token": raw_token,
                "expires_at": guest.expires_at,
                "retention_notice": (
                    "Guest projects, datasets, and jobs are deleted after this expiry. "
                    "The token is shown only in this response."
                ),
            },
            status=status.HTTP_201_CREATED,
        ))


class GuestClaimView(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsUserOrGuest]

    def post(self, request):
        raw_token = request.META.get(GUEST_HEADER, "").strip()
        if not raw_token:
            raise ValidationError({"guest_token": "Send X-Barracuda-Guest-Token."})
        claimed = claim_guest_session(raw_token, request.user)
        return Response({"claimed_projects": claimed})


class RegisterView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        token, _ = Token.objects.get_or_create(user=user)
        return no_store(Response(
            {"token": token.key, "user": {"id": user.pk, "username": user.username}},
            status=status.HTTP_201_CREATED,
        ))


class LoginView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = LoginSerializer(data=request.data, context={"request": request})
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data["user"]
        token, _ = Token.objects.get_or_create(user=user)
        return no_store(Response({"token": token.key, "user": {"id": user.pk, "username": user.username}}))


class LogoutView(APIView):
    authentication_classes = [TokenAuthentication, SessionAuthentication]
    permission_classes = [IsUserOrGuest]

    def post(self, request):
        if isinstance(request.auth, Token):
            request.auth.delete()
        logout(request)
        return Response(status=status.HTTP_204_NO_CONTENT)


class ProjectViewSet(viewsets.ModelViewSet):
    serializer_class = ProjectSerializer
    permission_classes = [IsUserOrGuest]
    http_method_names = ("get", "post", "patch", "delete", "head", "options")

    def get_queryset(self):
        return projects_owned_by(self.request.user)

    def perform_create(self, serializer):
        serializer.save(**owner_fields(self.request.user))


class DatasetViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    permission_classes = [IsUserOrGuest]
    throttle_scope = "api_read"
    throttle_classes = [ScopedRateThrottle]

    def get_throttles(self):
        self.throttle_scope = "dataset_upload" if self.action == "create" else "api_read"
        return super().get_throttles()

    def get_queryset(self):
        queryset = Dataset.objects.filter(project__in=projects_owned_by(self.request.user))
        project_id = optional_uuid_query(self.request, "project_id")
        if project_id:
            queryset = queryset.filter(project_id=project_id)
        return queryset.select_related("project")

    def get_serializer_class(self):
        return DatasetUploadSerializer if self.action == "create" else DatasetSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        dataset = serializer.save()
        return Response(
            DatasetSerializer(dataset, context=self.get_serializer_context()).data,
            status=status.HTTP_201_CREATED,
        )

    def perform_destroy(self, instance):
        if instance.jobs.exists():
            raise ValidationError("Datasets used by analysis jobs cannot be deleted separately.")
        instance.delete()

    @action(detail=True, methods=("get",))
    def download(self, request, pk=None):
        dataset = self.get_object()
        return FileResponse(
            open_dataset(dataset, "rb"),
            as_attachment=True,
            filename=dataset.original_name,
            content_type=dataset.content_type,
        )


class JobViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    viewsets.GenericViewSet,
):
    permission_classes = [IsUserOrGuest]
    throttle_classes = [ScopedRateThrottle]

    def get_throttles(self):
        self.throttle_scope = "job_create" if self.action == "create" else "job_read"
        return super().get_throttles()

    def get_queryset(self):
        queryset = AnalysisJob.objects.filter(project__in=projects_owned_by(self.request.user))
        project_id = optional_uuid_query(self.request, "project_id")
        if project_id:
            queryset = queryset.filter(project_id=project_id)
        return queryset.select_related("project", "dataset").prefetch_related("artifacts")

    def get_serializer_class(self):
        return AnalysisJobCreateSerializer if self.action == "create" else AnalysisJobSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        idempotency_key = validate_idempotency_key(request.headers.get("Idempotency-Key"))
        values = serializer.validated_data
        if idempotency_key:
            existing = AnalysisJob.objects.filter(
                project=values["project"], idempotency_key=idempotency_key
            ).first()
            if existing is not None:
                return Response(AnalysisJobSerializer(existing).data, status=status.HTTP_200_OK)
        try:
            with transaction.atomic():
                job = AnalysisJob.objects.create(
                    project=values["project"],
                    dataset=values["dataset"],
                    analysis_type=values["analysis_type"],
                    configuration=values.get("configuration", {}),
                    idempotency_key=idempotency_key,
                )
        except IntegrityError:
            job = AnalysisJob.objects.get(
                project=values["project"], idempotency_key=idempotency_key
            )
            return Response(AnalysisJobSerializer(job).data, status=status.HTTP_200_OK)
        try:
            dispatch_analysis_job(job)
        except Exception as exc:
            AnalysisJob.objects.filter(pk=job.pk).update(
                status=AnalysisJob.Status.FAILED,
                error_code="dispatch_failed",
                error_message="The analysis job could not be dispatched. Try again later.",
                completed_at=timezone.now(),
            )
            raise APIException("The analysis job could not be dispatched.") from exc
        job.refresh_from_db()
        return Response(AnalysisJobSerializer(job).data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=("post",))
    def cancel(self, request, pk=None):
        with transaction.atomic():
            job = AnalysisJob.objects.select_for_update().get(pk=self.get_object().pk)
            if job.status != AnalysisJob.Status.QUEUED:
                raise ValidationError("Only queued jobs can be cancelled.")
            job.status = AnalysisJob.Status.CANCELLED
            job.completed_at = timezone.now()
            job.save(update_fields=("status", "completed_at"))
        return Response(AnalysisJobSerializer(job).data)


class ArtifactViewSet(
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    viewsets.GenericViewSet,
):
    serializer_class = AnalysisArtifactSerializer
    permission_classes = [IsUserOrGuest]
    http_method_names = ("get", "head", "options")

    def get_queryset(self):
        queryset = AnalysisArtifact.objects.filter(
            job__project__in=projects_owned_by(self.request.user)
        )
        job_id = optional_uuid_query(self.request, "job_id")
        if job_id:
            queryset = queryset.filter(job_id=job_id)
        return queryset.select_related("job", "job__project")

    @action(detail=True, methods=("get",))
    def download(self, request, pk=None):
        artifact = self.get_object()
        return FileResponse(
            artifact.file.storage.open(artifact.file.name, "rb"),
            as_attachment=True,
            filename=artifact.filename,
            content_type=artifact.content_type,
        )


class SharedArtifactDownloadView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    def get(self, request, token: str, artifact_id):
        link = resolve_share_link(token)
        artifact = get_object_or_404(
            AnalysisArtifact,
            pk=artifact_id,
            job__project=link.project,
            shareable=True,
        )
        return no_store(FileResponse(
            artifact.file.storage.open(artifact.file.name, "rb"),
            as_attachment=True,
            filename=artifact.filename,
            content_type=artifact.content_type,
        ))


class ShareLinkViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    permission_classes = [IsUserOrGuest]

    def get_queryset(self):
        queryset = ProjectShareLink.objects.filter(project__in=projects_owned_by(self.request.user))
        project_id = optional_uuid_query(self.request, "project_id")
        if project_id:
            queryset = queryset.filter(project_id=project_id)
        return queryset.select_related("project")

    def get_serializer_class(self):
        return ShareLinkCreateSerializer if self.action == "create" else ShareLinkSerializer

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        link, raw_token = create_share_link(
            project=serializer.validated_data["project_id"],
            creator=request.user,
            expires_in_hours=serializer.validated_data.get("expires_in_hours"),
            allow_dataset_download=serializer.validated_data["allow_dataset_download"],
        )
        response = ShareLinkSerializer(link).data
        response["share_token"] = raw_token
        response["share_path"] = f"/api/v1/shared/{raw_token}/"
        return no_store(Response(response, status=status.HTTP_201_CREATED))

    def perform_destroy(self, instance):
        instance.revoked_at = timezone.now()
        instance.save(update_fields=("revoked_at",))


class SharedProjectView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    def get(self, request, token: str):
        link = resolve_share_link(token)
        project = (
            AnalysisProject.objects.annotate(dataset_count=Count("datasets"))
            .prefetch_related("datasets", "jobs__artifacts")
            .get(pk=link.project_id)
        )
        payload = SharedProjectSerializer(
            project,
            context={
                "share_token": token,
                "allow_dataset_download": link.allow_dataset_download,
            },
        ).data
        payload["share"] = {
            "expires_at": link.expires_at,
            "allow_dataset_download": link.allow_dataset_download,
        }
        return no_store(Response(payload))


class SharedDatasetDownloadView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]

    def get(self, request, token: str, dataset_id):
        link = resolve_share_link(token)
        if not link.allow_dataset_download:
            return Response(
                {
                    "error": {
                        "status": 403,
                        "code": "permission_denied",
                        "detail": "Dataset download is not enabled for this share link.",
                    }
                },
                status=status.HTTP_403_FORBIDDEN,
            )
        dataset = get_object_or_404(Dataset, pk=dataset_id, project=link.project)
        return no_store(FileResponse(
            open_dataset(dataset, "rb"),
            as_attachment=True,
            filename=dataset.original_name,
            content_type=dataset.content_type,
        ))
