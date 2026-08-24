from __future__ import annotations

from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    ArtifactViewSet,
    DatasetViewSet,
    GuestClaimView,
    GuestSessionCreateView,
    HealthView,
    JobViewSet,
    LoginView,
    LogoutView,
    ProjectViewSet,
    RegisterView,
    SharedDatasetDownloadView,
    SharedArtifactDownloadView,
    SharedProjectView,
    ShareLinkViewSet,
)


router = DefaultRouter()
router.register("projects", ProjectViewSet, basename="project")
router.register("datasets", DatasetViewSet, basename="dataset")
router.register("jobs", JobViewSet, basename="job")
router.register("artifacts", ArtifactViewSet, basename="artifact")
router.register("share-links", ShareLinkViewSet, basename="share-link")

urlpatterns = [
    path("health/", HealthView.as_view(), name="health"),
    path("guest-sessions/", GuestSessionCreateView.as_view(), name="guest-session-create"),
    path("guest-sessions/claim/", GuestClaimView.as_view(), name="guest-session-claim"),
    path("auth/register/", RegisterView.as_view(), name="register"),
    path("auth/login/", LoginView.as_view(), name="login"),
    path("auth/logout/", LogoutView.as_view(), name="logout"),
    path("shared/<str:token>/", SharedProjectView.as_view(), name="shared-project"),
    path(
        "shared/<str:token>/artifacts/<uuid:artifact_id>/download/",
        SharedArtifactDownloadView.as_view(),
        name="shared-artifact-download",
    ),
    path(
        "shared/<str:token>/datasets/<uuid:dataset_id>/download/",
        SharedDatasetDownloadView.as_view(),
        name="shared-dataset-download",
    ),
    path("", include(router.urls)),
]
