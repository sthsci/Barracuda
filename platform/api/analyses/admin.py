from __future__ import annotations

from django.contrib import admin

from .models import (
    AnalysisArtifact,
    AnalysisJob,
    AnalysisProject,
    Dataset,
    GuestSession,
    ProjectShareLink,
)


@admin.register(GuestSession)
class GuestSessionAdmin(admin.ModelAdmin):
    list_display = ("token_hint", "created_at", "expires_at", "revoked_at", "claimed_by")
    readonly_fields = ("token_digest", "token_hint", "created_at", "last_seen_at")


@admin.register(AnalysisProject)
class AnalysisProjectAdmin(admin.ModelAdmin):
    list_display = ("name", "owner_user", "owner_guest", "updated_at", "expires_at")
    search_fields = ("name",)


@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = ("original_name", "project", "row_count", "byte_size", "created_at")
    readonly_fields = ("sha256", "byte_size", "row_count", "column_count", "columns")


@admin.register(AnalysisJob)
class AnalysisJobAdmin(admin.ModelAdmin):
    list_display = ("analysis_type", "project", "status", "progress", "created_at")
    list_filter = ("analysis_type", "status")


@admin.register(AnalysisArtifact)
class AnalysisArtifactAdmin(admin.ModelAdmin):
    list_display = ("role", "job", "filename", "byte_size", "shareable", "created_at")
    readonly_fields = ("sha256", "byte_size", "created_at")


@admin.register(ProjectShareLink)
class ProjectShareLinkAdmin(admin.ModelAdmin):
    list_display = ("token_hint", "project", "expires_at", "revoked_at")
    readonly_fields = ("token_digest", "token_hint", "created_at")
