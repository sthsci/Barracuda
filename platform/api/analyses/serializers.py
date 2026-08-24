from __future__ import annotations

import json
import re

from django.contrib.auth import authenticate, get_user_model
from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers

from .access import projects_owned_by
from .models import (
    AnalysisArtifact,
    AnalysisJob,
    AnalysisProject,
    Dataset,
    ProjectShareLink,
)
from .services import create_dataset, inspect_csv_upload
from .schemas import validate_analysis_configuration


class RegisterSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    email = serializers.EmailField(required=False, allow_blank=True)
    password = serializers.CharField(write_only=True, trim_whitespace=False)

    def validate_username(self, value: str) -> str:
        value = value.strip()
        if get_user_model().objects.filter(username__iexact=value).exists():
            raise serializers.ValidationError("An account with this username already exists.")
        return value

    def validate(self, attrs):
        candidate = get_user_model()(username=attrs["username"], email=attrs.get("email", ""))
        validate_password(attrs["password"], user=candidate)
        return attrs

    def create(self, validated_data):
        return get_user_model().objects.create_user(**validated_data)


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True, trim_whitespace=False)

    def validate(self, attrs):
        user = authenticate(
            request=self.context.get("request"),
            username=attrs["username"],
            password=attrs["password"],
        )
        if user is None or not user.is_active:
            raise serializers.ValidationError("Invalid username or password.")
        attrs["user"] = user
        return attrs


class ProjectSerializer(serializers.ModelSerializer):
    owner_type = serializers.SerializerMethodField()

    class Meta:
        model = AnalysisProject
        fields = (
            "id",
            "name",
            "description",
            "visibility",
            "owner_type",
            "created_at",
            "updated_at",
            "expires_at",
        )
        read_only_fields = (
            "id",
            "visibility",
            "owner_type",
            "created_at",
            "updated_at",
            "expires_at",
        )

    def get_owner_type(self, instance) -> str:
        return "guest" if instance.owner_guest_id else "account"


class DatasetSerializer(serializers.ModelSerializer):
    class Meta:
        model = Dataset
        fields = (
            "id",
            "project",
            "original_name",
            "content_type",
            "byte_size",
            "sha256",
            "row_count",
            "column_count",
            "columns",
            "created_at",
        )
        read_only_fields = fields


class DatasetUploadSerializer(serializers.Serializer):
    project_id = serializers.UUIDField()
    file = serializers.FileField(write_only=True)

    def validate_project_id(self, value):
        try:
            return projects_owned_by(self.context["request"].user).get(pk=value)
        except AnalysisProject.DoesNotExist as exc:
            raise serializers.ValidationError("Project not found.") from exc

    def validate_file(self, value):
        return inspect_csv_upload(value)

    def create(self, validated_data):
        return create_dataset(
            project=validated_data["project_id"],
            inspected=validated_data["file"],
        )


class AnalysisArtifactSerializer(serializers.ModelSerializer):
    download_path = serializers.SerializerMethodField()

    class Meta:
        model = AnalysisArtifact
        fields = (
            "id",
            "job",
            "role",
            "filename",
            "content_type",
            "byte_size",
            "sha256",
            "shareable",
            "download_path",
            "created_at",
        )
        read_only_fields = fields

    def get_download_path(self, instance) -> str:
        return f"/api/v1/artifacts/{instance.pk}/download/"


class AnalysisJobSerializer(serializers.ModelSerializer):
    artifacts = AnalysisArtifactSerializer(many=True, read_only=True)

    class Meta:
        model = AnalysisJob
        fields = (
            "id",
            "project",
            "dataset",
            "analysis_type",
            "configuration",
            "status",
            "progress",
            "progress_detail",
            "result",
            "error_code",
            "error_message",
            "idempotency_key",
            "task_id",
            "created_at",
            "started_at",
            "completed_at",
            "artifacts",
        )
        read_only_fields = (
            "id",
            "status",
            "progress",
            "progress_detail",
            "result",
            "error_code",
            "error_message",
            "idempotency_key",
            "task_id",
            "created_at",
            "started_at",
            "completed_at",
            "artifacts",
        )


class AnalysisJobCreateSerializer(serializers.Serializer):
    project_id = serializers.UUIDField()
    dataset_id = serializers.UUIDField()
    analysis_type = serializers.ChoiceField(choices=AnalysisJob.AnalysisType.choices)
    configuration = serializers.JSONField(default=dict)

    def validate(self, attrs):
        owned = projects_owned_by(self.context["request"].user)
        try:
            project = owned.get(pk=attrs["project_id"])
        except AnalysisProject.DoesNotExist as exc:
            raise serializers.ValidationError({"project_id": "Project not found."}) from exc
        try:
            dataset = Dataset.objects.get(pk=attrs["dataset_id"], project=project)
        except Dataset.DoesNotExist as exc:
            raise serializers.ValidationError({"dataset_id": "Dataset not found in this project."}) from exc
        configuration = validate_analysis_configuration(
            attrs["analysis_type"], attrs.get("configuration", {})
        )
        if len(json.dumps(configuration, separators=(",", ":"))) > 32_000:
            raise serializers.ValidationError({"configuration": "Configuration is too large."})
        attrs["project"] = project
        attrs["dataset"] = dataset
        attrs["configuration"] = configuration
        return attrs


class ShareLinkSerializer(serializers.ModelSerializer):
    class Meta:
        model = ProjectShareLink
        fields = (
            "id",
            "project",
            "token_hint",
            "allow_dataset_download",
            "created_at",
            "expires_at",
            "revoked_at",
        )
        read_only_fields = fields


class ShareLinkCreateSerializer(serializers.Serializer):
    project_id = serializers.UUIDField()
    expires_in_hours = serializers.IntegerField(required=False, min_value=1)
    allow_dataset_download = serializers.BooleanField(default=False)

    def validate_project_id(self, value):
        try:
            return projects_owned_by(self.context["request"].user).get(pk=value)
        except AnalysisProject.DoesNotExist as exc:
            raise serializers.ValidationError("Project not found.") from exc


class SharedJobSerializer(serializers.ModelSerializer):
    artifacts = serializers.SerializerMethodField()

    class Meta:
        model = AnalysisJob
        fields = (
            "id",
            "analysis_type",
            "status",
            "progress",
            "result",
            "artifacts",
            "created_at",
            "started_at",
            "completed_at",
        )
        read_only_fields = fields

    def get_artifacts(self, instance) -> list[dict]:
        token = self.context.get("share_token", "")
        output = []
        for artifact in instance.artifacts.all():
            if not artifact.shareable:
                continue
            item = AnalysisArtifactSerializer(artifact).data
            item["download_path"] = (
                f"/api/v1/shared/{token}/artifacts/{artifact.pk}/download/"
            )
            output.append(item)
        return output


class SharedProjectSerializer(ProjectSerializer):
    jobs = SharedJobSerializer(many=True, read_only=True)
    dataset_count = serializers.IntegerField(read_only=True)
    datasets = serializers.SerializerMethodField()

    def get_datasets(self, instance) -> list[dict]:
        if not self.context.get("allow_dataset_download", False):
            return []
        token = self.context.get("share_token", "")
        return [
            {
                "id": str(dataset.pk),
                "original_name": dataset.original_name,
                "content_type": dataset.content_type,
                "byte_size": dataset.byte_size,
                "row_count": dataset.row_count,
                "column_count": dataset.column_count,
                "columns": dataset.columns,
                "download_path": f"/api/v1/shared/{token}/datasets/{dataset.pk}/download/",
            }
            for dataset in instance.datasets.all()
        ]

    class Meta(ProjectSerializer.Meta):
        fields = ProjectSerializer.Meta.fields + ("dataset_count", "datasets", "jobs")


def validate_idempotency_key(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    value = value.strip()
    if len(value) > 128 or not re.fullmatch(r"[A-Za-z0-9._:-]+", value):
        raise serializers.ValidationError(
            {"Idempotency-Key": "Use 1-128 letters, numbers, dots, underscores, colons, or hyphens."}
        )
    return value
