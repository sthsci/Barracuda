from __future__ import annotations

from datetime import timedelta
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import override_settings
from django.utils import timezone
from rest_framework.test import APITestCase

from analyses.models import AnalysisJob, AnalysisProject, Dataset, GuestSession


def error_detail(response):
    return response.data["error"]["detail"]


class ApiTestCase(APITestCase):
    def setUp(self) -> None:
        super().setUp()
        self.storage_directory = TemporaryDirectory(prefix="barracuda-api-test-")
        self.storage_override = override_settings(
            STORAGES={
                "default": {
                    "BACKEND": "django.core.files.storage.FileSystemStorage",
                    "OPTIONS": {"location": self.storage_directory.name},
                },
                "staticfiles": {
                    "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"
                },
                "datasets": {
                    "BACKEND": "django.core.files.storage.FileSystemStorage",
                    "OPTIONS": {"location": self.storage_directory.name},
                },
                "artifacts": {
                    "BACKEND": "django.core.files.storage.FileSystemStorage",
                    "OPTIONS": {"location": self.storage_directory.name},
                },
            },
            CELERY_TASK_ALWAYS_EAGER=True,
            CELERY_TASK_EAGER_PROPAGATES=True,
            BARRACUDA_ANALYSIS_ENGINE="analyses.execution.MockAnalysisEngine",
        )
        self.storage_override.enable()

    def tearDown(self) -> None:
        self.storage_override.disable()
        self.storage_directory.cleanup()
        super().tearDown()

    def create_guest(self) -> str:
        response = self.client.post("/api/v1/guest-sessions/", {}, format="json")
        self.assertEqual(response.status_code, 201)
        self.assertTrue(response.data["guest_token"].startswith("barracuda_g_"))
        return response.data["guest_token"]

    def use_guest(self, token: str) -> None:
        self.client.credentials(HTTP_X_BARRACUDA_GUEST_TOKEN=token)

    def create_project(self, name: str = "My analysis") -> dict:
        response = self.client.post(
            "/api/v1/projects/", {"name": name, "description": "Private"}, format="json"
        )
        self.assertEqual(response.status_code, 201, response.data)
        return response.data

    def upload_dataset(self, project_id: str, payload: bytes | None = None) -> dict:
        content = payload or b"cell_id,condition,count\na,Control,0\nb,Control,2\n"
        response = self.client.post(
            "/api/v1/datasets/",
            {
                "project_id": project_id,
                "file": SimpleUploadedFile("counts.csv", content, content_type="text/csv"),
            },
            format="multipart",
        )
        self.assertEqual(response.status_code, 201, response.data)
        return response.data


class HealthAndCorsTests(ApiTestCase):
    def test_health_checks_database_and_cors_is_limited_to_frontend_origin(self) -> None:
        response = self.client.get(
            "/api/v1/health/", HTTP_ORIGIN="http://127.0.0.1:8501"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, {"status": "ok", "database": "ok"})
        self.assertEqual(response["Access-Control-Allow-Origin"], "http://127.0.0.1:8501")

        denied_origin = self.client.get(
            "/api/v1/health/", HTTP_ORIGIN="https://untrusted.example"
        )
        self.assertNotIn("Access-Control-Allow-Origin", denied_origin)


class GuestAndAccountTests(ApiTestCase):
    def test_guest_projects_are_private_between_sessions(self) -> None:
        first = self.create_guest()
        self.use_guest(first)
        project = self.create_project()
        self.assertEqual(project["owner_type"], "guest")
        self.assertIsNotNone(project["expires_at"])

        second = self.create_guest()
        self.use_guest(second)
        hidden = self.client.get(f"/api/v1/projects/{project['id']}/")
        self.assertEqual(hidden.status_code, 404)

        self.use_guest(first)
        visible = self.client.get(f"/api/v1/projects/{project['id']}/")
        self.assertEqual(visible.status_code, 200)

    def test_guest_projects_can_be_claimed_by_an_optional_account(self) -> None:
        guest_token = self.create_guest()
        self.use_guest(guest_token)
        project = self.create_project()

        self.client.credentials()
        registration = self.client.post(
            "/api/v1/auth/register/",
            {
                "username": "researcher",
                "email": "researcher@example.org",
                "password": "Long-Random-Passphrase-238!",
            },
            format="json",
        )
        self.assertEqual(registration.status_code, 201, registration.data)
        account_token = registration.data["token"]
        self.client.credentials(
            HTTP_AUTHORIZATION=f"Token {account_token}",
            HTTP_X_BARRACUDA_GUEST_TOKEN=guest_token,
        )
        claimed = self.client.post("/api/v1/guest-sessions/claim/", {}, format="json")
        self.assertEqual(claimed.status_code, 200)
        self.assertEqual(claimed.data["claimed_projects"], 1)

        self.client.credentials(HTTP_AUTHORIZATION=f"Token {account_token}")
        projects = self.client.get("/api/v1/projects/")
        self.assertEqual(projects.data["count"], 1)
        self.assertEqual(projects.data["results"][0]["id"], project["id"])
        self.assertEqual(projects.data["results"][0]["owner_type"], "account")

        logout_response = self.client.post("/api/v1/auth/logout/", {}, format="json")
        self.assertEqual(logout_response.status_code, 204)
        denied = self.client.get("/api/v1/projects/")
        self.assertIn(denied.status_code, (401, 403))

    def test_expired_guest_cleanup_removes_database_records_and_blob(self) -> None:
        guest_token = self.create_guest()
        self.use_guest(guest_token)
        project = self.create_project()
        dataset = self.upload_dataset(project["id"])
        stored = Dataset.objects.get(pk=dataset["id"])
        stored_path = Path(self.storage_directory.name) / stored.file.name
        self.assertTrue(stored_path.exists())
        GuestSession.objects.update(expires_at=timezone.now() - timedelta(seconds=1))

        stdout = StringIO()
        call_command("purge_expired_guests", stdout=stdout)
        self.assertEqual(GuestSession.objects.count(), 0)
        self.assertEqual(AnalysisProject.objects.count(), 0)
        self.assertEqual(Dataset.objects.count(), 0)
        self.assertFalse(stored_path.exists())
        self.assertIn("Deleted 1 guest session", stdout.getvalue())


class DatasetTests(ApiTestCase):
    def test_csv_upload_records_metadata_and_private_download_preserves_bytes(self) -> None:
        guest_token = self.create_guest()
        self.use_guest(guest_token)
        project = self.create_project()
        payload = b"cell_id,condition,history\na,Control,001\nb,Control,\n"
        dataset = self.upload_dataset(project["id"], payload)

        self.assertEqual(dataset["row_count"], 2)
        self.assertEqual(dataset["column_count"], 3)
        self.assertEqual(dataset["columns"], ["cell_id", "condition", "history"])
        self.assertEqual(dataset["byte_size"], len(payload))

        download = self.client.get(f"/api/v1/datasets/{dataset['id']}/download/")
        self.assertEqual(download.status_code, 200)
        self.assertEqual(b"".join(download.streaming_content), payload)

        other_guest = self.create_guest()
        self.use_guest(other_guest)
        hidden = self.client.get(f"/api/v1/datasets/{dataset['id']}/download/")
        self.assertEqual(hidden.status_code, 404)

    def test_malformed_and_non_utf8_csv_files_are_rejected(self) -> None:
        guest_token = self.create_guest()
        self.use_guest(guest_token)
        project = self.create_project()
        malformed = self.client.post(
            "/api/v1/datasets/",
            {
                "project_id": project["id"],
                "file": SimpleUploadedFile("bad.csv", b"a,b\n1\n", content_type="text/csv"),
            },
            format="multipart",
        )
        self.assertEqual(malformed.status_code, 400)
        self.assertIn("expected 2", str(error_detail(malformed)))

        non_utf8 = self.client.post(
            "/api/v1/datasets/",
            {
                "project_id": project["id"],
                "file": SimpleUploadedFile("bad.csv", b"a\n\xff\n", content_type="text/csv"),
            },
            format="multipart",
        )
        self.assertEqual(non_utf8.status_code, 400)
        self.assertIn("UTF-8", str(error_detail(non_utf8)))


class ShareLinkTests(ApiTestCase):
    def test_share_link_is_read_only_revocable_and_download_is_explicit(self) -> None:
        guest_token = self.create_guest()
        self.use_guest(guest_token)
        project = self.create_project()
        dataset = self.upload_dataset(project["id"])
        share = self.client.post(
            "/api/v1/share-links/",
            {
                "project_id": project["id"],
                "expires_in_hours": 2,
                "allow_dataset_download": False,
            },
            format="json",
        )
        self.assertEqual(share.status_code, 201, share.data)
        raw_token = share.data["share_token"]

        self.client.credentials()
        shared = self.client.get(f"/api/v1/shared/{raw_token}/")
        self.assertEqual(shared.status_code, 200)
        self.assertEqual(shared.data["id"], project["id"])
        denied_download = self.client.get(
            f"/api/v1/shared/{raw_token}/datasets/{dataset['id']}/download/"
        )
        self.assertEqual(denied_download.status_code, 403)

        self.use_guest(guest_token)
        revoked = self.client.delete(f"/api/v1/share-links/{share.data['id']}/")
        self.assertEqual(revoked.status_code, 204)
        self.client.credentials()
        gone = self.client.get(f"/api/v1/shared/{raw_token}/")
        self.assertEqual(gone.status_code, 404)

    def test_account_owner_can_explicitly_share_csv_metadata_and_download(self) -> None:
        registration = self.client.post(
            "/api/v1/auth/register/",
            {
                "username": "spreadsheet-owner",
                "password": "Long-Random-Passphrase-839!",
            },
            format="json",
        )
        self.client.credentials(HTTP_AUTHORIZATION=f"Token {registration.data['token']}")
        project = self.create_project("Shared counts")
        dataset = self.upload_dataset(project["id"])
        share = self.client.post(
            "/api/v1/share-links/",
            {
                "project_id": project["id"],
                "expires_in_hours": 2,
                "allow_dataset_download": True,
            },
            format="json",
        )
        self.assertEqual(share.status_code, 201, share.data)

        self.client.credentials()
        shared = self.client.get(f"/api/v1/shared/{share.data['share_token']}/")
        self.assertEqual(shared.status_code, 200)
        self.assertTrue(shared.data["share"]["allow_dataset_download"])
        self.assertEqual(shared.data["datasets"][0]["id"], dataset["id"])
        self.assertNotIn("file", shared.data["datasets"][0])
        download = self.client.get(shared.data["datasets"][0]["download_path"])
        self.assertEqual(download.status_code, 200)


class QueryValidationTests(ApiTestCase):
    def test_invalid_uuid_filters_return_400_instead_of_500(self) -> None:
        guest = self.create_guest()
        self.use_guest(guest)
        for path in (
            "/api/v1/datasets/?project_id=not-a-uuid",
            "/api/v1/jobs/?project_id=not-a-uuid",
            "/api/v1/artifacts/?job_id=not-a-uuid",
            "/api/v1/share-links/?project_id=not-a-uuid",
        ):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 400, path)


class JobTests(ApiTestCase):
    def test_job_uses_eager_mock_adapter_and_idempotency_key(self) -> None:
        guest_token = self.create_guest()
        self.use_guest(guest_token)
        project = self.create_project()
        dataset = self.upload_dataset(project["id"])
        body = {
            "project_id": project["id"],
            "dataset_id": dataset["id"],
            "analysis_type": "trajectory_donor_ignorant",
            "configuration": {"particles": 64},
        }
        first = self.client.post(
            "/api/v1/jobs/", body, format="json", HTTP_IDEMPOTENCY_KEY="run-001"
        )
        self.assertEqual(first.status_code, 201, first.data)
        self.assertEqual(first.data["status"], "succeeded")
        self.assertEqual(first.data["progress"], 1.0)
        self.assertEqual(first.data["result"]["engine"], "mock")

        repeated = self.client.post(
            "/api/v1/jobs/", body, format="json", HTTP_IDEMPOTENCY_KEY="run-001"
        )
        self.assertEqual(repeated.status_code, 200)
        self.assertEqual(repeated.data["id"], first.data["id"])
        self.assertEqual(AnalysisJob.objects.count(), 1)

    def test_job_cannot_reference_a_dataset_from_another_project(self) -> None:
        guest_token = self.create_guest()
        self.use_guest(guest_token)
        first_project = self.create_project("First")
        second_project = self.create_project("Second")
        dataset = self.upload_dataset(first_project["id"])
        response = self.client.post(
            "/api/v1/jobs/",
            {
                "project_id": second_project["id"],
                "dataset_id": dataset["id"],
                "analysis_type": "event_count_donor_ignorant",
                "configuration": {},
            },
            format="json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("not found", str(error_detail(response)).lower())
