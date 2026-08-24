from __future__ import annotations

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from analyses.models import GuestSession, ProjectShareLink


class Command(BaseCommand):
    help = "Delete expired guest sessions and their projects, jobs, and stored datasets."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report eligible records without deleting them.",
        )

    def handle(self, *args, **options):
        now = timezone.now()
        guests = GuestSession.objects.filter(expires_at__lte=now)
        shares = ProjectShareLink.objects.filter(expires_at__lte=now)
        guest_count = guests.count()
        project_count = sum(guest.projects.count() for guest in guests.iterator())
        share_count = shares.count()
        if options["dry_run"]:
            self.stdout.write(
                f"Would delete {guest_count} guest session(s), "
                f"{project_count} project(s), and {share_count} expired share link(s)."
            )
            return
        with transaction.atomic():
            shares.delete()
            guests.delete()
        self.stdout.write(
            self.style.SUCCESS(
                f"Deleted {guest_count} guest session(s), "
                f"{project_count} project(s), and {share_count} expired share link(s)."
            )
        )
