"""Dash application factory for Orca."""

from __future__ import annotations

import os
from pathlib import Path

import diskcache
import psutil
from dash import Dash, DiskcacheManager, Input, Output, dcc, html

from webapp.analysis_ui import PROFILE_VALUES
from webapp.pages import bayes_101, donor_aware, event_counts, event_counts_overview, home, notebooks, synthetic_validation, trajectory


PAGES = [home, bayes_101, notebooks, event_counts_overview, event_counts, donor_aware, trajectory]
PAGE_BY_PATH = {page.PATH: page for page in PAGES}
# The former standalone validation URL now opens the merged donor ignorant
# workflow. Keeping the alias avoids breaking saved links without restoring a
# second copy of the page in navigation.
PAGE_BY_PATH[synthetic_validation.PATH] = event_counts
PAGE_BY_PATH["/donor-aware"] = donor_aware
NAV_GROUPS = [
    ("Overview", [(home, False), (bayes_101, False)]),
    ("Notebooks", [(notebooks, False)]),
    (
        "Event counts",
        [
            (event_counts_overview, False),
            (event_counts, True),
            (donor_aware, True),
        ],
    ),
    ("Trajectories", [(trajectory, False)]),
]
NAV_LABELS = {
    home.PATH: "Home",
    bayes_101.PATH: "Bayesian inference 101",
    notebooks.PATH: "Google Colab notebooks",
    event_counts_overview.PATH: "Event count analysis",
    event_counts.PATH: "Donor ignorant · Data and validation",
    donor_aware.PATH: "Donor aware · Condition analysis",
    trajectory.PATH: "Trajectory inference",
}
NAV_IDS = {page.PATH: f"nav-{index}" for index, page in enumerate(PAGES)}
NAV_BASE_CLASSES = {
    page.PATH: "orca-nav-link orca-nav-link-child" if is_child else "orca-nav-link"
    for _group, entries in NAV_GROUPS
    for page, is_child in entries
}


class OrcaDiskcacheManager(DiskcacheManager):
    """Collect completed jobs even when macOS blocks process-tree inspection.

    Dash normally asks ``psutil`` to inspect and terminate the worker process
    after its result has been read from diskcache. Sandboxed macOS sessions can
    deny that process-tree query. The worker has already written its result at
    this point, so cleanup failure must not discard an otherwise valid fit.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._orca_jobs: dict[int, object] = {}
        self._orca_completed_jobs: set[int] = set()

    def call_job_fn(self, key, job_fn, args, context):
        """Start a worker and retain its process handle in the server process."""

        from multiprocess import Process

        process = Process(
            target=job_fn,
            args=(key, self._make_progress_key(key), args, context),
        )
        process.start()
        if process.pid is None:
            return None
        self._orca_completed_jobs.discard(process.pid)
        self._orca_jobs[process.pid] = process
        return process.pid

    def job_running(self, job) -> bool:
        if job is None:
            return False
        job_id = int(job)
        if job_id in self._orca_completed_jobs:
            return False
        process = self._orca_jobs.get(job_id)
        if process is not None:
            if process.is_alive():
                return True
            process.join(timeout=0)
            self._orca_jobs.pop(job_id, None)
            self._orca_completed_jobs.add(job_id)
            return False
        try:
            return super().job_running(job_id)
        except (PermissionError, psutil.AccessDenied):
            return False

    def terminate_job(self, job) -> None:
        if job is None:
            return
        job_id = int(job)
        if job_id in self._orca_completed_jobs:
            return
        process = self._orca_jobs.pop(job_id, None)
        if process is not None:
            process.join(timeout=0.25)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
            self._orca_completed_jobs.add(job_id)
            return
        try:
            super().terminate_job(job_id)
        except (PermissionError, psutil.AccessDenied):
            # The completed worker exits on its own. In restricted macOS
            # sessions, process-tree cleanup is unavailable but the cached
            # callback result is still safe to return to the page.
            self._orca_completed_jobs.add(job_id)
            return


def _sidebar() -> html.Aside:
    groups: list = []
    for group, entries in NAV_GROUPS:
        groups.append(
            html.Div(
                [
                    html.Div(group, className="orca-nav-group-label"),
                    html.Nav(
                        [
                            dcc.Link(
                                NAV_LABELS[page.PATH],
                                href=page.PATH,
                                id=NAV_IDS[page.PATH],
                                className=NAV_BASE_CLASSES[page.PATH],
                            )
                            for page, _is_child in entries
                        ],
                        className="orca-nav-links",
                        **{"aria-label": f"{group} navigation"},
                    ),
                ],
                className="orca-nav-group",
            )
        )
    return html.Aside(
        [
            dcc.Link(
                [html.Span("O", className="orca-mark"), html.Div([html.Strong("Orca"), html.Small("Bayesian inference for immune cell decisions")])],
                href="/",
                className="orca-brand",
            ),
            html.Div(groups, className="orca-nav"),
            html.Div(
                [
                    html.Span("Data use", className="orca-preview-label"),
                    html.P("Use synthetic or approved anonymous data. Inputs are not intentionally retained.", className="orca-sidebar-warning"),
                ],
                className="orca-sidebar-footer",
            ),
        ],
        className="orca-sidebar",
    )


def _not_found(pathname: str) -> html.Div:
    return html.Div(
        [
            html.Span("404", className="orca-section-label"),
            html.H1("This Orca page does not exist"),
            html.P(f"No page is registered at {pathname!r}."),
            dcc.Link("Return home", href="/", className="orca-button primary"),
        ],
        className="orca-not-found",
    )


def create_app() -> Dash:
    asset_folder = Path(__file__).resolve().parent / "assets"
    cache_directory = Path(
        os.environ.get(
            "ORCA_BACKGROUND_CACHE",
            f"/tmp/orca-dash-background-{os.getpid()}",
        )
    )
    background_manager = OrcaDiskcacheManager(
        diskcache.Cache(str(cache_directory), size_limit=512 * 1024 * 1024)
    )
    app = Dash(
        __name__,
        assets_folder=str(asset_folder),
        title="Orca · Bayesian inference",
        suppress_callback_exceptions=True,
        update_title="Orca is working…",
        background_callback_manager=background_manager,
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"},
            {"name": "description", "content": "Bayesian inference for heterogeneity in immune cell decision making."},
            {"name": "theme-color", "content": "#304B3D"},
        ],
    )
    app.layout = html.Div(
        [
            dcc.Location(id="orca-location", refresh=False),
            _sidebar(),
            html.Main(html.Div(id="orca-page", className="orca-page-inner"), className="orca-main"),
        ],
        className="orca-shell",
    )

    nav_outputs = [Output(NAV_IDS[page.PATH], "className") for page in PAGES]

    @app.callback(Output("orca-page", "children"), *nav_outputs, Input("orca-location", "pathname"))
    def route(pathname: str | None):
        normalized = pathname or "/"
        page = PAGE_BY_PATH.get(normalized)
        content = page.layout() if page is not None else _not_found(normalized)
        active_path = page.PATH if page is not None else normalized
        classes = [
            f"{NAV_BASE_CLASSES[current.PATH]} active" if current.PATH == active_path else NAV_BASE_CLASSES[current.PATH]
            for current in PAGES
        ]
        return content, *classes

    for prefix in ("synthetic", "counts", "donor", "trajectory"):
        @app.callback(
            Output(f"{prefix}-particles", "value"),
            Output(f"{prefix}-chains", "value"),
            Output(f"{prefix}-cores", "value"),
            Input(f"{prefix}-profile", "value"),
        )
        def update_profile(profile: str, _prefix: str = prefix):
            del _prefix
            return PROFILE_VALUES.get(profile, PROFILE_VALUES["preview"])

    for page in PAGES:
        register = getattr(page, "register_callbacks", None)
        if register is not None:
            register(app)

    @app.server.route("/healthz")
    def health_check():
        return {"status": "ok", "application": "orca-dash"}, 200

    return app
