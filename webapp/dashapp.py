"""Dash application factory for Barracuda."""

from __future__ import annotations

import os
from pathlib import Path

import diskcache
import psutil
from dash import Dash, DiskcacheManager, Input, Output, dcc, html

from webapp.analysis_ui import PROFILE_VALUES
from webapp.account_ui import register_callbacks as register_account_callbacks
from webapp.account_ui import shell_components as account_shell_components
from webapp.pages import bayes_101, donor_aware, event_counts, event_counts_overview, home, notebooks, python_api, shared, synthetic_validation, trajectory, workspace
from webapp.ui import app_header


PAGES = [home, bayes_101, notebooks, python_api, event_counts_overview, event_counts, donor_aware, trajectory, workspace]
PAGE_BY_PATH = {page.PATH: page for page in PAGES}
# The former standalone validation URL now opens the merged donor ignorant
# workflow. Keeping the alias avoids breaking saved links without restoring a
# second copy of the page in navigation.
PAGE_BY_PATH[synthetic_validation.PATH] = event_counts
PAGE_BY_PATH["/donor-aware"] = donor_aware
PAGE_BY_PATH[shared.PATH] = shared
TOP_NAV = (
    ("Home", home.PATH, "nav-home"),
    ("Event counts", event_counts_overview.PATH, "nav-event-counts"),
    ("Contact histories", trajectory.PATH, "nav-contact-histories"),
    ("Learn", bayes_101.PATH, "nav-learn"),
    ("Resources", notebooks.PATH, "nav-resources"),
)
NAV_IDS = {path: item_id for _label, path, item_id in TOP_NAV}
NAV_SECTION = {
    home.PATH: home.PATH,
    event_counts_overview.PATH: event_counts_overview.PATH,
    event_counts.PATH: event_counts_overview.PATH,
    donor_aware.PATH: event_counts_overview.PATH,
    synthetic_validation.PATH: event_counts_overview.PATH,
    trajectory.PATH: trajectory.PATH,
    bayes_101.PATH: bayes_101.PATH,
    notebooks.PATH: notebooks.PATH,
    python_api.PATH: notebooks.PATH,
}
PAGE_TYPE = {
    home.PATH: "home",
    bayes_101.PATH: "learning",
    notebooks.PATH: "resource",
    python_api.PATH: "resource",
    event_counts_overview.PATH: "landing",
    event_counts.PATH: "analysis",
    donor_aware.PATH: "analysis",
    trajectory.PATH: "analysis",
    workspace.PATH: "workspace",
}


class BarracudaDiskcacheManager(DiskcacheManager):
    """Collect completed jobs even when macOS blocks process-tree inspection.

    Dash normally asks ``psutil`` to inspect and terminate the worker process
    after its result has been read from diskcache. Sandboxed macOS sessions can
    deny that process-tree query. The worker has already written its result at
    this point, so cleanup failure must not discard an otherwise valid fit.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._barracuda_jobs: dict[int, object] = {}
        self._barracuda_completed_jobs: set[int] = set()

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
        self._barracuda_completed_jobs.discard(process.pid)
        self._barracuda_jobs[process.pid] = process
        return process.pid

    def job_running(self, job) -> bool:
        if job is None:
            return False
        job_id = int(job)
        if job_id in self._barracuda_completed_jobs:
            return False
        process = self._barracuda_jobs.get(job_id)
        if process is not None:
            if process.is_alive():
                return True
            process.join(timeout=0)
            self._barracuda_jobs.pop(job_id, None)
            self._barracuda_completed_jobs.add(job_id)
            return False
        try:
            return super().job_running(job_id)
        except (PermissionError, psutil.AccessDenied):
            return False

    def terminate_job(self, job) -> None:
        if job is None:
            return
        job_id = int(job)
        if job_id in self._barracuda_completed_jobs:
            return
        process = self._barracuda_jobs.pop(job_id, None)
        if process is not None:
            process.join(timeout=0.25)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
            self._barracuda_completed_jobs.add(job_id)
            return
        try:
            super().terminate_job(job_id)
        except (PermissionError, psutil.AccessDenied):
            # The completed worker exits on its own. In restricted macOS
            # sessions, process-tree cleanup is unavailable but the cached
            # callback result is still safe to return to the page.
            self._barracuda_completed_jobs.add(job_id)
            return


def _not_found(pathname: str) -> html.Div:
    return html.Div(
        [
            html.Span("404", className="barracuda-section-label"),
            html.H1("This Barracuda page does not exist"),
            html.P(f"No page is registered at {pathname!r}."),
            dcc.Link("Return home", href="/", className="barracuda-button primary"),
        ],
        className="barracuda-not-found",
    )


def create_app() -> Dash:
    asset_folder = Path(__file__).resolve().parent / "assets"
    cache_directory = Path(
        os.environ.get(
            "BARRACUDA_BACKGROUND_CACHE",
            f"/tmp/barracuda-dash-background-{os.getpid()}",
        )
    )
    background_manager = BarracudaDiskcacheManager(
        diskcache.Cache(str(cache_directory), size_limit=512 * 1024 * 1024)
    )
    app = Dash(
        __name__,
        assets_folder=str(asset_folder),
        title="BARRACUDA",
        # Page controls are mounted only for the active route. Dash should not
        # report callbacks from the other scientific pages as browser errors
        # while their components are intentionally absent.
        suppress_callback_exceptions=True,
        update_title="BARRACUDA is working…",
        background_callback_manager=background_manager,
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"},
            {"name": "description", "content": "Bayesian inference for heterogeneity in immune cell decision making."},
            {"name": "theme-color", "content": "#F2F5F2"},
        ],
    )
    app.layout = html.Div(
        [
            dcc.Location(id="barracuda-location", refresh=False),
            html.A("Skip to content", href="#barracuda-main", className="barracuda-skip-link"),
            app_header(TOP_NAV, workspace_id="nav-workspace-utility"),
            html.Main(
                [
                    html.Div(id="barracuda-page", className="barracuda-page-inner"),
                    *account_shell_components(),
                ],
                id="barracuda-main",
                className="barracuda-main page-home",
                tabIndex=-1,
            ),
            html.Footer(
                [
                    html.Strong("Research use only."),
                    html.Span(" Use synthetic or approved anonymous data; inputs are not intentionally retained."),
                    dcc.Link("Privacy and workspace", href="/workspace"),
                ],
                className="barracuda-footer",
            ),
        ],
        className="barracuda-shell",
    )
    # Every scientific page is routed into ``barracuda-page``.  Supplying Dash with
    # the complete component tree prevents transient "nonexistent object"
    # errors while a route-specific page (notably the optional workspace) is
    # mounting, without keeping hidden duplicate controls in the live DOM.
    app.validation_layout = html.Div([app.layout, *(page.layout() for page in PAGES)])

    nav_outputs = [
        output
        for _label, _path, item_id in TOP_NAV
        for output in (
            Output(item_id, "className"),
            Output(f"{item_id}-mobile", "className"),
            Output(item_id, "aria-current"),
            Output(f"{item_id}-mobile", "aria-current"),
        )
    ]

    @app.callback(
        Output("barracuda-page", "children"),
        *nav_outputs,
        Output("nav-workspace-utility", "className"),
        Output("nav-workspace-utility", "aria-current"),
        Output("barracuda-account-panel", "className"),
        Output("barracuda-main", "className"),
        Input("barracuda-location", "pathname"),
    )
    def route(pathname: str | None):
        normalized = pathname or "/"
        page = PAGE_BY_PATH.get(normalized)
        content = page.layout() if page is not None else _not_found(normalized)
        active_path = page.PATH if page is not None else normalized
        active_section = NAV_SECTION.get(active_path)
        nav_state: list[str | None] = []
        for _label, path, _item_id in TOP_NAV:
            active = path == active_section
            nav_state.extend([
                "barracuda-nav-link active" if active else "barracuda-nav-link",
                "barracuda-nav-link active" if active else "barracuda-nav-link",
                "page" if active else None,
                "page" if active else None,
            ])
        workspace_active = active_path == workspace.PATH
        workspace_class = (
            "barracuda-account-workspace-shell"
            if workspace_active
            else "barracuda-account-workspace-shell is-hidden"
        )
        return (
            content,
            *nav_state,
            "barracuda-workspace-link active" if workspace_active else "barracuda-workspace-link",
            "page" if workspace_active else None,
            workspace_class,
            f"barracuda-main page-{PAGE_TYPE.get(active_path, 'resource')}",
        )

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

    shared.register_callbacks(app)

    register_account_callbacks(app)

    @app.server.route("/healthz")
    def health_check():
        return {"status": "ok", "application": "barracuda-dash"}, 200

    return app
