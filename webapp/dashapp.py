"""Dash application factory for Barracuda."""

from __future__ import annotations

import os
from pathlib import Path

import diskcache
import psutil
from dash import ClientsideFunction, Dash, DiskcacheManager, Input, Output, dcc, html

from webapp.analysis_ui import PROFILE_VALUES
from webapp.account_ui import register_callbacks as register_account_callbacks
from webapp.account_ui import shell_components as account_shell_components
from webapp.pages import bayes_101, donor_aware, event_counts, event_counts_overview, home, notebooks, python_api, shared, synthetic_validation, trajectory, workspace


PAGES = [home, bayes_101, notebooks, python_api, event_counts_overview, event_counts, donor_aware, trajectory, workspace]
PAGE_BY_PATH = {page.PATH: page for page in PAGES}
# The former standalone validation URL now opens the merged donor ignorant
# workflow. Keeping the alias avoids breaking saved links without restoring a
# second copy of the page in navigation.
PAGE_BY_PATH[synthetic_validation.PATH] = event_counts
PAGE_BY_PATH["/donor-aware"] = donor_aware
PAGE_BY_PATH[shared.PATH] = shared
NAV_GROUPS = [
    ("Overview", [(home, False), (bayes_101, False)]),
    ("Resources", [(notebooks, False), (python_api, False)]),
    (
        "Event counts",
        [
            (event_counts_overview, False),
            (event_counts, True),
            (donor_aware, True),
        ],
    ),
    ("Trajectories", [(trajectory, False)]),
    ("Workspace", [(workspace, False)]),
]
NAV_LABELS = {
    home.PATH: "Home",
    bayes_101.PATH: "Bayesian inference 101",
    notebooks.PATH: "Google Colab notebooks",
    python_api.PATH: "Python package API",
    event_counts_overview.PATH: "Event count analysis",
    event_counts.PATH: "Donor ignorant · Data and validation",
    donor_aware.PATH: "Donor aware · Condition analysis",
    trajectory.PATH: "Trajectory inference",
    workspace.PATH: "Account and CSV sharing",
}
NAV_IDS = {page.PATH: f"nav-{index}" for index, page in enumerate(PAGES)}
NAV_BASE_CLASSES = {
    page.PATH: "barracuda-nav-link barracuda-nav-link-child" if is_child else "barracuda-nav-link"
    for _group, entries in NAV_GROUPS
    for page, is_child in entries
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


def _sidebar() -> html.Aside:
    groups: list = []
    for group, entries in NAV_GROUPS:
        groups.append(
            html.Div(
                [
                    html.Div(group, className="barracuda-nav-group-label"),
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
                        className="barracuda-nav-links",
                        **{"aria-label": f"{group} navigation"},
                    ),
                ],
                className="barracuda-nav-group",
            )
        )
    return html.Aside(
        [
            dcc.Link(
                [html.Span("B", className="barracuda-mark"), html.Div([html.Strong("BARRACUDA"), html.Small("Inference for immune cell decisions")])],
                href="/",
                className="barracuda-brand",
            ),
            html.Div(groups, className="barracuda-nav"),
            html.Details(
                [
                    html.Summary("Figure display"),
                    html.Div(
                        [
                            html.Label(
                                [
                                    html.Span("Width", className="barracuda-field-label"),
                                    dcc.Slider(
                                        id="barracuda-figure-width",
                                        min=60,
                                        max=100,
                                        step=5,
                                        value=100,
                                        marks={60: "60%", 80: "80%", 100: "100%"},
                                        persistence=True,
                                        persistence_type="local",
                                    ),
                                ],
                                className="barracuda-field",
                            ),
                            html.Label(
                                [
                                    html.Span("Height", className="barracuda-field-label"),
                                    dcc.Slider(
                                        id="barracuda-figure-height-scale",
                                        min=0.75,
                                        max=1.75,
                                        step=0.05,
                                        value=1.0,
                                        marks={0.75: "75%", 1.0: "100%", 1.75: "175%"},
                                        persistence=True,
                                        persistence_type="local",
                                    ),
                                ],
                                className="barracuda-field",
                            ),
                            html.Button(
                                "Reset figure size",
                                id="barracuda-figure-reset",
                                n_clicks=0,
                                className="barracuda-button tertiary small full",
                            ),
                            html.P(
                                "Display only. Use each figure's export menu or download buttons to save it.",
                                className="barracuda-help",
                            ),
                        ],
                        className="barracuda-sidebar-figure-body",
                    ),
                ],
                className="barracuda-sidebar-figure-controls",
            ),
            html.Div(
                [
                    html.Span("Data use", className="barracuda-preview-label"),
                    html.P("Use synthetic or approved anonymous data. Inputs are not intentionally retained.", className="barracuda-sidebar-warning"),
                ],
                className="barracuda-sidebar-footer",
            ),
        ],
        className="barracuda-sidebar",
    )


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
            {"name": "theme-color", "content": "#304B3D"},
        ],
    )
    app.layout = html.Div(
        [
            dcc.Location(id="barracuda-location", refresh=False),
            dcc.Store(id="barracuda-figure-sizing-applied"),
            _sidebar(),
            html.Main(
                [
                    html.Div(id="barracuda-page", className="barracuda-page-inner"),
                    *account_shell_components(),
                ],
                className="barracuda-main",
            ),
        ],
        className="barracuda-shell",
    )
    # Every scientific page is routed into ``barracuda-page``.  Supplying Dash with
    # the complete component tree prevents transient "nonexistent object"
    # errors while a route-specific page (notably the optional workspace) is
    # mounting, without keeping hidden duplicate controls in the live DOM.
    app.validation_layout = html.Div([app.layout, *(page.layout() for page in PAGES)])

    nav_outputs = [Output(NAV_IDS[page.PATH], "className") for page in PAGES]

    @app.callback(
        Output("barracuda-page", "children"),
        *nav_outputs,
        Output("barracuda-account-panel", "className"),
        Input("barracuda-location", "pathname"),
    )
    def route(pathname: str | None):
        normalized = pathname or "/"
        page = PAGE_BY_PATH.get(normalized)
        content = page.layout() if page is not None else _not_found(normalized)
        active_path = page.PATH if page is not None else normalized
        classes = [
            f"{NAV_BASE_CLASSES[current.PATH]} active" if current.PATH == active_path else NAV_BASE_CLASSES[current.PATH]
            for current in PAGES
        ]
        workspace_class = (
            "barracuda-account-workspace-shell"
            if active_path == workspace.PATH
            else "barracuda-account-workspace-shell is-hidden"
        )
        return content, *classes, workspace_class

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

    app.clientside_callback(
        ClientsideFunction(
            namespace="barracudaFigureControls",
            function_name="apply",
        ),
        Output("barracuda-figure-sizing-applied", "data"),
        Input("barracuda-figure-width", "value"),
        Input("barracuda-figure-height-scale", "value"),
        Input("barracuda-location", "pathname"),
    )

    app.clientside_callback(
        """
        function(clicks) {
            if (!clicks) {
                return [window.dash_clientside.no_update, window.dash_clientside.no_update];
            }
            return [100, 1.0];
        }
        """,
        Output("barracuda-figure-width", "value"),
        Output("barracuda-figure-height-scale", "value"),
        Input("barracuda-figure-reset", "n_clicks"),
        prevent_initial_call=True,
    )

    @app.server.route("/healthz")
    def health_check():
        return {"status": "ok", "application": "barracuda-dash"}, 200

    return app
