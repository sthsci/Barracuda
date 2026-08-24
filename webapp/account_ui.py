"""Optional account, saved dataset and CSV sharing controls for Dash.

The scientific workflows stay inside Dash.  This module only talks to the
separate Django persistence API when the user explicitly opens the account
panel.  Tokens live in browser session storage so closing the tab returns the
application to its normal account-free behaviour.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd
from dash import Input, Output, State, ctx, dcc, html, no_update

from webapp.ui import hero


API_URL = os.environ.get("BARRACUDA_ACCOUNT_API_URL", "http://127.0.0.1:8000/api/v1").rstrip("/")
PUBLIC_API_URL = os.environ.get("BARRACUDA_PUBLIC_API_URL", API_URL).rstrip("/")
PUBLIC_DASH_URL = os.environ.get("BARRACUDA_PUBLIC_DASH_URL", "http://127.0.0.1:8501").rstrip("/")


def _message(title: str, body: str, *, error: bool = False) -> html.Div:
    return html.Div(
        [html.Strong(title), html.Span(body)],
        className=f"barracuda-account-message{' error' if error else ''}",
        role="alert" if error else "status",
    )


def _error_text(payload: Any, fallback: str) -> str:
    if not isinstance(payload, dict):
        return fallback
    detail = payload.get("error", {}).get("detail", payload.get("detail"))
    if isinstance(detail, dict):
        parts: list[str] = []
        for key, value in detail.items():
            if isinstance(value, list):
                value = " ".join(str(item) for item in value)
            parts.append(f"{key}: {value}")
        return " ".join(parts) or fallback
    return str(detail or fallback)


def _request(
    path: str,
    *,
    method: str = "GET",
    token: str | None = None,
    guest_token: str | None = None,
    payload: dict[str, Any] | None = None,
    file_payload: tuple[str, bytes] | None = None,
    timeout: float = 12,
) -> tuple[int, Any]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Token {token}"
    if guest_token:
        headers["X-Barracuda-Guest-Token"] = guest_token
    data: bytes | None = None
    if file_payload is not None:
        filename, content = file_payload
        boundary = "----BarracudaCsvBoundary"
        pieces = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"project_id\"\r\n\r\n{payload['project_id']}\r\n".encode(),
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                f"filename=\"{filename.replace(chr(34), '')}\"\r\nContent-Type: text/csv\r\n\r\n"
            ).encode(),
            content,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
        data = b"".join(pieces)
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
    elif payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(f"{API_URL}/{path.lstrip('/')}" , data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
            return response.status, json.loads(body) if body else None
    except HTTPError as exc:
        body = exc.read()
        try:
            parsed = json.loads(body) if body else None
        except json.JSONDecodeError:
            parsed = None
        return exc.code, parsed
    except (URLError, TimeoutError, OSError) as exc:
        raise ConnectionError(
            "The optional account service is not running. Dash analysis is still available without an account."
        ) from exc


def _auth_headers(session: dict[str, Any] | None) -> tuple[str | None, str | None]:
    session = session or {}
    return session.get("token"), session.get("guest_token")


def _download(path: str, *, token: str | None, guest_token: str | None) -> tuple[bytes, str]:
    headers = {"Accept": "text/csv"}
    if token:
        headers["Authorization"] = f"Token {token}"
    if guest_token:
        headers["X-Barracuda-Guest-Token"] = guest_token
    request = Request(f"{API_URL}/{path.lstrip('/')}", headers=headers, method="GET")
    try:
        with urlopen(request, timeout=12) as response:
            disposition = response.headers.get("Content-Disposition", "")
            filename = "bayesian_barracuda_saved.csv"
            if "filename=" in disposition:
                filename = disposition.split("filename=", 1)[1].strip().strip('"')
            return response.read(), filename
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise ConnectionError("The saved CSV could not be downloaded.") from exc


def _csv_from_records(records: list[dict[str, Any]] | None) -> bytes:
    if not records:
        raise ValueError("Load or enter a dataset before saving it.")
    return pd.DataFrame(records).to_csv(index=False).encode("utf-8")


def shell_components() -> list[Any]:
    """Persistent account state and the routed workspace controls.

    The controls are rendered on their own routed page. Persistent stores live
    in the application shell so a normalized analysis table survives ordinary
    Dash navigation until the user deliberately saves it.
    """

    return [
        dcc.Store(id="barracuda-account-session", storage_type="session"),
        dcc.Store(id="barracuda-account-projects", storage_type="session"),
        dcc.Store(id="barracuda-counts-snapshot"),
        dcc.Store(id="barracuda-donor-snapshot"),
        dcc.Store(id="barracuda-trajectory-snapshot"),
        dcc.Download(id="barracuda-saved-dataset-download"),
        html.Div(
            [
                html.Span("Optional service", className="barracuda-section-label"),
                html.H2("Account and CSV sharing"),
                html.P(
                    "You can keep using every analysis without registering. Sign in only to save CSV datasets for later use or create a controlled spreadsheet link.",
                    className="barracuda-help",
                ),
                html.Div(id="barracuda-account-summary"),
                html.Button("Sign out", id="barracuda-logout-submit", n_clicks=0, className="barracuda-button tertiary small"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Sign in"),
                                html.Div(
                                [
                                    html.Label([html.Span("Username", className="barracuda-field-label"), dcc.Input(id="barracuda-login-username", type="text", autoComplete="username")], className="barracuda-field"),
                                    html.Label([html.Span("Password", className="barracuda-field-label"), dcc.Input(id="barracuda-login-password", type="password", autoComplete="current-password")], className="barracuda-field"),
                                    html.Button("Sign in", id="barracuda-login-submit", n_clicks=0, className="barracuda-button primary full"),
                                ],
                                className="barracuda-account-form",
                            ),
                            ],
                            className="barracuda-account-auth-card",
                        ),
                        html.Div(
                            [
                                html.H3("Register"),
                                html.Div(
                                [
                                    html.Label([html.Span("Username", className="barracuda-field-label"), dcc.Input(id="barracuda-register-username", type="text", autoComplete="username")], className="barracuda-field"),
                                    html.Label([html.Span("Email (optional)", className="barracuda-field-label"), dcc.Input(id="barracuda-register-email", type="email", autoComplete="email")], className="barracuda-field"),
                                    html.Label([html.Span("Password", className="barracuda-field-label"), dcc.Input(id="barracuda-register-password", type="password", autoComplete="new-password")], className="barracuda-field"),
                                    html.Button("Create account", id="barracuda-register-submit", n_clicks=0, className="barracuda-button primary full"),
                                ],
                                className="barracuda-account-form",
                            ),
                            ],
                            className="barracuda-account-auth-card",
                        ),
                    ],
                    className="barracuda-account-auth-grid",
                ),
                html.Div(id="barracuda-account-auth-status"),
                html.H3("Saved projects"),
                html.Button("Refresh projects", id="barracuda-projects-refresh", n_clicks=0, className="barracuda-button secondary small"),
                dcc.Dropdown(id="barracuda-project-select", options=[], value=None, placeholder="Choose a saved project", clearable=True),
                html.Div(id="barracuda-projects-list"),
                html.H3("Saved CSV datasets"),
                dcc.Dropdown(id="barracuda-dataset-select", options=[], value=None, placeholder="Choose a saved CSV", clearable=True),
                html.Button("Download selected CSV", id="barracuda-dataset-download", n_clicks=0, className="barracuda-button secondary full"),
                html.Div(id="barracuda-datasets-status"),
                html.H3("Save a CSV dataset"),
                html.Label([html.Span("New project name", className="barracuda-field-label"), dcc.Input(id="barracuda-project-name", type="text", placeholder="My analysis")], className="barracuda-field"),
                dcc.Dropdown(
                    id="barracuda-save-source",
                    options=[
                        {"label": "Current donor ignorant count table", "value": "counts"},
                        {"label": "Current donor aware count table", "value": "donor"},
                        {"label": "Current trajectory table", "value": "trajectory"},
                    ],
                    value="counts",
                    clearable=False,
                ),
                html.Button("Save current CSV", id="barracuda-save-dataset", n_clicks=0, className="barracuda-button primary full"),
                html.Div(id="barracuda-save-status"),
                html.H3("Share a saved spreadsheet"),
                html.P("Only the CSV is shared. Imaging files are not accepted. Account holders can explicitly allow CSV download; guest projects never expose raw data.", className="barracuda-help"),
                dcc.Checklist(id="barracuda-share-allow-csv", options=[{"label": "Allow the recipient to download the CSV", "value": "allow"}], value=[]),
                html.Button("Create read-only link", id="barracuda-share-create", n_clicks=0, className="barracuda-button secondary full"),
                html.Div(id="barracuda-share-status"),
            ],
            id="barracuda-account-panel",
            className="barracuda-account-workspace-shell is-hidden",
            role="region",
            **{"aria-label": "Account and CSV sharing"},
        ),
    ]


def persistent_components() -> list[Any]:
    """State shared between scientific analysis pages and the workspace."""

    return [
        dcc.Store(id="barracuda-account-session", storage_type="session"),
        dcc.Store(id="barracuda-account-projects", storage_type="session"),
        dcc.Store(id="barracuda-counts-snapshot"),
        dcc.Store(id="barracuda-donor-snapshot"),
        dcc.Store(id="barracuda-trajectory-snapshot"),
        dcc.Download(id="barracuda-saved-dataset-download"),
    ]


def workspace_layout() -> html.Div:
    """Render account and CSV sharing without replacing the analysis UI."""

    return html.Div(
        [
            hero(
                "Optional workspace",
                "Save and share CSV data",
                "Every analysis works without an account. Sign in only when you want to keep a spreadsheet or create a read-only sharing link.",
                badge="Accounts optional · CSV only · No imaging files",
            ),
        ],
        className="barracuda-workspace-page",
    )


def register_callbacks(app) -> None:
    @app.callback(
        Output("barracuda-counts-snapshot", "data"),
        Input("counts-valid-data", "data", allow_optional=True),
        prevent_initial_call=True,
    )
    def remember_counts(records):
        return records

    @app.callback(
        Output("barracuda-donor-snapshot", "data"),
        Input("donor-valid-data", "data", allow_optional=True),
        prevent_initial_call=True,
    )
    def remember_donor(records):
        return records

    @app.callback(
        Output("barracuda-trajectory-snapshot", "data"),
        Input("trajectory-active-data", "data", allow_optional=True),
        prevent_initial_call=True,
    )
    def remember_trajectory(records):
        return records

    @app.callback(
        Output("barracuda-account-session", "data"),
        Output("barracuda-account-auth-status", "children"),
        Input("barracuda-login-submit", "n_clicks"),
        Input("barracuda-register-submit", "n_clicks"),
        Input("barracuda-logout-submit", "n_clicks"),
        State("barracuda-login-username", "value"),
        State("barracuda-login-password", "value"),
        State("barracuda-register-username", "value"),
        State("barracuda-register-email", "value"),
        State("barracuda-register-password", "value"),
        State("barracuda-account-session", "data"),
        prevent_initial_call=True,
    )
    def authenticate(_login, _register, _logout, login_username, login_password, register_username, register_email, register_password, current_session):
        if ctx.triggered_id == "barracuda-logout-submit":
            token, _guest_token = _auth_headers(current_session)
            if token:
                try:
                    _request("auth/logout/", method="POST", token=token, payload={})
                except ConnectionError:
                    pass
            return None, _message("Signed out", "Dash remains available without an account.")
        registering = ctx.triggered_id == "barracuda-register-submit"
        username = register_username if registering else login_username
        password = register_password if registering else login_password
        if not username or not password:
            return no_update, _message("Account details needed", "Enter both a username and password.", error=True)
        path = "auth/register/" if registering else "auth/login/"
        payload = {"username": username, "password": password}
        if registering:
            payload["email"] = register_email or ""
        try:
            status, response = _request(path, method="POST", payload=payload)
        except ConnectionError as exc:
            return no_update, _message("Account service unavailable", str(exc), error=True)
        if status not in {200, 201}:
            return no_update, _message("Account request failed", _error_text(response, "Check the details and try again."), error=True)
        session = {"token": response["token"], "username": response["user"]["username"]}
        return session, _message("Signed in", f"Signed in as {session['username']}.")

    @app.callback(
        Output("barracuda-account-summary", "children"),
        Input("barracuda-account-session", "data"),
    )
    def account_summary(session):
        if session and session.get("token"):
            return _message("Account active", f"Signed in as {session.get('username', 'researcher')}. Saved projects do not expire automatically.")
        return _message("Guest mode", "The scientific pages work normally. Nothing is saved unless you explicitly use this panel.")

    @app.callback(
        Output("barracuda-account-projects", "data"),
        Output("barracuda-project-select", "options"),
        Output("barracuda-projects-list", "children"),
        Input("barracuda-projects-refresh", "n_clicks", allow_optional=True),
        Input("barracuda-account-session", "data"),
    )
    def refresh_projects(_clicks, session):
        token, guest_token = _auth_headers(session)
        if not token and not guest_token:
            return [], [], html.P("Sign in to save projects across sessions.", className="barracuda-help")
        try:
            status, payload = _request("projects/", token=token, guest_token=guest_token)
        except ConnectionError as exc:
            return [], [], _message("Could not load projects", str(exc), error=True)
        if status != 200:
            return [], [], _message("Could not load projects", _error_text(payload, "Try signing in again."), error=True)
        projects = payload.get("results", payload) if isinstance(payload, dict) else payload
        projects = list(projects or [])
        options = [{"label": item["name"], "value": item["id"]} for item in projects]
        listing = html.Ul([html.Li(f"{item['name']} · {item['owner_type']}") for item in projects], className="barracuda-account-project-list") if projects else html.P("No saved projects yet.", className="barracuda-help")
        return projects, options, listing

    @app.callback(
        Output("barracuda-dataset-select", "options"),
        Output("barracuda-datasets-status", "children"),
        Input("barracuda-project-select", "value"),
        State("barracuda-account-session", "data"),
    )
    def refresh_datasets(project_id, session):
        if not project_id:
            return [], html.P("Choose a project to see its saved CSV files.", className="barracuda-help")
        token, guest_token = _auth_headers(session)
        try:
            status, payload = _request(
                f"datasets/?project_id={quote(str(project_id), safe='')}",
                token=token,
                guest_token=guest_token,
            )
        except ConnectionError as exc:
            return [], _message("Could not load CSV files", str(exc), error=True)
        if status != 200:
            return [], _message("Could not load CSV files", _error_text(payload, "Try signing in again."), error=True)
        datasets = payload.get("results", payload) if isinstance(payload, dict) else payload
        datasets = list(datasets or [])
        options = [
            {
                "label": f"{item['original_name']} · {item['row_count']:,} rows",
                "value": item["id"],
            }
            for item in datasets
        ]
        return options, html.P(f"{len(options)} saved CSV file{'s' if len(options) != 1 else ''}.", className="barracuda-help")

    @app.callback(
        Output("barracuda-saved-dataset-download", "data"),
        Output("barracuda-datasets-status", "children", allow_duplicate=True),
        Input("barracuda-dataset-download", "n_clicks"),
        State("barracuda-dataset-select", "value"),
        State("barracuda-account-session", "data"),
        prevent_initial_call=True,
    )
    def download_dataset(_clicks, dataset_id, session):
        if not dataset_id:
            return no_update, _message("Choose a CSV", "Select a saved dataset first.", error=True)
        token, guest_token = _auth_headers(session)
        try:
            payload, filename = _download(
                f"datasets/{quote(str(dataset_id), safe='')}/download/",
                token=token,
                guest_token=guest_token,
            )
        except ConnectionError as exc:
            return no_update, _message("Download failed", str(exc), error=True)
        return dcc.send_bytes(payload, filename), _message("Download ready", filename)

    @app.callback(
        Output("barracuda-save-status", "children"),
        Output("barracuda-projects-refresh", "n_clicks", allow_duplicate=True),
        Input("barracuda-save-dataset", "n_clicks"),
        State("barracuda-account-session", "data"),
        State("barracuda-project-select", "value"),
        State("barracuda-project-name", "value"),
        State("barracuda-save-source", "value"),
        State("barracuda-counts-snapshot", "data"),
        State("barracuda-donor-snapshot", "data"),
        State("barracuda-trajectory-snapshot", "data"),
        State("barracuda-projects-refresh", "n_clicks"),
        prevent_initial_call=True,
    )
    def save_dataset(_clicks, session, project_id, project_name, source, counts, donor, trajectory, refresh_clicks):
        token, guest_token = _auth_headers(session)
        if not token and not guest_token:
            return _message("Sign in first", "Create an account or sign in before saving a CSV.", error=True), no_update
        records = {"counts": counts, "donor": donor, "trajectory": trajectory}.get(source)
        try:
            csv_bytes = _csv_from_records(records)
            if not project_id:
                status, project = _request(
                    "projects/",
                    method="POST",
                    token=token,
                    guest_token=guest_token,
                    payload={"name": (project_name or "Saved analysis").strip(), "description": "Saved from the BARRACUDA Dash application."},
                )
                if status != 201:
                    raise ValueError(_error_text(project, "The project could not be created."))
                project_id = project["id"]
            status, dataset = _request(
                "datasets/",
                method="POST",
                token=token,
                guest_token=guest_token,
                payload={"project_id": project_id},
                file_payload=(f"bayesian_barracuda_{source}.csv", csv_bytes),
            )
            if status != 201:
                raise ValueError(_error_text(dataset, "The CSV could not be saved."))
        except (ConnectionError, ValueError) as exc:
            return _message("CSV not saved", str(exc), error=True), no_update
        return _message("CSV saved", f"Saved {dataset['original_name']} with {dataset['row_count']:,} rows."), int(refresh_clicks or 0) + 1

    @app.callback(
        Output("barracuda-share-status", "children"),
        Input("barracuda-share-create", "n_clicks"),
        State("barracuda-account-session", "data"),
        State("barracuda-project-select", "value"),
        State("barracuda-share-allow-csv", "value"),
        prevent_initial_call=True,
    )
    def create_share(_clicks, session, project_id, allow_values):
        if not project_id:
            return _message("Choose a project", "Select the saved project you want to share.", error=True)
        token, guest_token = _auth_headers(session)
        try:
            status, share = _request(
                "share-links/",
                method="POST",
                token=token,
                guest_token=guest_token,
                payload={"project_id": project_id, "expires_in_hours": 72, "allow_dataset_download": "allow" in (allow_values or [])},
            )
        except ConnectionError as exc:
            return _message("Share link not created", str(exc), error=True)
        if status != 201:
            return _message("Share link not created", _error_text(share, "Check the selected project."), error=True)
        raw_token = quote(share["share_token"], safe="")
        share_url = f"{PUBLIC_DASH_URL}/shared?token={raw_token}"
        return html.Div(
            [
                html.Strong("Read-only link created"),
                dcc.Input(value=share_url, readOnly=True, className="barracuda-account-share-url"),
                html.Small("The link expires automatically and can be revoked from the API. Raw CSV download is available only when explicitly enabled for an account-owned project."),
            ],
            className="barracuda-account-message",
        )


__all__ = [
    "API_URL",
    "PUBLIC_API_URL",
    "PUBLIC_DASH_URL",
    "persistent_components",
    "register_callbacks",
    "shell_components",
    "workspace_layout",
]
