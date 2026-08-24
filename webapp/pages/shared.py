"""Read-only CSV sharing page served by the existing Dash application."""

from __future__ import annotations

from urllib.parse import parse_qs, quote

from dash import Input, Output, dcc, html

from webapp.account_ui import PUBLIC_API_URL, _error_text, _request
from webapp.ui import hero, note


PATH = "/shared"
TITLE = "Shared spreadsheet"


def layout() -> html.Div:
    return html.Div(
        [
            hero(
                "Read-only share",
                "Shared BARRACUDA spreadsheet",
                "Inspect a spreadsheet that a researcher explicitly shared. The link expires automatically.",
                badge="CSV only · No imaging data",
            ),
            html.Div(id="barracuda-shared-content"),
        ]
    )


def register_callbacks(app) -> None:
    @app.callback(
        Output("barracuda-shared-content", "children"),
        Input("barracuda-location", "search"),
    )
    def resolve_shared_project(search):
        token = parse_qs((search or "").lstrip("?")).get("token", [""])[0]
        if not token:
            return note("Share token missing", "Use the complete link supplied by the project owner.", tone="amber")
        try:
            status, payload = _request(f"shared/{quote(token, safe='')}/")
        except ConnectionError as exc:
            return note("Sharing service unavailable", str(exc), tone="amber")
        if status != 200 or not isinstance(payload, dict):
            return note("Share link unavailable", _error_text(payload, "The link may have expired or been revoked."), tone="amber")
        datasets = payload.get("datasets") or []
        share = payload.get("share") or {}
        dataset_content = (
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Strong(item["original_name"]),
                                    html.Span(f"{item['row_count']:,} rows · {item['column_count']} columns"),
                                ]
                            ),
                            html.A(
                                "Download CSV",
                                href=f"{PUBLIC_API_URL}/{item['download_path'].lstrip('/')}",
                                className="barracuda-button secondary small",
                            ),
                        ],
                        className="barracuda-shared-dataset-row",
                    )
                    for item in datasets
                ],
                className="barracuda-shared-datasets",
            )
            if datasets
            else note(
                "Results only",
                "The owner did not enable spreadsheet download for this link.",
                tone="navy",
            )
        )
        return html.Div(
            [
                html.Span("Shared project", className="barracuda-section-label"),
                html.H2(payload.get("name", "BARRACUDA project")),
                html.P(payload.get("description") or "No project description was supplied."),
                note(
                    "Read only",
                    f"This link expires at {share.get('expires_at', 'the configured expiry time')}. It cannot modify the project.",
                    tone="teal",
                ),
                html.H3("Shared spreadsheets"),
                dataset_content,
            ],
            className="barracuda-workflow-panel",
        )


__all__ = ["PATH", "TITLE", "layout", "register_callbacks"]
