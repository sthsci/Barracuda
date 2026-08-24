"""Optional account, saved dataset, and CSV sharing workspace."""

from __future__ import annotations

from dash import html

from webapp.account_ui import workspace_layout


PATH = "/workspace"
TITLE = "Workspace"


def layout() -> html.Div:
    return workspace_layout()


__all__ = ["PATH", "TITLE", "layout"]
