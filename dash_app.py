"""Development and WSGI entrypoint for the Orca Dash application."""

from __future__ import annotations

import os

from webapp.dashapp import create_app


app = create_app()
server = app.server


if __name__ == "__main__":
    app.run(
        host=os.environ.get("ORCA_HOST", "127.0.0.1"),
        port=int(os.environ.get("ORCA_PORT", "8501")),
        debug=os.environ.get("ORCA_DEBUG", "0") == "1",
    )
