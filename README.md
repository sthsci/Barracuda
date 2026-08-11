# ORCA web application

This branch is the standalone Dash web application for ORCA. It intentionally
contains only the web interface, its static assets, deployment configuration,
and web tests.

The complete project remains on `main`. Manuscript assets are maintained on
`paper`, and the reusable Python distribution is maintained on `pypackage`.
The web application installs that package as its scientific inference backend.

## Run locally

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python dash_app.py
```

The application listens on `http://127.0.0.1:8501` by default. Environment
variables `ORCA_HOST`, `ORCA_PORT`, and `ORCA_DEBUG` control the development
server.

## Test

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

## Container

```bash
docker build -t orca-web .
docker run --rm -p 8501:8501 orca-web
```

Until `bayesorca` is published on PyPI, `requirements.txt` installs it directly
from the repository's `pypackage` branch. After the first PyPI release, replace
that direct URL with an exact release pin such as `bayesorca==0.1.0`.
