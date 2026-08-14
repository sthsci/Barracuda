# Development

## Branch scope

The `pypackage` branch is the standalone Python distribution. Package source,
tests, metadata, documentation, citation/license files, and release workflows
belong here. Manuscript assets remain on `paper`; the complete research tree is
on `main`; and the standalone web application is on `webpage`.

The standalone `_core` and `_backends` code originated in research/web paths.
Synchronize scientific changes deliberately and verify blob/API parity where
appropriate. Do not make hidden fixes only in a frontend copy.

## Set up

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,build,plot,docs]"
```

## Validate a change

```bash
python -m pytest -q
python -m build
python -m twine check dist/*
python -m mkdocs build --strict
```

For scientific code, add fast deterministic unit tests and a small mocked or
explicitly opt-in SMC integration test. Do not make manuscript-scale inference
a pull-request prerequisite.

## Public API and typing

- Add user-facing functions to a public module and its `__all__`.
- Keep implementation-only names under `_core`/`_backends` or prefix them `_`.
- Preserve public parameter terminology even when a backend uses a historical
  name.
- Add complete docstrings with parameters, return type, errors, direction
  conventions, cost, and scientific assumptions.
- Maintain `py.typed` and annotations.
- Update the README, relevant guide, API reference, tests, and changelog.

## Documentation locally

```bash
python -m mkdocs serve
```

The published site uses MkDocs Material and mkdocstrings. `mkdocs build
--strict` treats missing pages, invalid references, and warnings as failures.

## GitHub Pages

The workflow in `.github/workflows/docs.yml` builds on relevant pull requests
and pushes. It deploys only a push to `pypackage` through the official Pages
artifact actions. Before the first deployment, an administrator must set
**Settings → Pages → Source** to **GitHub Actions** and verify the
`github-pages` environment/organization policy.

## Release checklist

1. Update the version consistently in `pyproject.toml`, `CITATION.cff`, and the
   source-checkout fallback.
2. Update this changelog and stability notes.
3. Run the complete test, build, wheel-import, and strict-doc checks.
4. Inspect wheel and sdist contents for private/generated material.
5. Tag the intended `pypackage` commit and confirm the release workflow checks
   out that tag.
6. Publish through the protected `pypi` environment/trusted publisher.
7. Verify PyPI metadata and the Pages deployment.
