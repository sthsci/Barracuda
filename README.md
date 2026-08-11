# bayesorca Python package

This branch is the standalone Python distribution for ORCA. It intentionally
contains only package source, package tests, build metadata, citation and
licence files, and package release workflows.

The complete ORCA project remains on `main`. Manuscript assets are maintained
on `paper`, and the standalone Dash application is maintained on `webpage`.

## Install for development

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,build]"
python -m pytest -q
```

## Build

```bash
python -m build
python -m twine check dist/*
```

The public package guide and examples are in
[`PACKAGE_README.md`](PACKAGE_README.md). Package code lives entirely under
[`src/bayesorca`](src/bayesorca); it has no source-tree dependency on the paper
or web branches.
