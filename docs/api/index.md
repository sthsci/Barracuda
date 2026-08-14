# API reference

The reference is generated from source docstrings and type annotations by
mkdocstrings. Public modules are intentionally separated from `_core` and
`_backends`; names under those private packages may change without deprecation.

| Module | Scope |
|---|---|
| [`bayesorca.event_counts`](event-counts.md) | Count/donor/condition validation, simulation, inference, tables, archives |
| [`bayesorca.trajectories`](trajectories.md) | Trajectory I/O, simulation, inference, tables, archives |
| [`bayesorca.donors`](donors.md) | Donor simulation, posterior extraction, mixture moments, contrasts |
| [`bayesorca.evidence`](evidence.md) | Directed Bayes factors, model probabilities, independent aggregation, Savage–Dickey |
| [`bayesorca.validation`](validation.md) | Typed scenarios, recovery, coverage, superiority/ROPE, validation runners |
| [`bayesorca.scans`](scans.md) | Cumulative-prefix count and trajectory Bayes-factor scans |
| [`bayesorca.diagnostics`](diagnostics.md) | SMC, posterior, population, and trajectory diagnostics |
| [`bayesorca.plotting`](plotting.md) | Optional Matplotlib figures returning axes |
| [`bayesorca.io`](io.md) | Checksummed inference and scan persistence |
| [`bayesorca.progress`](progress.md) | Per-chain SMC stage/tempering progress bridge |

## Stability

Version 0.2 is alpha. Public names listed here are the intended user surface,
but signatures and result schemas can still evolve. Pin versions and read the
[changelog](../changelog.md). Private modules beginning with `_` are not part of
the compatibility contract.

## Top-level convenience exports

The package root re-exports common settings/result types and the primary event,
trajectory, donor, evidence, validation, and scan functions. Complete donor
analysis, diagnostics, persistence, plotting, and lower-frequency helpers remain
organized in the named modules above.

::: bayesorca
    options:
      members: true

## Error contract

Public validation and table helpers prefer `TypeError`, `ValueError`, and
`RuntimeError` with actionable messages. Inference may also propagate PyMC or
PyTensor exceptions. See [Settings, results, and errors](../reference/settings-results-errors.md).
