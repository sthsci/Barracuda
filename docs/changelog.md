# Changelog

This project follows semantic versioning where practical during alpha
development. Alpha releases can still contain intentionally breaking API
changes; pin exact versions.

## 0.2.0 — unreleased

### Added

- Directed Bayes-factor helpers, pairwise tables, posterior model
  probabilities, independent-evidence aggregation, and Savage–Dickey/history
  effect utilities.
- Typed count/trajectory validation scenarios, recovery and coverage summaries,
  superiority/ROPE probabilities, and complete validation runners.
- Cumulative-prefix count and trajectory Bayes-factor scans with schema
  validation and replicate summaries.
- Donor-aware simulation, posterior extraction, variance decomposition,
  leave-one-donor-out mixture sensitivity, and independent-condition contrasts.
- SMC/posterior/trajectory diagnostics.
- Optional UI-neutral Matplotlib plotting.
- Atomic inference persistence and checksummed scan bundles/archives.
- MkDocs Material documentation and artifact-based GitHub Pages deployment.

### Changed

- `README.md` is the canonical package and PyPI introduction.
- Package-scale validation ceilings replace inherited web-demo limits. These
  are broad safety bounds, not practical workload recommendations; frontends
  should impose tighter limits.
- Package version advanced to 0.2.0.

### Stability notes

- Version 0.2 remains alpha.
- Bayes-factor fields encode direction explicitly as `numerator_vs_denominator`.
- Scan sample sizes are nested cumulative prefixes within each
  scenario/replicate.
- Leave-one-donor-out moments recompute mixtures without refitting.

## 0.1.0

- Initial standalone package with event-count, donor-aware, condition-wise, and
  ordered-trajectory simulation/inference, result tables, and archives.
