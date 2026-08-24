# BARRACUDA

`barracuda` is the reusable scientific API for **BARRACUDA: Bayesian Analysis
Resolving Randomness and Alternative Causes Underlying Differential Activity**.
It supports event-count and ordered-trajectory simulation, PyMC inference,
donor-aware hierarchies, marginal-likelihood model comparison, validation,
cumulative Bayes-factor scans, diagnostics, plotting, and reproducible export.

!!! warning "Alpha research software"

    Version 0.2 is an alpha API accompanying a manuscript in preparation. Pin
    exact versions and retain inputs, priors, settings, seeds, raw posterior
    draws, and evidence-direction metadata for every reported analysis.

## Choose a workflow

| Goal | Start here |
|---|---|
| Simulate or fit event counts | [Event-count guide](guides/event-counts.md) |
| Separate donor and cellular variation | [Donor-aware guide](guides/donor-aware.md) |
| Fit ordered contact-kill histories | [Trajectory guide](guides/trajectories.md) |
| Test parameter recovery | [Scientific validation](guides/validation.md) |
| Measure evidence versus sample size | [Bayes-factor scans](guides/bayes-factor-scans.md) |
| Check or visualize results | [Plotting and diagnostics](guides/plotting-diagnostics.md) |
| Look up a function | [API reference](api/index.md) |

## Install

```bash
python -m pip install barracuda
```

BARRACUDA currently supports Python 3.12. Inference uses PyMC Sequential Monte Carlo
(SMC); runtime and memory can increase steeply with particles, chains, cells,
quadrature nodes, fitted models, scenarios, and replicates.

## Core principles

1. **Validate first.** Public validators return canonical copies and reject
   ambiguous or unsafe inputs before an expensive fit starts.
2. **Keep evidence directed.** A positive `log_BF_A_vs_B` supports `A`, the
   numerator named in the column.
3. **Retain posterior draws.** A plot or rounded table cannot reproduce a fit.
4. **Separate exploration from reporting.** Smoke settings are useful for code
   checks, not scientific conclusions.
5. **Treat privacy as a data-governance responsibility.** Local execution does
   not make sensitive data suitable for publication or CI logs.

## GitHub Pages activation

The repository includes an artifact-based Pages workflow. A repository
administrator must select **Settings → Pages → Build and deployment → Source:
GitHub Actions** before the first deployment. Pull requests and pushes build the
site strictly; only pushes to `pypackage` deploy it. Repository visibility and
organization policy determine who can view the published site.

Continue with [Getting started](getting-started.md), or read the canonical
[package README](https://github.com/sthsci/Barracuda/blob/pypackage/README.md).
