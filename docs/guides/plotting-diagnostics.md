# Plotting and diagnostics

Diagnostics return data; plots consume tidy data. Keeping the two layers
separate makes batch validation, alternative plotting libraries, and testing
possible without a display server.

## SMC evidence diagnostics

```python
from bayesorca.diagnostics import smc_evidence_summary, smc_log_evidence_by_chain

per_chain = smc_log_evidence_by_chain(idata)
summary = smc_evidence_summary(idata)
```

Historical `InferenceData` layouts with a stage dimension or attribute fallback
are accepted. Chains without a finite final estimate remain visible with
`NaN`; they are not silently dropped from the reported chain count.

## Posterior diagnostics

```python
from bayesorca.diagnostics import diagnostic_flags, posterior_diagnostics

table = posterior_diagnostics(idata, hdi_prob=0.95)
flagged = diagnostic_flags(
    table,
    min_ess_bulk=100,
    min_ess_tail=100,
    max_r_hat=1.01,
)
```

R-hat is unavailable with one chain. Missing R-hat produces a `limited` status,
not a false pass. ESS/R-hat thresholds are transparent review flags rather than
an automatic validity certificate, especially for SMC particles.

## Trajectory summaries

`population_p0_draws` samples cell-level baseline lethal probabilities from
posterior `mu_eta`/`sigma_eta` pairs while preserving parameter-draw rows.
`population_p0_summary` flattens those simulated probabilities into quantiles.

`trajectory_state_summary` counts contacts and lethal outcomes at every
pre-contact state. It accepts canonical/expanded trajectory data and returns a
tidy empirical probability table.

## Optional plotting contract

Matplotlib is installed with `bayesorca`. Every plotting function:

- accepts an optional existing `ax`;
- returns a Matplotlib `Axes`;
- never calls `show()`;
- never writes a file;
- validates required tidy columns;
- preserves explicitly directed Bayes-factor labels.

```python
from bayesorca.plotting import plot_bayes_factor_scan, plot_parameter_recovery

ax = plot_bayes_factor_scan(scan, scenario="No1")
ax.figure.savefig("scan.svg", bbox_inches="tight")

ax = plot_parameter_recovery(recovery, parameter="mu_lambda")
```

## Available figures

| Function | Input |
|---|---|
| `plot_event_count_distribution` | Count table, optionally with condition |
| `plot_rate_distribution` | Rate-law name and moments |
| `plot_model_evidence` | One-condition evidence table |
| `plot_bayes_factor_scan` | One-scenario long scan table |
| `plot_parameter_recovery` | One-parameter recovery table |
| `plot_posterior_intervals` | Generic tidy mean/HDI table |
| `plot_posterior_pair` | Paired posterior draws |
| `plot_trajectory_state_map` | Trajectory data or state summary |

`plot_posterior_pair` subsamples complete rows within groups, retaining joint
dependence. A marginal draw independently sampled for each axis would create a
scientifically false joint distribution.
