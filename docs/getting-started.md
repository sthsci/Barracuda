# Getting started

## Create an environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install barracuda
```

For a source checkout, use `python -m pip install -e ".[test,build,docs]"`.

## Run a small event-count workflow

```python
from barracuda import (
    InferenceSettings,
    evidence_table,
    run_count_models,
    simulate_event_counts,
)

frame, truth = simulate_event_counts(
    model_key="hetero3",
    n_cells=50,
    obs_time=1.0,
    mu_lambda=4.0,
    sigma_lambda=2.0,
    p_zero=0.2,
    seed=2026,
)

settings = InferenceSettings(draws=256, chains=1, cores=1, seed=2026)
fits = run_count_models(
    frame,
    observation_time=1.0,
    settings=settings,
    model_keys=["homo", "z2p", "dis2p", "hetero3"],
)
print(evidence_table(fits))
```

The small SMC settings make the example approachable. They are not a
publication recommendation. Review chain-level evidence and posterior
diagnostics, then choose settings through a documented sensitivity analysis.

## Understand the result mapping

`fits["hetero3"]` is an `InferenceResult` containing:

- `idata`: the ArviZ `InferenceData` with posterior and available prior/sample
  statistics;
- `model`: the PyMC model;
- `log_evidence`: the SMC log marginal likelihood;
- `elapsed_seconds`, `n_cells`, and `observation_time`;
- model labels and donor metadata.

Use package transformations instead of reaching into backend-specific variable
names:

```python
from barracuda import posterior_draw_table, summary_table

draws = posterior_draw_table(fits)
summary = summary_table(fits, hdi_prob=0.95)
```

## Check and plot

```python
from barracuda import posterior_diagnostics, smc_evidence_summary
from barracuda import plot_model_evidence

idata = fits["hetero3"].idata
print(smc_evidence_summary(idata))
print(posterior_diagnostics(idata))

ax = plot_model_evidence(evidence_table(fits))
ax.figure.savefig("evidence.svg", bbox_inches="tight")
```

Plot functions return axes and never show or save implicitly.

## Next steps

- Use [input schemas](reference/input-schemas.md) for your own data.
- Read [Evidence and Bayes factors](concepts/evidence.md) before interpreting
  model comparisons.
- Use [Scientific validation](guides/validation.md) to check recovery under
  known synthetic truths.
- Follow the [reproducibility checklist](reproducibility.md) before reporting a
  result.
