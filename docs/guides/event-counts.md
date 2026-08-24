# Event-count workflow

## 1. Validate or simulate

For observed data, start with `validate_count_frame`. For a known synthetic
truth, use `simulate_event_counts`:

```python
from barracuda import simulate_event_counts, validate_count_frame

frame, truth = simulate_event_counts(
    model_key="dis2p",
    n_cells=100,
    obs_time=1.0,
    mu_lambda=4.0,
    sigma_lambda=2.0,
    p_zero=0.0,
    seed=2026,
)
frame = validate_count_frame(frame)
```

`truth` records the generating model and parameters. Keep it with the data;
do not reconstruct truth from a filename.

`rate_distribution_curve` evaluates the engaging-cell rate distribution for a
display grid. It is a plotting aid, not posterior inference.

## 2. Configure inference

```python
from barracuda import InferenceSettings

settings = InferenceSettings(
    draws=256,
    chains=1,
    cores=1,
    seed=2026,
)
```

`draws` is the PyMC SMC particle count. Prior bounds and scales are part of the
scientific model; changing them changes the marginal likelihood. Store the
entire dataclass rather than only the particle count.

## 3. Fit a declared candidate set

```python
from barracuda import run_count_models

fits = run_count_models(
    frame,
    observation_time=1.0,
    settings=settings,
    model_keys=["homo", "z2p", "dis2p", "hetero3"],
)
```

Candidate order is retained in output tables. A missing plausible model changes
the posterior-model-probability interpretation. A failed model must not be
quietly omitted after seeing its result.

## 4. Extract evidence and posterior information

```python
from barracuda import (
    evidence_table,
    posterior_draw_table,
    summary_table,
)

evidence = evidence_table(fits)
summary = summary_table(fits, hdi_prob=0.95)
draws = posterior_draw_table(fits, max_draws_per_model=5_000)
```

Posterior-draw subsampling selects complete rows so joint dependence between
parameters is retained. The evidence table ranks models relative to the best
fit. Its model-versus-best log Bayes factors are zero or negative.

## 5. Diagnose and export

```python
from pathlib import Path

from barracuda import build_results_zip

archive = build_results_zip(
    fits,
    frame,
    observation_time=1.0,
    settings=settings,
    truth=truth,
)
Path("count-results.zip").write_bytes(archive)
```

Archive creation is deterministic and in-memory. Writing the returned bytes is
an explicit caller action. Before reporting, inspect chain evidence and
posterior diagnostics, retain `InferenceData`, and repeat sensitive evidence
comparisons with independent seeds.

## Practical computation

The broad package cell ceiling protects against an obviously accidental
allocation; it does not imply that a million-cell, four-model SMC fit is
practical. Benchmark a small candidate set, estimate memory and wall time, then
scale deliberately. Hosted frontends should enforce much tighter limits.
