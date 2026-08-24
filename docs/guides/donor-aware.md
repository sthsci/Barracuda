# Donor-aware workflow

Donor-aware inference separates population-level, between-donor, and
within-donor variation. Use it only when donor identifiers are scientifically
meaningful and approved for the analysis.

## Simulate heterogeneous donors

```python
from barracuda import DonorSimulationSpec, simulate_donor_event_counts

spec = DonorSimulationSpec(
    donor_sizes={"D1": 40, "D2": 55, "D3": 35},
    model_key="hetero3",
    mu_lambda={"D1": 3.0, "D2": 4.5, "D3": 5.0},
    sigma_lambda={"D1": 1.0, "D2": 1.4, "D3": 0.8},
    p_zero={"D1": 0.20, "D2": 0.10, "D3": 0.25},
    observation_time=1.0,
    seed=2026,
)
frame, truth = simulate_donor_event_counts(spec)
```

Each donor parameter may be a shared scalar, donor-labelled mapping, or
sequence aligned to `donor_sizes`. Model-incompatible values are rejected—for
example, `homo` requires zero dispersion and zero structural-zero probability.

## Fit donor-aware models

```python
from barracuda import InferenceSettings, run_donor_aware_models

settings = InferenceSettings(draws=256, chains=1, cores=1, seed=73)
fits = run_donor_aware_models(
    frame,
    observation_time=1.0,
    settings=settings,
    model_keys=["dis2p", "hetero3"],
)
```

The result mapping contains the same `InferenceResult` type as donor-ignorant
fits, with `donor_aware=True` and ordered `donor_labels`.

## Extract paired posterior draws

```python
from barracuda import donor_posterior_frame, population_posterior_frame

fit = fits["hetero3"]
population = population_posterior_frame(fit, max_draws=5_000)
donors = donor_posterior_frame(fit, max_draws=5_000)
```

Rows retain `chain` and `draw`, so parameters from one posterior particle remain
paired. Public frames use `p_zero_population` and `p_zero_donor` even when an
older backend stores `phi_0_*`.

## Variance decomposition

```python
from barracuda import population_variance_decomposition

moments = population_variance_decomposition(
    fit,
    donor_weights=[40, 55, 35],
)
```

The xarray result contains within-donor, between-donor, and total rate variance
for every posterior draw. In zero-inflated models, rate moments use active-cell
weights proportional to `w_d * (1 - p_zero_d)`.

Weights are normally observed donor cell counts. They are not inferred sample
weights, and a different weighting target answers a different question.

## Leave-one-donor-out sensitivity

```python
from barracuda import leave_one_donor_out_moments

sensitivity = leave_one_donor_out_moments(fit, [40, 55, 35])
```

!!! warning "No refitting occurs"

    This helper removes one fitted donor from each posterior mixture,
    renormalizes weights, and recomputes moments. It measures sensitivity of
    the fitted mixture summary. It is not leave-one-out predictive validation
    and does not include posterior changes that a refit would produce.

## Condition contrasts

Conditions fitted independently do not have paired chain/draw labels. Use
Cartesian comparisons:

```python
import pandas as pd

from barracuda import condition_contrast_frame, summarize_contrast_draws

all_draws = pd.concat([control_draws, treatment_draws], ignore_index=True)
contrasts = condition_contrast_frame(
    all_draws,
    treatment="Treatment",
    control="Control",
    scale="absolute",
)
contrast_summary = summarize_contrast_draws(contrasts, hdi_prob=0.95)
```

Exact Cartesian differences are used below the configured pair ceiling;
otherwise the function independently samples treatment/control indices with a
recorded deterministic seed. Percentage contrasts divide by the fixed control
posterior mean, never by individual near-zero particles. When `model_key` or
`donor_id` columns are present, both contrast construction and summary retain
them as groups by default so draws from different fitted models or donors are
not silently pooled. Pass an explicit empty `group_columns` sequence only when
pooling is the intended estimand.
