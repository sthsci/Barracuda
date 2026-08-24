# Ordered-trajectory workflow

## Canonical histories

```python
import pandas as pd

from barracuda import validate_trajectory_frame

frame = validate_trajectory_frame(
    pd.DataFrame(
        {
            "cell_id": ["cell_001", "cell_002", "cell_003"],
            "condition": ["Control", "Control", "Control"],
            "history": ["0,0,1,0", "1,1", ""],
        }
    )
)
```

A history is an ordered binary sequence: `0` is non-lethal and `1` is lethal.
A blank history is a valid zero-contact cell. Preserving these cells matters for
the contact-rate component.

`normalize_trajectory_frame` accepts supported compact, wide, and long forms.
`expanded_trajectory_frame` returns one row per contact with pre-contact state.

## Simulate known mechanisms

```python
from barracuda import TrajectorySimulationSpec, simulate_trajectory_frame

spec = TrajectorySimulationSpec(
    condition="Synthetic",
    n_cells=100,
    mu_lambda=4.0,
    sigma_lambda=2.0,
    p0=0.25,
    sigma_eta=0.75,
    beta_f=0.8,
    beta_s=-0.8,
    observation_time=1.0,
    seed=2026,
)
frame, truth = simulate_trajectory_frame([spec])
```

`truth_model_key` identifies the minimal trajectory model implied by
`sigma_eta`, `beta_f`, and `beta_s`.

## Configure and fit

```python
from barracuda import TrajectorySettings, run_trajectory_conditions

settings = TrajectorySettings(
    draws=256,
    chains=1,
    cores=1,
    seed=2026,
    n_quad=20,
    prior_draws=0,
)
fits = run_trajectory_conditions(
    frame,
    observation_time=1.0,
    settings=settings,
    model_keys=[
        "homogeneous_history_independent",
        "homogeneous_history_dependent",
        "heterogeneous_history_independent",
        "heterogeneous_history_dependent",
    ],
)
```

The outer mapping is condition and the inner mapping is model key. Each
condition uses a deterministic seed offset derived from the base setting.

Trajectory inference is especially sensitive to cells, contacts, particles,
chains, models, and quadrature nodes. Broad package ceilings are safety guards,
not workload advice; frontends should enforce smaller operational limits.

## Extract and diagnose

```python
from barracuda import (
    trajectory_evidence_frame,
    trajectory_posterior_draws,
    trajectory_summary_frame,
)

evidence = trajectory_evidence_frame(fits)
draws = trajectory_posterior_draws(fits, max_draws=6_000, seed=17)
summary = trajectory_summary_frame(fits, hdi_prob=0.95)
```

Subsampling retains paired posterior rows. History-independent models do not
gain zero-filled history coefficients; absent parameters remain absent.

Use `trajectory_state_summary` to aggregate empirical outcomes by pre-contact
state and `history_effect_bayes_factors` for carefully qualified point-null
evidence. Preserve public `beta_f`/`beta_s` terminology in reports.

## Export

`build_trajectory_archive` bundles canonical input, evidence, summaries,
posterior draws, settings, truth metadata, and NetCDF results. It returns bytes
and does not write to disk implicitly.
