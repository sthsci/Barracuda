# bayesorca

`bayesorca` is the reusable Python interface to ORCA, a Bayesian framework for
studying heterogeneity in immune cell cytotoxicity. It exposes the scientific
simulation and inference code used by the ORCA manuscript and web application
without requiring Dash.

The package contains three workflows:

- donor ignorant event count inference;
- donor aware hierarchical event count inference; and
- donor ignorant inference from ordered contact and kill trajectories.

All model comparison uses PyMC Sequential Monte Carlo (SMC). The returned
ArviZ `InferenceData` objects contain posterior draws, while ORCA records the
SMC marginal likelihood used to calculate Bayes factors.

## Install

```bash
python -m pip install bayesorca
```

ORCA is currently tested with Python 3.12. SMC inference can be computationally
expensive. Start with small particle and chain counts while checking a workflow,
then choose settings appropriate for the scientific analysis.

## Event count example

```python
import pandas as pd

from bayesorca.event_counts import (
    InferenceSettings,
    evidence_table,
    run_count_models,
)

counts = pd.DataFrame(
    {
        "cell_id": [f"cell_{index:03d}" for index in range(1, 9)],
        "count": [0, 1, 2, 0, 4, 1, 3, 2],
    }
)

settings = InferenceSettings(draws=256, chains=1, cores=1, seed=42)
results = run_count_models(
    counts,
    observation_time=1.0,
    settings=settings,
    model_keys=["homo", "z2p", "dis2p", "hetero3"],
)
print(evidence_table(results))
```

The four keys correspond to the homogeneous Poisson, zero inflated Poisson,
heterogeneous Gamma Poisson, and zero inflated heterogeneous Gamma Poisson
models in the paper.

## Donor aware example

```python
from bayesorca.event_counts import run_donor_aware_models

donor_results = run_donor_aware_models(
    donor_counts,  # columns: cell_id, donor_id, count
    observation_time=1.0,
    settings=settings,
    model_keys=["dis2p", "hetero3"],
)
```

Use `run_condition_models` when one table contains a `condition` column. Each
condition is inferred independently with identical models and prior settings.

`run_donor_ignorant_models` and `run_donor_aware_models` are descriptive aliases
for `run_count_models` and `run_donor_models` respectively.

## Trajectory example

```python
import pandas as pd

from bayesorca.trajectories import (
    TrajectorySettings,
    run_trajectory_conditions,
    trajectory_evidence_frame,
)

histories = pd.DataFrame(
    {
        "cell_id": ["cell_001", "cell_002", "cell_003"],
        "condition": ["Control", "Control", "Control"],
        "history": ["0,0,1,0", "1,1", ""],
    }
)

trajectory_results = run_trajectory_conditions(
    histories,
    settings=TrajectorySettings(draws=256, chains=1, cores=1, seed=42),
)
print(trajectory_evidence_frame(trajectory_results))
```

A blank history retains a cell with zero observed contacts. Public trajectory
notation uses previous failed contacts `f`, previous successful contacts `s`,
and coefficients `beta_f` and `beta_s`.

## Data and privacy

This library runs locally and does not transmit input data. The public ORCA web
application has additional upload limits and privacy guidance. Do not publish
identifying clinical or donor metadata in notebooks, issue reports, or package
examples.

## Research status

This is research software accompanying a manuscript in preparation. Version
`0.1.x` should be treated as an alpha API. Pin an exact version in reproducible
analyses and record the inference settings and random seed.

ORCA is released under the [MIT License](https://github.com/sthsci/Orca/blob/main/LICENSE).
