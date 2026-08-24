# Scientific validation

Schema validators answer “is this input admissible?” Scientific validation
answers “under a known generating mechanism, what does this inference workflow
recover?” The second question requires simulation, fitting, evidence, posterior
recovery, and enough independent replicates.

## Typed scenarios

```python
from barracuda import EventCountScenario

scenario = EventCountScenario(
    scenario="moderate_heterogeneity",
    label="Moderate heterogeneity and structural zeros",
    mu_lambda=4.0,
    sigma_lambda=2.0,
    p_zero=0.2,
    true_model="hetero3",
)
```

Scenario validation ensures that `true_model` is the minimal mechanism implied
by the parameters. `COUNT_SCENARIOS` and `TRAJECTORY_SCENARIOS` contain the four
canonical nested mechanisms for each workflow.

## Run one complete validation

```python
from barracuda import InferenceSettings
from barracuda import run_event_count_validation

result = run_event_count_validation(
    scenario,
    100,
    observation_time=1.0,
    replicate=1,
    base_seed=2026,
    settings=InferenceSettings(draws=256, chains=1, cores=1),
)

print(result.evidence)
print(result.recovery)
```

The runner derives separate stable simulation and inference seeds, simulates
one dataset, fits the declared model set, computes directed evidence versus the
true/best models, and builds recovery rows. The true model must be included.

`run_trajectory_validation` follows the same pattern with a
`TrajectoryScenario` and `TrajectorySettings`.

## Recovery table meaning

Each recovery row identifies condition, model, public parameter, backend
variable, truth, posterior mean/median/SD, HDI, error, absolute/relative error,
coverage, and finite draw count.

- `error = posterior_mean - truth`;
- `covered` uses the closed HDI;
- relative error is missing when truth is zero;
- recovery under a scientifically wrong model is still useful but must not be
  confused with recovery under the generating model.

## Aggregate coverage

```python
import pandas as pd

from barracuda import coverage_summary

all_recovery = pd.concat([run.recovery for run in validation_runs])
coverage = coverage_summary(
    all_recovery,
    group_by=("model_key", "parameter"),
)
```

Coverage from one or a handful of replicates is not a calibrated frequency.
Report the number of independent runs, generating scenarios, failures, and
Monte Carlo uncertainty.

## Boundary recovery

`boundary_recovery_summary` distinguishes true boundary values from interior
values and reports how often posterior estimates remain within a declared
tolerance. Choose tolerances before inspecting the answer. Boundaries such as
zero heterogeneity or zero history effect often produce asymmetric posteriors.

## Superiority and ROPE probabilities

```python
from barracuda import (
    posterior_rope_probabilities,
    posterior_superiority_probability,
)

p_superior = posterior_superiority_probability(treatment, control, margin=0.0)
rope = posterior_rope_probabilities(treatment, control, rope=(-0.1, 0.1))
```

Both helpers treat the supplied posterior samples as independent populations
and compute exact cross-draw probabilities without materializing a huge
Cartesian product. The ROPE is closed; below and above events are strict.

## Cost and failure reporting

Validation multiplies inference cost by scenarios, replicates, and models.
Start with a smoke run, but do not mix smoke and reporting settings in one
coverage summary. Preserve failed/interrupted attempts and their configurations;
silently dropping difficult fits biases validation.
