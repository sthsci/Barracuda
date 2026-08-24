# BARRACUDA

`barracuda` is the reusable Python implementation of **BARRACUDA: Bayesian
Analysis Resolving Randomness and Alternative Causes Underlying Differential
Activity**, a framework for studying heterogeneity and history dependence in
immune-cell cytotoxicity.
It provides simulation, PyMC inference, model evidence, donor-aware hierarchy,
ordered contact-kill trajectories, scientific validation, Bayes-factor scans,
diagnostics, plotting, and reproducible result export without requiring the
BARRACUDA web application.

> **Research-software status:** version 0.2 is an alpha API accompanying a
> manuscript in preparation. Pin the exact package version and record priors,
> inference settings, seeds, package versions, and input-data provenance in
> any reproducible analysis.

The complete documentation, including generated signatures and return types,
is published at **<https://sthsci.github.io/Barracuda/>** after GitHub Pages is
enabled for the repository. The source for those pages is in
[`docs/`](https://github.com/sthsci/Barracuda/tree/pypackage/docs).

## What is included

| Workflow | Main entry points | Typical outputs |
|---|---|---|
| Event-count simulation and inference | `simulate_event_counts`, `run_count_models` | validated `DataFrame`, truth metadata, model-keyed `InferenceResult` objects |
| Donor-aware inference | `run_donor_models`, `run_donor_aware_models` | hierarchical posterior draws and per-model evidence |
| Multiple experimental conditions | `run_condition_models` | condition → model → `InferenceResult` mapping |
| Ordered trajectories | `simulate_trajectory_frame`, `run_trajectory_conditions` | condition → model → `TrajectoryResult` mapping |
| Evidence and Bayes factors | `pairwise_bayes_factors`, `posterior_model_probabilities`, `combine_independent_evidence` | tidy evidence tables |
| Scientific validation | `run_event_count_validation`, `run_trajectory_validation`, recovery and coverage helpers | typed result objects and recovery tables |
| Cumulative Bayes-factor scans | `run_count_bf_scan`, `run_trajectory_bf_scan` | long-form scan `DataFrame` |
| Diagnostics | `posterior_diagnostics`, `smc_evidence_summary`, `trajectory_state_summary` | pandas/NumPy summaries |
| Plotting | `plot_model_evidence`, `plot_bayes_factor_scan`, `plot_parameter_recovery` | Matplotlib `Axes` |
| Reproducible export | `build_results_zip`, `build_condition_results_zip`, `build_trajectory_archive` | deterministic ZIP bytes |

The package code is entirely under
[`src/barracuda`](https://github.com/sthsci/Barracuda/tree/pypackage/src/barracuda).
The standalone `pypackage` branch deliberately excludes manuscript figures,
private data, notebooks, Dash components, and generated posterior files.

The installed distribution version is available as `barracuda.__version__`.

## Installation

BARRACUDA currently supports Python 3.12.

```bash
python -m pip install barracuda
```

For package development and documentation:

```bash
git clone --branch pypackage https://github.com/sthsci/Barracuda.git
cd Barracuda
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,build,docs]"
python -m pytest -q
python -m mkdocs build --strict
```

SMC inference is computationally expensive. The examples use small particle
and chain counts for orientation; they are not publication-grade settings.

## Model catalog

### Event-count models

All event-count models observe a non-negative integer count over one positive
observation time `T`.

| Key | Model | Interpretation | Main parameters |
|---|---|---|---|
| `homo` | Homogeneous Poisson | Every cell shares one event rate; count variation is Poisson noise. | `lambda` |
| `z2p` | Zero-inflated Poisson | A fraction of cells is non-engaging; engaging cells share one rate. | `lambda`, `p_zero` |
| `dis2p` | Gamma–Poisson heterogeneity | Engaging-cell rates follow a Gamma population distribution. | `mu_lambda`, `sigma_lambda` |
| `hetero3` | Zero-inflated Gamma–Poisson | Continuous rate heterogeneity plus a non-engaging fraction. | `mu_lambda`, `sigma_lambda`, `p_zero` |

Donor-aware fits use the same four mechanisms inside a hierarchy that separates
population-level, donor-level, and cell-level variation. A donor-aware result
does not make a donor-ignorant result conditionally equivalent: the priors and
latent structure differ.

### Ordered-trajectory models

Trajectory histories use `0` for a non-lethal contact and `1` for a lethal
contact. Public coefficients are `beta_f` for previous failed/non-lethal
contacts and `beta_s` for previous successful/lethal contacts.

| Key | Cell heterogeneity | History dependence | Main decision parameters |
|---|---:|---:|---|
| `homogeneous_history_independent` | No | No | `mu_eta` |
| `homogeneous_history_dependent` | No | Yes | `mu_eta`, `beta_f`, `beta_s` |
| `heterogeneous_history_independent` | Yes | No | `mu_eta`, `sigma_eta` |
| `heterogeneous_history_dependent` | Yes | Yes | `mu_eta`, `sigma_eta`, `beta_f`, `beta_s` |

Every trajectory model also estimates contact-rate population parameters. See
the [model catalog](https://sthsci.github.io/Barracuda/concepts/model-catalog/) for assumptions and
parameter notation.

## Event-count quick start

```python
from barracuda import (
    InferenceSettings,
    evidence_table,
    run_count_models,
    simulate_event_counts,
    summary_table,
)

counts, truth = simulate_event_counts(
    model_key="hetero3",
    n_cells=80,
    obs_time=1.0,
    mu_lambda=4.0,
    sigma_lambda=2.0,
    p_zero=0.2,
    seed=2026,
)

settings = InferenceSettings(draws=256, chains=1, cores=1, seed=2026)
fits = run_count_models(
    counts,
    observation_time=1.0,
    settings=settings,
    model_keys=["homo", "z2p", "dis2p", "hetero3"],
)

print(evidence_table(fits))
print(summary_table(fits, hdi_prob=0.95))
```

`fits` is an insertion-ordered mapping from model key to `InferenceResult`.
Each result contains the PyMC model, ArviZ `InferenceData`, finite log marginal
likelihood, elapsed time, cell count, observation time, and donor labels when
applicable. Retain the full `InferenceData`; rounded summary tables are not a
substitute for posterior draws.

### Required count schema

```csv
cell_id,count
cell_001,0
cell_002,3
cell_003,1
```

Use `validate_count_frame(frame)` before inference. Counts must be finite,
non-negative integers, cell identifiers must be non-empty and unique, and the
observation time must be finite and positive.

## Donor-aware and condition-wise inference

```python
import pandas as pd

from barracuda import (
    InferenceSettings,
    run_condition_models,
    run_donor_aware_models,
    validate_condition_frame,
)

donor_counts = pd.DataFrame(
    {
        "cell_id": ["A01", "A02", "A03", "B01", "B02", "B03"],
        "donor_id": ["A", "A", "A", "B", "B", "B"],
        "count": [0, 2, 1, 1, 4, 2],
    }
)

settings = InferenceSettings(draws=256, chains=1, cores=1, seed=73)
donor_fits = run_donor_aware_models(
    donor_counts,
    observation_time=1.0,
    settings=settings,
    model_keys=["dis2p", "hetero3"],
)
```

`run_donor_aware_models` is a descriptive alias for `run_donor_models`;
`run_donor_ignorant_models` aliases `run_count_models`.

When a table includes a `condition` column, validate and fit each condition
independently with identical settings:

```python
validated = validate_condition_frame(condition_table, donor_aware=True)
condition_fits = run_condition_models(
    validated,
    observation_time=1.0,
    settings=settings,
    model_keys=["dis2p", "hetero3"],
    donor_aware=True,
)
```

The package applies deterministic per-condition seed offsets when a base seed
is supplied. Conditions are independent fits; combining their evidence later
requires an explicit scientific independence assumption.

## Ordered-trajectory quick start

```python
import pandas as pd

from barracuda import (
    TrajectorySettings,
    run_trajectory_conditions,
    trajectory_evidence_frame,
    trajectory_summary_frame,
    validate_trajectory_frame,
)

histories = pd.DataFrame(
    {
        "cell_id": ["cell_001", "cell_002", "cell_003", "cell_004"],
        "condition": ["Control"] * 4,
        "history": ["0,0,1,0", "1,1", "", "0,1,1"],
    }
)

histories = validate_trajectory_frame(histories)
settings = TrajectorySettings(
    draws=256,
    chains=1,
    cores=1,
    seed=2026,
    n_quad=20,
    prior_draws=0,
)
trajectory_fits = run_trajectory_conditions(
    histories,
    observation_time=1.0,
    settings=settings,
)

print(trajectory_evidence_frame(trajectory_fits))
print(trajectory_summary_frame(trajectory_fits))
```

A blank history retains a cell with zero observed contacts. `read_trajectory_csv`
and `normalize_trajectory_frame` accept compact, wide, and long trajectory
representations and return the canonical three-column form. Use
`expanded_trajectory_frame` when one row per contact is needed.

## Evidence and Bayes-factor direction

BARRACUDA records log marginal likelihoods from PyMC SMC and performs arithmetic in
log space. For two models `A` and `B`:

```text
log_BF_A_vs_B = log p(data | A) - log p(data | B)
```

- Positive values support `A`, the named numerator model.
- Negative values support `B`.
- `log10_BF_A_vs_B = log_BF_A_vs_B / log(10)`.
- `BF_A_vs_B` can overflow; prefer log values for computation and storage.
- Bayes factors depend on the fitted priors and are not posterior effect sizes.

```python
import pandas as pd

from barracuda import (
    combine_independent_evidence,
    pairwise_bayes_factors,
    posterior_model_probabilities,
)
from barracuda import evidence_table

log_evidence = {key: result.log_evidence for key, result in fits.items()}
pairs = pairwise_bayes_factors(log_evidence)
model_probabilities = posterior_model_probabilities(log_evidence)

condition_evidence = pd.concat(
    [
        evidence_table(group).assign(condition=condition)
        for condition, group in condition_fits.items()
    ],
    ignore_index=True,
)
# Valid only if conditions are scientifically independent datasets fitted with
# the same model definitions and compatible priors.
combined = combine_independent_evidence(condition_evidence)
```

`savage_dickey_ratio` estimates point-null evidence from prior and posterior
draws. Its `bf_01` supports the point null and `bf_10` supports the alternative.
The identity requires compatible nuisance-parameter priors, and the KDE helper
is not boundary-corrected. `history_effect_bayes_factors` applies the same
calculation to `beta_f` and `beta_s` when those variables are present.

## Scientific validation and recovery

Input validation (`validate_*`) checks schemas. Scientific validation asks
whether simulated ground truth can be recovered and whether the correct model
is selected under repeated datasets.

The `barracuda.validation` module provides:

- typed `EventCountScenario` and `TrajectoryScenario` definitions;
- `run_event_count_validation` and `run_trajectory_validation`;
- posterior recovery tables with truth, mean, HDI, bias, relative error, and
  interval coverage;
- `coverage_summary` and boundary-aware recovery summaries;
- posterior superiority probabilities and ROPE probabilities across two
  independently fitted posterior populations;
- deterministic `stable_seed` derivation for scenario/replicate/model tasks.

Validation results retain the scenario, replicate, derived seeds, simulated
data, truth metadata, fitted results, evidence, and recovery tables. The caller
must preserve the supplied settings alongside them. They are scientific result
containers, not pass/fail certificates. Coverage estimates need enough
independent replicates to be meaningful, and non-identifiability near
nested-model boundaries should be reported rather than hidden.

See the [validation guide](https://sthsci.github.io/Barracuda/guides/validation/) and
generated [`barracuda.validation` reference](https://sthsci.github.io/Barracuda/api/validation/).

## Cumulative Bayes-factor scans

```python
from barracuda import run_count_bf_scan, summarize_bf_scan
from barracuda import COUNT_SCENARIOS

scan = run_count_bf_scan(
    [25, 50, 100],
    scenarios=COUNT_SCENARIOS[:1],
    replicates=3,
    observation_time=1.0,
    base_seed=2026,
    settings=settings,
)
summary = summarize_bf_scan(scan)
```

For each scenario and replicate, a scan simulates **one dataset at the largest
requested sample size** and fits cumulative `.iloc[:N]` prefixes. Thus `N=25`,
`N=50`, and `N=100` are nested views of one replicate, not three independent
datasets. Replicates are independent; adjacent sample sizes within a replicate
are not. This design measures the accumulation of evidence along a growing
dataset and prevents changes from being confounded with resimulation.

Scan tables use lowercase `model_key`, `true_model`, and `best_model`. A column
named `log_bf_model_vs_true` is always
`log_evidence(model) - log_evidence(true)`: positive values support the row's
candidate model. Model-versus-best columns use the same numerator/denominator
convention and are therefore non-positive except for numerical ties.

Scan cost grows approximately with:

```text
scenarios × replicates × sample sizes × fitted models × SMC cost per fit
```

Begin with one scenario, one replicate, a few sample sizes, one chain, and low
particles. Estimate runtime and storage before starting a manuscript-scale
grid. Interruptions do not make partial scientific comparisons complete; keep
the long table and configuration metadata together.

## Diagnostics

```python
from barracuda import (
    diagnostic_flags,
    posterior_diagnostics,
    smc_evidence_summary,
    smc_log_evidence_by_chain,
)

idata = fits["hetero3"].idata
chain_evidence = smc_log_evidence_by_chain(idata)
evidence_stability = smc_evidence_summary(idata)
posterior_table = posterior_diagnostics(idata, hdi_prob=0.95)
flagged = diagnostic_flags(posterior_table)
```

R-hat is unavailable for one-chain output and remains missing; the package does
not silently treat it as a pass. ESS values for weighted/resampled SMC draws
must be interpreted cautiously. Diagnostics identify reasons to review a fit;
they do not prove model adequacy or biological validity.

Trajectory-specific diagnostics include population-level baseline lethal
probability draws/summaries and empirical state aggregation through
`trajectory_state_summary`.

## Plotting

Plotting functions import Matplotlib lazily, return a `matplotlib.axes.Axes`,
and never call `show()` or write files.

```python
from barracuda import plot_model_evidence, plot_posterior_intervals
from barracuda import trajectory_summary_frame

ax = plot_model_evidence(evidence_table(fits))
ax.figure.savefig("model-evidence.svg", bbox_inches="tight")

trajectory_summary = trajectory_summary_frame(trajectory_fits)
ax = plot_posterior_intervals(trajectory_summary)
```

The plotting module also provides count distributions, rate distributions,
cumulative Bayes-factor trajectories, parameter recovery, paired posterior
scatter plots, and trajectory state maps. Functions validate required tidy
columns and raise a clear `ImportError` if a damaged environment is missing
Matplotlib.

## Public API map

### `barracuda.event_counts`

| API | Purpose |
|---|---|
| `InferenceSettings`, `InferenceResult`, `ModelSpec`, `ConditionResults` | Validated settings and typed result/model metadata |
| `MODEL_SPECS`, `MODEL_LABELS`, `PAPER_RATE_DISTRIBUTIONS` | Canonical model metadata |
| `COUNT_COLUMNS`, `DONOR_COLUMNS`, `CONDITION_COLUMN`, `MAX_CONDITIONS` | Canonical schema metadata and broad condition safety ceiling |
| `APPLE_COLOUR_PRESETS`, `default_condition_colours`, `sanitize_condition_colours` | Deterministic optional condition-colour metadata |
| `validate_count_frame`, `validate_donor_frame`, `validate_condition_frame`, `validate_observation_time` | Strict schema/value checks |
| `normalize_condition_frame`, `split_condition_frame` | Canonicalize and partition multi-condition data |
| `sample_count_frame`, `sample_donor_frame` | Small anonymous example frames |
| `simulate_event_counts`, `paper_rate_distribution_for_model`, `rate_distribution_curve` | Generate counts and inspect the configured rate law |
| `run_count_models`, `run_donor_models`, `run_condition_models` | Fit candidate models with SMC |
| `evidence_table`, `summary_table`, `posterior_draw_table` | Convert fits to tidy tables |
| `build_results_zip`, `build_condition_results_zip` | Build deterministic result archives |

### `barracuda.trajectories`

| API | Purpose |
|---|---|
| `TrajectorySettings`, `TrajectorySimulationSpec`, `TrajectoryResult`, `TrajectoryModelSpec` | Validated simulation/inference types |
| `TRAJECTORY_MODEL_SPECS`, `PUBLIC_PARAMETERS` | Model and parameter metadata |
| `CANONICAL_COLUMNS`, `DEFAULT_CONDITION` | Canonical compact-history schema metadata |
| `PUBLIC_TO_BACKEND_PARAMETER`, `BACKEND_TO_PUBLIC_PARAMETER` | Explicit public/research parameter-name translation |
| `read_trajectory_csv`, `normalize_trajectory_frame`, `validate_trajectory_frame` | Read and canonicalize histories |
| `expanded_trajectory_frame` | Convert histories to one row per contact |
| `simulate_trajectory_frame`, `truth_model_key` | Simulate conditions and identify the minimal truth model |
| `run_trajectory_conditions` | Fit selected trajectory models per condition |
| `trajectory_evidence_frame`, `trajectory_summary_frame`, `trajectory_posterior_draws` | Tidy result extraction |
| `build_trajectory_archive` | Reproducible trajectory export |

### `barracuda.donors`

| API | Purpose |
|---|---|
| `DONOR_MODEL_KEYS`, `ContrastScale` | Canonical model set and accepted contrast-scale type |
| `DonorSimulationSpec`, `simulate_donor_event_counts` | Typed unequal-donor simulation and exact truth metadata |
| `canonical_donor_model_key` | Normalize accepted donor-model aliases |
| `population_posterior_frame`, `donor_posterior_frame` | Extract paired public posterior draws |
| `population_variance_decomposition` | Reconstruct active-weighted within/between donor moments |
| `leave_one_donor_out_moments` | Recompute mixture sensitivity without refitting |
| `cartesian_contrast_draws`, `condition_contrast_frame` | Compare independently fitted posterior populations |
| `summarize_contrast_draws` | Tidy HDIs and sign probabilities for contrasts |

### `barracuda.evidence`

| API | Purpose |
|---|---|
| `log_bayes_factor`, `bayes_factor` | Directed two-model comparison |
| `classify_bayes_factor` | Descriptive strength label based on absolute log BF |
| `pairwise_bayes_factors` | Every unordered model pair in a tidy table |
| `posterior_model_probabilities` | Normalize evidence and model prior weights |
| `combine_independent_evidence` | Sum log evidence across independent datasets with complete-coverage checks |
| `smc_log_evidence` | Extract the mean final finite chain evidence from `InferenceData` |
| `evidence_from_inference_data` | Rank a model-keyed mapping of raw `InferenceData` objects |
| `SavageDickeyResult`, `savage_dickey_ratio` | Point-null density-ratio evidence |
| `history_effect_bayes_factors` | Point-null evidence for `beta_f` and `beta_s` |

### `barracuda.validation` and `barracuda.scans`

| API | Purpose |
|---|---|
| `EventCountScenario`, `TrajectoryScenario` | Typed ground-truth scenarios |
| `COUNT_MODEL_KEYS`, `TRAJECTORY_MODEL_KEYS`, `COUNT_SCENARIOS`, `TRAJECTORY_SCENARIOS` | Canonical candidate sets and nested truth scenarios |
| `EventCountValidationResult`, `TrajectoryValidationResult` | Complete validation run containers |
| `PosteriorProbabilityResult` | Exact below/ROPE/above cross-draw probabilities |
| `posterior_recovery_table`, `event_count_recovery_table`, `trajectory_recovery_table` | Truth-versus-posterior tables |
| `coverage_summary`, `boundary_recovery_summary` | Repeated-run calibration summaries |
| `posterior_superiority_probability`, `posterior_rope_probabilities` | Paired posterior comparisons |
| `run_event_count_validation`, `run_trajectory_validation` | End-to-end simulation, fitting, evidence, and recovery |
| `simulate_event_count_data`, `simulate_trajectory_data`, `fit_event_count_models`, `fit_trajectory_models` | Replaceable adapters used by validation and scan orchestration |
| `plan_count_ground_truth_grid` | Construct one-at-a-time typed count scenarios around a baseline |
| `run_count_bf_scan`, `run_trajectory_bf_scan` | Cumulative-prefix evidence scans |
| `validate_bf_scan_schema`, `summarize_bf_scan` | Validate and summarize long scan tables |
| `ScanProgressCallback` | Callback signature for scan task progress |

### `barracuda.diagnostics`, `barracuda.plotting`, and `barracuda.progress`

| API | Purpose |
|---|---|
| `smc_log_evidence_by_chain`, `smc_evidence_summary` | Check chain-level marginal-likelihood stability |
| `posterior_diagnostics`, `diagnostic_flags` | ArviZ summaries and transparent review flags |
| `population_p0_draws`, `population_p0_summary` | Baseline lethal-probability population summaries |
| `trajectory_state_summary` | Empirical decision summaries by history state |
| `plot_event_count_distribution`, `plot_rate_distribution` | Data and generative-distribution plots |
| `MODEL_COLOURS`, `BF_THRESHOLDS_LOG10` | Stable default model palette and plotted BF guide thresholds |
| `plot_model_evidence`, `plot_bayes_factor_scan` | Model-comparison plots |
| `plot_parameter_recovery`, `plot_posterior_intervals`, `plot_posterior_pair` | Posterior/recovery plots |
| `plot_trajectory_state_map` | Empirical trajectory state map |
| `SMCProgressCallback`, `run_with_smc_progress` | Genuine per-chain PyMC SMC tempering progress |

### `barracuda.io`

| API | Purpose |
|---|---|
| `canonical_json`, `configuration_fingerprint`, `dataframe_checksum` | Deterministic configuration/input identity |
| `SCAN_SCHEMA_VERSION` | Version of the persisted scan bundle schema |
| `save_inference_data`, `load_inference_data` | Atomic ArviZ NetCDF persistence |
| `ScanBundle`, `save_scan_bundle`, `load_scan_bundle` | Checksummed CSV/manifest persistence with resume verification |
| `build_scan_archive` | Deterministic portable scan ZIP bytes |

Generated signatures, public docstrings, workflow contracts, return conventions,
and error guidance are in the [API reference](https://sthsci.github.io/Barracuda/api/).

## Return types and errors

The public API favors explicit, inspectable objects:

- input/simulation/recovery/evidence/scan outputs are pandas `DataFrame`s;
- compact scalar summaries are pandas `Series` or frozen dataclasses;
- posterior samples are ArviZ `InferenceData`/xarray objects;
- inference collections are insertion-ordered mappings keyed by condition and
  model;
- plot functions return Matplotlib `Axes`;
- archive functions return `bytes` and do not write implicitly.

Common errors are deliberately conventional:

- `TypeError` for the wrong object category, such as a non-DataFrame table;
- `ValueError` for invalid schemas, keys, priors, ranges, missing columns, or
  non-finite values;
- `RuntimeError` when expected inference evidence/diagnostic content is absent;
- `ImportError` when a required runtime dependency is missing from a damaged
  environment.

Inference exceptions raised by PyMC/PyTensor can still propagate. Catching an
exception should not be interpreted as a valid negative scientific result.

## Reproducibility, computation, and privacy

For every reported analysis, preserve:

1. exact `barracuda`, Python, PyMC, PyTensor, ArviZ, NumPy, and SciPy versions;
2. validated input data or an approved content hash;
3. model keys and observation time;
4. the full settings dataclass and priors;
5. every base/derived seed;
6. raw `InferenceData`, not only figures or rounded CSV summaries;
7. evidence direction and whether datasets were assumed independent;
8. failures, interrupted fits, exclusions, and convergence limitations.

`barracuda` runs locally and does not transmit data. That is not permission to
publish sensitive material. Do not put names, clinical identifiers, dates,
unapproved donor metadata, raw microscopy, or private paths into examples,
notebooks, archives, issue reports, CI logs, or public Pages builds. Use
anonymous study identifiers and follow the applicable ethics, retention, and
data-access agreements.

See [Reproducibility and privacy](https://sthsci.github.io/Barracuda/reproducibility/) for the full
checklist.

## Development, packaging, and documentation

```bash
python -m pytest -q
python -m build
python -m twine check dist/*
python -m mkdocs build --strict
```

GitHub Pages deployment is defined in
[`docs.yml`](https://github.com/sthsci/Barracuda/blob/pypackage/.github/workflows/docs.yml).
A repository administrator must first
open **Settings → Pages** and set **Build and deployment → Source** to
**GitHub Actions**. The workflow builds strictly on pull requests and pushes,
but deploys only from a push to `pypackage`. Private-repository visibility and
organization policy determine who can view the resulting site.

The complete research repository remains on `main`; manuscript assets live on
`paper`, and the standalone web application lives on `webpage`. Package changes
should be developed against `pypackage` and synchronized deliberately with the
research implementation to avoid backend drift.

See [Development](https://sthsci.github.io/Barracuda/development/) and the
[changelog](https://sthsci.github.io/Barracuda/changelog/).

## Citation and license

Use [`CITATION.cff`](https://github.com/sthsci/Barracuda/blob/pypackage/CITATION.cff)
when citing the software and cite the BARRACUDA manuscript for the scientific
framework. `barracuda` is released under the
[MIT License](https://github.com/sthsci/Barracuda/blob/pypackage/LICENSE).
