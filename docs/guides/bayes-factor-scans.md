# Bayes-factor scans

Bayes-factor scans evaluate how model evidence develops across sample size or a
ground-truth parameter grid. They are expensive orchestration workflows, not a
single vectorized statistic.

## Plan scenarios

Use typed scenarios directly or construct a count ground-truth grid with
`plan_count_ground_truth_grid`. The planner varies `sigma_lambda` and `p_zero`
one at a time around the supplied baseline, deduplicating the overlapping
baseline point. It intentionally returns this one-at-a-time union, not a
Cartesian product. Scenario identifiers, labels, truth models, and seed offsets
are retained in every output row.

## Cumulative-prefix semantics

For each scenario and replicate, a runner:

1. validates and sorts the requested sample sizes;
2. simulates one dataset with `max(sample_sizes)` cells;
3. fits the cumulative `.iloc[:N]` prefix at each requested `N`;
4. fits every declared candidate model with deterministically derived seeds;
5. returns one long-form row per scenario/replicate/sample-size/model.

Adjacent `N` values within one replicate are nested and statistically
dependent. Replicates are independently simulated. The scan describes an
evidence accumulation path; it is not a collection of independent experiments
at every sample size.

## Run a count scan

```python
from bayesorca.event_counts import InferenceSettings
from bayesorca.scans import run_count_bf_scan, summarize_bf_scan
from bayesorca.validation import COUNT_SCENARIOS

scan = run_count_bf_scan(
    [25, 50, 100],
    scenarios=COUNT_SCENARIOS[:1],
    replicates=3,
    observation_time=1.0,
    base_seed=2026,
    settings=InferenceSettings(draws=256, chains=1, cores=1),
)
summary = summarize_bf_scan(scan)
```

`run_trajectory_bf_scan` has the same orchestration shape with trajectory
scenarios, model keys, and settings.

## Direction and schema

`validate_bf_scan_schema` verifies identifiers, finite evidence, sample-size
ordering, true-model presence, and directionally consistent comparison fields.
Key columns include:

- `scenario`, `replicate`, `n_cells`, `model_key`, `true_model`, `best_model`;
- `log_evidence`;
- `log_bf_model_vs_true = log_evidence(model) - log_evidence(true)`;
- `log_bf_model_vs_best = log_evidence(model) - log_evidence(best)`;
- base-10 counterparts and best/true flags.

Positive `model_vs_true` supports the candidate row model. `model_vs_best` is
non-positive except for ties.

## Summaries

`summarize_bf_scan` aggregates replicates by workflow/scenario/sample size/model
and returns the declared central interval. Interpret its interval as
between-replicate variation at one cumulative size—not uncertainty from
independent neighboring sample sizes.

## Persistence and resume safety

```python
from bayesorca.io import load_scan_bundle, save_scan_bundle

configuration = {
    "workflow": "event_count",
    "sample_sizes": [25, 50, 100],
    "replicates": 3,
    "base_seed": 2026,
}
bundle = save_scan_bundle(scan, "results/count-scan", configuration=configuration)
loaded = load_scan_bundle(
    "results/count-scan",
    expected_configuration=configuration,
)
```

Bundles include schema version, configuration fingerprint, result checksum,
row count, and columns. Existing files are not silently reused or overwritten.
Configuration verification protects against accidentally resuming a different
scientific experiment.

## Budget before running

Approximate fit count is:

```text
scenarios × replicates × sample sizes × models
```

Each cell is an SMC fit. Benchmark one representative fit, account for chains
and artifacts, then choose concurrency that respects memory. Broad package
ceilings are not feasible scan defaults, and frontends should impose much
tighter limits.
