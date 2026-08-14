# Settings, results, and errors

## Count inference settings

`InferenceSettings` is a frozen validated dataclass. It records SMC particles
(`draws`), chains, cores, seed, SMC thresholds, log-rate prior bounds,
zero-inflation prior parameters, dispersion scale, and donor-deviation scales.

Construct settings once and store them with every output. Invalid integers,
seeds, bounds, probabilities, and prior scales fail immediately.

## Trajectory settings

`TrajectorySettings` additionally records the contact-rate dispersion prior,
baseline-probability prior, `sigma_eta` prior, history-coefficient prior scale,
Gauss–Hermite quadrature nodes, and optional prior-predictive draw count. Its
`particles` property is the user-facing alias for `draws`.

The package's large hard ceilings prevent obvious accidental overflow; they are
not practical recommendations. Runtime grows substantially with particles,
chains, cells, model complexity, and quadrature nodes.

## Result containers

`InferenceResult` contains one event-count/donor fit:

- `model_key`, label, and donor-aware flag;
- ArviZ `idata` and the PyMC `model`;
- `log_evidence` and elapsed time;
- cell count, observation time, and ordered donor labels.

`TrajectoryResult` contains one model for one condition:

- condition and model metadata;
- ArviZ `idata`;
- log evidence, elapsed time, cell and event counts;
- observation time.

Validation result dataclasses retain the scenario, replicate, derived seeds,
simulated data, truth, fitted result mappings, and evidence/recovery tables.
Callers must preserve the supplied settings separately. The dataclasses are
containers for audit and reuse rather than Boolean validity decisions.

## Collection shapes

- Event-count inference returns `dict[model_key, InferenceResult]`.
- Condition inference returns `dict[condition, dict[model_key, InferenceResult]]`.
- Trajectory inference returns `dict[condition, dict[model_key, TrajectoryResult]]`.
- Evidence, recovery, diagnostic, contrast, and scan functions return tidy
  pandas tables unless their reference documents another explicit type.
- Variance and leave-one-donor-out moment helpers return xarray datasets.
- Plotting returns Matplotlib axes.
- Archive builders return bytes and do not write files implicitly.

## Progress callbacks

High-level inference functions accept operation-level and sampler-level
callbacks. The sampler callback receives genuine per-chain SMC stage and
tempering information. Treat callbacks as observers: do not mutate inputs,
models, results, or random-number state from a callback.

## Error behavior

| Exception | Typical meaning |
|---|---|
| `TypeError` | Wrong object category, such as a list where a DataFrame is required |
| `ValueError` | Invalid schema, key, prior, range, missing column, or non-finite value |
| `RuntimeError` | Expected SMC evidence or diagnostic content is absent/incomplete |
| `ImportError` | A required dependency is missing from the environment |

PyMC and PyTensor errors can propagate during model construction or sampling.
Do not convert an exception, interrupted fit, or non-finite evidence value into
a scientific model-comparison row.
