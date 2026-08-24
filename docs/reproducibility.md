# Reproducibility and privacy

## Minimum analysis record

Preserve these items for every result:

1. `barracuda` version and commit, Python version, operating system, and core
   dependency versions;
2. validated input data or an approved checksum plus data provenance;
3. workflow, model keys, observation time, and candidate-set rationale;
4. the complete settings dataclass and every prior/bound;
5. base and derived random seeds;
6. raw ArviZ `InferenceData` and log marginal likelihoods;
7. evidence table with explicit numerator/denominator direction;
8. diagnostics, reruns, failures, interruptions, and exclusions;
9. code used for transformations and plots;
10. any independence assumption used to aggregate evidence.

`save_inference_data`, scan bundles, and deterministic archive builders support
this record. A checksum verifies content consistency; it does not prove that a
dataset is correct or ethically authorized.

## Randomness

Use explicit integer seeds. Validation and scan runners derive stable task seeds
from workflow, scenario, replicate, sample size, model, and a base seed. This
prevents Python hash randomization and execution order from changing a task's
seed.

Reusing a seed does not remove Monte Carlo error. Repeat sensitive SMC evidence
comparisons with independent base seeds.

## Cumulative scans

One maximum-size dataset is simulated per scenario/replicate and smaller sample
sizes are cumulative prefixes. Record this dependency in the configuration and
do not combine adjacent sizes as independent evidence.

## Computational reporting

Report particles, chains, cores, quadrature nodes, prior draws, number of cells
and events, scenarios, replicates, candidate models, hardware, elapsed time, and
whether any run was retried. Broad package ceilings are safety guards, not
recommended analysis sizes. Frontends should enforce operational limits suited
to their resources.

## Privacy and governance

The package executes locally and has no telemetry or upload code. This does not
relax governance obligations.

- Use anonymous study identifiers.
- Do not commit or publish names, clinical identifiers, dates, unapproved donor
  metadata, raw microscopy, private paths, or credentials.
- Treat NetCDF and ZIP outputs as potentially sensitive derived data.
- Do not attach real data to public issues or CI artifacts.
- Follow ethics approvals, data-processing agreements, access controls,
  retention schedules, and institutional policy.
- Review documentation examples and GitHub Pages artifacts as public content
  unless repository policy explicitly guarantees restricted access.

## Interpretation limits

Passing a validator, diagnostic threshold, recovery run, or coverage target does
not establish biological validity. Model assumptions, observation processes,
selection effects, donor sampling, and prior sensitivity remain scientific
responsibilities.
