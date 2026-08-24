# Scientific worker and adapter contract

## Boundary

The worker is an orchestrator, not a second implementation of Barracuda. It should
install a pinned `barracuda` wheel and call its public adapters. Do not copy
PyMC code from `section_1`, `section_2` or `section_3` into `platform/worker`.
Do not import Dash pages or reconstruct likelihoods from frontend types.

The extraction target is a UI-neutral package with an interface resembling:

```python
class ScientificAdapter(Protocol):
    kind: str
    schema_version: int

    def normalize(self, input_path: Path, request: AnalysisRequest) -> ValidatedInput: ...
    def run(
        self,
        validated: ValidatedInput,
        request: AnalysisRequest,
        emit: Callable[[ProgressEvent], None],
        output_dir: Path,
    ) -> ArtifactManifest: ...
```

The package name must not be `platform`; that collides with Python's standard
library. If the wheel extraction is staged, the first compatibility adapter
may call existing `webapp.core` functions, but the worker dependency must still
be pinned to an exact source revision/wheel hash.

## Mapping to the current code

The current UI-neutral functions already define most of the scientific seam:

| Analysis mode | Normalize/validate | Run | Package |
|---|---|---|---|
| Donor-ignorant event counts | `webapp.core.conditions.normalize_condition_frame(..., donor_aware=False)` then `validate_condition_frame` | `webapp.core.condition_inference.run_condition_models(..., donor_aware=False)` | `build_condition_results_zip` |
| Donor-aware event counts | same functions with `donor_aware=True` | same runner with `donor_aware=True` | `build_condition_results_zip` |
| Donor-ignorant trajectories | `webapp.core.trajectory.read_trajectory_csv`, `normalize_trajectory_frame`, `validate_trajectory_frame` | `run_trajectory_conditions` | `build_trajectory_archive` |
| Synthetic event counts | `webapp.core.simulation.simulate_event_counts`, then count validation | same count runner | same count packager, including ground truth |
| Synthetic trajectories | `webapp.core.trajectory.simulate_trajectory_frame`, then trajectory validation | same trajectory runner | same trajectory packager, including ground truth |

These adapters lazy-import the research backends. The worker wheel/image must
therefore include the corresponding scientific modules and exact pinned
versions of PyMC, PyTensor, ArviZ, NumPy, pandas, SciPy, Matplotlib and xarray.
The existing Dash Dockerfile is not a sufficient trajectory worker image: it
does not copy `section_3/src`.

The extracted package should replace source-layout imports with stable package
imports. The research modules remain the sole definitions of the likelihoods;
the adapter package owns schema conversion, limits, progress translation,
evidence extraction and portable artifacts.

## Queue envelope

Keep the broker message small and non-sensitive:

```json
{
  "schema_version": 1,
  "job_id": "01950f8e-7ad8-7d71-9d7e-f4e4787b02bc",
  "attempt": 1
}
```

The orchestrator obtains the immutable request and server-derived input object
from the authoritative database/object store after atomically claiming the
job. It must reject an unsupported schema version rather than guessing.

## Immutable runner request

The orchestrator writes a local request file for a fresh scientific subprocess.
It contains no service credentials and refers only to paths inside that job's
workspace:

```json
{
  "schema_version": 1,
  "job_id": "01950f8e-7ad8-7d71-9d7e-f4e4787b02bc",
  "analysis": {
    "kind": "event-counts",
    "donor_aware": false,
    "source": "upload"
  },
  "input": {
    "path": "input/normalized.csv",
    "sha256": "<64 lowercase hex characters>",
    "bytes": 8421,
    "observation_time": 1.0
  },
  "models": ["homo", "z2p", "dis2p", "hetero3"],
  "sampler": {
    "particles": 64,
    "chains": 1,
    "cores": 1,
    "seed": 2026,
    "threshold": 0.5,
    "correlation_threshold": 0.01
  },
  "priors": {
    "lambda_log10_bounds": [-1.5, 1.5],
    "continuous_sd_scale": 3.0
  }
}
```

Use explicit analysis variants rather than inferring donor awareness from
columns. Valid model keys and prior fields differ for count and trajectory
requests, so validation should be a tagged/discriminated union. Unknown fields
are rejected in release 1. Colours, chart selections, titles and share settings
do not belong in this scientific request.

The API validates limits, but the wheel validates them again. Keep the current
caps as the initial public authority: at most four conditions and 1,000 cells;
count limits/donor requirements from `webapp.core.data`; and trajectory caps of
20,000 events, 250 events per cell, 2,000 particles, four chains/cores and 80
quadrature nodes. The staged UI may expose stricter values.

## Progress protocol

Both count and trajectory adapters already accept model and native sampler
callbacks. Translate them to a common event without inventing SMC stages:

```json
{
  "schema_version": 1,
  "sequence": 37,
  "job_id": "01950f8e-7ad8-7d71-9d7e-f4e4787b02bc",
  "attempt": 1,
  "timestamp": "2026-08-12T09:12:31.248Z",
  "phase": "sampling",
  "condition": {"index": 1, "total": 2, "label": "Control"},
  "model": {"index": 3, "total": 4, "key": "dis2p"},
  "chain": {"index": 0, "total": 1, "stage": 4, "beta": 0.731}
}
```

`beta` is PyMC's native tempering value from zero (prior) to one (posterior).
The number of SMC stages is not known in advance. Overall display progress may
be derived from completed condition/model units plus mean chain beta, but the
event retains the raw values.

Publish reconnectable events to a bounded Redis Stream such as
`barracuda:job:{job_id}:events` using monotonically increasing `sequence`. Trim it
to a small maximum (for example 1,000 events), throttle repetitive updates to
roughly 4 Hz and set the retention TTL. The Django SSE endpoint authorizes the
request and reads the stream. Redis progress is advisory; PostgreSQL terminal
state is authoritative.

Progress payloads must not contain posterior values, input rows, donor/cell
identifiers, object keys or exception traces. A condition label is potentially
pseudonymous; the safer default is a condition index plus a sanitized display
label already approved by input validation.

## Artifact manifest

The scientific subprocess writes artifacts under its output directory and
returns a manifest. It never uploads them itself:

```json
{
  "schema_version": 1,
  "job_id": "01950f8e-7ad8-7d71-9d7e-f4e4787b02bc",
  "attempt": 1,
  "status": "succeeded",
  "scientific_runtime": {
    "package": "barracuda",
    "version": "0.1.0",
    "wheel_sha256": "<digest>",
    "request_sha256": "<digest>",
    "python": "3.12.x",
    "pymc": "5.25.1"
  },
  "artifacts": [
    {
      "role": "results-archive",
      "path": "output/barracuda-results.zip",
      "media_type": "application/zip",
      "bytes": 1837421,
      "sha256": "<digest>"
    },
    {
      "role": "model-evidence",
      "path": "output/model-evidence.csv",
      "media_type": "text/csv",
      "bytes": 992,
      "sha256": "<digest>"
    }
  ]
}
```

Paths must be relative, normalized, contain no `..` and resolve below the job
output root. The orchestrator rehashes and size-checks every file before upload,
uses server-derived MinIO keys and then records the final object digest in the
database. Reject symlinks, devices and undeclared files. Apply a total output
byte limit before packaging can fill the worker volume.

Keep NetCDF/ArviZ files as downloadable scientific artifacts and expose compact
CSV/JSON tables for the Dash and Plotly views. Do not pickle PyMC models or pandas
objects into Redis/Celery. An `InferenceResult` contains live Python/PyMC
objects and must remain inside the scientific process.

## Execution and isolation

The queue task should be a supervisor:

1. atomically claim the job/attempt;
2. create a workspace under a fixed root using a server-generated UUID and
   mode `0700`;
3. fetch the expected input object, enforce size and verify SHA-256;
4. create normalized input/request files with `umask 077`;
5. launch a new process group with a clean allowlisted environment;
6. forward bounded progress events and poll cancellation;
7. enforce wall time, terminate then kill the process group if necessary;
8. validate and upload the artifact manifest;
9. atomically mark the attempt ready or failed; and
10. delete the workspace and job-specific caches in `finally`.

Set `MPLCONFIGDIR`, `XDG_CACHE_HOME` and PyTensor's `base_compiledir` to
job-specific paths before the subprocess imports scientific libraries. Set
thread counts deliberately (`OMP_NUM_THREADS`, BLAS variables) so PyMC chains
do not multiply hidden BLAS threads. Use deterministic per-condition/model seed
derivation already provided by the adapters.

The subprocess should have no database, Redis, MinIO or Django secret. In local
Compose, the worker orchestrator necessarily reaches those services; the child
contract still prevents accidental use. In production, run the scientific
process in a separate network-denied batch container/pod.

## Failure contract

Return stable private error categories, for example:

- `INPUT_INVALID`
- `SCHEMA_UNSUPPORTED`
- `RESOURCE_LIMIT`
- `SAMPLER_FAILED`
- `TIMED_OUT`
- `CANCELLED`
- `ARTIFACT_INVALID`
- `INTERNAL_ERROR`

The API returns a safe user message and correlation ID, not `str(exc)` or a
traceback. Restricted logs may contain the traceback but must redact tokens,
paths, object keys and submitted labels/data. Failed attempts never publish a
partially written artifact as current.

## Contract and parity tests

The wheel and worker need tests at three levels:

1. **Contract tests:** reject unknown versions/fields, path traversal, symlinks,
   digest mismatch, oversized inputs/outputs and invalid resource settings.
2. **Adapter parity tests:** fixed tiny datasets and seeds produce the same
   canonical input, model keys, public parameter names, evidence-table shape
   and archive members as the current `webapp.core` path. Numerical SMC checks
   use tolerances; schema and labels are exact.
3. **Process tests:** native chain/stage/beta progress crosses the subprocess
   boundary, cancellation kills descendants, timeout and OOM become safe error
   codes, duplicate delivery is idempotent, and workspaces are removed for
   success and every failure path.

Record request schema, wheel version/hash, source revision, dependency versions,
seed and settings in every result archive. Reproducibility metadata is part of
the scientific result, not optional observability.
