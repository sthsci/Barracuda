# Platform architecture and local deployment

## Decision summary

The first platform release keeps the established Dash interface and adds the
optional persistence services behind it:

| Service | Responsibility | Local address | Persistent data |
|---|---|---|---|
| `dash` | Existing BARRACUDA analysis interface and Plotly visualisation | `http://localhost:8501` | browser session state and temporary callback cache only |
| `api` | Django REST API, authorization, validation, job and artifact metadata | internal `api:8000`; host `http://localhost:8000/api/v1` for API development only | PostgreSQL and MinIO references |
| `worker` | Queue consumer and scientific subprocess supervisor | no host port | temporary per-job workspace only |
| `postgres` | Authoritative jobs, capabilities, share grants and retention state | internal only | named database volume |
| `redis` | Celery broker and short-lived progress streams | internal only | none required for the first release |
| `minio` | Private uploaded inputs and result artifacts | internal only | named object-store volume |

Dash is the only product interface. In local development, Dash binds to 8501
and Django binds to loopback port 8000 for authorized CSV downloads. In an
Imperial deployment, one HTTPS reverse proxy should expose Dash and route the
API paths without exposing PostgreSQL, Redis or MinIO. Do not deploy the
archived Next.js experiment in `platform/frontend`.

```text
browser
  |
  v
Dash :8501 ------server-side------> Django :8000 --> PostgreSQL
   |                                   |  \------> private MinIO
   |                                   \---------> Redis broker
   |                                                   |
   +---- existing local inference                      v
                                                scientific worker
```

The Django database is the authority for state. Redis progress and queue data
are disposable. MinIO contains bytes but never decides whether a requester is
allowed to read them.

## Job lifecycle

Use explicit internal states even if the interface initially groups some of
them under `running` or `failed`:

```text
uploading -> validated -> queued -> starting -> running -> packaging -> ready
                    \          \          \             \
                     expired    cancelled  failed         failed

ready | failed | cancelled -> expiring -> deleting -> deleted
```

Recommended invariants:

1. The API creates a random job ID and immutable job specification only after
   parsing and validating the uploaded bytes itself.
2. Enqueue the task with `transaction.on_commit`; a worker must never observe
   a job whose database transaction can still roll back.
3. The queue message contains only `{schema_version, job_id, attempt}`. It does
   not contain CSV data, object-store credentials, ownership tokens or user
   controlled object keys.
4. A worker claims a queued job with a conditional update or row lock. A
   duplicate delivery either resumes an idempotent attempt or exits without
   running the model twice.
5. Terminal state is written only after every artifact has been uploaded and
   its digest recorded. A result is not `ready` while its ZIP or NetCDF files
   are still being copied.
6. Cancellation and expiry are states, not deletion by side effect. Cleanup is
   retriable and records enough non-sensitive information to finish deleting
   an interrupted object prefix.

Use a public UUID/ULID solely as an identifier. Knowledge of that identifier
is not authorization; ownership and share capabilities are separate.

## API boundary

Version all endpoints under `/api/v1`. A minimal contract is:

```text
POST   /api/v1/uploads                         multipart CSV upload + validation
POST   /api/v1/analyses                       create immutable analysis spec
POST   /api/v1/analyses/{id}/runs             queue inference
GET    /api/v1/analyses/{id}                  authorized state and summary
GET    /api/v1/analyses/{id}/events           SSE progress stream
GET    /api/v1/analyses/{id}/artifacts        authorized artifact manifest
GET    /api/v1/analyses/{id}/artifacts/{aid}  authorized download proxy
DELETE /api/v1/analyses/{id}                  early deletion request
POST   /api/v1/analyses/{id}/shares           create read-only share grant
DELETE /api/v1/analyses/{id}/shares/{sid}     revoke share grant
GET    /api/v1/shares/{token}                 exchange token, then redirect
```

Dash validation state is presentation data only. The API
must not accept that object as evidence that a CSV was validated. It must bind
analysis creation to the server-side upload record, content digest and
canonical normalized data. Likewise, colours and plot selections are display
preferences and must not be allowed to alter an immutable inference request.

## Docker Compose baseline

The Compose file should encode the following rather than relying on developer
discipline:

- Pin image versions and, for release builds, image digests. Commit Python and
  JavaScript lock files.
- Use a one-shot `migrate` service or explicit migration command. Do not let
  every API replica race to apply migrations at startup.
- Wait for health checks, not just container creation. Use `pg_isready`, Redis
  `PING`, MinIO `/minio/health/live`, API `/api/v1/health/` and Dash `/healthz`.
- Create private MinIO buckets with a one-shot bootstrap container. Do not use
  public bucket policies.
- Run application containers as fixed non-root UIDs with
  `read_only: true`, `cap_drop: [ALL]`, `security_opt:
  [no-new-privileges:true]`, a bounded `tmpfs`, `pids_limit`, CPU/memory limits
  and file descriptor limits.
- Put the services on one private backend network initially. Publish Dash on
  `127.0.0.1:8501` and, for local CSV downloads, Django on
  `127.0.0.1:8000`. Production should put both behind the Imperial HTTPS
  reverse proxy.
- Give `worker` a job scratch volume, but do not share that volume with Dash
  or the API. Give each data service its own named volume.
- Never mount `/var/run/docker.sock` into a worker. It turns one scientific
  code execution bug into host control.
- Store local secrets outside the committed Compose file. Compose secrets or
  `_FILE` variables are preferable; `.env.example` contains names and safe
  placeholders only.
- Configure Redis with `maxmemory-policy noeviction`. Broker messages must not
  disappear under memory pressure. Redis is not the artifact or durable job
  database.
- Keep MinIO, PostgreSQL and Redis off host ports in the default profile.

The repository already excludes `data/`, figures, notebooks and result files
from the Docker context. Preserve that denylist. Add `platform`-specific build
contexts where possible so private research folders cannot be sent to the
Docker daemon accidentally.

## Worker process model

PyMC SMC starts its own child processes. A Celery prefork child is therefore a
poor place to call it directly: nested process pools can fail, leak children
or multiply configured concurrency. For local Compose:

- run one queue consumer per worker container;
- use a non-prefork/solo orchestrator with concurrency one;
- have the orchestrator launch one fresh scientific subprocess in its own
  process group for every attempt;
- let the validated `chains`/`cores` setting control PyMC's internal
  parallelism; and
- scale worker containers horizontally rather than stacking Celery and PyMC
  concurrency in one container.

Celery's soft time limit is not sufficient with a solo pool. The orchestrator
must enforce wall time, terminate the whole subprocess process group, wait a
short grace period and then issue a hard kill. Cleanup belongs in `finally`.

Production should run each analysis in a dedicated batch container/pod with a
read-only image, per-job scratch volume, resource quota and denied outbound
network. Local Compose is a functional development topology, not equivalent
tenant isolation.

## Python packaging boundary

Do not add `platform/__init__.py`. A top-level importable package named
`platform` shadows Python's standard-library `platform` module, which the
current Barracuda adapters use to record runtime versions.

Package the scientific boundary as a separately named wheel, for example
`barracuda`, and install that wheel into the worker image with an exact version
and hash. The wheel owns model adapters and their scientific dependencies;
`platform/worker` owns orchestration only. Until extraction is complete, a
compatibility wheel may expose the existing UI-neutral functions, but the
worker must not import Dash pages.

The worker and API images may share a base image, but the API should not install
PyMC, compilers or the research runtime. A smaller API image reduces startup
time and attack surface.

## Staged delivery order

1. Freeze request, progress and artifact manifest schema version 1.
2. Produce a reproducible `barracuda` wheel from the current scientific commit
   and run parity/golden tests against existing `webapp.core` adapters.
3. Implement one donor-ignorant event-count worker path end to end.
4. Add donor-aware counts, then donor-ignorant trajectories through the same
   adapter protocol.
5. Add cancellation, expiry and share-token tests before exposing guest jobs.
6. Run destructive retention tests against disposable PostgreSQL/MinIO data.
7. Treat authentication, longer retention and editor sharing as later features,
   not extensions of a guest capability token.
