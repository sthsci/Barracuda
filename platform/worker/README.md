# Worker boundary

`platform/worker` is reserved for queue orchestration, scientific subprocess
supervision, progress transport and artifact verification. It must not contain
copied PyMC likelihoods or simulator implementations.

The local Compose worker installs `barracuda` from the checked-out source and
runs a single Celery consumer. It revalidates every CSV through the public
package API, emits native SMC chain/stage/beta progress, produces compact
result JSON, and returns verified CSV/ZIP artifact bytes for the API to store.
For a release image, replace the source install with a versioned, hash-pinned
wheel.

See [the worker adapter contract](../docs/worker-adapter-contract.md) before
adding executable code here.
