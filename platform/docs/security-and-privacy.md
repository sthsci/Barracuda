# Guest security, privacy, retention and sharing

## Scope and threat model

The first release accepts anonymous scientific CSV files and runs expensive
PyMC models. Its important assets are uploaded data, donor and condition
labels, posterior artifacts, compute capacity, capability tokens, object-store
credentials and model integrity.

Design for these realistic threats:

- a user uploads malformed or deliberately expensive data;
- one guest guesses or receives another guest's analysis ID;
- a share URL leaks through logs, browser history, screenshots or referrers;
- a duplicated queue delivery runs the same expensive job twice;
- scientific code consumes excessive CPU, memory, processes, disk or time;
- a failed/cancelled task leaves data in `/tmp`, Redis or MinIO;
- spreadsheet software executes a formula embedded in a downloaded identifier;
- a dependency or model error exposes a traceback, filesystem path or another
  job's data; and
- a developer publishes database, Redis or MinIO ports from Compose.

This is not a suitable service for direct identifiers, clinical metadata or
raw microscopy. The upload screen and privacy notice should state that clearly,
but warnings do not replace technical controls.

## Release-blocking controls

1. **Server-authoritative validation.** Accept the CSV as bounded multipart
   bytes. The API parses and normalizes it again using the scientific adapter.
   Never create a job from a client-supplied `valid: true`, row count, donor
   flag, condition list or object key.
2. **Capability separation.** An analysis ID is not a secret. Guest ownership,
   share access and internal worker authorization use separate credentials.
3. **Private storage.** MinIO buckets and objects are private. Downloads pass
   through an authorization check; local deployment should proxy bytes through
   Django rather than expose MinIO.
4. **Bounded jobs.** Validate file size, rows, conditions, cells, donors,
   events, counts, models, particles, chains, cores, quadrature nodes and wall
   time before enqueueing. Enforce the limits again in the worker.
5. **Process isolation.** One fresh scientific subprocess and workspace per
   attempt, non-root, restrictive umask, process-group cancellation and
   container resource limits.
6. **Deterministic deletion.** Every guest analysis has `expires_at` when it is
   created. Access and sharing never extend it. A tested cleanup loop deletes
   the object prefix and sensitive database fields.
7. **No sensitive logs.** Do not log CSV bodies, original filenames, cell,
   donor or condition labels, capability/share tokens, presigned URLs, full
   object keys, posterior samples or raw exception text returned to users.

## Explicit guest retention defaults

Use these defaults for the staged guest-only release:

| Data | Default retention |
|---|---|
| Incomplete multipart upload or validation staging object | 1 hour |
| Accepted guest input, normalized input and successful artifacts | 24 hours from analysis creation |
| Failed or cancelled job input/artifacts | 24 hours from terminal state |
| Worker scratch, PyTensor cache and transient NetCDF files | delete immediately in `finally` |
| Redis progress stream | 1 hour after terminal state; hard TTL 24 hours |
| Read-only guest share grant | at most 24 hours and never beyond analysis expiry |
| Non-sensitive operational tombstone | 30 days; no title, filename, labels, object key or token digest |
| Encrypted infrastructure backups | 7 days, then automatic expiry |

Set a hard job wall time of 45 minutes for the staged preview and a maximum
queue age of 2 hours. A job that reaches either bound becomes failed/expired
and follows the 24-hour deletion path. These are policy defaults, not magic
constants: define them once in server settings, surface the expiry timestamp to
the user and test overrides.

For the local signed-in prototype, projects may remain until the user deletes
them. That is not a production retention policy. Before signed-in production
use, agree and publish a separate retention period, inactivity rule, backup
window and account-deletion behavior. A signed-in default share may be 72
hours, but a guest share requesting the same duration is truncated to the guest
analysis expiry and therefore can never remain valid for more than 24 hours.

Deletion worker behavior:

1. select expired rows in bounded batches with a lock that skips already
   claimed rows;
2. mark the row `deleting` and revoke owner/share capabilities;
3. delete every object/version under the server-derived job prefix;
4. remove Redis streams and broker metadata for the job;
5. erase sensitive database fields and either delete the row or keep only the
   minimal tombstone; and
6. retry partial deletion without constructing a prefix from user input.

MinIO lifecycle rules are a second safety net, not the primary cleanup
mechanism. If object versioning is enabled, cleanup must delete versions and
delete markers too. The privacy notice must explain that deleted data can
remain in encrypted backups until the seven-day backup window expires.

## Guest owner capabilities

On first analysis creation, generate at least 256 random bits with a
cryptographically secure generator. Return the owner capability only in a
`Secure`, `HttpOnly`, `SameSite=Lax`, path-scoped cookie. Store only a
server-peppered HMAC-SHA-256 digest with constant-time comparison; do not store
the plaintext token.

The owner capability may view, retry within limits, revoke shares and delete
that analysis. It does not grant access to any other analysis and is not a
login session. Rotate it after a suspected leak. CSRF protection remains
required for state-changing cookie-authenticated requests.

Rate limit uploads, validation, job creation, status polling, share creation
and token exchange by a combination of capability, IP prefix and global queue
pressure. Return `429` with retry guidance before the queue is exhausted.

## Share-link threat model and design

Guest shares should be **read-only**. `editor` sharing requires authenticated
accounts, an audit trail and a separate authorization design; it must not ship
as a larger guest capability. A guest share may view the result summary and
download explicitly approved result spreadsheets/artifacts, but should not
view the original uploaded CSV, rerun inference, create another share or delete
the analysis. Raw uploaded CSV files remain private even when results are
shared.

Generate an independent, single-purpose token with at least 192 random bits
(256 preferred). Store only its peppered digest along with:

- analysis ID;
- scope (`results:read` in release 1);
- `created_at`, mandatory `expires_at` and optional `revoked_at`; and
- last-used timestamp or coarse access count only if the privacy notice needs
  it.

Do not derive tokens from analysis IDs, access names or timestamps. A share
expires at `min(now + requested_duration, analysis.expires_at)` and analysis
deletion revokes it immediately.

Because URL secrets leak easily, use an exchange flow:

1. `GET /api/v1/shares/{token}` performs a constant-time digest lookup and
   rate-limit check.
2. On success, set a short-lived, `Secure`, `HttpOnly`, `SameSite=Lax`
   read-only share-session cookie.
3. Respond `303` to a tokenless analysis URL.
4. Send `Cache-Control: no-store` and `Referrer-Policy: no-referrer` on the
   exchange and shared pages. Scrub request paths in reverse-proxy/APM logs.

Return the same generic not-found response for unknown, expired and revoked
tokens. Do not reveal whether an analysis ID exists. Apply a restrictive CSP,
`frame-ancestors 'none'`, `X-Content-Type-Options: nosniff` and a minimal
Permissions Policy. A default `strict-origin-when-cross-origin`
is insufficient for a page whose URL contains a capability; the redirect must
remove the token and the exchange route must use `no-referrer`.

## Upload and download handling

- Use multipart upload, not a JSON field containing the entire CSV. Enforce a
  5 MiB transport limit in the reverse proxy and Django for the staged release;
  scientific row/cell/event caps remain substantially stricter.
- Stream to a server-selected temporary object while calculating SHA-256.
  Ignore browser MIME type and original path. Decode as UTF-8/UTF-8 with BOM
  only unless another encoding is explicitly supported.
- Parse with bounded rows and columns. Current scientific limits remain the
  authority: up to four conditions, 1,000 cells, bounded counts/events and the
  donor/trajectory constraints exposed by the adapter package.
- Reject NULs and control characters in identifiers. Limit identifier/label
  lengths. Treat cells, donors and conditions as pseudonymous data, not safe
  log strings.
- Protect exported CSVs from spreadsheet formula injection. Text fields that
  begin (after whitespace) with `=`, `+`, `-`, `@`, tab or carriage return
  should be rejected at upload or safely neutralized in the spreadsheet-facing
  export. CSV quoting alone is not protection.
- Set downloads to `application/octet-stream` or their exact safe media type,
  `Content-Disposition: attachment`, `X-Content-Type-Options: nosniff`, and a
  server-generated filename. Never render user CSV/HTML inline.
- Store a digest and byte count for every artifact. Before marking a job ready,
  verify the uploaded object against the manifest.

## Worker isolation and secrets

Create the workspace from a trusted job UUID beneath one fixed scratch root,
verify the resolved path remains under that root and use `umask 077`. Do not
reuse a PyTensor compilation directory across guests. The scientific process
must start with job-scoped values for `MPLCONFIGDIR`, `XDG_CACHE_HOME` and
`PYTENSOR_FLAGS=base_compiledir=...` before importing PyMC/PyTensor.

The subprocess receives only normalized input, an immutable request file and
an output directory. It must not receive Django/MinIO/Redis credentials. The
orchestrator uploads its outputs after validating the manifest. In production,
deny the scientific container outbound network. In local Compose, document
that the shared worker network is weaker isolation and never mount host or
Docker control sockets.

Run as non-root, drop Linux capabilities, disable privilege escalation, use a
read-only root filesystem and bound memory, CPU, pids, open files, output bytes
and wall time. Send cancellation to the entire process group; kill descendants
after a short grace period. A user-visible error contains a stable code and
safe explanation, while a private correlation ID links to the restricted
traceback.

Secrets belong in secret files or a managed secret store. Rotate the Django
secret key, database password, Redis credentials, MinIO credentials and token
HMAC pepper independently. Never use one of them to derive another.

## Verification gates

Before guest access, automate at least these tests:

- cross-analysis owner and share tokens receive `404`/`403` consistently;
- expired/revoked shares fail and cannot outlive analysis expiry;
- share pages cannot fetch raw input or invoke mutation endpoints;
- client-forged validation metadata cannot create a job;
- duplicate queue delivery produces one scientific attempt/artifact set;
- cancellation kills sampler descendants and removes the workspace;
- CPU, memory, pids, disk and wall limits fail closed;
- cleanup removes MinIO objects, Redis stream, token digests and sensitive DB
  columns, including partial-retry and versioned-object cases;
- logs contain no submitted CSV data, tokens, presigned URLs or labels;
- formula-like identifiers cannot become active spreadsheet formulas; and
- an analysis cannot read or overwrite another job's object prefix.
