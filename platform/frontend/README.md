# Archived frontend experiment

This Next.js shell is not the BARRACUDA product interface and is not used
by the local or deployment stack. The established Dash application remains
the visual and scientific interface. This directory is retained only as an
architecture experiment and should not be presented to users or deployed.

An isolated Next.js migration shell for the BARRACUDA web platform. It does not import or modify the existing Dash application or the scientific Python packages.

## Included UX

- Public, guest-first landing page with a direct **Start as a guest** route.
- Optional account affordance; no authentication wall blocks the workspace.
- Responsive analysis dashboard, saved analysis cards and example posterior plot.
- CSV workflow for event counts and ordered contact trajectories, including local schema checks and safe examples.
- Accessible share modal with viewer/copy permissions and expiry controls.
- Typed API interface with HTTP and deterministic in-memory mock adapters.
- Responsive Plotly visualisation with image export.

The data and inference results shown in mock mode are illustrative. Uploaded data are held only in component memory; no file is submitted to a server until the HTTP API adapter is selected.

## Run locally

```bash
cd platform/frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The default `mock` API mode works without a backend.

```bash
cp .env.example .env.local
```

## API boundary

Browser requests use the same-origin prefix `/api/v1`. Next proxies that prefix to `BARRACUDA_API_INTERNAL_URL`, avoiding browser CORS and cross-origin cookie handling.

| Variable | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_API_MODE` | `mock` | Use `mock` or `http`. |
| `NEXT_PUBLIC_API_URL` | `/api/v1` | Browser-visible API prefix. |
| `BARRACUDA_API_INTERNAL_URL` | `http://api:8000` in `.env.example` | Server-side proxy destination. Use `http://localhost:8000` outside containers. |

The HTTP adapter includes cookies on every request and sends the `barracuda_csrf` cookie as `X-CSRF-Token` for mutating requests. Contracts live in `lib/api/types.ts`; switch adapters without changing components.

Expected endpoints:

```text
GET  /api/v1/analyses
POST /api/v1/uploads/validate
POST /api/v1/analyses
POST /api/v1/analyses/{analysis_id}/shares
```

## Quality checks

```bash
npm run typecheck
npm run lint
npm test
npm run build
```

## Container

```bash
docker build -t barracuda-frontend platform/frontend
docker run --rm -p 3000:3000 \
  -e NEXT_PUBLIC_API_MODE=mock \
  barracuda-frontend
```

The image defaults to HTTP mode with `/api/v1` proxied to `http://api:8000`. Override build-time routing when needed:

```bash
docker build platform/frontend \
  --build-arg BARRACUDA_API_INTERNAL_URL=http://api:8000 \
  --build-arg NEXT_PUBLIC_API_URL=/api/v1 \
  --build-arg NEXT_PUBLIC_API_MODE=http
```

Next resolves rewrites while building, so `BARRACUDA_API_INTERNAL_URL` must be supplied as a build argument in container deployments.

## Migration notes

- The frontend deliberately treats sign-in, durable storage and managed share links as optional enhancements.
- Mock state is process-local and resets on reload; this prevents the prototype from implying persistence it does not provide.
- Production deployment still needs the platform authentication, job queue, retention policy, privacy notice, audit logging and accessibility review.
