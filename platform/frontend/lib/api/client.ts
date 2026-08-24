import { validateCsvLocally } from "./csv";
import {
  ApiError, type AnalysisJob, type AnalysisKind, type AnalysisListResponse, type AnalysisStatus,
  type AnalysisSummary, type ApiClient, type AuthIdentity, type CreateAnalysisRequest,
  type CsvValidationRequest, type CsvValidationResult, type ShareAnalysisRequest, type ShareLink,
  type SharedAnalysis,
} from "./types";

export interface HttpApiOptions { baseUrl?: string; fetcher?: typeof fetch; storage?: Storage; }

const GUEST_KEY = "barracuda.guest-token";
const ACCOUNT_KEY = "barracuda.account";
const accentFor = (kind: AnalysisKind) => kind === "trajectory" ? "#8A4331" : "#486857";
const isRecord = (value: unknown): value is Record<string, unknown> => !!value && typeof value === "object" && !Array.isArray(value);
const asString = (value: unknown, fallback = "") => typeof value === "string" ? value : fallback;
const asNumber = (value: unknown, fallback = 0) => typeof value === "number" && Number.isFinite(value) ? value : fallback;

interface ApiProject { id: string; name: string; description: string; owner_type: "guest" | "account"; created_at: string; updated_at: string; expires_at: string | null; }
interface ApiDataset { id: string; project: string; original_name: string; row_count: number; column_count: number; columns: string[]; created_at: string; }
interface ApiArtifact { id: string; role: string; filename: string; content_type: string; byte_size: number; sha256: string; shareable: boolean; }
interface ApiJob { id: string; project: string; dataset: string; analysis_type: string; configuration: Record<string, unknown>; status: string; progress: number; progress_detail?: AnalysisJob["progressDetail"]; result: Record<string, unknown> | null; error_code: string; error_message: string; created_at: string; started_at: string | null; completed_at: string | null; artifacts?: ApiArtifact[]; }
interface Page<T> { count: number; results: T[]; }

function browserStorage(provided?: Storage) {
  if (provided) return provided;
  return typeof window === "undefined" ? undefined : window.localStorage;
}
function parseIdentity(storage?: Storage): AuthIdentity | null {
  try { const raw = storage?.getItem(ACCOUNT_KEY); return raw ? JSON.parse(raw) as AuthIdentity : null; } catch { return null; }
}
function toKind(type: string): AnalysisKind { return type.startsWith("trajectory") ? "trajectory" : "event-counts"; }
function toStatus(status: string): AnalysisStatus {
  if (status === "succeeded") return "ready";
  if (["queued", "running", "failed", "cancelled"].includes(status)) return status as AnalysisStatus;
  return "draft";
}
function toJob(job: ApiJob): AnalysisJob {
  return { id: job.id, projectId: job.project, datasetId: job.dataset, analysisType: job.analysis_type,
    configuration: isRecord(job.configuration) ? job.configuration : {}, status: toStatus(job.status),
    progress: Math.max(0, Math.min(1, asNumber(job.progress))), progressDetail: isRecord(job.progress_detail) ? job.progress_detail : undefined,
    result: isRecord(job.result) ? job.result : null,
    errorCode: asString(job.error_code), errorMessage: asString(job.error_message), createdAt: job.created_at,
    startedAt: job.started_at, completedAt: job.completed_at,
    artifacts: (job.artifacts ?? []).map((artifact) => ({ id: artifact.id, role: artifact.role, filename: artifact.filename, contentType: artifact.content_type, byteSize: artifact.byte_size, sha256: artifact.sha256, shareable: artifact.shareable })) };
}
function resultConditions(result: Record<string, unknown> | null): string[] {
  const conditions = result?.conditions;
  return Array.isArray(conditions) ? conditions.filter((item): item is string => typeof item === "string") : [];
}
function buildSummary(project: ApiProject, datasets: ApiDataset[], jobs: AnalysisJob[]): AnalysisSummary {
  const dataset = datasets[0] ?? null;
  const job = jobs[0] ?? null;
  const kind = job ? toKind(job.analysisType) : (dataset?.columns.includes("history") ? "trajectory" : "event-counts");
  const conditions = resultConditions(job?.result ?? null);
  return { id: project.id, projectId: project.id, datasetId: dataset?.id ?? null, job, title: project.name,
    kind, status: job?.status ?? "draft", createdAt: project.created_at, updatedAt: job?.completedAt ?? job?.startedAt ?? project.updated_at,
    cellCount: dataset?.row_count ?? 0, conditionCount: conditions.length, conditions,
    modelCount: Array.isArray(job?.configuration.models) ? job!.configuration.models.length : 4,
    ownerLabel: project.owner_type === "guest" ? "Guest workspace" : "Account workspace", isGuestOwned: project.owner_type === "guest",
    expiresAt: project.expires_at, accent: accentFor(kind) };
}

export function createHttpApiClient(options: HttpApiOptions = {}): ApiClient {
  const baseUrl = (options.baseUrl ?? "/api/v1").replace(/\/$/, "");
  const fetcher = options.fetcher ?? fetch;
  const storage = browserStorage(options.storage);
  const apiUrl = (path: string) => `${baseUrl}${path}`;
  const identity = () => parseIdentity(storage);
  const clearGuest = () => storage?.removeItem(GUEST_KEY);

  async function raw(path: string, init: RequestInit = {}, { guest = true }: { guest?: boolean } = {}): Promise<unknown> {
    const headers = new Headers(init.headers);
    const account = identity();
    if (account?.token) headers.set("Authorization", `Token ${account.token}`);
    if (guest) {
      const guestToken = storage?.getItem(GUEST_KEY);
      if (guestToken) headers.set("X-Barracuda-Guest-Token", guestToken);
    }
    const response = await fetcher(apiUrl(path), { ...init, credentials: "same-origin", headers });
    const payload = response.status === 204 ? null : await response.json().catch(() => null);
    if (!response.ok) {
      const detail = errorDetail(payload);
      throw new ApiError(detail || "The BARRACUDA API request failed.", response.status, payload);
    }
    return payload;
  }
  async function ensureGuest(signal?: AbortSignal) {
    if (identity()?.token || storage?.getItem(GUEST_KEY)) return;
    const payload = await raw("/guest-sessions/", { method: "POST", signal, headers: { "Content-Type": "application/json" } }, { guest: false }) as { guest_token?: unknown };
    const token = asString(payload?.guest_token);
    if (!token) throw new ApiError("The server did not create a guest session.", 502, payload);
    storage?.setItem(GUEST_KEY, token);
  }
  async function request<T>(path: string, init: RequestInit = {}, opts: { guest?: boolean; needsGuest?: boolean } = {}): Promise<T> {
    if (opts.needsGuest ?? true) await ensureGuest(init.signal ?? undefined);
    return raw(path, init, { guest: opts.guest }) as Promise<T>;
  }
  async function downloadArtifact(artifactId: string, signal?: AbortSignal): Promise<Blob> {
    await ensureGuest(signal);
    const headers = new Headers();
    const account = identity();
    if (account?.token) headers.set("Authorization", `Token ${account.token}`);
    const guestToken = storage?.getItem(GUEST_KEY);
    if (guestToken) headers.set("X-Barracuda-Guest-Token", guestToken);
    const response = await fetcher(apiUrl(`/artifacts/${encodeURIComponent(artifactId)}/download/`), { signal, credentials: "same-origin", headers });
    if (!response.ok) throw new ApiError("The result file could not be downloaded.", response.status);
    return response.blob();
  }
  const json = (method: string, body: unknown, signal?: AbortSignal, headers?: HeadersInit) => ({ method, signal, body: JSON.stringify(body), headers: { "Content-Type": "application/json", ...headers } });
  async function all<T>(path: string, signal?: AbortSignal): Promise<T[]> {
    const payload = await request<Page<T> | T[]>(path, { signal });
    return Array.isArray(payload) ? payload : payload.results;
  }
  async function listAnalyses(signal?: AbortSignal): Promise<AnalysisListResponse> {
    const [projects, datasets, apiJobs] = await Promise.all([all<ApiProject>("/projects/", signal), all<ApiDataset>("/datasets/", signal), all<ApiJob>("/jobs/", signal)]);
    const jobs = apiJobs.map(toJob);
    const analyses = projects.map((project) => buildSummary(project, datasets.filter((item) => item.project === project.id), jobs.filter((item) => item.projectId === project.id)));
    return { analyses, total: analyses.length };
  }
  async function createAnalysis(requestBody: CreateAnalysisRequest, signal?: AbortSignal): Promise<AnalysisSummary> {
    if (!requestBody.upload.valid) throw new ApiError("The CSV must be valid before starting an analysis.", 400);
    if (!requestBody.file) throw new ApiError("Keep the selected CSV in this browser until the analysis starts.", 400);
    const project = await request<ApiProject>("/projects/", json("POST", { name: requestBody.title.trim() || requestBody.upload.filename.replace(/\.csv$/i, ""), description: "Created from the BARRACUDA web workspace." }, signal));
    try {
      const form = new FormData(); form.set("project_id", project.id); form.set("file", requestBody.file, requestBody.file.name);
      const dataset = await request<ApiDataset>("/datasets/", { method: "POST", body: form, signal });
      const analysisType = requestBody.kind === "trajectory" ? "trajectory_donor_ignorant" : requestBody.upload.donorAware ? "event_count_donor_aware" : "event_count_donor_ignorant";
      const idempotencyKey = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
      const apiJob = await request<ApiJob>("/jobs/", json("POST", { project_id: project.id, dataset_id: dataset.id, analysis_type: analysisType, configuration: {} }, signal, { "Idempotency-Key": idempotencyKey }));
      const summary = buildSummary(project, [dataset], [toJob(apiJob)]);
      return { ...summary, conditions: requestBody.upload.conditions, conditionCount: requestBody.upload.conditionCount };
    } catch (error) {
      // A project without a dataset/job has no scientific value; best-effort cleanup keeps guest storage tidy.
      void request(`/projects/${encodeURIComponent(project.id)}/`, { method: "DELETE" }, { needsGuest: false }).catch(() => undefined);
      throw error;
    }
  }
  async function setIdentity(payload: { token?: unknown; user?: { username?: unknown } }) {
    const next = { token: asString(payload.token), username: asString(payload.user?.username) };
    if (!next.token || !next.username) throw new ApiError("The server returned an incomplete account response.", 502, payload);
    storage?.setItem(ACCOUNT_KEY, JSON.stringify(next));
    const guestToken = storage?.getItem(GUEST_KEY);
    if (guestToken) {
      try { await raw("/guest-sessions/claim/", json("POST", {})); clearGuest(); }
      catch (error) { storage?.removeItem(ACCOUNT_KEY); throw error; }
    }
    return next;
  }
  return {
    listAnalyses,
    validateCsv: async (body: CsvValidationRequest) => validateCsvLocally(body),
    createAnalysis,
    getJob: async (jobId, signal) => toJob(await request<ApiJob>(`/jobs/${encodeURIComponent(jobId)}/`, { signal })),
    downloadArtifact,
    createShareLink: async (body: ShareAnalysisRequest, signal) => {
      const value = await request<{ id: string; expires_at: string | null; share_token: string }>("/share-links/", json("POST", { project_id: body.analysisId, expires_in_hours: 24, allow_dataset_download: false }, signal));
      const origin = typeof window === "undefined" ? "" : window.location.origin;
      return { id: value.id, analysisId: body.analysisId, access: "viewer", expiresAt: value.expires_at, url: `${origin}/shared/${encodeURIComponent(value.share_token)}` };
    },
    getSharedAnalysis: async (token, signal) => {
      const value = await raw(`/shared/${encodeURIComponent(token)}/`, { signal }, { guest: false }) as Record<string, unknown>;
      const share = isRecord(value.share) ? value.share : {};
      const apiJobs = Array.isArray(value.jobs) ? value.jobs.filter(isRecord) as unknown as ApiJob[] : [];
      return { id: asString(value.id), name: asString(value.name), description: asString(value.description), datasetCount: asNumber(value.dataset_count), jobs: apiJobs.map(toJob), expiresAt: asString(share.expires_at) };
    },
    getIdentity: identity,
    register: async (username, password, email, signal) => setIdentity(await raw("/auth/register/", json("POST", { username, password, ...(email ? { email } : {}) }, signal), { guest: false }) as { token?: unknown; user?: { username?: unknown } }),
    login: async (username, password, signal) => setIdentity(await raw("/auth/login/", json("POST", { username, password }, signal), { guest: false }) as { token?: unknown; user?: { username?: unknown } }),
    logout: async (signal) => { try { await raw("/auth/logout/", { method: "POST", signal, headers: { "Content-Type": "application/json" } }); } finally { storage?.removeItem(ACCOUNT_KEY); } },
  };
}

function errorDetail(payload: unknown): string {
  if (!isRecord(payload)) return "";
  if (isRecord(payload.error)) return asString(payload.error.detail);
  if (typeof payload.detail === "string") return payload.detail;
  for (const [field, value] of Object.entries(payload)) {
    if (typeof value === "string") return `${field}: ${value}`;
    if (Array.isArray(value) && typeof value[0] === "string") return `${field}: ${value[0]}`;
    if (isRecord(value)) {
      const nested = errorDetail(value);
      if (nested) return nested;
    }
  }
  return "";
}
