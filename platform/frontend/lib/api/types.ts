export type AnalysisKind = "event-counts" | "trajectory";
export type AnalysisStatus = "draft" | "queued" | "running" | "ready" | "failed" | "cancelled";
export type AccessLevel = "viewer";

export interface AnalysisArtifact {
  id: string;
  role: string;
  filename: string;
  contentType: string;
  byteSize: number;
  sha256: string;
  shareable: boolean;
}

export interface AnalysisJob {
  id: string;
  projectId: string;
  datasetId: string;
  analysisType: string;
  configuration: Record<string, unknown>;
  status: AnalysisStatus;
  progress: number;
  progressDetail?: {
    phase?: string;
    message?: string;
    condition_index?: number;
    condition_total?: number;
    condition?: string;
    model_index?: number;
    model_total?: number;
    model?: string;
    chain?: number;
    stage?: number;
    beta?: number;
  };
  result: Record<string, unknown> | null;
  errorCode: string;
  errorMessage: string;
  createdAt: string;
  startedAt: string | null;
  completedAt: string | null;
  artifacts: AnalysisArtifact[];
}

export interface AnalysisSummary {
  id: string;
  projectId: string;
  datasetId: string | null;
  job: AnalysisJob | null;
  title: string;
  kind: AnalysisKind;
  status: AnalysisStatus;
  createdAt: string;
  updatedAt: string;
  cellCount: number;
  conditionCount: number;
  conditions: string[];
  modelCount: number;
  ownerLabel: string;
  isGuestOwned: boolean;
  expiresAt: string | null;
  accent: string;
}

export interface AnalysisListResponse { analyses: AnalysisSummary[]; total: number; }
export interface CsvValidationRequest { kind: AnalysisKind; filename: string; content: string; }
export interface CsvPreviewRow { [column: string]: string; }
export interface CsvValidationResult {
  valid: boolean; kind: AnalysisKind; filename: string; columns: string[]; rowCount: number;
  conditionCount: number; conditions: string[]; donorAware: boolean; preview: CsvPreviewRow[];
  warnings: string[]; errors: string[];
}
export interface CreateAnalysisRequest {
  title: string;
  kind: AnalysisKind;
  upload: CsvValidationResult;
  /** The original CSV. Required by the HTTP adapter and intentionally never persisted in the UI. */
  file?: File;
}
export interface ShareAnalysisRequest { analysisId: string; access: AccessLevel; expiresInDays: 1; }
export interface ShareLink { id: string; analysisId: string; url: string; access: AccessLevel; expiresAt: string | null; }
export interface AuthIdentity { username: string; token: string; }
export interface SharedAnalysis { id: string; name: string; description: string; datasetCount: number; jobs: AnalysisJob[]; expiresAt: string; }

export interface ApiClient {
  listAnalyses(signal?: AbortSignal): Promise<AnalysisListResponse>;
  validateCsv(request: CsvValidationRequest, signal?: AbortSignal): Promise<CsvValidationResult>;
  createAnalysis(request: CreateAnalysisRequest, signal?: AbortSignal): Promise<AnalysisSummary>;
  getJob(jobId: string, signal?: AbortSignal): Promise<AnalysisJob>;
  downloadArtifact(artifactId: string, signal?: AbortSignal): Promise<Blob>;
  createShareLink(request: ShareAnalysisRequest, signal?: AbortSignal): Promise<ShareLink>;
  getSharedAnalysis(token: string, signal?: AbortSignal): Promise<SharedAnalysis>;
  getIdentity(): AuthIdentity | null;
  register(username: string, password: string, email?: string, signal?: AbortSignal): Promise<AuthIdentity>;
  login(username: string, password: string, signal?: AbortSignal): Promise<AuthIdentity>;
  logout(signal?: AbortSignal): Promise<void>;
}

export class ApiError extends Error {
  constructor(message: string, public readonly status: number, public readonly details?: unknown) {
    super(message); this.name = "ApiError";
  }
}
