import { validateCsvLocally } from "./csv";
import { seedAnalyses } from "./seed";
import type {
  AnalysisListResponse,
  AnalysisJob,
  AnalysisSummary,
  ApiClient,
  AuthIdentity,
  CreateAnalysisRequest,
  CsvValidationRequest,
  ShareAnalysisRequest,
  ShareLink,
  SharedAnalysis,
} from "./types";

export interface MockApiOptions {
  latencyMs?: number;
  analyses?: AnalysisSummary[];
  baseUrl?: string;
}

const wait = (milliseconds: number, signal?: AbortSignal) =>
  new Promise<void>((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Aborted", "AbortError"));
    const timer = setTimeout(resolve, milliseconds);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(new DOMException("Aborted", "AbortError"));
      },
      { once: true },
    );
  });

export function createMockApiClient(options: MockApiOptions = {}): ApiClient {
  const latency = options.latencyMs ?? 180;
  const analyses = (options.analyses ?? seedAnalyses).map((analysis) => ({ ...analysis }));
  const baseUrl = options.baseUrl ?? "https://app.barracuda.org/share";
  let identity: AuthIdentity | null = null;

  return {
    async listAnalyses(signal): Promise<AnalysisListResponse> {
      await wait(latency, signal);
      return { analyses: analyses.map((analysis) => ({ ...analysis })), total: analyses.length };
    },

    async validateCsv(request: CsvValidationRequest, signal) {
      await wait(latency, signal);
      return validateCsvLocally(request);
    },

    async createAnalysis(request: CreateAnalysisRequest, signal) {
      await wait(latency, signal);
      if (!request.upload.valid) throw new Error("The upload must be valid before creating an analysis.");
      const now = new Date().toISOString();
      const analysis: AnalysisSummary = {
        id: `ana_${Math.random().toString(36).slice(2, 10)}`,
        projectId: `ana_${Math.random().toString(36).slice(2, 10)}`,
        datasetId: `data_${Math.random().toString(36).slice(2, 10)}`,
        job: null,
        title: request.title.trim() || request.upload.filename.replace(/\.csv$/i, ""),
        kind: request.kind,
        status: "draft",
        createdAt: now,
        updatedAt: now,
        cellCount: request.upload.rowCount,
        conditionCount: request.upload.conditionCount,
        conditions: request.upload.conditions,
        modelCount: request.kind === "trajectory" ? 4 : 4,
        ownerLabel: "Guest workspace",
        isGuestOwned: true,
        expiresAt: new Date(Date.now() + 86_400_000).toISOString(),
        accent: request.kind === "trajectory" ? "#8A4331" : "#486857",
      };
      analyses.unshift(analysis);
      return { ...analysis };
    },

    async createShareLink(request: ShareAnalysisRequest, signal): Promise<ShareLink> {
      await wait(latency, signal);
      if (!analyses.some((analysis) => analysis.id === request.analysisId)) {
        throw new Error("Analysis not found.");
      }
      const token = `${request.analysisId.slice(-8)}-${request.access}`;
      const expiresAt = request.expiresInDays
        ? new Date(Date.now() + request.expiresInDays * 86_400_000).toISOString()
        : null;
      return {
        id: `shr_${token}`,
        analysisId: request.analysisId,
        url: `${baseUrl}/${token}`,
        access: request.access,
        expiresAt,
      };
    },

    async getJob(jobId: string, signal): Promise<AnalysisJob> {
      await wait(latency, signal);
      const analysis = analyses.find((item) => item.job?.id === jobId);
      if (!analysis?.job) throw new Error("Analysis job not found.");
      return { ...analysis.job };
    },
    async downloadArtifact(): Promise<Blob> {
      return new Blob(["Mock BARRACUDA result\n"], { type: "text/plain" });
    },

    async getSharedAnalysis(token: string, signal): Promise<SharedAnalysis> {
      await wait(latency, signal);
      const analysis = analyses.find((item) => token.includes(item.id.slice(-8))) ?? analyses[0];
      if (!analysis) throw new Error("Shared analysis not found.");
      return { id: analysis.id, name: analysis.title, description: "Illustrative read-only result.", datasetCount: 1, jobs: analysis.job ? [{ ...analysis.job }] : [], expiresAt: new Date(Date.now() + 86_400_000).toISOString() };
    },

    getIdentity: () => identity,
    async register(username: string, _password: string, _email?: string, signal?: AbortSignal) {
      await wait(latency, signal);
      identity = { username, token: "mock-account-token" };
      return identity;
    },
    async login(username: string, _password: string, signal?: AbortSignal) {
      await wait(latency, signal);
      identity = { username, token: "mock-account-token" };
      return identity;
    },
    async logout(signal?: AbortSignal) { await wait(latency, signal); identity = null; },
  };
}
