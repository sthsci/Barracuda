import { describe, expect, it, vi } from "vitest";
import { createHttpApiClient, createMockApiClient } from "@/lib/api";

describe("typed API clients", () => {
  it("creates a guest analysis and controlled share link in mock mode", async () => {
    const client = createMockApiClient({ latencyMs: 0, analyses: [] });
    const upload = await client.validateCsv({
      kind: "event-counts",
      filename: "counts.csv",
      content: "cell_id,condition,count\na,Control,0\nb,Treatment,2\n",
    });
    const analysis = await client.createAnalysis({ title: "Test counts", kind: "event-counts", upload });
    const share = await client.createShareLink({ analysisId: analysis.id, access: "viewer", expiresInDays: 1 });
    expect(analysis.isGuestOwned).toBe(true);
    expect(analysis.conditionCount).toBe(2);
    expect(share.url).toContain("viewer");
    expect(share.expiresAt).not.toBeNull();
  });

  it("uses the versioned same-origin API and includes browser credentials", async () => {
    const fetcher = vi.fn<typeof fetch>().mockResolvedValue(
      new Response(JSON.stringify({ analyses: [], total: 0 }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    const client = createHttpApiClient({ fetcher });
    await client.listAnalyses();
    expect(fetcher).toHaveBeenCalledWith(
      "/api/v1/analyses",
      expect.objectContaining({ credentials: "include" }),
    );
  });
});
