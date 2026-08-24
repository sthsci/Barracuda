import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { createMockApiClient } from "@/lib/api";
import { WorkspaceDashboard } from "@/components/WorkspaceDashboard";

vi.mock("@/components/PosteriorPreview", () => ({
  PosteriorPreview: () => <div aria-label="Mock posterior plot" />,
}));

beforeAll(() => {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation(() => ({ matches: false, addListener: vi.fn(), removeListener: vi.fn() })),
  });
});

describe("guest workspace", () => {
  it("renders without authentication and keeps sign-in optional", async () => {
    render(<WorkspaceDashboard client={createMockApiClient({ latencyMs: 0 })} />);
    expect(screen.getByText("Continue without an account.")).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /new analysis/i }).length).toBeGreaterThan(0);
    expect(await screen.findByText("Control and rituximab event counts")).toBeInTheDocument();
    expect(screen.getAllByText(/optional sign in/i).length).toBeGreaterThan(0);
  });

  it("opens a share dialog from a saved analysis", async () => {
    const user = userEvent.setup();
    render(<WorkspaceDashboard client={createMockApiClient({ latencyMs: 0 })} />);
    await screen.findByText("Control and rituximab event counts");
    await user.click(screen.getByRole("button", { name: /share control and rituximab/i }));
    expect(screen.getByRole("dialog", { name: /share analysis/i })).toBeInTheDocument();
    expect(screen.getByText(/guest sharing is temporary/i)).toBeInTheDocument();
    expect(screen.getByText(/expires after 24 hours/i)).toBeInTheDocument();
    expect(screen.queryByText(/use as a starting point/i)).not.toBeInTheDocument();
  });
});
