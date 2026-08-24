import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { createMockApiClient } from "@/lib/api";
import { UploadWizard } from "@/components/UploadWizard";

describe("UploadWizard", () => {
  it("takes a guest from analysis choice through safe example review", async () => {
    const user = userEvent.setup();
    render(
      <UploadWizard
        client={createMockApiClient({ latencyMs: 0, analyses: [] })}
        onCreated={vi.fn()}
        onCancel={vi.fn()}
      />,
    );
    await user.click(screen.getByRole("button", { name: /ordered binary histories/i }));
    await user.click(screen.getByRole("button", { name: /use a safe example csv/i }));
    expect(await screen.findByText(/4 cells · 2 conditions/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/analysis name/i)).toHaveValue("barracuda trajectory example");
    expect(screen.getByRole("button", { name: /create analysis/i })).toBeEnabled();
  });

  it("creates an event-count analysis", async () => {
    const user = userEvent.setup();
    const created = vi.fn();
    render(
      <UploadWizard
        client={createMockApiClient({ latencyMs: 0, analyses: [] })}
        initialKind="event-counts"
        onCreated={created}
        onCancel={vi.fn()}
      />,
    );
    await user.click(screen.getByRole("button", { name: /use a safe example csv/i }));
    await screen.findByText(/4 cells · 2 conditions/i);
    await user.click(screen.getByRole("button", { name: /create analysis/i }));
    expect(created).toHaveBeenCalledWith(expect.objectContaining({ kind: "event-counts", status: "draft" }));
  });
});
