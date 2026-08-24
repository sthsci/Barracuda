import type { Metadata } from "next";
import { WorkspaceDashboard } from "@/components/WorkspaceDashboard";

export const metadata: Metadata = {
  title: "Guest workspace",
  description: "Start an event-count or contact-trajectory analysis without creating an account.",
};

export default function WorkspacePage() {
  return <WorkspaceDashboard />;
}
