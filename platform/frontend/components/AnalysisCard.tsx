"use client";

import type { AnalysisSummary } from "@/lib/api";
import { Icon } from "./icons";

const labels = {
  "event-counts": "Event counts",
  trajectory: "Contact trajectory",
} as const;

export function AnalysisCard({
  analysis,
  onShare,
  onOpen,
}: {
  analysis: AnalysisSummary;
  onShare: (analysis: AnalysisSummary) => void;
  onOpen: (analysis: AnalysisSummary) => void;
}) {
  const date = new Intl.DateTimeFormat("en-GB", {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(new Date(analysis.updatedAt));
  return (
    <article className="analysisCard">
      <div className="analysisAccent" style={{ background: analysis.accent }} />
      <div className="analysisCardTop">
        <span className={`analysisKind kind-${analysis.kind}`}>
          <Icon name={analysis.kind === "trajectory" ? "trajectory" : "count"} />
          {labels[analysis.kind]}
        </span>
        <span className={`statusBadge status-${analysis.status}`}>
          <i /> {analysis.status === "ready" ? "Complete" : analysis.status}
        </span>
      </div>
      <h3>{analysis.title}</h3>
      <p>{analysis.conditions.join(" · ")}</p>
      <div className="analysisStats">
        <span><strong>{analysis.cellCount.toLocaleString()}</strong> cells</span>
        <span><strong>{analysis.conditionCount}</strong> condition{analysis.conditionCount === 1 ? "" : "s"}</span>
        <span><strong>{analysis.modelCount}</strong> models</span>
      </div>
      <div className="analysisCardFooter">
        <span><Icon name="clock" />Updated {date}</span>
        <div>
          <button className="iconButton subtle" type="button" onClick={() => onShare(analysis)} aria-label={`Share ${analysis.title}`}>
            <Icon name="share" />
          </button>
          <button className="openAnalysis" type="button" onClick={() => onOpen(analysis)}>Open <Icon name="arrow" /></button>
        </div>
      </div>
    </article>
  );
}
