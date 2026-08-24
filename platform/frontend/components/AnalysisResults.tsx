"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import type { Config, Data, Layout } from "plotly.js";
import type { AnalysisArtifact, AnalysisJob, AnalysisSummary, ApiClient } from "@/lib/api";
import { Icon } from "./icons";

const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => <div className="plotLoading" aria-label="Loading result plot" />,
});

type Row = Record<string, unknown>;
const META = new Set(["condition", "model", "model_key", "posterior_draw", "chain", "draw"]);
const COLOURS = ["#304B3D", "#B8512B", "#31556A", "#9B712D", "#7D5A8A", "#63806C"];

const record = (value: unknown): Row => value && typeof value === "object" && !Array.isArray(value) ? value as Row : {};
const rows = (value: unknown): Row[] => Array.isArray(value) ? value.filter((item): item is Row => !!item && typeof item === "object" && !Array.isArray(item)) : [];
const number = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? value : null;
const text = (value: unknown, fallback = "") => typeof value === "string" ? value : fallback;
const truthy = (value: unknown) => value === true || value === "true" || value === "True" || value === 1;
const label = (key: string) => ({
  mu_lambda: "Mean event rate μλ",
  sigma_lambda: "Continuous cell-to-cell heterogeneity σλ",
  p_zero: "Non-engaging fraction φ₀",
  lambda_: "Event rate λ",
  lambda: "Event rate λ",
  eta: "History effect η",
  beta: "History dependence β",
  p0: "Initial lethal-contact probability p₀",
}[key] ?? key.replaceAll("_", " "));

function resultParts(job: AnalysisJob) {
  const root = record(job.result);
  const summary = record(root.summary);
  return {
    evidence: rows(summary.evidence ?? root.evidence),
    posterior: rows(summary.posterior_draws ?? root.posterior_draws),
    posteriorSummary: rows(summary.posterior_summary ?? root.posterior_summary),
    normalization: record(root.normalization),
  };
}

function ProgressPanel({ job }: { job: AnalysisJob }) {
  const percent = Math.round(job.progress * 100);
  const detail = job.progressDetail;
  const condition = detail?.condition;
  const model = detail?.model;
  return (
    <section className="jobProgress" aria-live="polite">
      <div className="jobProgressTitle"><span className="statusBadge status-running"><i />{job.status}</span><strong>{percent}%</strong></div>
      <div className="progressTrack" aria-label={`Inference ${percent}% complete`} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}><span style={{ width: `${percent}%` }} /></div>
      <div className="progressFacts">
        <span><small>Phase</small>{detail?.phase ?? (job.status === "queued" ? "Waiting for a worker" : "PyMC sampling")}</span>
        {condition && <span><small>Condition</small>{condition}</span>}
        {model && <span><small>Model</small>{model}</span>}
        {typeof detail?.chain === "number" && <span><small>SMC update</small>Chain {detail.chain + 1} · stage {detail.stage ?? "–"} · β {typeof detail.beta === "number" ? detail.beta.toFixed(3) : "–"}</span>}
      </div>
      <p>Live progress comes from the PyMC sampler. You can leave this page open while inference runs.</p>
    </section>
  );
}

function EvidencePlot({ evidence }: { evidence: Row[] }) {
  const conditions = Array.from(new Set(evidence.map((row) => text(row.condition, "All cells"))));
  const [condition, setCondition] = useState(conditions[0] ?? "All cells");
  const filtered = evidence.filter((row) => text(row.condition, "All cells") === condition);
  const values = filtered.map((row) => number(row.log10_BF_best_vs_model) ?? 0);
  const names = filtered.map((row) => text(row.short_model) || text(row.model) || text(row.model_key, "Model"));
  const max = Math.max(3, ...values.map((value) => Math.ceil(value * 10) / 10));
  const bands: NonNullable<Partial<Layout>["shapes"]> = [
    { type: "rect", x0: 0, x1: Math.log10(3), y0: -0.5, y1: names.length - 0.5, fillcolor: "#EEEAE1", line: { width: 0 }, layer: "below" },
    { type: "rect", x0: Math.log10(3), x1: 1, y0: -0.5, y1: names.length - 0.5, fillcolor: "#FFF1BE", line: { width: 0 }, layer: "below" },
    { type: "rect", x0: 1, x1: 2, y0: -0.5, y1: names.length - 0.5, fillcolor: "#F9D99D", line: { width: 0 }, layer: "below" },
    { type: "rect", x0: 2, x1: max, y0: -0.5, y1: names.length - 0.5, fillcolor: "#EBA58F", line: { width: 0 }, layer: "below" },
  ];
  const data: Data[] = [{
    type: "bar", orientation: "h", y: names, x: values,
    marker: { color: filtered.map((row) => truthy(row.is_best) ? "#304B3D" : "#6EAED1"), line: { color: "#25231F", width: 0.7 } },
    customdata: filtered.map((row) => [text(row.model), number(row.log_evidence), truthy(row.is_best)] as Plotly.Datum[]),
    hovertemplate: "%{customdata[0]}<br>log₁₀ BF(best/model) = %{x:.3f}<br>log evidence = %{customdata[1]:.2f}<extra></extra>",
    text: filtered.map((row, index) => truthy(row.is_best) ? "Best model" : values[index].toFixed(2)),
    textposition: "outside", cliponaxis: false,
  }];
  const layout: Partial<Layout> = {
    autosize: true, height: Math.max(330, 95 + names.length * 58), margin: { l: 180, r: 76, t: 52, b: 72 },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "#FAF8F2", shapes: bands,
    font: { family: "Avenir Next, Segoe UI, sans-serif", color: "#4E493F", size: 11 },
    xaxis: { title: { text: "log₁₀ BF(best model / fitted model)" }, range: [0, max], tickmode: "array", tickvals: [0, Math.log10(3), 1, 2, max], ticktext: ["0", "log₁₀3", "1", "2", max === 3 ? "≥3" : max.toFixed(1)], gridcolor: "#D8D0C2", zeroline: false },
    yaxis: { autorange: "reversed", automargin: true }, showlegend: false,
    annotations: [
      { x: Math.log10(3) / 2, y: 1.1, xref: "x", yref: "paper", text: "Anecdotal", showarrow: false, font: { size: 9 } },
      { x: (Math.log10(3) + 1) / 2, y: 1.1, xref: "x", yref: "paper", text: "Moderate", showarrow: false, font: { size: 9 } },
      { x: 1.5, y: 1.1, xref: "x", yref: "paper", text: "Strong", showarrow: false, font: { size: 9 } },
      { x: Math.min(max, 2.5), y: 1.1, xref: "x", yref: "paper", text: "Extreme", showarrow: false, font: { size: 9 } },
    ],
  };
  return <section className="resultPanel"><div className="resultPanelHeader"><div><span className="sectionLabel">Model evidence</span><h2>Bayes factors</h2></div>{conditions.length > 1 && <label><span>Condition</span><select value={condition} onChange={(event) => setCondition(event.target.value)}>{conditions.map((item) => <option key={item}>{item}</option>)}</select></label>}</div><p>Evidence is relative to the best model. The horizontal scale is the exact log₁₀ Bayes factor; thresholds are placed at log₁₀3, 1 and 2.</p><div className="resultPlot"><Plot data={data} layout={layout} config={plotConfig("barracuda-bayes-factors")} useResizeHandler style={{ width: "100%", height: "100%" }} /></div></section>;
}

function PosteriorPlot({ posterior, summary }: { posterior: Row[]; summary: Row[] }) {
  const parameters = Array.from(new Set(posterior.flatMap((row) => Object.keys(row).filter((key) => !META.has(key) && number(row[key]) !== null))));
  const summaryParameters = Array.from(new Set(summary.map((row) => text(row.parameter)).filter(Boolean)));
  const available = parameters.length ? parameters : summaryParameters;
  const [parameter, setParameter] = useState(available[0] ?? "");
  const conditions = Array.from(new Set(posterior.map((row) => text(row.condition, "All cells"))));
  const groups = Array.from(new Set(posterior.map((row) => `${text(row.condition, "All cells")}|||${text(row.model, text(row.model_key, "Model"))}`)));
  const data: Data[] = groups.map((group, index): Data | null => {
    const [condition, model] = group.split("|||");
    const values = posterior.filter((row) => text(row.condition, "All cells") === condition && text(row.model, text(row.model_key, "Model")) === model).map((row) => number(row[parameter])).filter((value): value is number => value !== null);
    if (!values.length) return null;
    return { type: "histogram" as const, histnorm: "probability density", x: values, name: conditions.length > 1 ? `${condition} · ${model}` : model, opacity: 0.48, marker: { color: COLOURS[index % COLOURS.length] }, hovertemplate: `${condition} · ${model}<br>${label(parameter)}: %{x:.3f}<br>Density: %{y:.3f}<extra></extra>` };
  }).filter((trace): trace is Data => trace !== null);
  const layout: Partial<Layout> = { autosize: true, height: 430, margin: { l: 62, r: 20, t: 36, b: 70 }, barmode: "overlay", paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "#F5F1E8", font: { family: "Avenir Next, Segoe UI, sans-serif", color: "#4E493F", size: 10 }, legend: { orientation: "h", y: 1.14 }, xaxis: { title: { text: label(parameter) }, gridcolor: "#D8D0C2" }, yaxis: { title: { text: "Posterior density" }, gridcolor: "#D8D0C2", rangemode: "tozero" } };
  return <section className="resultPanel"><div className="resultPanelHeader"><div><span className="sectionLabel">Parameter inference</span><h2>Posterior distributions</h2></div>{available.length > 0 && <label><span>Parameter</span><select value={parameter} onChange={(event) => setParameter(event.target.value)}>{available.map((item) => <option key={item} value={item}>{label(item)}</option>)}</select></label>}</div>{data.length ? <><p>Each curve uses posterior particles from the selected model and condition, rather than means of separate analyses.</p><div className="resultPlot"><Plot data={data} layout={layout} config={plotConfig("barracuda-posterior")} useResizeHandler style={{ width: "100%", height: "100%" }} /></div></> : <SummaryTable summary={summary} />}</section>;
}

function SummaryTable({ summary }: { summary: Row[] }) {
  if (!summary.length) return <div className="emptyState"><Icon name="chart" /><h3>Posterior output is not available</h3><p>The worker completed without a reportable posterior table.</p></div>;
  return <div className="resultTable"><table><thead><tr><th>Condition</th><th>Model</th><th>Parameter</th><th>Mean</th><th>HDI</th></tr></thead><tbody>{summary.slice(0, 80).map((row, index) => { const lowKey = Object.keys(row).find((key) => key.startsWith("hdi_") && key.endsWith("%")); const highKey = Object.keys(row).filter((key) => key.startsWith("hdi_") && key.endsWith("%")).at(-1); return <tr key={index}><td>{text(row.condition, "All cells")}</td><td>{text(row.model, text(row.model_key))}</td><td>{label(text(row.parameter))}</td><td>{number(row.mean)?.toFixed(3) ?? "–"}</td><td>{lowKey && highKey ? `${number(row[lowKey])?.toFixed(3) ?? "–"} to ${number(row[highKey])?.toFixed(3) ?? "–"}` : "–"}</td></tr>; })}</tbody></table></div>;
}

const plotConfig = (filename: string): Partial<Config> => ({ responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"], toImageButtonOptions: { format: "png", filename, scale: 2 } });

function ArtifactDownloads({ artifacts, client }: { artifacts: AnalysisArtifact[]; client: ApiClient }) {
  const [error, setError] = useState("");
  async function download(artifact: AnalysisArtifact) {
    setError("");
    try {
      const blob = await client.downloadArtifact(artifact.id);
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url; anchor.download = artifact.filename; anchor.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The result file could not be downloaded.");
    }
  }
  if (!artifacts.length) return null;
  return <section className="resultDownloads"><div><span className="sectionLabel">Exports</span><h2>Download inference results</h2><p>Archives include evidence tables, posterior summaries, paired draws and ArviZ data where available.</p></div><div>{artifacts.map((artifact) => <button className="button buttonQuiet" type="button" onClick={() => void download(artifact)} key={artifact.id}><Icon name="download" />{artifact.filename}</button>)}</div>{error && <p className="formError" role="alert">{error}</p>}</section>;
}

export function AnalysisResults({ analysis, client, onBack }: { analysis: AnalysisSummary; client: ApiClient; onBack: () => void }) {
  const [job, setJob] = useState(analysis.job);
  useEffect(() => {
    if (!job || (job.status !== "queued" && job.status !== "running")) return;
    const controller = new AbortController();
    const timer = window.setInterval(() => {
      client.getJob(job.id, controller.signal).then(setJob).catch((reason: unknown) => {
        if ((reason as { name?: string })?.name !== "AbortError") window.clearInterval(timer);
      });
    }, 1_500);
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [client, job]);
  const parts = useMemo(() => job ? resultParts(job) : { evidence: [], posterior: [], posteriorSummary: [], normalization: {} }, [job]);
  return <div className="analysisDetail"><button className="textButton resultBack" type="button" onClick={onBack}><Icon name="arrow" />Back to analyses</button><header className="analysisDetailHeader"><div><span className="sectionLabel">{analysis.kind === "trajectory" ? "Contact trajectory" : "Event counts"}</span><h1>{analysis.title}</h1><p>{analysis.cellCount.toLocaleString()} rows · {analysis.conditionCount || parts.normalization.conditions as number || 1} condition{analysis.conditionCount === 1 ? "" : "s"}</p></div>{job && <span className={`statusBadge status-${job.status}`}><i />{job.status === "ready" ? "Complete" : job.status}</span>}</header>{!job ? <div className="emptyState"><Icon name="clock" /><h3>No inference job</h3><p>Create a new analysis to run inference.</p></div> : job.status === "queued" || job.status === "running" ? <ProgressPanel job={job} /> : job.status === "failed" ? <div className="validationErrors" role="alert"><strong>Inference did not complete</strong><span>{job.errorMessage || "The worker could not complete this analysis."}</span><span>Job ID: {job.id}</span></div> : <><EvidencePlot evidence={parts.evidence} /><PosteriorPlot posterior={parts.posterior} summary={parts.posteriorSummary} /><ArtifactDownloads artifacts={job.artifacts} client={client} /></>}</div>;
}
