"use client";

import { useRef, useState } from "react";
import type { AnalysisKind, AnalysisSummary, ApiClient, CsvValidationResult } from "@/lib/api";
import { Icon } from "./icons";

const examples: Record<AnalysisKind, { name: string; content: string }> = {
  "event-counts": {
    name: "barracuda_event_counts_example.csv",
    content: "cell_id,condition,count\ncell_001,Control,0\ncell_002,Control,3\ncell_003,Treatment,1\ncell_004,Treatment,4\n",
  },
  trajectory: {
    name: "barracuda_trajectory_example.csv",
    content: 'cell_id,condition,history\ncell_001,Control,"0,0,1"\ncell_002,Control,"1,1"\ncell_003,Treatment,"0,1,0"\ncell_004,Treatment,"1"\n',
  },
};

export function UploadWizard({
  client,
  initialKind,
  onCreated,
  onCancel,
}: {
  client: ApiClient;
  initialKind?: AnalysisKind;
  onCreated: (analysis: AnalysisSummary) => void;
  onCancel: () => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [kind, setKind] = useState<AnalysisKind | null>(initialKind ?? null);
  const [validation, setValidation] = useState<CsvValidationResult | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");

  async function validate(name: string, content: string, file?: File) {
    if (!kind) return;
    setBusy(true);
    setError("");
    try {
      const result = await client.validateCsv({ kind, filename: name, content });
      setValidation(result);
      setSelectedFile(file ?? new File([content], name, { type: "text/csv" }));
      setTitle(name.replace(/\.csv$/i, "").replace(/[_-]+/g, " "));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The CSV could not be checked.");
    } finally {
      setBusy(false);
    }
  }

  async function acceptFiles(files: FileList | null) {
    const file = files?.[0];
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) return setError("Choose a .csv file.");
    if (file.size > 1_000_000) return setError("Choose a CSV smaller than 1 MB for the web preview.");
    await validate(file.name, await file.text(), file);
  }

  async function create() {
    if (!kind || !validation?.valid) return;
    setBusy(true);
    setError("");
    try {
      onCreated(await client.createAnalysis({ title, kind, upload: validation, file: selectedFile ?? undefined }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The analysis could not be created.");
    } finally {
      setBusy(false);
    }
  }

  const step = validation?.valid ? 3 : kind ? 2 : 1;
  return (
    <section className="uploadWizard" aria-labelledby="new-analysis-title">
      <div className="wizardHeader">
        <div><span className="sectionLabel">New analysis</span><h2 id="new-analysis-title">Start with a CSV</h2><p>Your file is checked before any analysis is created.</p></div>
        <button className="iconButton" type="button" onClick={onCancel} aria-label="Close new analysis"><Icon name="close" /></button>
      </div>
      <ol className="stepper" aria-label={`Step ${step} of 3`}>
        {([
          [1, "Analysis"], [2, "Data"], [3, "Review"],
        ] as const).map(([number, label]) => <li key={number} className={step >= number ? "active" : ""}><span>{step > number ? <Icon name="check" /> : number}</span>{label}</li>)}
      </ol>

      {!kind ? (
        <div className="kindChooser">
          <button type="button" onClick={() => setKind("event-counts")}><Icon name="count" /><span className="sectionLabel">Event counts</span><strong>Counts per immune cell</strong><p>Compare stochastic, inactive and continuously heterogeneous populations.</p><small>Required: cell_id, count</small></button>
          <button type="button" onClick={() => setKind("trajectory")}><Icon name="trajectory" /><span className="sectionLabel">Contact trajectory</span><strong>Ordered binary histories</strong><p>Separate stable killing propensity from previous-contact effects.</p><small>Required: cell_id, history</small></button>
        </div>
      ) : validation?.valid ? (
        <div className="reviewUpload">
          <div className="uploadSuccess"><span><Icon name="check" /></span><div><strong>{validation.filename}</strong><p>{validation.rowCount} cells · {validation.conditionCount} condition{validation.conditionCount === 1 ? "" : "s"}{validation.donorAware ? " · donor aware" : ""}</p></div><button className="textButton" type="button" onClick={() => { setValidation(null); setSelectedFile(null); }}>Replace</button></div>
          <div className="reviewGrid">
            <label><span className="fieldLabel">Analysis name</span><input value={title} onChange={(event) => setTitle(event.target.value)} /></label>
            <div><span className="fieldLabel">Detected conditions</span><div className="conditionPills">{validation.conditions.map((condition) => <span key={condition}>{condition}</span>)}</div></div>
          </div>
          <div className="previewTableWrap"><table><thead><tr>{validation.columns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{validation.preview.slice(0, 3).map((row, index) => <tr key={index}>{validation.columns.map((column) => <td key={column}>{row[column] || <em>blank</em>}</td>)}</tr>)}</tbody></table></div>
          {validation.warnings.map((warning) => <p className="formWarning" key={warning}>{warning}</p>)}
          <div className="wizardActions"><button className="button buttonQuiet" type="button" onClick={() => { setValidation(null); setSelectedFile(null); }}>Back</button><button className="button buttonPrimary" type="button" onClick={create} disabled={busy}>{busy ? "Creating…" : "Start analysis"}<Icon name="arrow" /></button></div>
        </div>
      ) : (
        <div className="uploadStep">
          <div className="selectedKind"><Icon name={kind === "trajectory" ? "trajectory" : "count"} /><div><span className="sectionLabel">Selected analysis</span><strong>{kind === "trajectory" ? "Contact trajectory" : "Event counts"}</strong></div><button className="textButton" type="button" onClick={() => { setKind(null); setValidation(null); }}>Change</button></div>
          <div
            className={`dropzone ${dragging ? "isDragging" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => { event.preventDefault(); setDragging(false); void acceptFiles(event.dataTransfer.files); }}
          >
            <span className="uploadMark"><Icon name="upload" /></span>
            <h3>{busy ? "Checking your CSV…" : "Drop a CSV here"}</h3>
            <p>or choose a file from your computer</p>
            <button className="button buttonQuiet" type="button" onClick={() => input.current?.click()} disabled={busy}>Browse files</button>
            <input ref={input} type="file" accept=".csv,text/csv" hidden onChange={(event) => void acceptFiles(event.target.files)} />
            <small>CSV only · up to 1 MB · do not upload names or clinical identifiers</small>
          </div>
          <button className="exampleLink" type="button" onClick={() => void validate(examples[kind].name, examples[kind].content)} disabled={busy}><Icon name="spark" />Use a safe example CSV instead</button>
          {validation && !validation.valid && <div className="validationErrors" role="alert"><strong>Check the CSV and try again</strong>{validation.errors.map((message) => <span key={message}>{message}</span>)}</div>}
          {error && <p className="formError" role="alert">{error}</p>}
        </div>
      )}
    </section>
  );
}
