"use client";

import { useEffect, useRef, useState } from "react";
import type { AnalysisSummary, ApiClient, ShareLink } from "@/lib/api";
import { Icon } from "./icons";

export function ShareModal({
  analysis,
  client,
  onClose,
}: {
  analysis: AnalysisSummary;
  client: ApiClient;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDivElement>(null);
  const [share, setShare] = useState<ShareLink | null>(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    document.addEventListener("keydown", closeOnEscape);
    dialog.current?.focus();
    return () => document.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  async function createLink() {
    setLoading(true);
    setError("");
    try {
      setShare(await client.createShareLink({ analysisId: analysis.id, access: "viewer", expiresInDays: 1 }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "The share link could not be created.");
    } finally {
      setLoading(false);
    }
  }

  async function copyLink() {
    if (!share) return;
    await navigator.clipboard?.writeText(share.url);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  }

  return (
    <div className="modalBackdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby="share-title" tabIndex={-1} ref={dialog}>
        <div className="modalHeader">
          <div><span className="sectionLabel">Collaboration</span><h2 id="share-title">Share analysis</h2></div>
          <button className="iconButton" type="button" onClick={onClose} aria-label="Close share dialog"><Icon name="close" /></button>
        </div>
        <div className="shareAnalysisName"><Icon name={analysis.kind === "trajectory" ? "trajectory" : "count"} /><div><strong>{analysis.title}</strong><span>{analysis.conditionCount} conditions · {analysis.cellCount} cells</span></div></div>
        {!share ? (
          <>
            <div className="guestSharePolicy">
              <div><Icon name="check" /><span><strong>Read-only results</strong><small>Recipients can inspect this result but cannot edit or copy it.</small></span></div>
              <div><Icon name="clock" /><span><strong>Expires after 24 hours</strong><small>Guest links are deliberately short-lived and cannot be extended.</small></span></div>
            </div>
            <div className="modalNote"><Icon name="lock" /><span><strong>Guest sharing is temporary and viewer-only.</strong> Account-managed sharing will be available after authentication is connected. Continuing as a guest always remains available.</span></div>
            {error && <p className="formError" role="alert">{error}</p>}
            <div className="modalActions"><button className="button buttonQuiet" type="button" onClick={onClose}>Cancel</button><button className="button buttonPrimary" type="button" onClick={createLink} disabled={loading}>{loading ? "Creating…" : "Create link"}<Icon name="share" /></button></div>
          </>
        ) : (
          <div className="shareSuccess">
            <span className="successMark"><Icon name="check" /></span>
            <h3>Link ready to share</h3>
            <p>Anyone with this link can view the shared result for the next 24 hours.</p>
            <div className="copyField"><input aria-label="Share link" readOnly value={share.url} /><button type="button" onClick={copyLink}><Icon name={copied ? "check" : "copy"} />{copied ? "Copied" : "Copy"}</button></div>
            <button className="button buttonPrimary fullWidth" type="button" onClick={onClose}>Done</button>
          </div>
        )}
      </div>
    </div>
  );
}
