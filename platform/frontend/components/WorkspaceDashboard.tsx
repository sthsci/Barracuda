"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { apiClient, type AnalysisKind, type AnalysisSummary, type ApiClient } from "@/lib/api";
import { AnalysisCard } from "./AnalysisCard";
import { AnalysisResults } from "./AnalysisResults";
import { AuthModal } from "./AuthModal";
import { Brand } from "./Brand";
import { Icon } from "./icons";
import { PosteriorPreview } from "./PosteriorPreview";
import { ShareModal } from "./ShareModal";
import { UploadWizard } from "./UploadWizard";

type View = "overview" | "new" | "saved";

export function WorkspaceDashboard({ client = apiClient }: { client?: ApiClient }) {
  const [analyses, setAnalyses] = useState<AnalysisSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [view, setView] = useState<View>("overview");
  const [newKind, setNewKind] = useState<AnalysisKind | undefined>();
  const [shareAnalysis, setShareAnalysis] = useState<AnalysisSummary | null>(null);
  const [authOpen, setAuthOpen] = useState(false);
  const [, setIdentity] = useState(() => client.getIdentity());
  const [activeAnalysis, setActiveAnalysis] = useState<AnalysisSummary | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [kindFilter, setKindFilter] = useState<"all" | AnalysisKind>("all");

  useEffect(() => {
    const controller = new AbortController();
    client
      .listAnalyses(controller.signal)
      .then((response) => setAnalyses(response.analyses))
      .catch((reason: unknown) => {
        if ((reason as { name?: string })?.name !== "AbortError") {
          setLoadError("Example analyses are temporarily unavailable. You can still start a new analysis.");
        }
      })
      .finally(() => setLoading(false));
    const authTimer = window.setTimeout(() => {
      if (new URLSearchParams(window.location.search).get("signin") === "1") setAuthOpen(true);
    }, 0);
    return () => {
      controller.abort();
      window.clearTimeout(authTimer);
    };
  }, [client]);

  const filtered = useMemo(() => {
    const token = query.trim().toLowerCase();
    return analyses.filter((analysis) =>
      (kindFilter === "all" || analysis.kind === kindFilter) &&
      (!token || `${analysis.title} ${analysis.conditions.join(" ")}`.toLowerCase().includes(token)),
    );
  }, [analyses, kindFilter, query]);

  const ready = analyses.filter((analysis) => analysis.status === "ready").length;
  const running = analyses.filter((analysis) => analysis.status === "running" || analysis.status === "queued").length;
  const cells = analyses.reduce((total, analysis) => total + analysis.cellCount, 0);

  function startNew(kind?: AnalysisKind) {
    setNewKind(kind);
    setView("new");
    setSidebarOpen(false);
  }

  function handleCreated(analysis: AnalysisSummary) {
    setAnalyses((current) => [analysis, ...current]);
    setActiveAnalysis(analysis);
    setView("saved");
    setNewKind(undefined);
  }

  return (
    <div className="workspaceShell">
      <aside className={`workspaceSidebar ${sidebarOpen ? "isOpen" : ""}`}>
        <div className="sidebarTop"><Brand /><button type="button" className="iconButton mobileClose" aria-label="Close menu" onClick={() => setSidebarOpen(false)}><Icon name="close" /></button></div>
        <nav className="workspaceNav" aria-label="Workspace">
          <span className="navSection">Workspace</span>
          <button className={view === "overview" ? "active" : ""} onClick={() => { setView("overview"); setSidebarOpen(false); }} type="button"><Icon name="grid" />Overview</button>
          <button className={view === "new" ? "active" : ""} onClick={() => startNew()} type="button"><Icon name="plus" />New analysis</button>
          <button className={view === "saved" ? "active" : ""} onClick={() => { setView("saved"); setSidebarOpen(false); }} type="button"><Icon name="file" />Analyses <span>{analyses.length}</span></button>
          <span className="navSection">Learn</span>
          <Link href="/#analyses"><Icon name="help" />Analysis guide</Link>
          <Link href="/#research"><Icon name="file" />Research context</Link>
        </nav>
        <div className="guestAccountCard" id="account">
          <span className="guestAvatar"><Icon name="user" /></span>
          <div><strong>Guest workspace</strong><small>Temporary on this device</small></div>
          <button type="button" onClick={() => setAuthOpen(true)}>Sign in</button>
        </div>
        <div className="workspaceSafety"><Icon name="lock" /><span>Use synthetic or approved anonymous data only.</span></div>
      </aside>

      <main className="workspaceMain">
        <header className="workspaceHeader">
          <button className="iconButton workspaceMenu" type="button" onClick={() => setSidebarOpen(true)} aria-label="Open menu"><Icon name="menu" /></button>
          <div className="mobileBrand"><Brand compact /></div>
          <div className="workspaceHeaderActions">
            <button className="helpButton" type="button"><Icon name="help" />Help</button>
            <button className="accountButton" type="button" onClick={() => setAuthOpen(true)}><span>G</span><div><strong>Guest</strong><small>Optional sign in</small></div><Icon name="chevron" /></button>
          </div>
        </header>

        <div className="workspaceContent">
          {activeAnalysis ? (
            <AnalysisResults analysis={activeAnalysis} client={client} onBack={() => setActiveAnalysis(null)} />
          ) : view === "new" ? (
            <UploadWizard client={client} initialKind={newKind} onCreated={handleCreated} onCancel={() => setView("overview")} />
          ) : (
            <>
              <div className="workspaceTitleRow">
                <div><span className="sectionLabel">Guest workspace</span><h1>{view === "saved" ? "Your analyses" : "Analysis overview"}</h1><p>{view === "saved" ? "Search, reopen and share your exploratory work." : "Start a new analysis or return to a recent result."}</p></div>
                <button className="button buttonPrimary" type="button" onClick={() => startNew()}><Icon name="plus" />New analysis</button>
              </div>

              <div className="guestBanner">
                <Icon name="lock" />
                <div><strong>Continue without an account.</strong><span>Your guest workspace is ready now. Sign in only to save across devices and manage shared links.</span></div>
                <button type="button" onClick={() => setAuthOpen(true)}>Optional sign in <Icon name="arrow" /></button>
              </div>

              {view === "overview" && (
                <>
                  <section className="metricGrid" aria-label="Workspace summary">
                    <div><span className="metricIcon green"><Icon name="file" /></span><p>All analyses</p><strong>{analyses.length}</strong><small>{ready} completed</small></div>
                    <div><span className="metricIcon ochre"><Icon name="spark" /></span><p>In progress</p><strong>{running}</strong><small>Live PyMC status</small></div>
                    <div><span className="metricIcon rust"><Icon name="count" /></span><p>Cells explored</p><strong>{cells.toLocaleString()}</strong><small>Across guest analyses</small></div>
                    <div><span className="metricIcon navy"><Icon name="share" /></span><p>Shared links</p><strong>0</strong><small>Create from a result</small></div>
                  </section>

                  <section className="quickStart">
                    <div className="dashboardSectionTitle"><div><span className="sectionLabel">Quick start</span><h2>What would you like to analyse?</h2></div></div>
                    <div className="quickGrid">
                      <button type="button" onClick={() => startNew("event-counts")}><span className="quickIcon"><Icon name="count" /></span><div><span className="sectionLabel">Event counts</span><h3>Variation across cells</h3><p>Upload per-cell counts for one to four conditions.</p><small>Start analysis <Icon name="arrow" /></small></div></button>
                      <button type="button" onClick={() => startNew("trajectory")}><span className="quickIcon rust"><Icon name="trajectory" /></span><div><span className="sectionLabel">Trajectories</span><h3>Interaction history</h3><p>Upload ordered lethal and non-lethal contacts.</p><small>Start analysis <Icon name="arrow" /></small></div></button>
                      <button type="button" className="exampleQuick"><span className="quickIcon paper"><Icon name="spark" /></span><div><span className="sectionLabel">Not ready to upload?</span><h3>Explore an example</h3><p>Open a complete analysis with safe synthetic data.</p><small>View examples <Icon name="arrow" /></small></div></button>
                    </div>
                  </section>
                </>
              )}

              <section className="savedSection">
                <div className="dashboardSectionTitle">
                  <div><span className="sectionLabel">{view === "saved" ? "Library" : "Recent work"}</span><h2>{view === "saved" ? "Saved in this guest session" : "Recent analyses"}</h2></div>
                  {view === "overview" ? <button className="textButton" type="button" onClick={() => setView("saved")}>View all <Icon name="arrow" /></button> : <div className="analysisFilters"><label><Icon name="search" /><input aria-label="Search analyses" placeholder="Search analyses" value={query} onChange={(event) => setQuery(event.target.value)} /></label><select aria-label="Filter by analysis type" value={kindFilter} onChange={(event) => setKindFilter(event.target.value as "all" | AnalysisKind)}><option value="all">All types</option><option value="event-counts">Event counts</option><option value="trajectory">Trajectories</option></select></div>}
                </div>
                {loadError && <p className="formWarning" role="status">{loadError}</p>}
                {loading ? (
                  <div className="cardSkeletons" aria-label="Loading analyses"><i /><i /><i /></div>
                ) : filtered.length ? (
                  <div className="analysisGrid">{(view === "overview" ? filtered.slice(0, 3) : filtered).map((analysis) => <AnalysisCard analysis={analysis} onShare={setShareAnalysis} onOpen={setActiveAnalysis} key={analysis.id} />)}</div>
                ) : (
                  <div className="emptyState"><Icon name="search" /><h3>No analyses match</h3><p>Clear the search or begin a new analysis.</p></div>
                )}
              </section>

              {view === "overview" && (
                <section className="insightGrid">
                  <article className="insightPlot"><div className="dashboardSectionTitle"><div><span className="sectionLabel">Example result</span><h2>Posterior distributions</h2></div><span className="statusBadge status-ready"><i />Interactive</span></div><PosteriorPreview compact /></article>
                  <article className="activityCard"><span className="sectionLabel">Workspace activity</span><h2>Latest updates</h2><ul><li><span className="activityIcon green"><Icon name="check" /></span><div><strong>Ordered contact histories completed</strong><p>4 models · 3 conditions</p><small>Today</small></div></li><li><span className="activityIcon ochre"><Icon name="spark" /></span><div><strong>Anonymous donor panel is sampling</strong><p>SMC stage 4 · β = 0.81</p><small>8 min ago</small></div></li><li><span className="activityIcon paper"><Icon name="file" /></span><div><strong>Control event-count CSV checked</strong><p>486 cells recognised</p><small>Yesterday</small></div></li></ul></article>
                </section>
              )}
            </>
          )}
        </div>
      </main>

      {shareAnalysis && <ShareModal analysis={shareAnalysis} client={client} onClose={() => setShareAnalysis(null)} />}
      {authOpen && <AuthModal client={client} onAuthenticated={setIdentity} onClose={() => setAuthOpen(false)} />}
    </div>
  );
}
