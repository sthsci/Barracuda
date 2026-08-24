"use client";

import { useEffect, useRef, useState } from "react";
import type { ApiClient, AuthIdentity } from "@/lib/api";
import { Icon } from "./icons";

export function AuthModal({ client, onAuthenticated, onClose }: { client: ApiClient; onAuthenticated: (identity: AuthIdentity) => void; onClose: () => void }) {
  const dialog = useRef<HTMLDivElement>(null);
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    const close = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    document.addEventListener("keydown", close); dialog.current?.focus();
    return () => document.removeEventListener("keydown", close);
  }, [onClose]);
  async function submit(event: React.FormEvent) {
    event.preventDefault(); setBusy(true); setError("");
    try {
      const identity = mode === "login" ? await client.login(username, password) : await client.register(username, password, email);
      onAuthenticated(identity); onClose();
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Sign-in failed. Please try again."); }
    finally { setBusy(false); }
  }
  return (
    <div className="modalBackdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <div className="modal authModal" role="dialog" aria-modal="true" aria-labelledby="auth-title" tabIndex={-1} ref={dialog}>
        <div className="modalHeader"><div><span className="sectionLabel">Optional account</span><h2 id="auth-title">{mode === "login" ? "Sign in to save work" : "Create an account"}</h2></div><button className="iconButton" onClick={onClose} type="button" aria-label="Close sign-in dialog"><Icon name="close" /></button></div>
        <p>Your guest workspace works without an account. A username and password lets you claim this guest work and save future work across devices.</p>
        <p className="authSsoNote"><span className="imperialMark">I</span> Imperial SSO is planned, but is not connected in this release.</p>
        <form onSubmit={submit}>
          <label className="fieldLabel" htmlFor="username">Username</label>
          <input id="username" autoComplete="username" value={username} onChange={(event) => setUsername(event.target.value)} required />
          {mode === "register" && <><label className="fieldLabel" htmlFor="email">Email (optional)</label><input id="email" type="email" autoComplete="email" value={email} onChange={(event) => setEmail(event.target.value)} /></>}
          <label className="fieldLabel" htmlFor="password">Password</label>
          <input id="password" type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} value={password} onChange={(event) => setPassword(event.target.value)} required />
          {mode === "register" && <small className="authFineprint">Use a strong passphrase. The API checks Django’s password policy.</small>}
          {error && <p className="formError" role="alert">{error}</p>}
          <button className="button buttonPrimary fullWidth" type="submit" disabled={busy}>{busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}</button>
        </form>
        <button className="continueGuest" type="button" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(""); }}>{mode === "login" ? "Need an account? Register" : "Already have an account? Sign in"}</button>
        <button className="continueGuest" type="button" onClick={onClose}>Continue without an account <Icon name="arrow" /></button>
      </div>
    </div>
  );
}
