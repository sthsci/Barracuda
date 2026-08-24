"use client";

import Link from "next/link";
import { useState } from "react";
import { Brand } from "./Brand";
import { Icon } from "./icons";

export function AppHeader() {
  const [open, setOpen] = useState(false);
  return (
    <header className="siteHeader">
      <div className="headerInner">
        <Brand />
        <button
          className="iconButton mobileMenu"
          type="button"
          aria-label="Toggle navigation"
          aria-expanded={open}
          onClick={() => setOpen((value) => !value)}
        >
          <Icon name={open ? "close" : "menu"} />
        </button>
        <nav className={`siteNav ${open ? "isOpen" : ""}`} aria-label="Primary navigation">
          <Link href="/#analyses" onClick={() => setOpen(false)}>Analyses</Link>
          <Link href="/#research" onClick={() => setOpen(false)}>Research</Link>
          <span className="navRule" aria-hidden="true" />
          <Link className="textButton" href="/workspace?signin=1">
            <Icon name="user" />
            Sign in
          </Link>
          <Link className="button buttonPrimary buttonSmall" href="/workspace">
            Open workspace
            <Icon name="arrow" />
          </Link>
        </nav>
      </div>
    </header>
  );
}
