from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st


_ASSET_ROOT = Path(__file__).resolve().parent / "assets"


def load_styles() -> None:
    css_path = _ASSET_ROOT / "styles.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def hero(kicker: str, title: str, lead: str, badge: str | None = None) -> None:
    badge_html = f'<span class="orca-badge">{escape(badge)}</span>' if badge else ""
    st.markdown(
        f"""
        <section class="orca-hero">
          <div class="orca-kicker">{escape(kicker)}</div>
          <h1>{escape(title)}</h1>
          <p>{escape(lead)}</p>
          {badge_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def note(title: str, body: str, tone: str = "teal") -> None:
    safe_tone = tone if tone in {"teal", "amber", "navy"} else "teal"
    st.markdown(
        f"""
        <div class="orca-note orca-note-{safe_tone}">
          <strong>{escape(title)}</strong>
          <span>{escape(body)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def step_card(number: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="orca-step">
          <span>{escape(number)}</span>
          <h3>{escape(title)}</h3>
          <p>{escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def research_warning() -> None:
    note(
        "Exploratory research software",
        "Small-particle preview runs are useful for learning and interface testing, not for publication-grade conclusions.",
        tone="amber",
    )
