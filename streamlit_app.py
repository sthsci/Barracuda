from __future__ import annotations

import streamlit as st

from webapp.ui import load_styles


st.set_page_config(
    page_title="ORCA Bayesian Lab",
    page_icon=":material/experiment:",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_styles()

pages = {
    "Learn": [
        st.Page(
            "webapp/pages/bayes_101.py",
            title="0 · Bayesian inference 101",
            icon=":material/school:",
            default=True,
        ),
    ],
    "Experiment": [
        st.Page(
            "webapp/pages/synthetic_validation.py",
            title="1 · Synthetic validation",
            icon=":material/science:",
        ),
        st.Page(
            "webapp/pages/event_counts.py",
            title="2 · Event-count inference",
            icon=":material/bar_chart:",
        ),
        st.Page(
            "webapp/pages/donor_aware.py",
            title="3 · Donor-aware inference",
            icon=":material/groups:",
        ),
    ],
    "Roadmap": [
        st.Page(
            "webapp/pages/trajectory.py",
            title="4 · Trajectory model",
            icon=":material/timeline:",
        ),
    ],
}

with st.sidebar:
    st.markdown("### ORCA")
    st.caption("Bayesian event-count laboratory")
    st.markdown("---")
    st.caption(
        "Research preview · Inputs are processed in the running session and are not intentionally retained."
    )
    st.warning(
        "Local evaluation only. Use synthetic or approved anonymous data until "
        "Imperial privacy, accessibility and security reviews are complete."
    )

navigation = st.navigation(pages, position="sidebar", expanded=True)
navigation.run()
