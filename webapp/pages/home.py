from __future__ import annotations

import streamlit as st

from webapp.ui import hero, note, step_card


hero(
    "Project home",
    "Orca",
    "Bayesian inference for heterogeneity in immune-cell decision-making",
    badge="Single-cell interaction histories · research preview",
)

st.markdown(
    """
    <div class="orca-question">
      Why do immune cells that share an experimental condition make such different
      decisions - stochastic chance, stable cell-to-cell differences, donor variation,
      accumulated interaction history, or a combination of these mechanisms?
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "Orca is a quantitative framework for connecting time-lapse observations of "
    "natural killer (NK) cells to the hidden mechanisms that shape target contact "
    "and cytotoxic decisions. It combines synthetic validation, Bayesian parameter "
    "inference, and model comparison with analyses of experimental single-cell histories."
)

st.header("One question, three levels of information")
framework_columns = st.columns(3)
with framework_columns[0]:
    step_card(
        "01",
        "Event counts",
        "Use contacts or kills per cell to distinguish stochastic variation, continuous heterogeneity, and structural inactivity.",
    )
with framework_columns[1]:
    step_card(
        "02",
        "Donor hierarchy",
        "Estimate donor-specific behaviour and separate variation within donors from systematic differences between donors.",
    )
with framework_columns[2]:
    step_card(
        "03",
        "Ordered trajectories",
        "Retain lethal and non-lethal contact order to separate stable killing propensity from interaction-history effects.",
    )

note(
    "Current web release",
    "Event-count and donor-aware inference are available now. The trajectory code exists in the research repository, while its public web interface remains in development.",
    tone="amber",
)

st.header("What you can find in each section")

first_row = st.columns(3)
with first_row[0]:
    st.markdown('<span class="orca-route-label">Section 0</span>', unsafe_allow_html=True)
    step_card(
        "0",
        "Bayesian inference 101",
        "Bayes' theorem, priors, likelihoods, posteriors, MCMC, SMC, marginal likelihoods, and Bayes factors.",
    )
    st.page_link("webapp/pages/bayes_101.py", label="Open the primer", icon=":material/arrow_forward:")
with first_row[1]:
    st.markdown('<span class="orca-route-label">Section 1</span>', unsafe_allow_html=True)
    step_card(
        "1",
        "Synthetic validation",
        "Choose a ground truth, generate event counts, run inference, and compare posterior recovery and model evidence.",
    )
    st.page_link("webapp/pages/synthetic_validation.py", label="Open synthetic validation", icon=":material/arrow_forward:")
with first_row[2]:
    st.markdown('<span class="orca-route-label">Section 2</span>', unsafe_allow_html=True)
    step_card(
        "2",
        "Event-count inference",
        "Analyse an example, upload a small CSV, or enter cell-level counts directly in an editable table.",
    )
    st.page_link("webapp/pages/event_counts.py", label="Open event-count inference", icon=":material/arrow_forward:")

second_row = st.columns(2)
with second_row[0]:
    st.markdown('<span class="orca-route-label">Section 3</span>', unsafe_allow_html=True)
    step_card(
        "3",
        "Donor-aware inference",
        "Fit a hierarchical extension that estimates population and donor-specific event-rate distributions.",
    )
    st.page_link("webapp/pages/donor_aware.py", label="Open donor-aware inference", icon=":material/arrow_forward:")
with second_row[1]:
    st.markdown('<span class="orca-route-label">Section 4</span>', unsafe_allow_html=True)
    step_card(
        "4",
        "Trajectory roadmap",
        "See the planned ordered contact-kill input, mechanistic parameters, and outputs for the next web release.",
    )
    st.page_link("webapp/pages/trajectory.py", label="Open trajectory roadmap", icon=":material/arrow_forward:")

st.header("Research context")
st.markdown(
    "The accompanying manuscript applies the framework to untreated, rituximab-treated, "
    "and bispecific-antibody-treated NK cells. The web interface is designed for learning "
    "and exploratory analysis; manuscript-scale conclusions require larger SMC runs, "
    "diagnostic checks, and an appropriate experimental design."
)

link_columns = st.columns([0.35, 0.65])
with link_columns[0]:
    st.link_button(
        "View the research repository",
        "https://github.com/sthsci/Orca",
        icon=":material/code:",
        width="stretch",
    )
with link_columns[1]:
    st.caption(
        "Developed at Imperial College London with research collaborators. "
        "Use synthetic or approved anonymous data in this preview."
    )
