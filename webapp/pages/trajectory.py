from __future__ import annotations

import pandas as pd
import streamlit as st

from webapp.ui import hero, note, step_card


hero(
    "4 · Roadmap",
    "Trajectory inference is in development",
    "The next release will use the ordered sequence of successful and unsuccessful contacts to separate stable cellular heterogeneity from interaction-history effects.",
    badge="Planned for a future release",
)

note(
    "Why it is not enabled yet",
    "The trajectory likelihood, priors, validation thresholds and compute budget are still being finalised. The event-count tools remain available in this release.",
    tone="amber",
)

st.header("What the trajectory model retains")
st.markdown(
    "Event counts remember *how many* kills occurred. A trajectory also remembers *when* they occurred and what happened before each decision."
)
st.latex(r"x_{ij}\sim\mathrm{Bernoulli}(p_{ij})")
st.latex(r"\mathrm{logit}(p_{ij})=\eta_i+\beta_f f_{ij}+\beta_s s_{ij}")
st.markdown(
    "Here, **ηᵢ** is cell *i*'s baseline killing propensity; **fᵢⱼ** and **sᵢⱼ** count previous failed and successful contacts. The coefficients **βf** and **βs** describe history effects."
)

columns = st.columns(3)
with columns[0]:
    step_card("01", "Ordered input", "One row per contact, with cell ID, contact order and binary outcome.")
with columns[1]:
    step_card("02", "Competing mechanisms", "Compare homogeneous and heterogeneous populations with or without history dependence.")
with columns[2]:
    step_card("03", "Decision maps", "Summarise how prior successes and failures shift future killing probability.")

st.subheader("Planned input format")
example = pd.DataFrame(
    {
        "cell_id": ["cell_001", "cell_001", "cell_001", "cell_002"],
        "contact_index": [1, 2, 3, 1],
        "outcome": [0, 1, 1, 0],
        "donor_id": ["donor_A", "donor_A", "donor_A", "donor_B"],
    }
)
st.dataframe(example, hide_index=True, width="stretch")

st.button(
    "Trajectory inference is not yet available",
    disabled=True,
    width="stretch",
    help="This control will be enabled after the trajectory validation work is complete.",
)
st.caption("In the meantime, use event-count inference for the number of contacts or kills per cell.")
