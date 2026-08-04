from __future__ import annotations

import pandas as pd
import streamlit as st

from webapp.analysis_ui import (
    data_overview,
    inference_controls,
    model_selector,
    normalize_uploaded_frame,
    read_uploaded_csv,
    render_results,
)
from webapp.core.data import sample_donor_frame, validate_donor_frame
from webapp.core.inference import run_donor_models
from webapp.ui import hero, note


hero(
    "3 · Hierarchical extension",
    "Donor-aware event-count inference",
    "Allow event-rate, continuous heterogeneity and non-engaging fractions to vary between donors while estimating population-level quantities.",
    badge="Prototype hierarchy · 2–12 donors",
)

note(
    "Treat donor codes carefully",
    "A label such as donor_01 can still be pseudonymised personal data when a separate key can reconnect it to an individual. Use synthetic or approved anonymised data in this demo.",
    tone="amber",
)

st.header("A. Provide donor-labelled counts")
source = st.radio(
    "Input method",
    ["Example", "Upload CSV", "Edit spreadsheet"],
    horizontal=True,
    key="donor_source",
)

candidate: pd.DataFrame | None = None
if source == "Example":
    candidate = sample_donor_frame()
    st.caption("A fully synthetic example with three donor labels.")
elif source == "Upload CSV":
    uploaded = st.file_uploader("Upload donor-aware CSV", type=["csv"], help="Maximum size: 1 MB.")
    if uploaded is not None:
        try:
            raw = read_uploaded_csv(uploaded)
            candidate = normalize_uploaded_frame(raw, prefix="donor_upload", donor_aware=True)
        except ValueError as exc:
            st.error(str(exc))
else:
    starter = sample_donor_frame().groupby("donor_id", group_keys=False).head(4)
    candidate = st.data_editor(
        starter,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "cell_id": st.column_config.TextColumn("Cell ID", required=True),
            "donor_id": st.column_config.TextColumn("Donor ID", required=True),
            "count": st.column_config.NumberColumn("Event count", min_value=0, step=1, required=True),
        },
        key="donor_editor",
    )

observation_time = st.number_input(
    "Common observation time",
    min_value=0.01,
    max_value=100.0,
    value=1.0,
    step=0.25,
    key="donor_observation_time",
    help="All cells must currently share one observation duration.",
)

valid_frame: pd.DataFrame | None = None
if candidate is not None:
    try:
        valid_frame = validate_donor_frame(candidate.dropna(how="all"))
    except ValueError as exc:
        st.error(f"Please correct the input: {exc}")
    else:
        st.success("The donor-aware dataset passed the demo validation checks.")
        data_overview(valid_frame, donor_aware=True)

if valid_frame is None:
    st.stop()

st.header("B. Configure the hierarchical fit")
selected_models = model_selector("donor", default=["dis2p"])
settings = inference_controls("donor", donor_aware=True)

if st.button(
    "Fit selected donor-aware models",
    type="primary",
    width="stretch",
    disabled=settings is None or not selected_models,
):
    st.session_state.pop("donor_results", None)
    progress = st.progress(0.0, text="Preparing donor-aware inference")

    def update_progress(index: int, total: int, label: str) -> None:
        progress.progress((index - 1) / total, text=f"Fitting {label} ({index}/{total})")

    try:
        results = run_donor_models(
            valid_frame,
            float(observation_time),
            settings=settings,
            model_keys=selected_models,
            progress_callback=update_progress,
        )
    except Exception as exc:
        progress.empty()
        st.error(f"Donor-aware inference did not complete: {exc}")
    else:
        progress.progress(1.0, text="Inference complete")
        st.session_state["donor_results"] = results
        st.session_state["donor_result_data"] = valid_frame.copy()
        st.session_state["donor_result_time"] = float(observation_time)
        st.session_state["donor_result_settings"] = settings

if "donor_results" in st.session_state:
    result_data = st.session_state["donor_result_data"]
    result_time = st.session_state["donor_result_time"]
    result_settings = st.session_state["donor_result_settings"]
    result_models = list(st.session_state["donor_results"])
    if (
        not valid_frame.equals(result_data)
        or float(observation_time) != result_time
        or settings != result_settings
        or selected_models != result_models
    ):
        st.warning(
            "The input, model selection or inference settings have changed since "
            "the last fit. Run inference again to view matching donor-aware results."
        )
    else:
        render_results(
            st.session_state["donor_results"],
            data=result_data,
            observation_time=result_time,
            settings=result_settings,
            download_name="orca_donor_aware_results.zip",
        )
