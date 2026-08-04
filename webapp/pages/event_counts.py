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
from webapp.core.data import sample_count_frame, validate_count_frame
from webapp.core.inference import run_count_models
from webapp.ui import hero, note


hero(
    "2 · Donor-ignorant analysis",
    "Event-count inference",
    "Analyse one outcome and one condition at a time. Upload a CSV, edit a spreadsheet in the browser, or begin with a synthetic example.",
    badge="Derived integer counts only · maximum 1,000 cells",
)

note(
    "Input scope",
    "Use one row per cell and one count outcome—such as contacts or kills. Do not upload names, clinical metadata, raw microscopy, or other identifiers.",
    tone="navy",
)

st.header("A. Provide a small dataset")
source = st.radio("Input method", ["Example", "Upload CSV", "Edit spreadsheet"], horizontal=True)

candidate: pd.DataFrame | None = None
if source == "Example":
    candidate = sample_count_frame()
    st.caption("A synthetic example included with the demo.")
elif source == "Upload CSV":
    uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Maximum size: 1 MB.")
    if uploaded is not None:
        try:
            raw = read_uploaded_csv(uploaded)
            candidate = normalize_uploaded_frame(raw, prefix="counts_upload", donor_aware=False)
        except ValueError as exc:
            st.error(str(exc))
else:
    starter = sample_count_frame().head(12)
    candidate = st.data_editor(
        starter,
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config={
            "cell_id": st.column_config.TextColumn("Cell ID", required=True),
            "count": st.column_config.NumberColumn("Event count", min_value=0, step=1, required=True),
        },
        key="count_editor",
    )

observation_time = st.number_input(
    "Common observation time",
    min_value=0.01,
    max_value=100.0,
    value=1.0,
    step=0.25,
    help="All cells must currently share one observation duration.",
)

valid_frame: pd.DataFrame | None = None
if candidate is not None:
    try:
        valid_frame = validate_count_frame(candidate.dropna(how="all"))
    except ValueError as exc:
        st.error(f"Please correct the input: {exc}")
    else:
        st.success("The dataset passed the demo validation checks.")
        data_overview(valid_frame)

if valid_frame is None:
    st.stop()

st.header("B. Configure and run inference")
selected_models = model_selector("counts")
settings = inference_controls("counts")

if st.button(
    "Fit selected event-count models",
    type="primary",
    width="stretch",
    disabled=settings is None or not selected_models,
):
    st.session_state.pop("count_results", None)
    progress = st.progress(0.0, text="Preparing inference")

    def update_progress(index: int, total: int, label: str) -> None:
        progress.progress((index - 1) / total, text=f"Fitting {label} ({index}/{total})")

    try:
        results = run_count_models(
            valid_frame,
            float(observation_time),
            settings=settings,
            model_keys=selected_models,
            progress_callback=update_progress,
        )
    except Exception as exc:
        progress.empty()
        st.error(f"Inference did not complete: {exc}")
    else:
        progress.progress(1.0, text="Inference complete")
        st.session_state["count_results"] = results
        st.session_state["count_result_data"] = valid_frame.copy()
        st.session_state["count_result_time"] = float(observation_time)
        st.session_state["count_result_settings"] = settings

if "count_results" in st.session_state:
    result_data = st.session_state["count_result_data"]
    result_time = st.session_state["count_result_time"]
    result_settings = st.session_state["count_result_settings"]
    result_models = list(st.session_state["count_results"])
    if (
        not valid_frame.equals(result_data)
        or float(observation_time) != result_time
        or settings != result_settings
        or selected_models != result_models
    ):
        st.warning(
            "The input, model selection or inference settings have changed since "
            "the last fit. Run inference again to view matching results."
        )
    else:
        render_results(
            st.session_state["count_results"],
            data=result_data,
            observation_time=result_time,
            settings=result_settings,
            download_name="orca_event_count_results.zip",
        )
