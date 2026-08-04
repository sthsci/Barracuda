from __future__ import annotations

import pandas as pd
import streamlit as st

from webapp.analysis_ui import (
    MODEL_LABELS,
    data_overview,
    inference_controls,
    model_selector,
    parse_optional_seed,
    render_results,
)
from webapp.core.data import validate_count_frame
from webapp.core.inference import run_count_models
from webapp.core.simulation import simulate_event_counts
from webapp.ui import hero, note


hero(
    "1 · Simulate → infer → compare",
    "Synthetic data validation",
    "Choose a known population structure, generate event counts, then ask whether Bayesian inference recovers the parameters and ranks the generating model.",
    badge="Ground truth is visible",
)

note(
    "A demonstration is not calibration",
    "One successful dataset is encouraging, but formal validation repeats simulations across many truths, datasets and seeds.",
    tone="amber",
)
st.caption(
    "The public demo accepts generated counts up to 100. If an extreme rate, "
    "duration or heterogeneity setting exceeds that compute-safe limit, reduce "
    "the setting and generate again. Choose priors that cover the truth you set."
)

st.header("A. Choose the ground truth")
ground_label = st.selectbox("Generating model", list(MODEL_LABELS.values()), index=3)
ground_key = {label: key for key, label in MODEL_LABELS.items()}[ground_label]

first, second, third = st.columns(3)
with first:
    n_cells = st.number_input("Number of cells", 10, 1_000, 100, step=10)
    observation_time = st.number_input(
        "Observation time",
        min_value=0.01,
        max_value=100.0,
        value=1.0,
        step=0.25,
        help="Counts are modelled as Poisson events with expected count equal to rate × observation time.",
    )
with second:
    mu_lambda = st.number_input("Mean active-cell rate μλ", 0.01, 100.0, 4.0, step=0.25)
    sigma_lambda = st.number_input(
        "Continuous heterogeneity σλ",
        0.0,
        50.0,
        3.0 if ground_key in {"dis2p", "hetero3"} else 0.0,
        step=0.25,
        disabled=ground_key not in {"dis2p", "hetero3"},
    )
with third:
    p_zero = st.slider(
        "Non-engaging fraction p₀",
        0.0,
        0.95,
        0.2 if ground_key in {"z2p", "hetero3"} else 0.0,
        step=0.05,
        disabled=ground_key not in {"z2p", "hetero3"},
    )
    simulation_seed_raw = st.text_input(
        "Simulation seed (optional)",
        value="",
        placeholder="Blank = a new dataset",
        help="Set a seed when you want to reproduce exactly the same simulated dataset.",
    )

if st.button("Generate synthetic data", type="primary", width="stretch"):
    for key in (
        "synthetic_frame",
        "synthetic_truth",
        "synthetic_time",
        "synthetic_results",
        "synthetic_settings",
    ):
        st.session_state.pop(key, None)
    try:
        simulation_seed = parse_optional_seed(simulation_seed_raw)
        frame, truth = simulate_event_counts(
            model_key=ground_key,
            n_cells=int(n_cells),
            obs_time=float(observation_time),
            mu_lambda=float(mu_lambda),
            sigma_lambda=float(sigma_lambda),
            p_zero=float(p_zero),
            seed=simulation_seed,
        )
        frame = validate_count_frame(frame)
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.session_state["synthetic_frame"] = frame
        st.session_state["synthetic_truth"] = truth
        st.session_state["synthetic_time"] = float(observation_time)

frame = st.session_state.get("synthetic_frame")
truth = st.session_state.get("synthetic_truth")
stored_time = st.session_state.get("synthetic_time")
if frame is None:
    st.info("Choose the ground truth and generate a dataset to continue.")
    st.stop()

st.header("B. Inspect the generated dataset")
truth_frame = pd.DataFrame(
    {
        "Quantity": ["Generating model", "Mean rate μλ", "Heterogeneity σλ", "Non-engaging p₀", "Seed used"],
        "Ground truth": [
            MODEL_LABELS.get(str(truth.get("model_key")), truth.get("model_key")),
            f"{float(truth.get('mu_lambda', 0)):.4g}",
            f"{float(truth.get('sigma_lambda', 0)):.4g}",
            f"{float(truth.get('p_zero', 0)):.4g}",
            (
                str(truth.get("seed"))
                if truth.get("seed") is not None
                else "Not fixed (new random stream)"
            ),
        ],
    }
)
st.dataframe(truth_frame, hide_index=True, width="stretch")
data_overview(frame)
st.download_button(
    "Download this synthetic dataset",
    frame.to_csv(index=False).encode("utf-8"),
    "orca_synthetic_counts.csv",
    "text/csv",
)

st.header("C. Run Bayesian inference")
selected_models = model_selector("synthetic")
settings = inference_controls("synthetic")
run_disabled = settings is None or not selected_models

if st.button(
    "Fit selected models",
    type="primary",
    width="stretch",
    disabled=run_disabled,
):
    st.session_state.pop("synthetic_results", None)
    st.session_state.pop("synthetic_settings", None)
    progress = st.progress(0.0, text="Preparing inference")

    def update_progress(index: int, total: int, label: str) -> None:
        progress.progress((index - 1) / total, text=f"Fitting {label} ({index}/{total})")

    try:
        results = run_count_models(
            frame,
            float(stored_time),
            settings=settings,
            model_keys=selected_models,
            progress_callback=update_progress,
        )
    except Exception as exc:
        progress.empty()
        st.error(f"Inference did not complete: {exc}")
    else:
        progress.progress(1.0, text="Inference complete")
        st.session_state["synthetic_results"] = results
        st.session_state["synthetic_settings"] = settings

if "synthetic_results" in st.session_state:
    result_settings = st.session_state["synthetic_settings"]
    result_models = list(st.session_state["synthetic_results"])
    if settings != result_settings or selected_models != result_models:
        st.warning(
            "The model selection or inference settings have changed since the "
            "last fit. Run inference again to view matching results."
        )
    else:
        render_results(
            st.session_state["synthetic_results"],
            data=frame,
            observation_time=float(stored_time),
            settings=result_settings,
            truth=truth,
            download_name="orca_synthetic_validation.zip",
        )
