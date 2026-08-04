from __future__ import annotations

from collections.abc import Mapping
from io import StringIO

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from webapp.palette import DONOR_TEAL
from webapp.core.inference import (
    InferenceSettings,
    build_results_zip,
    evidence_table,
    summary_table,
)
from webapp.ui import note, research_warning


MODEL_LABELS = {
    "homo": "Homogeneous Poisson",
    "z2p": "Zero-inflated Poisson",
    "dis2p": "Gamma–Poisson",
    "hetero3": "Zero-inflated Gamma–Poisson",
}

MODEL_HELP = {
    "homo": "One shared event rate for every cell.",
    "z2p": "A shared active-cell rate plus a non-engaging fraction.",
    "dis2p": "A continuous Gamma distribution of cell-specific event rates.",
    "hetero3": "Continuous rate heterogeneity plus a non-engaging fraction.",
}


def parse_optional_seed(raw: str) -> int | None:
    value = raw.strip()
    if not value:
        return None
    try:
        seed = int(value)
    except ValueError as exc:
        raise ValueError("Seed must be a whole number or left blank.") from exc
    if not 0 <= seed <= 4_294_967_295:
        raise ValueError("Seed must be between 0 and 4,294,967,295.")
    return seed


def inference_controls(prefix: str, *, donor_aware: bool = False) -> InferenceSettings | None:
    research_warning()
    preset = st.selectbox(
        "Compute profile",
        ["Preview", "Demo", "Custom"],
        index=0,
        key=f"{prefix}_preset",
        help=(
            "Preview is the quickest illustrative run. Demo uses more particles and independent chains. "
            "Neither replaces a validated publication-scale analysis."
        ),
    )
    preset_values = {
        "Preview": (64, 1, 1),
        "Demo": (256, 2, 2),
        "Custom": (128, 2, 1),
    }
    default_particles, default_chains, default_cores = preset_values[preset]

    with st.expander("Inference settings and what they mean", expanded=preset == "Custom"):
        first, second, third = st.columns(3)
        with first:
            particles = st.number_input(
                "SMC particles per chain",
                min_value=32,
                max_value=1_000,
                value=default_particles,
                step=32,
                key=f"{prefix}_particles_{preset}",
                help="More particles reduce Monte Carlo noise but increase runtime and memory use.",
            )
        with second:
            chains = st.number_input(
                "Independent chains",
                min_value=1,
                max_value=2,
                value=default_chains,
                step=1,
                key=f"{prefix}_chains_{preset}",
                help="Independent SMC runs help assess the repeatability of posterior and evidence estimates.",
            )
        with third:
            cores = st.number_input(
                "CPU cores",
                min_value=1,
                max_value=2,
                value=min(default_cores, default_chains),
                step=1,
                key=f"{prefix}_cores_{preset}",
                help="Cores can shorten runtime. They do not add statistical information.",
            )

        seed_raw = st.text_input(
            "Inference seed (optional)",
            value="",
            key=f"{prefix}_seed",
            placeholder="Leave blank for a new pseudo-random run",
            help="A fixed seed makes the pseudo-random computation reproducible; it does not improve the inference.",
        )
        advanced_a, advanced_b = st.columns(2)
        with advanced_a:
            threshold = st.slider(
                "Tempering threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.6 if donor_aware else 0.5,
                step=0.05,
                key=f"{prefix}_threshold",
                help="Controls SMC tempering increments; a higher value generally creates more intermediate stages.",
            )
            correlation_threshold = st.number_input(
                "Mutation correlation threshold",
                min_value=0.001,
                max_value=0.2,
                value=0.01,
                step=0.005,
                format="%.3f",
                key=f"{prefix}_correlation",
                help="A smaller value generally requests more particle-mutation effort.",
            )
        with advanced_b:
            prior_lower, prior_upper = st.slider(
                "log10 rate-prior bounds",
                min_value=-6.0,
                max_value=3.0,
                value=(-1.5, 1.5) if donor_aware else (-5.0, 2.0),
                step=0.5,
                key=f"{prefix}_prior_bounds",
                help="Uniform bounds on the base-10 logarithm of the event rate.",
            )
            sigma_prior = st.number_input(
                "Heterogeneity prior scale",
                min_value=0.1,
                max_value=10.0,
                value=3.0 if donor_aware else 1.0,
                step=0.1,
                key=f"{prefix}_sigma_prior",
                help="Half-Normal scale for continuous rate heterogeneity. Bayes factors can be sensitive to this prior.",
            )

        donor_deviation_prior = (0.2, 0.2, 0.5)
        if donor_aware:
            st.markdown("**Donor-deviation prior scales**")
            st.caption(
                "These hierarchical priors affect shrinkage and marginal likelihoods; "
                "include their values when reporting a Bayes-factor analysis."
            )
            donor_columns = st.columns(3)
            with donor_columns[0]:
                donor_mean_scale = st.number_input(
                    "Log-rate deviation",
                    min_value=0.05,
                    max_value=2.0,
                    value=0.2,
                    step=0.05,
                    key=f"{prefix}_donor_mean_scale",
                    help="Prior standard deviation for donor departures in mean log event rate.",
                )
            with donor_columns[1]:
                donor_sigma_scale = st.number_input(
                    "Log-heterogeneity deviation",
                    min_value=0.05,
                    max_value=2.0,
                    value=0.2,
                    step=0.05,
                    key=f"{prefix}_donor_sigma_scale",
                    help="Prior standard deviation for donor departures in log rate heterogeneity.",
                )
            with donor_columns[2]:
                donor_zero_scale = st.number_input(
                    "Zero-logit deviation",
                    min_value=0.05,
                    max_value=3.0,
                    value=0.5,
                    step=0.05,
                    key=f"{prefix}_donor_zero_scale",
                    help="Prior standard deviation for donor departures in the non-engaging logit.",
                )
            donor_deviation_prior = (
                float(donor_mean_scale),
                float(donor_sigma_scale),
                float(donor_zero_scale),
            )

    try:
        seed = parse_optional_seed(seed_raw)
    except ValueError as exc:
        st.error(str(exc))
        return None

    return InferenceSettings(
        draws=int(particles),
        chains=int(chains),
        cores=min(int(cores), int(chains)),
        seed=seed,
        threshold=float(threshold),
        correlation_threshold=float(correlation_threshold),
        lambda_prior_bounds=(float(prior_lower), float(prior_upper)),
        p_prior_bounds=(1.0, 1.0),
        std_prior_factor=float(sigma_prior),
        donor_deviation_prior=donor_deviation_prior,
    )


def model_selector(prefix: str, *, default: list[str] | None = None) -> list[str]:
    default = default or ["homo", "z2p", "dis2p", "hetero3"]
    chosen_labels = st.multiselect(
        "Models to fit",
        options=list(MODEL_LABELS.values()),
        default=[MODEL_LABELS[key] for key in default],
        key=f"{prefix}_models",
        help="Fit at least two models to obtain a comparative Bayes-factor table.",
    )
    reverse = {label: key for key, label in MODEL_LABELS.items()}
    for key, label in MODEL_LABELS.items():
        if label in chosen_labels:
            st.caption(f"**{label}:** {MODEL_HELP[key]}")
    return [reverse[label] for label in chosen_labels]


def data_overview(frame: pd.DataFrame, *, donor_aware: bool = False) -> None:
    metrics = st.columns(4 if donor_aware else 3)
    metrics[0].metric("Cells", f"{len(frame):,}")
    metrics[1].metric("Mean count", f"{frame['count'].mean():.2f}")
    metrics[2].metric("Zero-count cells", f"{(frame['count'] == 0).mean():.1%}")
    if donor_aware:
        metrics[3].metric("Donors", f"{frame['donor_id'].nunique():,}")

    count_frequency = (
        frame["count"].value_counts().sort_index().rename_axis("Event count").rename("Number of cells")
    )
    st.bar_chart(count_frequency, color=DONOR_TEAL, x_label="Event count", y_label="Number of cells")
    st.dataframe(frame, hide_index=True, width="stretch", height=280)

    if donor_aware:
        donor_table = (
            frame.groupby("donor_id", sort=True)["count"]
            .agg(cells="size", mean_count="mean", median_count="median", zero_fraction=lambda x: (x == 0).mean())
            .reset_index()
        )
        st.caption("Donor-level input summary")
        st.dataframe(
            donor_table.style.format({"mean_count": "{:.2f}", "median_count": "{:.1f}", "zero_fraction": "{:.1%}"}),
            hide_index=True,
            width="stretch",
        )
        small_groups = donor_table.loc[donor_table["cells"] < 20, "donor_id"]
        if not small_groups.empty:
            st.warning(
                "The following donors have fewer than 20 cells, so donor-specific "
                "estimates may be weakly identified: "
                + ", ".join(map(str, small_groups.tolist()))
            )


def _result_idata(result: object):
    idata = getattr(result, "idata", None)
    if idata is None and isinstance(result, Mapping):
        idata = result.get("idata")
    return idata


def render_results(
    results: Mapping[str, object],
    *,
    data: pd.DataFrame,
    observation_time: float,
    settings: InferenceSettings,
    truth: Mapping[str, object] | None = None,
    download_name: str = "orca_results.zip",
) -> None:
    if not results:
        return
    st.success("Inference completed for all selected models.")

    st.header("Model comparison")
    evidence = evidence_table(results)
    st.dataframe(evidence, hide_index=True, width="stretch")
    st.caption(
        "log10 BF vs best is zero for the highest-evidence fitted model and negative for alternatives. "
        "With small SMC runs, repeatability across seeds and particle counts is essential."
    )

    st.header("Posterior summaries")
    summary = summary_table(results)
    st.dataframe(summary, hide_index=True, width="stretch", height=420)

    if truth is not None:
        parameter_truth_keys = {
            "lambda": "mu_lambda",
            "mu_lambda": "mu_lambda",
            "sigma_lambda": "sigma_lambda",
            "p_zero": "p_zero",
        }
        hdi_columns = [
            column for column in summary.columns if str(column).startswith("hdi_")
        ]
        recovery_rows: list[dict[str, object]] = []
        if len(hdi_columns) >= 2:
            lower_column, upper_column = hdi_columns[:2]
            for row in summary.to_dict("records"):
                truth_key = parameter_truth_keys.get(str(row["parameter"]))
                if truth_key is None or truth_key not in truth:
                    continue
                truth_value = float(truth[truth_key])
                recovery_rows.append(
                    {
                        "Fitted model": row["model"],
                        "Parameter": row["parameter"],
                        "Ground truth": truth_value,
                        "Posterior mean": row.get("mean"),
                        "95% HDI lower": row[lower_column],
                        "95% HDI upper": row[upper_column],
                        "Truth in 95% HDI": (
                            float(row[lower_column])
                            <= truth_value
                            <= float(row[upper_column])
                        ),
                    }
                )
        if recovery_rows:
            st.subheader("Ground-truth recovery check")
            st.dataframe(
                pd.DataFrame(recovery_rows),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "Coverage in one simulated dataset is a useful check, not formal calibration. "
                "Misspecified fitted models can target parameters differently from the generating model."
            )

    tabs = st.tabs([MODEL_LABELS.get(key, key) for key in results])
    for tab, (key, result) in zip(tabs, results.items()):
        with tab:
            idata = _result_idata(result)
            if idata is None or getattr(idata, "posterior", None) is None:
                st.info("Posterior samples are unavailable for this result.")
                continue
            variables = list(idata.posterior.data_vars)
            preferred = [
                name
                for name in variables
                if name
                in {
                    "lambda",
                    "mu_lambda",
                    "sigma_lambda",
                    "p_zero",
                    "mu_lambda_population",
                    "sigma_lambda_population",
                    "phi_0_population",
                    "mu_lambda_ref",
                }
            ]
            default_index = variables.index(preferred[0]) if preferred else 0
            selected = st.selectbox(
                "Posterior variable",
                variables,
                index=default_index,
                key=f"posterior_variable_{key}_{id(result)}",
            )
            try:
                truth_key = {
                    "lambda": "mu_lambda",
                    "mu_lambda": "mu_lambda",
                    "sigma_lambda": "sigma_lambda",
                    "p_zero": "p_zero",
                }.get(selected)
                reference = (
                    float(truth[truth_key])
                    if truth is not None
                    and truth_key is not None
                    and truth_key in truth
                    else None
                )
                axes = az.plot_posterior(
                    idata,
                    var_names=[selected],
                    hdi_prob=0.95,
                    ref_val=reference,
                )
                figure = np.asarray(axes).ravel()[0].figure
                st.pyplot(figure, width="stretch")
                plt.close(figure)
            except Exception as exc:  # plotting should not hide numerical summaries
                st.warning(f"The posterior plot could not be rendered: {exc}")

    archive = build_results_zip(
        results,
        data,
        observation_time,
        settings,
        truth=dict(truth) if truth is not None else None,
    )
    st.download_button(
        "Download results and configuration",
        data=archive,
        file_name=download_name,
        mime="application/zip",
        width="stretch",
    )


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    if uploaded_file.size > 1_000_000:
        raise ValueError("The demo accepts CSV files up to 1 MB.")
    try:
        raw = uploaded_file.getvalue().decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The CSV must use UTF-8 text encoding.") from exc
    try:
        return pd.read_csv(StringIO(raw))
    except Exception as exc:
        raise ValueError(f"Could not read the CSV file: {exc}") from exc


def normalize_uploaded_frame(
    raw: pd.DataFrame,
    *,
    prefix: str,
    donor_aware: bool,
) -> pd.DataFrame:
    if raw.empty:
        return raw
    columns = list(raw.columns)
    cell_default = columns.index("cell_id") if "cell_id" in columns else 0
    count_default = columns.index("count") if "count" in columns else min(1, len(columns) - 1)
    mapping_columns = st.columns(3 if donor_aware else 2)
    with mapping_columns[0]:
        cell_column = st.selectbox("Cell ID column", columns, index=cell_default, key=f"{prefix}_cell_column")
    with mapping_columns[1]:
        count_column = st.selectbox("Count column", columns, index=count_default, key=f"{prefix}_count_column")
    normalized = pd.DataFrame({"cell_id": raw[cell_column], "count": raw[count_column]})
    if donor_aware:
        donor_default = columns.index("donor_id") if "donor_id" in columns else min(2, len(columns) - 1)
        with mapping_columns[2]:
            donor_column = st.selectbox("Donor column", columns, index=donor_default, key=f"{prefix}_donor_column")
        normalized["donor_id"] = raw[donor_column]
    return normalized
