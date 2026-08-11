"""Pure reporting helpers for donor-aware event-count inference.

The donor-aware notebooks fit every condition independently with the hierarchy
in :mod:`bayesorca._backends.donor.inference_donor_relative`. This module keeps the
scientific rules used by those notebooks separate from Dash callbacks:

* donor-level draws from one fit remain paired by ``chain`` and ``draw``;
* condition-level posteriors are independent and are compared with all
  treatment-control particle pairs (or independent Monte Carlo pairs);
* percentage contrasts use the fixed posterior mean of the control samples as
  their denominator; and
* population heterogeneity is the variance of the cell-weighted donor mixture,
  split into within-donor and between-donor components.

The functions return pandas/xarray objects or Plotly figures and do not mutate
application state, register callbacks, or depend on a running Dash app.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from io import StringIO
from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xarray as xr

from webapp.palette import (
    CONDITION_BISPECIFIC,
    CONDITION_CONTROL,
    CONDITION_RITUXIMAB,
    DONOR_GOLD,
    DONOR_RUST,
    DONOR_SAGE,
    DONOR_TEAL,
    MODEL_GAMMA,
    MODEL_HOMOGENEOUS,
    MODEL_ZERO_INFLATED,
    MODEL_ZERO_INFLATED_GAMMA,
    PAPER_INK,
    PAPER_LINE,
    PAPER_MIST,
    PAPER_SPINE,
    PAPER_WARM,
)


MODEL_ORDER = ("homo", "z2p", "dis2p", "hetero3")
MODEL_ALIASES = {
    "homo": "homo",
    "homogeneous": "homo",
    "z2p": "z2p",
    "zi": "z2p",
    "dis2p": "dis2p",
    "gamma": "dis2p",
    "hetero3": "hetero3",
    "zir": "hetero3",
    "zigamma": "hetero3",
}
MODEL_LABELS = {
    "homo": "𝓜_homo · Homogeneous Poisson",
    "z2p": "𝓜_ZI · Zero inflated Poisson",
    "dis2p": "𝓜_Γ · Heterogeneous Gamma Poisson",
    "hetero3": "𝓜_ZIΓ · Zero inflated heterogeneous Gamma Poisson",
}
MODEL_SHORT_LABELS = {
    "homo": "𝓜_homo",
    "z2p": "𝓜_ZI",
    "dis2p": "𝓜_Γ",
    "hetero3": "𝓜_ZIΓ",
}
MODEL_COLOURS = {
    "homo": MODEL_HOMOGENEOUS,
    "z2p": MODEL_ZERO_INFLATED,
    "dis2p": MODEL_GAMMA,
    "hetero3": MODEL_ZERO_INFLATED_GAMMA,
}

POPULATION_PARAMETERS = (
    "mu_lambda_population",
    "sigma_lambda_population",
    "phi_0_population",
)
DONOR_PARAMETERS = (
    "mu_lambda_donor",
    "sigma_lambda_donor",
    "phi_0_donor",
)
PARAMETER_LABELS = {
    "mu_lambda_population": "Population mean event rate, μλ",
    "sigma_lambda_population": "Population cell-to-cell heterogeneity, σλ",
    "phi_0_population": "Population nonengaging fraction, φ₀",
    "mu_lambda_donor": "Donor mean event rate, μλ,d",
    "sigma_lambda_donor": "Within-donor cell-to-cell heterogeneity, σλ,d",
    "phi_0_donor": "Donor nonengaging fraction, φ₀,d",
    "delta_mu_lambda": "Change in mean event rate, Δμλ",
    "delta_sigma_lambda": "Change in cell-to-cell heterogeneity, Δσλ",
    "percent_delta_mu_lambda": "Change in mean event rate (% of control mean)",
    "percent_delta_sigma_lambda": (
        "Change in cell-to-cell heterogeneity (% of control mean)"
    ),
}

CONDITION_COLOURS = {
    "No treatment": CONDITION_CONTROL,
    "Control": CONDITION_CONTROL,
    "Rituximab": CONDITION_RITUXIMAB,
    "Bispecific antibody": CONDITION_BISPECIFIC,
    "Bispecific": CONDITION_BISPECIFIC,
    "Bispecific Ab": CONDITION_BISPECIFIC,
}
DONOR_COLOURS = (
    DONOR_TEAL,
    DONOR_SAGE,
    DONOR_GOLD,
    DONOR_RUST,
    "#3A86FF",
    "#8338EC",
    "#FF006E",
    "#FB5607",
    "#577590",
    "#43AA8B",
    "#F9844A",
    "#6D597A",
)

BF3_LOG10 = float(np.log10(3.0))
BF_BANDS = (
    ("Anecdotal", 0.0, BF3_LOG10, PAPER_MIST),
    ("Moderate", BF3_LOG10, 1.0, "#FFF3B0"),
    ("Strong", 1.0, 2.0, "#FFD28A"),
    ("Extreme", 2.0, None, "#E76F51"),
)


def canonical_model_name(model: object) -> str:
    """Return the canonical key used by the four event-count models."""

    key = str(model).strip().lower().replace("_", "").replace("-", "")
    if key not in MODEL_ALIASES:
        raise ValueError(f"unknown donor-aware model: {model!r}")
    return MODEL_ALIASES[key]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required column(s): {', '.join(missing)}")


def _validated_log_evidence(table: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame) or table.empty:
        raise ValueError("log-evidence table must contain at least one row")
    _require_columns(table, ("condition", "outcome", "model", "logml"))

    result = table.loc[:, ["condition", "outcome", "model", "logml"]].copy()
    result["condition"] = result["condition"].astype(str)
    result["outcome"] = result["outcome"].astype(str)
    result["model"] = result["model"].map(canonical_model_name)
    result["logml"] = pd.to_numeric(result["logml"], errors="coerce")
    if not np.isfinite(result["logml"].to_numpy(dtype=float)).all():
        raise ValueError("logml values must be finite numbers")
    if result.duplicated(["condition", "outcome", "model"]).any():
        raise ValueError("duplicate condition-outcome-model evidence rows")
    return result


def _add_bayes_factors(
    table: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    evidence_column: str,
) -> pd.DataFrame:
    result = table.copy()
    best_column = (
        "best_logml" if evidence_column == "logml" else "best_total_log_evidence"
    )
    result[best_column] = result.groupby(list(group_columns), sort=False)[
        evidence_column
    ].transform("max")
    result["delta_logml_vs_best"] = result[evidence_column] - result[best_column]
    result["log10_BF_model_vs_best"] = result["delta_logml_vs_best"] / np.log(10.0)
    result["log10_BF_best_vs_model"] = -result["log10_BF_model_vs_best"]
    result["BF_best_vs_model"] = np.exp(
        np.clip(-result["delta_logml_vs_best"].to_numpy(dtype=float), -745.0, 709.0)
    )
    result["is_best"] = np.isclose(
        result[evidence_column].to_numpy(dtype=float),
        result[best_column].to_numpy(dtype=float),
    )

    ranked = result.sort_values(
        [*group_columns, evidence_column],
        ascending=[True] * len(group_columns) + [False],
        kind="stable",
    )
    best = (
        ranked.groupby(list(group_columns), sort=False, as_index=False)
        .first()[[*group_columns, "model"]]
        .rename(columns={"model": "best_model"})
    )
    result = result.merge(best, on=list(group_columns), how="left", validate="many_to_one")
    result["model_display"] = result["model"].map(MODEL_LABELS)
    result["best_model_display"] = result["best_model"].map(MODEL_LABELS)
    return result


def condition_model_evidence(log_evidence: pd.DataFrame) -> pd.DataFrame:
    """Compute condition-specific Bayes factors from SMC log evidence.

    The returned positive ``log10_BF_best_vs_model`` is plotted on an ordinary
    linear axis.  Consequently the evidence boundaries remain at
    ``log10(3)``, ``1`` and ``2`` rather than being converted into equal-width
    categorical bands.
    """

    table = _validated_log_evidence(log_evidence)
    result = _add_bayes_factors(
        table,
        group_columns=("condition", "outcome"),
        evidence_column="logml",
    )
    return result.sort_values(
        ["outcome", "condition", "logml"],
        ascending=[True, True, False],
        kind="stable",
    ).reset_index(drop=True)


def aggregate_model_evidence(
    log_evidence: pd.DataFrame,
    *,
    conditions: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Sum independent condition log evidence and compute Bayes factors.

    Aggregation is allowed only when each outcome-model combination contains
    exactly one evidence value for every requested condition.  This prevents a
    model from appearing better merely because one or more difficult conditions
    were omitted.
    """

    table = _validated_log_evidence(log_evidence)
    expected = (
        tuple(dict.fromkeys(str(condition) for condition in conditions))
        if conditions is not None
        else tuple(dict.fromkeys(table["condition"].tolist()))
    )
    if not expected:
        raise ValueError("at least one condition is required")
    unexpected = set(table["condition"]).difference(expected)
    if unexpected:
        raise ValueError(
            "evidence contains conditions outside the requested set: "
            + ", ".join(sorted(unexpected))
        )

    expected_set = set(expected)
    for (outcome, model), subset in table.groupby(["outcome", "model"], sort=False):
        observed = set(subset["condition"])
        if observed != expected_set:
            missing = sorted(expected_set.difference(observed))
            extra = sorted(observed.difference(expected_set))
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if extra:
                detail.append("unexpected " + ", ".join(extra))
            raise ValueError(
                f"incomplete condition coverage for outcome={outcome!r}, "
                f"model={model!r}: {'; '.join(detail)}"
            )

    aggregate = (
        table.groupby(["outcome", "model"], as_index=False, sort=False)
        .agg(total_log_evidence=("logml", "sum"), n_conditions=("condition", "nunique"))
    )
    result = _add_bayes_factors(
        aggregate,
        group_columns=("outcome",),
        evidence_column="total_log_evidence",
    )
    result["delta_log_evidence_vs_best"] = result["delta_logml_vs_best"]
    return result.sort_values(
        ["outcome", "total_log_evidence"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)


def _evidence_axis_upper(values: Sequence[float] | np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    maximum = max(float(finite.max(initial=0.0)), 0.0)
    return max(3.0, maximum * 1.08)


def _evidence_axis_ticks(upper: float) -> tuple[list[float], list[str]]:
    if upper <= 4.0:
        values = [0.0, BF3_LOG10, 1.0, 2.0]
        labels = ["0", "log₁₀ 3", "1", "2"]
        for value in (3.0, 4.0):
            if value <= upper + 1e-9:
                values.append(value)
                labels.append(f"{value:g}")
        return values, labels

    rough_step = max(float(upper) / 5.0, 1e-9)
    magnitude = 10.0 ** np.floor(np.log10(rough_step))
    normalised = rough_step / magnitude
    if normalised <= 1.0:
        multiplier = 1.0
    elif normalised <= 2.0:
        multiplier = 2.0
    elif normalised <= 5.0:
        multiplier = 5.0
    else:
        multiplier = 10.0
    step = multiplier * magnitude
    values = [0.0]
    value = step
    while value <= upper + 1e-9:
        values.append(float(value))
        value += step
    return values, [f"{value:g}" for value in values]


def bayes_factor_figure(
    comparison: pd.DataFrame,
    *,
    title: str | None = None,
) -> go.Figure:
    """Plot one condition/outcome (or aggregate outcome) comparison."""

    if not isinstance(comparison, pd.DataFrame) or comparison.empty:
        raise ValueError("Bayes-factor comparison must contain at least one row")
    _require_columns(comparison, ("model", "log10_BF_best_vs_model"))
    grouping_columns = [
        column for column in ("condition", "outcome") if column in comparison.columns
    ]
    for column in grouping_columns:
        if comparison[column].nunique(dropna=False) != 1:
            raise ValueError(f"select one {column} before plotting Bayes factors")

    table = comparison.copy()
    table["model"] = table["model"].map(canonical_model_name)
    table["log10_BF_best_vs_model"] = pd.to_numeric(
        table["log10_BF_best_vs_model"], errors="coerce"
    )
    values = table["log10_BF_best_vs_model"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < -1e-10):
        raise ValueError("log10_BF_best_vs_model must contain finite nonnegative values")
    values = np.maximum(values, 0.0)
    table = table.assign(_bf=values).sort_values("_bf", ascending=False, kind="stable")

    if "is_best" in table.columns:
        is_best = table["is_best"].astype(bool).to_numpy()
    else:
        is_best = np.isclose(table["_bf"].to_numpy(dtype=float), 0.0)
    labels = [
        MODEL_SHORT_LABELS[model] + (" · Best model" if best else "")
        for model, best in zip(table["model"], is_best)
    ]
    texts = [
        "Best model · 0.00" if best else f"{value:.2f}"
        for value, best in zip(table["_bf"], is_best)
    ]
    upper = _evidence_axis_upper(table["_bf"].to_numpy(dtype=float))

    figure = go.Figure()
    for name, lower, raw_upper, colour in BF_BANDS:
        band_upper = upper if raw_upper is None else min(raw_upper, upper)
        if band_upper <= lower:
            continue
        figure.add_vrect(
            x0=lower,
            x1=band_upper,
            fillcolor=colour,
            opacity=0.52,
            line_width=0,
            layer="below",
        )
        figure.add_annotation(
            x=(lower + band_upper) / 2.0,
            y=1.02,
            xref="x",
            yref="paper",
            text=name,
            showarrow=False,
            font={"size": 11, "color": PAPER_SPINE},
        )
    for boundary in (BF3_LOG10, 1.0, 2.0):
        if boundary < upper:
            figure.add_vline(
                x=boundary,
                line={"color": PAPER_SPINE, "width": 1, "dash": "dot"},
                layer="below",
            )

    figure.add_trace(
        go.Bar(
            x=table["_bf"],
            y=labels,
            orientation="h",
            marker={
                "color": [
                    PAPER_INK if best else MODEL_COLOURS[model]
                    for model, best in zip(table["model"], is_best)
                ],
                "line": {"color": PAPER_INK, "width": 0.7},
            },
            text=texts,
            textposition="outside",
            cliponaxis=False,
            customdata=np.column_stack(
                [
                    table["model"].to_numpy(dtype=object),
                    table["_bf"].to_numpy(dtype=float),
                ]
            ),
            hovertemplate=(
                "%{customdata[0]}<br>log₁₀ BF(best / model) = "
                "%{customdata[1]:.3f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    figure.update_layout(
        title=title,
        height=max(330, 76 * len(table) + 150),
        margin={"l": 120, "r": 55, "t": 75, "b": 65},
        paper_bgcolor=PAPER_WARM,
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": PAPER_INK},
        bargap=0.20,
    )
    tick_values, tick_labels = _evidence_axis_ticks(upper)
    figure.update_xaxes(
        title="log₁₀ BF(best model / candidate model)",
        range=[0.0, upper],
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_labels,
        gridcolor=PAPER_LINE,
        zeroline=False,
    )
    figure.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return figure


def _posterior(idata: az.InferenceData) -> xr.Dataset:
    posterior = getattr(idata, "posterior", None)
    if posterior is None or not isinstance(posterior, xr.Dataset):
        raise ValueError("InferenceData must contain a posterior group")
    if "chain" not in posterior.dims or "draw" not in posterior.dims:
        raise ValueError("posterior variables must contain chain and draw dimensions")
    return posterior


def _fit_from_mapping(
    posterior_data: Mapping[tuple[str, str, str], az.InferenceData],
    condition: str,
    outcome: str,
    model: str,
) -> az.InferenceData:
    requested_model = canonical_model_name(model)
    matches = [
        idata
        for (stored_condition, stored_outcome, stored_model), idata in posterior_data.items()
        if str(stored_condition) == str(condition)
        and str(stored_outcome) == str(outcome)
        and canonical_model_name(stored_model) == requested_model
    ]
    if len(matches) != 1:
        raise KeyError(
            f"expected one posterior for condition={condition!r}, outcome={outcome!r}, "
            f"model={model!r}; found {len(matches)}"
        )
    return matches[0]


def _available_conditions(
    posterior_data: Mapping[tuple[str, str, str], az.InferenceData],
    *,
    outcome: str,
    model: str,
) -> list[str]:
    requested_model = canonical_model_name(model)
    return list(
        dict.fromkeys(
            str(condition)
            for condition, stored_outcome, stored_model in posterior_data
            if str(stored_outcome) == str(outcome)
            and canonical_model_name(stored_model) == requested_model
        )
    )


def _subsample_paired_draws(
    frame: pd.DataFrame,
    max_draws: int | None,
    *,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if max_draws is None:
        return frame
    max_draws = int(max_draws)
    if max_draws <= 0:
        raise ValueError("max_draws_per_fit must be positive")
    draw_keys = frame.loc[:, ["chain", "draw"]].drop_duplicates()
    if len(draw_keys) <= max_draws:
        return frame
    selected_positions = np.sort(rng.choice(len(draw_keys), size=max_draws, replace=False))
    selected = draw_keys.iloc[selected_positions]
    return frame.merge(selected, on=["chain", "draw"], how="inner", validate="many_to_one")


def _population_fit_frame(idata: az.InferenceData) -> pd.DataFrame:
    posterior = _posterior(idata)
    variables = [name for name in POPULATION_PARAMETERS if name in posterior]
    if "mu_lambda_population" not in variables:
        raise ValueError("posterior is missing mu_lambda_population")

    n_chain = int(posterior.sizes["chain"])
    n_draw = int(posterior.sizes["draw"])
    frame = pd.DataFrame(
        {
            "chain": np.repeat(np.asarray(posterior.coords["chain"]), n_draw),
            "draw": np.tile(np.asarray(posterior.coords["draw"]), n_chain),
        }
    )
    for name in variables:
        variable = posterior[name]
        if set(variable.dims) != {"chain", "draw"}:
            raise ValueError(f"{name} must be scalar for each chain and draw")
        frame[name] = variable.transpose("chain", "draw").values.reshape(-1)
    finite_columns = [name for name in variables]
    finite = np.isfinite(frame[finite_columns].to_numpy(dtype=float)).all(axis=1)
    return frame.loc[finite].reset_index(drop=True)


def _donor_dimension(variable: xr.DataArray, name: str) -> str:
    extra = [dimension for dimension in variable.dims if dimension not in {"chain", "draw"}]
    if len(extra) != 1:
        raise ValueError(f"{name} must contain exactly one donor dimension")
    return extra[0]


def _donor_fit_frame(
    idata: az.InferenceData,
    *,
    donor_labels: Sequence[object] | Mapping[object, object] | None = None,
) -> pd.DataFrame:
    posterior = _posterior(idata)
    variables = [name for name in DONOR_PARAMETERS if name in posterior]
    if "mu_lambda_donor" not in variables:
        raise ValueError("posterior is missing mu_lambda_donor")

    donor_dimension = _donor_dimension(posterior["mu_lambda_donor"], "mu_lambda_donor")
    donor_coordinates = np.asarray(posterior["mu_lambda_donor"].coords[donor_dimension])
    n_chain = int(posterior.sizes["chain"])
    n_draw = int(posterior.sizes["draw"])
    n_donor = int(posterior.sizes[donor_dimension])
    frame = pd.DataFrame(
        {
            "chain": np.repeat(np.asarray(posterior.coords["chain"]), n_draw * n_donor),
            "draw": np.tile(np.repeat(np.asarray(posterior.coords["draw"]), n_donor), n_chain),
            "donor_coordinate": np.tile(donor_coordinates, n_chain * n_draw),
        }
    )
    for name in variables:
        variable = posterior[name]
        variable_donor_dimension = _donor_dimension(variable, name)
        if int(variable.sizes[variable_donor_dimension]) != n_donor:
            raise ValueError(f"{name} uses a different number of donors")
        frame[name] = variable.transpose("chain", "draw", variable_donor_dimension).values.reshape(-1)

    if donor_labels is None:
        frame["donor_id"] = frame["donor_coordinate"].astype(str)
    elif isinstance(donor_labels, Mapping):
        frame["donor_id"] = frame["donor_coordinate"].map(donor_labels)
        if frame["donor_id"].isna().any():
            raise ValueError("donor_labels mapping does not cover every donor coordinate")
    else:
        labels = list(donor_labels)
        if len(labels) != n_donor:
            raise ValueError(f"expected {n_donor} donor labels, received {len(labels)}")
        label_map = dict(zip(donor_coordinates.tolist(), labels))
        frame["donor_id"] = frame["donor_coordinate"].map(label_map)

    finite = np.isfinite(frame[variables].to_numpy(dtype=float)).all(axis=1)
    return frame.loc[finite].reset_index(drop=True)


def population_posterior_frame(
    posterior_data: Mapping[tuple[str, str, str], az.InferenceData],
    *,
    outcome: str,
    model: str = "dis2p",
    conditions: Sequence[str] | None = None,
    max_draws_per_fit: int | None = None,
    random_seed: int = 307,
) -> pd.DataFrame:
    """Collect condition-specific population posterior draws without pooling them."""

    canonical_model = canonical_model_name(model)
    selected_conditions = (
        list(dict.fromkeys(str(condition) for condition in conditions))
        if conditions is not None
        else _available_conditions(posterior_data, outcome=outcome, model=canonical_model)
    )
    if not selected_conditions:
        raise ValueError("no matching population posteriors were found")

    rng = np.random.default_rng(random_seed)
    frames = []
    for condition in selected_conditions:
        fit = _fit_from_mapping(posterior_data, condition, outcome, canonical_model)
        frame = _population_fit_frame(fit)
        frame = _subsample_paired_draws(frame, max_draws_per_fit, rng=rng)
        frame.insert(0, "model", canonical_model)
        frame.insert(0, "outcome", str(outcome))
        frame.insert(0, "condition", condition)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def donor_posterior_frame(
    posterior_data: Mapping[tuple[str, str, str], az.InferenceData],
    *,
    outcome: str,
    model: str = "dis2p",
    conditions: Sequence[str] | None = None,
    donor_labels: Sequence[object] | Mapping[object, object] | None = None,
    max_draws_per_fit: int | None = None,
    random_seed: int = 307,
) -> pd.DataFrame:
    """Collect paired donor draws for condition and donor comparison plots."""

    canonical_model = canonical_model_name(model)
    selected_conditions = (
        list(dict.fromkeys(str(condition) for condition in conditions))
        if conditions is not None
        else _available_conditions(posterior_data, outcome=outcome, model=canonical_model)
    )
    if not selected_conditions:
        raise ValueError("no matching donor posteriors were found")

    rng = np.random.default_rng(random_seed)
    frames = []
    for condition in selected_conditions:
        fit = _fit_from_mapping(posterior_data, condition, outcome, canonical_model)
        frame = _donor_fit_frame(fit, donor_labels=donor_labels)
        frame = _subsample_paired_draws(frame, max_draws_per_fit, rng=rng)
        frame.insert(0, "model", canonical_model)
        frame.insert(0, "outcome", str(outcome))
        frame.insert(0, "condition", condition)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _validated_condition_results(
    results: Mapping[str, Mapping[str, object]],
) -> tuple[list[str], list[str]]:
    if not isinstance(results, Mapping) or not results:
        raise ValueError("condition results must contain at least one condition")
    conditions = [str(condition) for condition in results]
    first_models: list[str] | None = None
    for condition, condition_results in results.items():
        if not isinstance(condition_results, Mapping) or not condition_results:
            raise ValueError(f"condition {condition!r} contains no inference results")
        models = [canonical_model_name(model) for model in condition_results]
        if first_models is None:
            first_models = models
        elif models != first_models:
            raise ValueError("every condition must contain inference results for the same models in the same order")
        for model, result in condition_results.items():
            if not bool(getattr(result, "donor_aware", False)):
                raise ValueError(
                    f"condition {condition!r}, model {model!r} is not donor aware"
                )
            if getattr(result, "idata", None) is None:
                raise ValueError(
                    f"condition {condition!r}, model {model!r} has no InferenceData"
                )
    return conditions, list(first_models or [])


def condition_results_log_evidence(
    results: Mapping[str, Mapping[str, object]],
    *,
    outcome: str = "count",
) -> pd.DataFrame:
    """Convert web ``ConditionResults`` into the notebook evidence schema."""

    _validated_condition_results(results)
    rows = []
    for condition, condition_results in results.items():
        for stored_model, result in condition_results.items():
            rows.append(
                {
                    "condition": str(condition),
                    "outcome": str(outcome),
                    "model": canonical_model_name(stored_model),
                    "logml": float(getattr(result, "log_evidence")),
                }
            )
    return _validated_log_evidence(pd.DataFrame(rows))


def condition_results_population_frame(
    results: Mapping[str, Mapping[str, object]],
    *,
    model: str,
    outcome: str = "count",
    max_draws_per_fit: int | None = 5_000,
    random_seed: int = 307,
) -> pd.DataFrame:
    """Collect population posteriors from web condition-inference results."""

    conditions, models = _validated_condition_results(results)
    model_key = canonical_model_name(model)
    if model_key not in models:
        raise KeyError(f"model {model!r} is not present in the condition results")
    rng = np.random.default_rng(random_seed)
    frames = []
    for condition in conditions:
        result = next(
            value
            for key, value in results[condition].items()
            if canonical_model_name(key) == model_key
        )
        frame = _population_fit_frame(getattr(result, "idata"))
        frame = _subsample_paired_draws(frame, max_draws_per_fit, rng=rng)
        frame.insert(0, "model", model_key)
        frame.insert(0, "outcome", str(outcome))
        frame.insert(0, "condition", condition)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def condition_results_donor_frame(
    results: Mapping[str, Mapping[str, object]],
    *,
    model: str,
    outcome: str = "count",
    max_draws_per_fit: int | None = 5_000,
    random_seed: int = 307,
) -> pd.DataFrame:
    """Collect donor posteriors and apply each fit's stored donor labels."""

    conditions, models = _validated_condition_results(results)
    model_key = canonical_model_name(model)
    if model_key not in models:
        raise KeyError(f"model {model!r} is not present in the condition results")
    rng = np.random.default_rng(random_seed)
    frames = []
    for condition in conditions:
        result = next(
            value
            for key, value in results[condition].items()
            if canonical_model_name(key) == model_key
        )
        labels = tuple(map(str, getattr(result, "donor_labels", ())))
        frame = _donor_fit_frame(
            getattr(result, "idata"),
            donor_labels=labels or None,
        )
        frame["donor_id"] = frame["donor_id"].astype(str)
        frame = _subsample_paired_draws(frame, max_draws_per_fit, rng=rng)
        frame.insert(0, "model", model_key)
        frame.insert(0, "outcome", str(outcome))
        frame.insert(0, "condition", condition)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def _normalised_donor_weights(
    values: Sequence[float] | np.ndarray,
    *,
    n_donors: int,
) -> np.ndarray:
    weights = np.asarray(values, dtype=float)
    if weights.shape != (n_donors,):
        raise ValueError(f"expected {n_donors} donor weights or cell counts")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("donor weights or cell counts must be finite and positive")
    return weights / weights.sum()


def decompose_population_heterogeneity(
    idata: az.InferenceData,
    donor_weights: Sequence[float] | np.ndarray,
    *,
    tolerance: float = 1e-10,
) -> xr.Dataset:
    r"""Split every population variance draw into within and between donors.

    With cell-count weights :math:`w_d`, zero-inflated models first use active
    cell weights

    .. math:: \widetilde w_d = \frac{w_d(1-\phi_d)}{\sum_j w_j(1-\phi_j)}.

    Then

    .. math:: V_\mathrm{within}=\sum_d\widetilde w_d\sigma_d^2,
              \quad
              V_\mathrm{between}=\sum_d\widetilde w_d(\mu_d-\mu_\mathrm{pop})^2.

    Their sum is the population mixture variance.  The reconstructed moments
    are checked against saved deterministic population parameters when those
    variables are available.
    """

    posterior = _posterior(idata)
    for name in ("mu_lambda_donor", "sigma_lambda_donor"):
        if name not in posterior:
            raise ValueError(f"posterior is missing {name}")
    mu = posterior["mu_lambda_donor"]
    sigma = posterior["sigma_lambda_donor"]
    donor_dimension = _donor_dimension(mu, "mu_lambda_donor")
    if _donor_dimension(sigma, "sigma_lambda_donor") != donor_dimension:
        raise ValueError("donor parameter arrays use different donor dimensions")
    n_donors = int(mu.sizes[donor_dimension])
    weights = _normalised_donor_weights(donor_weights, n_donors=n_donors)
    weight = xr.DataArray(
        weights,
        dims=(donor_dimension,),
        coords={donor_dimension: mu.coords[donor_dimension]},
    )

    if "phi_0_donor" in posterior:
        phi = posterior["phi_0_donor"]
        if _donor_dimension(phi, "phi_0_donor") != donor_dimension:
            raise ValueError("phi_0_donor uses a different donor dimension")
        active_mass = weight * (1.0 - phi)
        active_total = active_mass.sum(donor_dimension)
        if np.any(active_total.to_numpy() <= 0):
            raise ValueError("every posterior draw must retain positive active-cell mass")
        active_weight = active_mass / active_total
        phi_population = (weight * phi).sum(donor_dimension)
    else:
        active_weight = weight.broadcast_like(mu)
        phi_population = None

    mu_population = (active_weight * mu).sum(donor_dimension)
    variance_within = (active_weight * sigma**2).sum(donor_dimension)
    variance_between = (
        active_weight * (mu - mu_population) ** 2
    ).sum(donor_dimension)
    variance_total = variance_within + variance_between
    sigma_population = np.sqrt(variance_total.clip(min=0.0))
    fraction_within = xr.where(variance_total > 0, variance_within / variance_total, np.nan)
    fraction_between = xr.where(variance_total > 0, variance_between / variance_total, np.nan)

    checks = {
        "mu_lambda_population": mu_population,
        "sigma_lambda_population": sigma_population,
    }
    if phi_population is not None:
        checks["phi_0_population"] = phi_population
    for saved_name, reconstructed in checks.items():
        if saved_name not in posterior:
            continue
        difference = np.abs(posterior[saved_name] - reconstructed)
        maximum = float(difference.max().to_numpy())
        if not np.isfinite(maximum) or maximum > float(tolerance):
            raise ValueError(
                f"reconstructed {saved_name} differs from the saved posterior "
                f"by {maximum:.3g} (tolerance {tolerance:g})"
            )

    return xr.Dataset(
        {
            "active_donor_weight": active_weight,
            "mu_lambda_population": mu_population,
            "variance_within_donor": variance_within,
            "variance_between_donor": variance_between,
            "variance_total": variance_total,
            "sigma_lambda_population": sigma_population,
            "fraction_within_donor": fraction_within,
            "fraction_between_donor": fraction_between,
        }
    )


ContrastScale = Literal["absolute", "percent_of_control_mean"]


def _finite_sample_matrix(samples: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(samples, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2:
        raise ValueError(f"{name} samples must be a one- or two-dimensional array")
    matrix = matrix[np.isfinite(matrix).all(axis=1)]
    if not len(matrix):
        raise ValueError(f"{name} samples contain no finite rows")
    return matrix


def cartesian_contrast_draws(
    treatment_samples: Sequence[float] | np.ndarray,
    control_samples: Sequence[float] | np.ndarray,
    *,
    scale: ContrastScale = "absolute",
    control_mean: Sequence[float] | np.ndarray | None = None,
    max_exact_pairs: int = 500_000,
    approximate_pairs: int = 100_000,
    random_seed: int = 307,
) -> tuple[np.ndarray, dict[str, object]]:
    """Compare independent condition posteriors with all or sampled pairs.

    Exact comparisons contain every treatment-control draw pair.  When this is
    too large, treatment and control indices are sampled independently and
    uniformly.  For percentage contrasts the denominator is the fixed control
    posterior mean, matching the final donor-summary notebook implementation.
    """

    treatment = _finite_sample_matrix(treatment_samples, "treatment")
    control = _finite_sample_matrix(control_samples, "control")
    if treatment.shape[1] != control.shape[1]:
        raise ValueError("treatment and control samples must contain the same parameters")
    max_exact_pairs = int(max_exact_pairs)
    approximate_pairs = int(approximate_pairs)
    if max_exact_pairs <= 0 or approximate_pairs <= 0:
        raise ValueError("pair limits must be positive")

    if scale == "percent_of_control_mean":
        denominator = (
            control.mean(axis=0)
            if control_mean is None
            else np.asarray(control_mean, dtype=float)
        )
        if denominator.shape != (control.shape[1],):
            raise ValueError("control_mean must contain one value per parameter")
        if not np.isfinite(denominator).all() or np.any(denominator <= 0):
            raise ValueError("control posterior means must be finite and positive")
    elif scale == "absolute":
        denominator = None
    else:
        raise ValueError(f"unknown contrast scale: {scale!r}")

    possible_pairs = int(len(treatment) * len(control))
    if possible_pairs <= max_exact_pairs:
        differences = (
            treatment[:, None, :] - control[None, :, :]
        ).reshape(possible_pairs, treatment.shape[1])
        exact = True
    else:
        rng = np.random.default_rng(random_seed)
        treatment_index = rng.integers(0, len(treatment), size=approximate_pairs)
        control_index = rng.integers(0, len(control), size=approximate_pairs)
        differences = treatment[treatment_index] - control[control_index]
        exact = False

    values = differences if denominator is None else 100.0 * differences / denominator
    metadata: dict[str, object] = {
        "scale": scale,
        "exact_cartesian": exact,
        "possible_pairs": possible_pairs,
        "returned_pairs": int(len(values)),
        "control_mean": None if denominator is None else denominator.copy(),
    }
    return values, metadata


def condition_control_contrast_frame(
    posterior_data: Mapping[tuple[str, str, str], az.InferenceData],
    *,
    treatment: str,
    control: str,
    outcome: str,
    model: str = "dis2p",
    donor_labels: Sequence[object] | Mapping[object, object] | None = None,
    scale: ContrastScale = "absolute",
    max_exact_pairs: int = 500_000,
    approximate_pairs: int = 100_000,
    random_seed: int = 307,
) -> pd.DataFrame:
    """Build draw-level treatment-control contrasts separately for each donor."""

    frame = donor_posterior_frame(
        posterior_data,
        outcome=outcome,
        model=model,
        conditions=(control, treatment),
        donor_labels=donor_labels,
    )
    required = ("mu_lambda_donor", "sigma_lambda_donor")
    _require_columns(frame, required)
    control_donors = set(frame.loc[frame["condition"] == control, "donor_id"])
    treatment_donors = set(frame.loc[frame["condition"] == treatment, "donor_id"])
    if control_donors != treatment_donors:
        raise ValueError("treatment and control inference results must contain the same donors")

    value_columns = (
        ("delta_mu_lambda", "delta_sigma_lambda")
        if scale == "absolute"
        else ("percent_delta_mu_lambda", "percent_delta_sigma_lambda")
    )
    frames = []
    for donor_index, donor in enumerate(sorted(control_donors, key=str)):
        treatment_values = frame.loc[
            (frame["condition"] == treatment) & (frame["donor_id"] == donor), required
        ].to_numpy(dtype=float)
        control_values = frame.loc[
            (frame["condition"] == control) & (frame["donor_id"] == donor), required
        ].to_numpy(dtype=float)
        values, metadata = cartesian_contrast_draws(
            treatment_values,
            control_values,
            scale=scale,
            max_exact_pairs=max_exact_pairs,
            approximate_pairs=approximate_pairs,
            random_seed=random_seed + donor_index,
        )
        donor_frame = pd.DataFrame(values, columns=value_columns)
        donor_frame.insert(0, "donor_id", donor)
        donor_frame.insert(0, "outcome", str(outcome))
        donor_frame.insert(0, "control", str(control))
        donor_frame.insert(0, "treatment", str(treatment))
        donor_frame["exact_cartesian"] = bool(metadata["exact_cartesian"])
        donor_frame["possible_pairs"] = int(metadata["possible_pairs"])
        if metadata["control_mean"] is not None:
            denominator = np.asarray(metadata["control_mean"], dtype=float)
            donor_frame["control_mean_mu_lambda"] = denominator[0]
            donor_frame["control_mean_sigma_lambda"] = denominator[1]
        frames.append(donor_frame)
    return pd.concat(frames, ignore_index=True)


def donor_frame_condition_contrasts(
    donor_draws: pd.DataFrame,
    *,
    treatment: str,
    control: str,
    scale: ContrastScale = "percent_of_control_mean",
    max_exact_pairs: int = 500_000,
    approximate_pairs: int = 100_000,
    random_seed: int = 307,
) -> pd.DataFrame:
    """Compare two conditions already collected by ``donor_posterior_frame``.

    Donor parameters remain paired within each condition.  Across conditions,
    chain/draw labels have no common meaning, so this function deliberately
    sends the two sample matrices through :func:`cartesian_contrast_draws`.
    """

    required = (
        "condition",
        "donor_id",
        "mu_lambda_donor",
        "sigma_lambda_donor",
    )
    _require_columns(donor_draws, required)
    treatment = str(treatment)
    control = str(control)
    if treatment == control:
        raise ValueError("treatment and control conditions must be different")
    available = set(donor_draws["condition"].astype(str))
    missing = {treatment, control}.difference(available)
    if missing:
        raise ValueError("unknown condition(s): " + ", ".join(sorted(missing)))

    control_donors = set(
        donor_draws.loc[
            donor_draws["condition"].astype(str) == control,
            "donor_id",
        ].astype(str)
    )
    treatment_donors = set(
        donor_draws.loc[
            donor_draws["condition"].astype(str) == treatment,
            "donor_id",
        ].astype(str)
    )
    if control_donors != treatment_donors:
        raise ValueError(
            "treatment and control must contain the same donor labels for "
            "donor-specific contrasts"
        )

    value_columns = (
        ("delta_mu_lambda", "delta_sigma_lambda")
        if scale == "absolute"
        else ("percent_delta_mu_lambda", "percent_delta_sigma_lambda")
    )
    parameter_columns = ("mu_lambda_donor", "sigma_lambda_donor")
    frames = []
    for donor_index, donor_id in enumerate(sorted(control_donors, key=str)):
        treatment_values = donor_draws.loc[
            (donor_draws["condition"].astype(str) == treatment)
            & (donor_draws["donor_id"].astype(str) == donor_id),
            parameter_columns,
        ].to_numpy(dtype=float)
        control_values = donor_draws.loc[
            (donor_draws["condition"].astype(str) == control)
            & (donor_draws["donor_id"].astype(str) == donor_id),
            parameter_columns,
        ].to_numpy(dtype=float)
        values, metadata = cartesian_contrast_draws(
            treatment_values,
            control_values,
            scale=scale,
            max_exact_pairs=max_exact_pairs,
            approximate_pairs=approximate_pairs,
            random_seed=random_seed + donor_index,
        )
        frame = pd.DataFrame(values, columns=value_columns)
        frame.insert(0, "donor_id", donor_id)
        frame.insert(0, "control", control)
        frame.insert(0, "treatment", treatment)
        frame["exact_cartesian"] = bool(metadata["exact_cartesian"])
        frame["possible_pairs"] = int(metadata["possible_pairs"])
        frame["returned_pairs"] = int(metadata["returned_pairs"])
        if metadata["control_mean"] is not None:
            denominator = np.asarray(metadata["control_mean"], dtype=float)
            frame["control_mean_mu_lambda"] = denominator[0]
            frame["control_mean_sigma_lambda"] = denominator[1]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize_contrast_draws(
    contrasts: pd.DataFrame,
    *,
    group_columns: Sequence[str] = ("donor_id",),
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    """Summarise every numeric contrast column with HDI and sign probability."""

    if not 0 < float(hdi_prob) < 1:
        raise ValueError("hdi_prob must lie strictly between 0 and 1")
    _require_columns(contrasts, group_columns)
    value_columns = [
        column
        for column in contrasts.columns
        if column.startswith("delta_") or column.startswith("percent_delta_")
    ]
    if not value_columns:
        raise ValueError("contrast table contains no contrast columns")

    rows: list[dict[str, object]] = []
    grouper: object = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for group_key, group in contrasts.groupby(grouper, sort=False, dropna=False):
        keys = (group_key,) if len(group_columns) == 1 else tuple(group_key)
        identity = dict(zip(group_columns, keys))
        for parameter in value_columns:
            values = pd.to_numeric(group[parameter], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            interval = np.asarray(az.hdi(values, hdi_prob=hdi_prob), dtype=float)
            rows.append(
                {
                    **identity,
                    "parameter": parameter,
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "hdi_lower": float(interval[0]),
                    "hdi_upper": float(interval[1]),
                    "probability_above_zero": float(np.mean(values > 0)),
                    "n_draws": int(len(values)),
                    "hdi_prob": float(hdi_prob),
                }
            )
    return pd.DataFrame(rows)


def _colour_map(groups: Sequence[object], group_column: str) -> dict[object, str]:
    unique = list(dict.fromkeys(groups))
    if group_column == "model":
        return {group: MODEL_COLOURS[canonical_model_name(group)] for group in unique}
    if group_column == "condition":
        fallback = (CONDITION_CONTROL, CONDITION_RITUXIMAB, CONDITION_BISPECIFIC)
        return {
            group: CONDITION_COLOURS.get(str(group), fallback[index % len(fallback)])
            for index, group in enumerate(unique)
        }
    return {group: DONOR_COLOURS[index % len(DONOR_COLOURS)] for index, group in enumerate(unique)}


def joint_posterior_figure(
    draws: pd.DataFrame,
    *,
    group_column: str,
    parameters: Sequence[str],
    title: str | None = None,
    zero_reference: bool = False,
    colours: Mapping[object, str] | None = None,
) -> go.Figure:
    """Draw marginal densities and paired lower-triangle joint contours."""

    if not isinstance(draws, pd.DataFrame) or draws.empty:
        raise ValueError("posterior draw table must contain at least one row")
    parameter_list = list(dict.fromkeys(str(parameter) for parameter in parameters))
    if not parameter_list:
        raise ValueError("at least one posterior parameter is required")
    _require_columns(draws, (group_column, *parameter_list))
    groups = list(dict.fromkeys(draws[group_column].tolist()))
    default_colours = _colour_map(groups, group_column)
    colour_map = {
        group: str((colours or {}).get(group, default_colours[group]))
        for group in groups
    }
    size = len(parameter_list)
    figure = make_subplots(
        rows=size,
        cols=size,
        horizontal_spacing=0.07,
        vertical_spacing=0.08,
    )

    for row_index, row_parameter in enumerate(parameter_list, start=1):
        for column_index, column_parameter in enumerate(parameter_list, start=1):
            if column_index > row_index:
                figure.update_xaxes(visible=False, row=row_index, col=column_index)
                figure.update_yaxes(visible=False, row=row_index, col=column_index)
                continue
            for group_index, group in enumerate(groups):
                subset = draws.loc[draws[group_column] == group]
                if row_index == column_index:
                    values = pd.to_numeric(subset[row_parameter], errors="coerce")
                    values = values[np.isfinite(values)]
                    if values.empty:
                        continue
                    figure.add_trace(
                        go.Histogram(
                            x=values,
                            histnorm="probability density",
                            nbinsx=32,
                            opacity=0.24,
                            marker={"color": colour_map[group]},
                            name=str(group),
                            legendgroup=str(group),
                            showlegend=row_index == 1,
                            hovertemplate=f"{group}<br>{row_parameter}=%{{x:.3g}}<extra></extra>",
                        ),
                        row=row_index,
                        col=column_index,
                    )
                else:
                    joint = subset.loc[:, [column_parameter, row_parameter]].apply(
                        pd.to_numeric, errors="coerce"
                    ).dropna()
                    if joint.empty:
                        continue
                    figure.add_trace(
                        go.Histogram2dContour(
                            x=joint[column_parameter],
                            y=joint[row_parameter],
                            ncontours=5,
                            contours={"coloring": "none", "showlabels": False},
                            line={"color": colour_map[group], "width": 2},
                            name=str(group),
                            legendgroup=str(group),
                            showlegend=False,
                            hoverinfo="skip",
                        ),
                        row=row_index,
                        col=column_index,
                    )

            if zero_reference:
                figure.add_vline(
                    x=0.0,
                    line={"color": PAPER_SPINE, "width": 1, "dash": "dash"},
                    row=row_index,
                    col=column_index,
                )
                if row_index != column_index:
                    figure.add_hline(
                        y=0.0,
                        line={"color": PAPER_SPINE, "width": 1, "dash": "dash"},
                        row=row_index,
                        col=column_index,
                    )

            if row_index == size:
                figure.update_xaxes(
                    title=PARAMETER_LABELS.get(column_parameter, column_parameter),
                    row=row_index,
                    col=column_index,
                )
            if column_index == 1:
                ylabel = (
                    "Posterior density"
                    if row_index == column_index
                    else PARAMETER_LABELS.get(row_parameter, row_parameter)
                )
                figure.update_yaxes(title=ylabel, row=row_index, col=column_index)

    figure.update_layout(
        title=title,
        barmode="overlay",
        height=max(430, 345 * size),
        paper_bgcolor=PAPER_WARM,
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": PAPER_INK},
        legend={"orientation": "h", "y": 1.04, "x": 0.0},
        margin={"l": 80, "r": 35, "t": 95, "b": 65},
    )
    figure.update_xaxes(gridcolor=PAPER_LINE, zeroline=False)
    figure.update_yaxes(gridcolor=PAPER_LINE, zeroline=False)
    return figure


def population_joint_posterior_figure(
    draws: pd.DataFrame,
    *,
    title: str | None = None,
    condition_colours: Mapping[object, str] | None = None,
) -> go.Figure:
    """Overlay condition-specific population posteriors for available parameters."""

    parameters = [name for name in POPULATION_PARAMETERS if name in draws.columns]
    return joint_posterior_figure(
        draws,
        group_column="condition",
        parameters=parameters,
        title=title,
        colours=condition_colours,
    )


def donor_joint_posterior_figure(
    draws: pd.DataFrame,
    *,
    group_by: Literal["donor_id", "condition"] = "donor_id",
    title: str | None = None,
    colours: Mapping[object, str] | None = None,
) -> go.Figure:
    """Plot one condition across donors or one donor across conditions."""

    if group_by not in {"donor_id", "condition"}:
        raise ValueError("group_by must be 'donor_id' or 'condition'")
    parameters = [name for name in DONOR_PARAMETERS if name in draws.columns]
    return joint_posterior_figure(
        draws,
        group_column=group_by,
        parameters=parameters,
        title=title,
        colours=colours,
    )


def contrast_joint_posterior_figure(
    contrasts: pd.DataFrame,
    *,
    title: str | None = None,
) -> go.Figure:
    """Plot donor-specific absolute or percentage condition contrasts."""

    parameters = [
        name
        for name in (
            "delta_mu_lambda",
            "delta_sigma_lambda",
            "percent_delta_mu_lambda",
            "percent_delta_sigma_lambda",
        )
        if name in contrasts.columns
    ]
    return joint_posterior_figure(
        contrasts,
        group_column="donor_id",
        parameters=parameters,
        title=title,
        zero_reference=True,
    )


def donor_frame_condition_contrast_figure(
    donor_draws: pd.DataFrame,
    *,
    treatment: str,
    control: str,
    scale: ContrastScale = "percent_of_control_mean",
    max_exact_pairs: int = 500_000,
    approximate_pairs: int = 100_000,
    random_seed: int = 307,
    title: str | None = None,
) -> go.Figure:
    """Calculate all-particle donor contrasts and transmit compact densities.

    The potentially large Cartesian/Monte Carlo draw matrix is reduced to
    one-dimensional histograms and a two-dimensional density grid before it is
    placed in the Plotly figure.  Thus the scientific comparison still uses the
    notebook's complete pairing rule without sending hundreds of thousands of
    raw points to the browser.
    """

    required = (
        "condition",
        "donor_id",
        "mu_lambda_donor",
        "sigma_lambda_donor",
    )
    _require_columns(donor_draws, required)
    treatment = str(treatment)
    control = str(control)
    if treatment == control:
        raise ValueError("treatment and control conditions must be different")
    available = set(donor_draws["condition"].astype(str))
    missing = {treatment, control}.difference(available)
    if missing:
        raise ValueError("unknown condition(s): " + ", ".join(sorted(missing)))
    control_donors = set(
        donor_draws.loc[
            donor_draws["condition"].astype(str) == control, "donor_id"
        ].astype(str)
    )
    treatment_donors = set(
        donor_draws.loc[
            donor_draws["condition"].astype(str) == treatment, "donor_id"
        ].astype(str)
    )
    if control_donors != treatment_donors:
        raise ValueError(
            "treatment and control must contain the same donor labels for "
            "donor-specific contrasts"
        )

    donors = sorted(control_donors, key=str)
    colours = {
        donor: DONOR_COLOURS[index % len(DONOR_COLOURS)]
        for index, donor in enumerate(donors)
    }
    parameters = (
        ("delta_mu_lambda", "delta_sigma_lambda")
        if scale == "absolute"
        else ("percent_delta_mu_lambda", "percent_delta_sigma_lambda")
    )
    source_parameters = ("mu_lambda_donor", "sigma_lambda_donor")
    figure = make_subplots(
        rows=2,
        cols=2,
        horizontal_spacing=0.10,
        vertical_spacing=0.10,
    )
    figure.update_xaxes(visible=False, row=1, col=2)
    figure.update_yaxes(visible=False, row=1, col=2)
    exact_flags: list[bool] = []

    for donor_index, donor_id in enumerate(donors):
        treatment_values = donor_draws.loc[
            (donor_draws["condition"].astype(str) == treatment)
            & (donor_draws["donor_id"].astype(str) == donor_id),
            source_parameters,
        ].to_numpy(dtype=float)
        control_values = donor_draws.loc[
            (donor_draws["condition"].astype(str) == control)
            & (donor_draws["donor_id"].astype(str) == donor_id),
            source_parameters,
        ].to_numpy(dtype=float)
        values, metadata = cartesian_contrast_draws(
            treatment_values,
            control_values,
            scale=scale,
            max_exact_pairs=max_exact_pairs,
            approximate_pairs=approximate_pairs,
            random_seed=random_seed + donor_index,
        )
        exact_flags.append(bool(metadata["exact_cartesian"]))
        marginal_data = (
            (values[:, 0], 1, 1, parameters[0]),
            (values[:, 1], 2, 2, parameters[1]),
        )
        for marginal, row, column, parameter in marginal_data:
            density, edges = np.histogram(marginal, bins=42, density=True)
            centres = 0.5 * (edges[:-1] + edges[1:])
            figure.add_trace(
                go.Scatter(
                    x=centres,
                    y=density,
                    mode="lines",
                    line={"color": colours[donor_id], "width": 2},
                    fill="tozeroy",
                    fillcolor=colours[donor_id],
                    opacity=0.48,
                    name=donor_id,
                    legendgroup=donor_id,
                    showlegend=row == 1,
                    hovertemplate=(
                        f"{donor_id}<br>{parameter}=%{{x:.3g}}"
                        "<br>Density=%{y:.3g}<extra></extra>"
                    ),
                ),
                row=row,
                col=column,
            )

        joint_density, x_edges, y_edges = np.histogram2d(
            values[:, 0],
            values[:, 1],
            bins=42,
            density=True,
        )
        figure.add_trace(
            go.Contour(
                x=0.5 * (x_edges[:-1] + x_edges[1:]),
                y=0.5 * (y_edges[:-1] + y_edges[1:]),
                z=joint_density.T,
                contours={"coloring": "none", "showlabels": False},
                ncontours=6,
                line={"color": colours[donor_id], "width": 2},
                showscale=False,
                name=donor_id,
                legendgroup=donor_id,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )

    for row, column in ((1, 1), (2, 1), (2, 2)):
        figure.add_vline(
            x=0.0,
            line={"color": PAPER_SPINE, "width": 1, "dash": "dash"},
            row=row,
            col=column,
        )
    figure.add_hline(
        y=0.0,
        line={"color": PAPER_SPINE, "width": 1, "dash": "dash"},
        row=2,
        col=1,
    )
    figure.update_xaxes(
        title=PARAMETER_LABELS[parameters[0]],
        row=2,
        col=1,
    )
    figure.update_yaxes(
        title=PARAMETER_LABELS[parameters[1]],
        row=2,
        col=1,
    )
    figure.update_xaxes(
        title=PARAMETER_LABELS[parameters[1]],
        row=2,
        col=2,
    )
    method = "All particle pairs" if all(exact_flags) else "100,000 Monte Carlo pairs per donor"
    figure.update_layout(
        title=title or f"{treatment} minus {control} · {method}",
        height=700,
        paper_bgcolor=PAPER_WARM,
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": PAPER_INK},
        legend={"orientation": "h", "y": 1.04, "x": 0.0},
        margin={"l": 90, "r": 35, "t": 100, "b": 75},
    )
    figure.update_xaxes(gridcolor=PAPER_LINE, zeroline=False)
    figure.update_yaxes(gridcolor=PAPER_LINE, zeroline=False)
    return figure


def variance_decomposition_summary(
    decomposition: xr.Dataset,
    *,
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    """Summarise within- and between-donor variance draws."""

    if not 0 < float(hdi_prob) < 1:
        raise ValueError("hdi_prob must lie strictly between 0 and 1")
    rows = []
    for variable, label in (
        ("variance_within_donor", "Within donors"),
        ("variance_between_donor", "Between donors"),
    ):
        if variable not in decomposition:
            raise ValueError(f"decomposition is missing {variable}")
        values = np.asarray(decomposition[variable], dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        interval = np.asarray(az.hdi(values, hdi_prob=hdi_prob), dtype=float)
        rows.append(
            {
                "component": variable,
                "label": label,
                "posterior_mean": float(values.mean()),
                "posterior_median": float(np.median(values)),
                "hdi_lower": float(interval[0]),
                "hdi_upper": float(interval[1]),
                "hdi_prob": float(hdi_prob),
            }
        )
    return pd.DataFrame(rows)


def variance_decomposition_figure(
    summaries: pd.DataFrame,
    *,
    category_column: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Plot posterior-mean population variance as within/between stacked bars."""

    required = ["component", "posterior_mean"]
    if category_column is not None:
        required.append(category_column)
    _require_columns(summaries, required)
    categories = [""] if category_column is None else list(dict.fromkeys(summaries[category_column]))
    figure = go.Figure()
    for component, label, colour in (
        ("variance_within_donor", "Within donors", DONOR_TEAL),
        ("variance_between_donor", "Between donors", DONOR_GOLD),
    ):
        subset = summaries.loc[summaries["component"] == component]
        if category_column is None:
            lookup = {"": float(subset["posterior_mean"].iloc[0])} if len(subset) else {}
        else:
            lookup = dict(zip(subset[category_column], subset["posterior_mean"]))
        figure.add_trace(
            go.Bar(
                x=categories,
                y=[lookup.get(category, np.nan) for category in categories],
                name=label,
                marker={"color": colour, "line": {"color": PAPER_INK, "width": 0.6}},
                hovertemplate=f"{label}<br>variance=%{{y:.3g}}<extra></extra>",
            )
        )
    figure.update_layout(
        title=title,
        barmode="stack",
        height=390,
        paper_bgcolor=PAPER_WARM,
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": PAPER_INK},
        yaxis_title="Posterior mean population variance",
        legend={"orientation": "h", "y": 1.06},
        margin={"l": 75, "r": 30, "t": 80, "b": 55},
    )
    figure.update_yaxes(gridcolor=PAPER_LINE, rangemode="tozero")
    return figure


def _graph_component(dcc: Any, figure: go.Figure, filename: str, class_name: str) -> Any:
    return dcc.Graph(
        figure=figure,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {
                "format": "png",
                "filename": filename,
                "scale": 2,
            },
        },
        responsive=True,
        className=class_name,
        style={"height": f"{int(figure.layout.height or 430)}px"},
    )


def _result_for_model(
    results: Mapping[str, Mapping[str, object]],
    condition: str,
    model_key: str,
) -> object:
    return next(
        result
        for stored_model, result in results[condition].items()
        if canonical_model_name(stored_model) == model_key
    )


def _variance_summaries_for_model(
    results: Mapping[str, Mapping[str, object]],
    data: pd.DataFrame,
    model_key: str,
) -> pd.DataFrame:
    from webapp.core.conditions import split_condition_frame

    groups = split_condition_frame(data, donor_aware=True)
    summaries = []
    for condition, group in groups.items():
        result = _result_for_model(results, condition, model_key)
        posterior = _posterior(getattr(result, "idata"))
        if "sigma_lambda_donor" not in posterior:
            continue
        donor_labels = tuple(map(str, getattr(result, "donor_labels", ())))
        if not donor_labels:
            donor_dimension = _donor_dimension(
                posterior["mu_lambda_donor"],
                "mu_lambda_donor",
            )
            donor_labels = tuple(
                map(str, np.asarray(posterior.coords[donor_dimension]).tolist())
            )
        cell_counts = (
            group["donor_id"]
            .astype(str)
            .value_counts(sort=False)
            .reindex(donor_labels)
        )
        if cell_counts.isna().any():
            raise ValueError(
                f"condition {condition!r} does not contain every inferred donor label"
            )
        decomposition = decompose_population_heterogeneity(
            getattr(result, "idata"),
            cell_counts.to_numpy(dtype=float),
        )
        summary = variance_decomposition_summary(decomposition)
        summary.insert(0, "condition", condition)
        summaries.append(summary)
    if not summaries:
        return pd.DataFrame()
    return pd.concat(summaries, ignore_index=True)


def _donor_model_panel(
    *,
    dcc: Any,
    html: Any,
    results: Mapping[str, Mapping[str, object]],
    data: pd.DataFrame,
    model_key: str,
    condition_colours: Mapping[str, str],
    prefix: str,
) -> Any:
    population = condition_results_population_frame(results, model=model_key)
    donors = condition_results_donor_frame(results, model=model_key)
    conditions = list(dict.fromkeys(donors["condition"].astype(str)))
    donor_ids = list(dict.fromkeys(donors["donor_id"].astype(str)))
    donor_colours = {
        donor: DONOR_COLOURS[index % len(DONOR_COLOURS)]
        for index, donor in enumerate(donor_ids)
    }

    population_figure = population_joint_posterior_figure(
        population,
        title=f"{MODEL_SHORT_LABELS[model_key]} · Population mixture moments",
        condition_colours=condition_colours,
    )
    initial_condition = conditions[0]
    within_condition_figure = donor_joint_posterior_figure(
        donors.loc[donors["condition"] == initial_condition],
        group_by="donor_id",
        title=f"{initial_condition} · Donor posteriors",
        colours=donor_colours,
    )
    initial_donor = donor_ids[0]
    across_condition_figure = donor_joint_posterior_figure(
        donors.loc[donors["donor_id"] == initial_donor],
        group_by="condition",
        title=f"{initial_donor} · Independent inference by condition",
        colours=condition_colours,
    )

    variance_summary = _variance_summaries_for_model(results, data, model_key)
    variance_content: list[Any] = []
    if not variance_summary.empty:
        variance_figure = variance_decomposition_figure(
            variance_summary,
            category_column="condition",
            title="Population heterogeneity within and between donors",
        )
        variance_content = [
            html.H4("Where population heterogeneity comes from"),
            html.P(
                "The total population variance is split into within-donor and "
                "between-donor components for every posterior draw. Bars show "
                "posterior means; zero-inflated models use the inferred active-cell "
                "weights.",
                className="orca-help",
            ),
            _graph_component(
                dcc,
                variance_figure,
                f"orca_{model_key}_variance_decomposition",
                "orca-variance-decomposition-plot",
            ),
        ]

    payload = {
        "frame": donors.to_json(orient="split", double_precision=15),
        "condition_colours": {str(key): str(value) for key, value in condition_colours.items()},
        "donor_colours": donor_colours,
    }
    return html.Div(
        [
            html.H3(MODEL_LABELS[model_key]),
            *variance_content,
            html.H4("Population posterior by condition"),
            html.P(
                "Each curve is the population mixture moment from one independently "
                "condition inferred independently. These are not the shared reference parameters of "
                "the donor hierarchy.",
                className="orca-help",
            ),
            _graph_component(
                dcc,
                population_figure,
                f"orca_{model_key}_population_conditions",
                "orca-joint-posterior-plot",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Compare donors within one condition"),
                            html.P(
                                "Mean, heterogeneity and fraction of nonengaging cells "
                                "remain paired by SMC chain and draw within this inference run.",
                                className="orca-help",
                            ),
                            dcc.Dropdown(
                                id={
                                    "type": f"{prefix}-donor-condition-select",
                                    "index": model_key,
                                },
                                options=[
                                    {"label": condition, "value": condition}
                                    for condition in conditions
                                ],
                                value=initial_condition,
                                clearable=False,
                            ),
                            dcc.Graph(
                                id={
                                    "type": f"{prefix}-donor-condition-graph",
                                    "index": model_key,
                                },
                                figure=within_condition_figure,
                                config={"displaylogo": False, "responsive": True},
                                responsive=True,
                                className="orca-joint-posterior-plot",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.H4("Compare conditions for one donor"),
                            html.P(
                                "Conditions are independent posterior distributions. "
                                "The plot overlays them but never pairs their chain or "
                                "draw indices.",
                                className="orca-help",
                            ),
                            dcc.Dropdown(
                                id={
                                    "type": f"{prefix}-donor-id-select",
                                    "index": model_key,
                                },
                                options=[
                                    {"label": donor, "value": donor}
                                    for donor in donor_ids
                                ],
                                value=initial_donor,
                                clearable=False,
                            ),
                            dcc.Graph(
                                id={
                                    "type": f"{prefix}-donor-id-graph",
                                    "index": model_key,
                                },
                                figure=across_condition_figure,
                                config={"displaylogo": False, "responsive": True},
                                responsive=True,
                                className="orca-joint-posterior-plot",
                            ),
                        ]
                    ),
                ],
                className="orca-form-grid two",
            ),
            dcc.Store(
                id={
                    "type": f"{prefix}-donor-draw-store",
                    "index": model_key,
                },
                data=payload,
            ),
        ],
        id={"type": f"{prefix}-model-panel", "index": model_key},
        className="orca-model-result-panel",
    )


def render_donor_condition_results(
    results: Mapping[str, Mapping[str, object]],
    *,
    data: pd.DataFrame,
    observation_time: float,
    settings: object,
    condition_colours: Mapping[str, str],
    prefix: str,
) -> tuple[Any, Any]:
    """Render the donor-aware multi-condition report used by the Dash page.

    This is the only Dash-specific entry point in the module.  All numerical
    transformations and figures above remain independently testable.
    """

    import base64

    from dash import dcc, html

    from webapp.analysis_ui import csv_download_link
    from webapp.condition_reporting import condition_bayes_factor_figure
    from webapp.core.condition_inference import build_condition_results_zip
    from webapp.donor_interactive import donor_contrast_section
    from webapp.ui import note

    conditions, fitted_models = _validated_condition_results(results)
    evidence_source = condition_results_log_evidence(results)
    evidence = condition_model_evidence(evidence_source).round(6)
    condition_figure = condition_bayes_factor_figure(results)
    aggregate = aggregate_model_evidence(
        evidence_source,
        conditions=conditions,
    ).round(6)
    aggregate_figure = bayes_factor_figure(
        aggregate,
        title="Evidence combined across independent conditions",
    )
    model_panels = [
        _donor_model_panel(
            dcc=dcc,
            html=html,
            results=results,
            data=data,
            model_key=model_key,
            condition_colours=condition_colours,
            prefix=prefix,
        )
        for model_key in fitted_models
    ]

    archive = build_condition_results_zip(
        results,
        data,
        observation_time,
        settings,
        donor_aware=True,
    )
    encoded_archive = base64.b64encode(archive).decode("ascii")
    download = html.A(
        "Download all results and InferenceData files",
        href=f"data:application/zip;base64,{encoded_archive}",
        download="orca_donor_aware_condition_analysis.zip",
        className="orca-button primary download",
    )

    content = html.Div(
        [
            note(
                "Inference complete",
                "Inference was run independently for each condition with the same donor aware "
                "models and prior settings.",
                tone="teal",
            ),
            html.Section(
                [
                    html.Span("Model evidence", className="orca-section-label"),
                    html.H3("Bayes factors by experimental condition"),
                    html.P(
                        "Every condition has its own best model. The horizontal axis "
                        "uses the unmodified log₁₀ BF(best model / candidate model) "
                        "scale, with exact boundaries at log₁₀(3), 1 and 2.",
                        className="orca-help",
                    ),
                    _graph_component(
                        dcc,
                        condition_figure,
                        "orca_donor_condition_bayes_factors",
                        "orca-bayes-factor-plot",
                    ),
                    csv_download_link(
                        evidence,
                        "orca_donor_condition_model_evidence.csv",
                        "Download condition Bayes factor CSV",
                    ),
                    html.H3("Evidence combined across conditions"),
                    html.P(
                        "Because inference was run independently for each condition, their log "
                        "marginal likelihoods add. Aggregation is performed only after "
                        "checking that every model has one result for every condition.",
                        className="orca-help",
                    ),
                    _graph_component(
                        dcc,
                        aggregate_figure,
                        "orca_donor_aggregate_bayes_factors",
                        "orca-bayes-factor-plot",
                    ),
                    csv_download_link(
                        aggregate,
                        "orca_donor_aggregate_model_evidence.csv",
                        "Download aggregate Bayes factor CSV",
                    ),
                ],
                className="orca-result-section orca-figure-result",
            ),
            html.Section(
                [
                    html.Span("Posterior results", className="orca-section-label"),
                    html.H3("Choose inference results to visualise"),
                    dcc.Checklist(
                        id=f"{prefix}-model-view",
                        options=[
                            {"label": MODEL_LABELS[model_key], "value": model_key}
                            for model_key in fitted_models
                        ],
                        value=fitted_models,
                        inline=True,
                        className="orca-posterior-model-options",
                        inputClassName="orca-check-input",
                        labelClassName="orca-posterior-model-option",
                    ),
                    html.P(
                        "Each model panel reports population mixture moments first, "
                        "then donor posteriors. Parameters absent from a model are not "
                        "shown.",
                        className="orca-help",
                    ),
                    html.Div(model_panels, className="orca-condition-model-panels"),
                ],
                className="orca-result-section",
            ),
            donor_contrast_section(results, prefix=prefix),
        ],
        className="orca-results orca-condition-results orca-donor-results",
    )
    return content, download


def register_donor_reporting_callbacks(app: Any, *, prefix: str) -> None:
    """Register the two lightweight selectors inside donor result panels."""

    from dash import MATCH, Input, Output, State

    @app.callback(
        Output(
            {"type": f"{prefix}-donor-condition-graph", "index": MATCH},
            "figure",
        ),
        Input(
            {"type": f"{prefix}-donor-condition-select", "index": MATCH},
            "value",
        ),
        State(
            {"type": f"{prefix}-donor-draw-store", "index": MATCH},
            "data",
        ),
        prevent_initial_call=True,
    )
    def update_donors_within_condition(condition: object, payload: Mapping[str, object]):
        frame = pd.read_json(StringIO(str(payload["frame"])), orient="split")
        condition = str(condition)
        subset = frame.loc[frame["condition"].astype(str) == condition]
        return donor_joint_posterior_figure(
            subset,
            group_by="donor_id",
            title=f"{condition} · Donor posteriors",
            colours=payload.get("donor_colours", {}),
        )

    @app.callback(
        Output(
            {"type": f"{prefix}-donor-id-graph", "index": MATCH},
            "figure",
        ),
        Input(
            {"type": f"{prefix}-donor-id-select", "index": MATCH},
            "value",
        ),
        State(
            {"type": f"{prefix}-donor-draw-store", "index": MATCH},
            "data",
        ),
        prevent_initial_call=True,
    )
    def update_conditions_for_donor(donor_id: object, payload: Mapping[str, object]):
        frame = pd.read_json(StringIO(str(payload["frame"])), orient="split")
        donor_id = str(donor_id)
        subset = frame.loc[frame["donor_id"].astype(str) == donor_id]
        return donor_joint_posterior_figure(
            subset,
            group_by="condition",
            title=f"{donor_id} · Independent inference by condition",
            colours=payload.get("condition_colours", {}),
        )

__all__ = [
    "BF3_LOG10",
    "DONOR_PARAMETERS",
    "MODEL_LABELS",
    "POPULATION_PARAMETERS",
    "aggregate_model_evidence",
    "bayes_factor_figure",
    "canonical_model_name",
    "cartesian_contrast_draws",
    "condition_control_contrast_frame",
    "condition_model_evidence",
    "condition_results_donor_frame",
    "condition_results_log_evidence",
    "condition_results_population_frame",
    "contrast_joint_posterior_figure",
    "decompose_population_heterogeneity",
    "donor_joint_posterior_figure",
    "donor_frame_condition_contrasts",
    "donor_frame_condition_contrast_figure",
    "donor_posterior_frame",
    "joint_posterior_figure",
    "population_joint_posterior_figure",
    "population_posterior_frame",
    "register_donor_reporting_callbacks",
    "render_donor_condition_results",
    "summarize_contrast_draws",
    "variance_decomposition_figure",
    "variance_decomposition_summary",
]
