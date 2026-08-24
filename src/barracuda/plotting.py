"""Matplotlib plots for BARRACUDA's tidy result tables.

Matplotlib is imported only when a plotting function is called, so simulation
and inference users avoid its import-time cost.  Every function returns an
``Axes`` object and never calls :func:`matplotlib.pyplot.show`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes


MODEL_COLOURS: Final[dict[str, str]] = {
    "homo": "#5C677D",
    "z2p": "#4C78A8",
    "dis2p": "#59A14F",
    "hetero3": "#E45756",
    "homogeneous_history_independent": "#5C677D",
    "homogeneous_history_dependent": "#4C78A8",
    "heterogeneous_history_independent": "#59A14F",
    "heterogeneous_history_dependent": "#E45756",
}
BF_THRESHOLDS_LOG10: Final[tuple[float, float, float]] = (
    float(np.log10(3.0)),
    1.0,
    2.0,
)


def _matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise ImportError(
            "Matplotlib is a required barracuda dependency; reinstall barracuda"
        ) from exc
    return plt


def _axes(ax: Any = None, *, figsize: tuple[float, float] = (7.2, 4.4)):
    plt = _matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    return ax


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")


def _model_column(frame: pd.DataFrame) -> str:
    for column in ("model_key", "model"):
        if column in frame.columns:
            return column
    raise ValueError("table must contain a model_key or model column")


def _colour(label: Any, index: int = 0) -> str:
    key = str(label)
    if key in MODEL_COLOURS:
        return MODEL_COLOURS[key]
    fallback = ("#4C78A8", "#F28E2B", "#59A14F", "#E15759", "#B279A2")
    return fallback[index % len(fallback)]


def plot_event_count_distribution(
    frame: pd.DataFrame,
    *,
    condition_column: str = "condition",
    normalize: bool = False,
    ax: Any = None,
) -> "Axes":
    """Plot an empirical count distribution, optionally split by condition."""

    _require_columns(frame, ("count",), "frame")
    if frame.empty:
        raise ValueError("frame must contain at least one row")
    counts = pd.to_numeric(frame["count"], errors="raise")
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts % 1 != 0).any():
        raise ValueError("count must contain finite non-negative integers")
    axis = _axes(ax)
    if condition_column in frame.columns:
        groups = list(frame.assign(count=counts.astype(int)).groupby(
            condition_column, sort=False, observed=True
        ))
    else:
        groups = [("All cells", frame.assign(count=counts.astype(int)))]
    maximum = int(counts.max())
    x = np.arange(maximum + 1)
    width = min(0.82 / max(len(groups), 1), 0.82)
    offsets = (np.arange(len(groups)) - (len(groups) - 1) / 2.0) * width
    for index, ((condition, group), offset) in enumerate(zip(groups, offsets)):
        frequencies = group["count"].value_counts().reindex(x, fill_value=0).to_numpy(float)
        if normalize:
            frequencies /= frequencies.sum()
        axis.bar(
            x + offset,
            frequencies,
            width=width,
            label=str(condition),
            color=_colour(condition, index),
            edgecolor="white",
            linewidth=0.5,
        )
    axis.set_xlabel("Events per cell")
    axis.set_ylabel("Proportion of cells" if normalize else "Number of cells")
    axis.set_xticks(x)
    if len(groups) > 1:
        axis.legend(frameon=False, title="Condition")
    return axis


def plot_rate_distribution(
    rate_distribution: str,
    mu_lambda: float,
    sigma_lambda: float,
    *,
    points: int = 320,
    ax: Any = None,
) -> "Axes":
    """Plot the engaging-cell rate law used by an event-count simulation."""

    from .event_counts import rate_distribution_curve

    x, density = rate_distribution_curve(
        rate_distribution,
        mu_lambda,
        sigma_lambda,
        points=points,
    )
    axis = _axes(ax)
    if x.size == 1:
        axis.axvline(float(x[0]), color="#4C78A8", linewidth=2.2)
    else:
        axis.plot(x, density, color="#4C78A8", linewidth=2.2)
        axis.fill_between(x, density, color="#4C78A8", alpha=0.18)
    axis.set_xlabel("Engaging-cell event rate, λ")
    axis.set_ylabel("Density")
    axis.set_title(str(rate_distribution).replace("_", " ").title())
    return axis


def plot_model_evidence(
    evidence: pd.DataFrame,
    *,
    condition: str | None = None,
    ax: Any = None,
    title: str | None = None,
) -> "Axes":
    """Plot positive ``log10 BF(best / model)`` from any BARRACUDA evidence table."""

    if not isinstance(evidence, pd.DataFrame) or evidence.empty:
        raise ValueError("evidence must be a non-empty pandas DataFrame")
    table = evidence.copy()
    if condition is not None:
        if "condition" not in table:
            raise ValueError("condition was supplied but the table has no condition column")
        table = table.loc[table["condition"].astype(str) == str(condition)].copy()
        if table.empty:
            raise ValueError(f"condition {condition!r} is not present")
    if "condition" in table and table["condition"].nunique(dropna=False) > 1:
        raise ValueError("select one condition before plotting model evidence")
    model_column = _model_column(table)
    if "log10_BF_best_vs_model" in table:
        values = pd.to_numeric(table["log10_BF_best_vs_model"], errors="raise")
    elif "log10_bf_best_vs_model" in table:
        values = pd.to_numeric(table["log10_bf_best_vs_model"], errors="raise")
    elif "log10_BF_model_vs_best" in table:
        values = -pd.to_numeric(table["log10_BF_model_vs_best"], errors="raise")
    elif "log10_bf_model_vs_best" in table:
        values = -pd.to_numeric(table["log10_bf_model_vs_best"], errors="raise")
    else:
        raise ValueError("evidence contains no recognized log10 Bayes-factor column")
    values = values.to_numpy(float)
    if not np.isfinite(values).all() or np.any(values < -1e-10):
        raise ValueError("best-vs-model log10 Bayes factors must be finite and non-negative")
    values = np.maximum(values, 0.0)
    table = table.assign(_value=values).sort_values(
        "_value", ascending=True, kind="stable"
    )
    axis = _axes(ax, figsize=(7.4, max(3.4, 0.58 * len(table) + 1.8)))
    labels = table[model_column].astype(str).tolist()
    colours = [_colour(label, index) for index, label in enumerate(labels)]
    axis.barh(labels, table["_value"], color=colours, alpha=0.92)
    maximum = max(float(table["_value"].max()), BF_THRESHOLDS_LOG10[0])
    for boundary, label in zip(
        BF_THRESHOLDS_LOG10,
        ("BF=3", "BF=10", "BF=100"),
    ):
        if boundary <= maximum * 1.12 + 1e-12:
            axis.axvline(boundary, color="#777777", linestyle=":", linewidth=1)
            axis.text(
                boundary,
                1.01,
                label,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#555555",
            )
    axis.set_xlabel("log₁₀ BF(best model / candidate model)")
    axis.set_ylabel("")
    if title:
        axis.set_title(title)
    return axis


def _scan_value_column(frame: pd.DataFrame) -> str:
    candidates = (
        "log10_bf_true_vs_model",
        "log10_BF_true_vs_model",
        "log10_bf_model_vs_true",
        "log10_BF_model_vs_true",
        "log10_bf_reference_vs_model",
        "log10_BF_best_vs_model",
        "log10_bf_best_vs_model",
    )
    for column in candidates:
        if column in frame:
            return column
    raise ValueError(
        "scan contains no recognized model/true/reference/best log10 BF column"
    )


def plot_bayes_factor_scan(
    scan: pd.DataFrame,
    *,
    scenario: str | None = None,
    model_keys: Sequence[str] | None = None,
    interval: float = 0.90,
    value_column: str | None = None,
    ax: Any = None,
) -> "Axes":
    """Plot median Bayes-factor trajectories and replicate intervals.

    Sample sizes are cumulative prefixes within each scenario/replicate, as in
    the package scan runners.  The shaded interval therefore describes
    between-replicate variation, not independent samples at adjacent sizes.
    """

    _require_columns(scan, ("n_cells",), "scan")
    if scan.empty:
        raise ValueError("scan must contain at least one row")
    probability = float(interval)
    if not np.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("interval must be between zero and one")
    table = scan.copy()
    if scenario is not None:
        if "scenario" not in table:
            raise ValueError("scenario was supplied but scan has no scenario column")
        table = table.loc[table["scenario"].astype(str) == str(scenario)].copy()
        if table.empty:
            raise ValueError(f"scenario {scenario!r} is not present")
    if "scenario" in table and table["scenario"].nunique(dropna=False) > 1:
        raise ValueError("select one scenario before plotting a Bayes-factor scan")
    model_column = _model_column(table)
    if model_keys is not None:
        requested = {str(model) for model in model_keys}
        table = table.loc[table[model_column].astype(str).isin(requested)].copy()
        missing = requested.difference(table[model_column].astype(str))
        if missing:
            raise ValueError("model keys not present: " + ", ".join(sorted(missing)))
    value = value_column or _scan_value_column(table)
    if value not in table:
        raise ValueError(f"value column {value!r} is not present")
    table[value] = pd.to_numeric(table[value], errors="raise")
    table["n_cells"] = pd.to_numeric(table["n_cells"], errors="raise")
    if not np.isfinite(table[[value, "n_cells"]].to_numpy(float)).all():
        raise ValueError("sample sizes and Bayes factors must be finite")
    alpha = (1.0 - probability) / 2.0
    summary = (
        table.groupby([model_column, "n_cells"], sort=False, observed=True)[value]
        .agg(
            median="median",
            lower=lambda series: series.quantile(alpha),
            upper=lambda series: series.quantile(1.0 - alpha),
            n_replicates="size",
        )
        .reset_index()
    )
    axis = _axes(ax)
    for index, (model, group) in enumerate(
        summary.groupby(model_column, sort=False, observed=True)
    ):
        group = group.sort_values("n_cells", kind="stable")
        colour = _colour(model, index)
        x = group["n_cells"].to_numpy(float)
        axis.plot(
            x,
            group["median"].to_numpy(float),
            marker="o",
            linewidth=1.8,
            color=colour,
            label=str(model),
        )
        axis.fill_between(
            x,
            group["lower"].to_numpy(float),
            group["upper"].to_numpy(float),
            color=colour,
            alpha=0.16,
            linewidth=0,
        )
    axis.axhline(0.0, color="#555555", linewidth=1)
    axis.set_xlabel("Cumulative number of cells")
    axis.set_ylabel(value.replace("_", " "))
    axis.legend(frameon=False, title="Model")
    return axis


def plot_parameter_recovery(
    recovery: pd.DataFrame,
    *,
    parameter: str | None = None,
    model_key: str | None = None,
    mean_column: str | None = None,
    ax: Any = None,
) -> "Axes":
    """Plot posterior means and HDIs against generating truth."""

    estimate_column = mean_column or (
        "mean" if "mean" in recovery.columns else "posterior_mean"
    )
    required = ("parameter", "truth", estimate_column, "hdi_lower", "hdi_upper")
    _require_columns(recovery, required, "recovery")
    table = recovery.copy()
    if parameter is not None:
        table = table.loc[table["parameter"].astype(str) == str(parameter)].copy()
    elif table["parameter"].nunique(dropna=False) > 1:
        raise ValueError("select one parameter before plotting recovery")
    if model_key is not None:
        if "model_key" not in table:
            raise ValueError("model_key was supplied but recovery has no model_key column")
        table = table.loc[table["model_key"].astype(str) == str(model_key)].copy()
    if table.empty:
        raise ValueError("no recovery rows match the requested selection")
    numeric = table.loc[:, ["truth", estimate_column, "hdi_lower", "hdi_upper"]].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("recovery estimates must be finite")
    axis = _axes(ax)
    labels = (
        table["model_key"].astype(str)
        if "model_key" in table
        else pd.Series([f"run {index + 1}" for index in range(len(table))])
    )
    x = np.arange(len(table), dtype=float)
    means = numeric[estimate_column].to_numpy(float)
    lower = numeric["hdi_lower"].to_numpy(float)
    upper = numeric["hdi_upper"].to_numpy(float)
    truth = numeric["truth"].to_numpy(float)
    axis.errorbar(
        x,
        means,
        yerr=np.vstack([means - lower, upper - means]),
        fmt="o",
        color="#4C78A8",
        ecolor="#4C78A8",
        capsize=4,
        label="Posterior mean and HDI",
    )
    axis.scatter(x, truth, marker="x", s=55, color="#E45756", label="Truth", zorder=3)
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_ylabel(str(table["parameter"].iloc[0]))
    axis.legend(frameon=False)
    return axis


def plot_posterior_intervals(
    summary: pd.DataFrame,
    *,
    parameter_column: str = "parameter",
    mean_column: str = "mean",
    lower_column: str = "hdi_lower",
    upper_column: str = "hdi_upper",
    ax: Any = None,
) -> "Axes":
    """Plot any tidy posterior interval table as horizontal error bars."""

    required = (parameter_column, mean_column, lower_column, upper_column)
    _require_columns(summary, required, "summary")
    if summary.empty:
        raise ValueError("summary must contain at least one row")
    table = summary.loc[:, required].copy()
    numeric = table.loc[:, [mean_column, lower_column, upper_column]].apply(
        pd.to_numeric, errors="raise"
    )
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("posterior intervals must be finite")
    means = numeric[mean_column].to_numpy(float)
    lower = numeric[lower_column].to_numpy(float)
    upper = numeric[upper_column].to_numpy(float)
    if np.any(lower > means) or np.any(means > upper):
        raise ValueError("posterior means must lie within the supplied intervals")
    y = np.arange(len(table))
    axis = _axes(ax, figsize=(7.0, max(3.2, 0.45 * len(table) + 1.5)))
    axis.errorbar(
        means,
        y,
        xerr=np.vstack([means - lower, upper - means]),
        fmt="o",
        color="#4C78A8",
        ecolor="#4C78A8",
        capsize=3,
    )
    axis.set_yticks(y, table[parameter_column].astype(str))
    axis.invert_yaxis()
    axis.set_xlabel("Posterior estimate")
    return axis


def plot_trajectory_state_map(
    frame: pd.DataFrame,
    *,
    condition: str | None = None,
    ax: Any = None,
) -> "Axes":
    """Plot empirical lethal probability at each pre-contact history state."""

    summary_columns = {
        "condition",
        "x_before",
        "y_before",
        "n_contacts",
        "empirical_lethal_probability",
    }
    if summary_columns.issubset(frame.columns):
        summary = frame.copy()
    else:
        from .diagnostics import trajectory_state_summary

        summary = trajectory_state_summary(frame)
    if condition is not None:
        summary = summary.loc[
            summary["condition"].astype(str) == str(condition)
        ].copy()
    elif summary["condition"].nunique(dropna=False) > 1:
        raise ValueError("select one condition before plotting a state map")
    if summary.empty:
        raise ValueError("the selected trajectories contain no contact events")
    axis = _axes(ax, figsize=(6.2, 5.3))
    sizes = 45.0 + 45.0 * np.log1p(summary["n_contacts"].to_numpy(float))
    points = axis.scatter(
        summary["x_before"],
        summary["y_before"],
        c=summary["empirical_lethal_probability"],
        s=sizes,
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        edgecolor="#333333",
        linewidth=0.6,
        zorder=3,
    )
    for row in summary.itertuples(index=False):
        probability = float(row.empirical_lethal_probability)
        axis.annotate(
            "",
            xy=(float(row.x_before) + 0.72, float(row.y_before)),
            xytext=(float(row.x_before) + 0.12, float(row.y_before)),
            arrowprops={"arrowstyle": "->", "alpha": max(0.12, 1.0 - probability), "color": "#4C78A8"},
        )
        axis.annotate(
            "",
            xy=(float(row.x_before), float(row.y_before) + 0.72),
            xytext=(float(row.x_before), float(row.y_before) + 0.12),
            arrowprops={"arrowstyle": "->", "alpha": max(0.12, probability), "color": "#E45756"},
        )
    axis.figure.colorbar(points, ax=axis, label="Empirical lethal probability")
    axis.set_xlabel("Previous non-lethal contacts")
    axis.set_ylabel("Previous lethal contacts")
    axis.set_aspect("equal", adjustable="datalim")
    return axis


def plot_posterior_pair(
    draws: pd.DataFrame,
    x: str,
    y: str,
    *,
    group: str | None = "model_key",
    max_points_per_group: int | None = 5_000,
    seed: int | None = 17,
    ax: Any = None,
) -> "Axes":
    """Plot paired posterior draws without breaking joint dependence."""

    _require_columns(draws, (x, y), "draws")
    if max_points_per_group is not None and int(max_points_per_group) <= 0:
        raise ValueError("max_points_per_group must be positive or None")
    table = draws.copy()
    table[x] = pd.to_numeric(table[x], errors="coerce")
    table[y] = pd.to_numeric(table[y], errors="coerce")
    table = table.loc[np.isfinite(table[x]) & np.isfinite(table[y])]
    if table.empty:
        raise ValueError("draws contain no finite paired values")
    if group is not None and group not in table:
        raise ValueError(f"group column {group!r} is not present")
    grouped = [("Posterior", table)] if group is None else list(
        table.groupby(group, sort=False, observed=True)
    )
    rng = np.random.default_rng(seed)
    axis = _axes(ax)
    for index, (label, subset) in enumerate(grouped):
        if max_points_per_group is not None and len(subset) > int(max_points_per_group):
            positions = np.sort(
                rng.choice(len(subset), size=int(max_points_per_group), replace=False)
            )
            subset = subset.iloc[positions]
        axis.scatter(
            subset[x],
            subset[y],
            s=10,
            alpha=0.22,
            color=_colour(label, index),
            label=str(label),
            rasterized=True,
        )
    axis.set_xlabel(x)
    axis.set_ylabel(y)
    if len(grouped) > 1:
        axis.legend(frameon=False)
    return axis


__all__ = [
    "BF_THRESHOLDS_LOG10",
    "MODEL_COLOURS",
    "plot_bayes_factor_scan",
    "plot_event_count_distribution",
    "plot_model_evidence",
    "plot_parameter_recovery",
    "plot_posterior_intervals",
    "plot_posterior_pair",
    "plot_rate_distribution",
    "plot_trajectory_state_map",
]
