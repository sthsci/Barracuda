"""Pure figures and Dash result components for contact-trajectory inference.

The functions in this module deliberately consume ordinary data frames.  They
do not import the Section 3 inference backend, so report rendering remains
testable while inference adapters evolve and can safely run in a Dash
background worker.
"""

from __future__ import annotations

from ast import literal_eval
import base64
from collections.abc import Mapping, Sequence
from html import escape
import math
from typing import Final

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from dash import dcc, html


INK: Final[str] = "#25231F"
PAPER: Final[str] = "#F3EDDF"
SHEET: Final[str] = "#FBF7ED"
RULE: Final[str] = "#887B66"
GRID: Final[str] = "#D6CCBA"
TRUTH: Final[str] = "#9A4938"
SERIF: Final[str] = (
    "Iowan Old Style, Baskerville, Palatino Linotype, Palatino, Georgia, serif"
)
BF3_LOG10: Final[float] = float(np.log10(3.0))

TRAJECTORY_MODEL_LABELS: Final[dict[str, str]] = {
    "homogeneous_history_independent": (
        "𝓜_Hom-HI · Homogeneous, history independent"
    ),
    "homogeneous_history_dependent": (
        "𝓜_Hom-HD · Homogeneous, history dependent"
    ),
    "heterogeneous_history_independent": (
        "𝓜_Het-HI · Heterogeneous, history independent"
    ),
    "heterogeneous_history_dependent": (
        "𝓜_Het-HD · Heterogeneous, history dependent"
    ),
}

PARAMETER_LABELS: Final[dict[str, str]] = {
    "mu_lambda": "Mean contact rate, μλ",
    "sigma_lambda": "Contact-rate SD, σλ",
    "mu_eta": "Mean baseline killing propensity, μη",
    "mu_p0": "Central baseline lethal probability, logit⁻¹(μη)",
    "p0_centre": "Central baseline lethal probability, p₀",
    "sigma_eta": "Cell-to-cell heterogeneity in baseline killing propensity, ση",
    "beta_f": "Previous non-lethal contact effect, βf",
    "beta_s": "Previous lethal contact effect, βs",
    "beta_x": "Previous non-lethal contact effect, βf",
    "beta_y": "Previous lethal contact effect, βs",
    "odds_ratio_x": "Previous non-lethal contact odds ratio, exp(βf)",
    "odds_ratio_y": "Previous lethal contact odds ratio, exp(βs)",
}

PARAMETER_AXIS_LABELS: Final[dict[str, str]] = {
    "mu_lambda": "Mean contact rate<br>μλ",
    "sigma_lambda": "Contact-rate SD<br>σλ",
    "mu_eta": "Mean baseline killing<br>propensity μη",
    "mu_p0": "Central baseline lethal<br>probability",
    "p0_centre": "Central baseline lethal<br>probability p₀",
    "sigma_eta": "Baseline killing-propensity<br>heterogeneity ση",
    "beta_f": "Previous non-lethal<br>effect βf",
    "beta_s": "Previous lethal<br>effect βs",
    "beta_x": "Previous non-lethal<br>effect βf",
    "beta_y": "Previous lethal<br>effect βs",
    "odds_ratio_x": "Previous non-lethal<br>odds ratio",
    "odds_ratio_y": "Previous lethal<br>odds ratio",
}

MODEL_PARAMETERS: Final[dict[str, tuple[str, ...]]] = {
    "homogeneous_history_independent": (
        "mu_lambda",
        "sigma_lambda",
        "mu_eta",
    ),
    "homogeneous_history_dependent": (
        "mu_lambda",
        "sigma_lambda",
        "mu_eta",
        "beta_f",
        "beta_s",
    ),
    "heterogeneous_history_independent": (
        "mu_lambda",
        "sigma_lambda",
        "mu_eta",
        "sigma_eta",
    ),
    "heterogeneous_history_dependent": (
        "mu_lambda",
        "sigma_lambda",
        "mu_eta",
        "sigma_eta",
        "beta_f",
        "beta_s",
    ),
}

PROBABILITY_SCALE: Final[list[list[object]]] = [
    [0.0, "#2C7BB6"],
    [0.5, "#F7F7F7"],
    [1.0, "#D7191C"],
]
BF_BANDS: Final[tuple[tuple[str, float | None, float, str], ...]] = (
    ("Extreme", None, -2.0, "#E76F51"),
    ("Strong", -2.0, -1.0, "#FFD28A"),
    ("Moderate", -1.0, -BF3_LOG10, "#FFF3B0"),
    ("Anecdotal", -BF3_LOG10, 0.0, "#F7F7F7"),
)

_DRAW_ID_COLUMNS: Final[tuple[str, ...]] = (
    "condition",
    "model_key",
    "model",
    "chain",
    "draw",
    "posterior_draw",
)


def _normalise_history(raw: object) -> tuple[int, ...]:
    """Normalise a cell history without importing the inference backend."""

    if raw is None:
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        if not text or text.lower() in {"nan", "none", "null", "[]", "()"}:
            return ()
        compact = "".join(text.split())
        if set(compact).issubset({"0", "1"}):
            values: object = list(compact)
        else:
            try:
                values = literal_eval(text)
            except (SyntaxError, ValueError):
                stripped = text.strip("[]()")
                values = [part.strip() for part in stripped.split(",") if part.strip()]
    else:
        try:
            missing = pd.isna(raw)
            if isinstance(missing, (bool, np.bool_)) and bool(missing):
                return ()
        except (TypeError, ValueError):
            pass
        try:
            values = list(raw)  # type: ignore[arg-type]
        except TypeError:
            values = [raw]

    if isinstance(values, str) or np.isscalar(values):
        values = [values]
    output: list[int] = []
    for value in values:  # type: ignore[union-attr]
        try:
            if bool(pd.isna(value)):
                continue
        except (TypeError, ValueError):
            pass
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("each history must contain only 0 and 1") from exc
        if not numeric.is_integer() or int(numeric) not in (0, 1):
            raise ValueError("each history must contain only 0 and 1")
        output.append(int(numeric))
    return tuple(output)


def expanded_history_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Expand one-row-per-cell histories into ordered contact-level records."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("trajectory data must be a pandas DataFrame")
    required = ("cell_id", "condition", "history")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError("missing trajectory columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("trajectory data must contain at least one cell")

    keys = frame[["condition", "cell_id"]].astype("string")
    keys = keys.apply(lambda column: column.str.strip())
    if keys.isna().any().any() or (keys == "").any().any():
        raise ValueError("cell_id and condition must not be blank")
    if keys.duplicated().any():
        raise ValueError("cell_id must be unique within each condition")

    rows: list[dict[str, object]] = []
    for row_number, record in enumerate(
        frame.loc[:, required].itertuples(index=False)
    ):
        cell_id = str(keys.iloc[row_number]["cell_id"])
        condition = str(keys.iloc[row_number]["condition"])
        x_before = 0
        y_before = 0
        for contact_index, outcome in enumerate(
            _normalise_history(record.history),
            start=1,
        ):
            rows.append(
                {
                    "cell_id": cell_id,
                    "condition": condition,
                    "contact_index": contact_index,
                    "x_before": x_before,
                    "y_before": y_before,
                    "outcome": outcome,
                }
            )
            if outcome:
                y_before += 1
            else:
                x_before += 1

    return pd.DataFrame(
        rows,
        columns=[
            "cell_id",
            "condition",
            "contact_index",
            "x_before",
            "y_before",
            "outcome",
        ],
    )


def _as_expanded(frame: pd.DataFrame) -> pd.DataFrame:
    expanded_columns = [
        "cell_id",
        "condition",
        "contact_index",
        "x_before",
        "y_before",
        "outcome",
    ]
    if set(expanded_columns).issubset(frame.columns):
        output = frame.loc[:, expanded_columns].copy()
        return output
    paper_columns = {
        "previous_nonlethal_contacts": "x_before",
        "previous_lethal_contacts": "y_before",
    }
    if {
        "cell_id",
        "condition",
        "contact_index",
        "previous_nonlethal_contacts",
        "previous_lethal_contacts",
        "outcome",
    }.issubset(frame.columns):
        return frame.rename(columns=paper_columns).loc[:, expanded_columns].copy()
    return expanded_history_frame(frame)


def empirical_state_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate empirical lethal decisions at each pre-contact state."""

    summary_columns = {
        "condition",
        "x_before",
        "y_before",
        "n_cells",
        "n_contacts",
        "n_lethal",
        "empirical_lethal_probability",
    }
    if summary_columns.issubset(frame.columns):
        output = frame.copy()
        if "n_nonlethal" not in output:
            output["n_nonlethal"] = (
                output["n_contacts"].astype(int)
                - output["n_lethal"].astype(int)
            )
        if "log2_n_cells" not in output:
            output["log2_n_cells"] = np.log2(
                output["n_cells"].astype(float).clip(lower=1.0)
            )
        return output.sort_values(
            ["condition", "x_before", "y_before"],
            kind="stable",
        ).reset_index(drop=True)

    expanded = _as_expanded(frame)
    if expanded.empty:
        return pd.DataFrame(
            columns=[
                "condition",
                "x_before",
                "y_before",
                "n_cells",
                "n_contacts",
                "n_lethal",
                "n_nonlethal",
                "empirical_lethal_probability",
                "log2_n_cells",
            ]
        )
    outcomes = pd.to_numeric(expanded["outcome"], errors="raise")
    if not outcomes.isin([0, 1]).all():
        raise ValueError("outcome must contain only 0 and 1")
    expanded = expanded.assign(outcome=outcomes.astype(int))
    grouped = (
        expanded.groupby(
            ["condition", "x_before", "y_before"],
            sort=False,
            observed=True,
        )
        .agg(
            n_cells=("cell_id", "nunique"),
            n_contacts=("outcome", "size"),
            n_lethal=("outcome", "sum"),
        )
        .reset_index()
    )
    grouped["n_nonlethal"] = grouped["n_contacts"] - grouped["n_lethal"]
    grouped["empirical_lethal_probability"] = (
        grouped["n_lethal"] / grouped["n_contacts"]
    )
    grouped["log2_n_cells"] = np.log2(
        grouped["n_cells"].astype(float).clip(lower=1.0)
    )
    return grouped


def _probability_colour(probability: float) -> str:
    return str(sample_colorscale(PROBABILITY_SCALE, [float(probability)])[0])


def _state_arrow_length(
    n_cells: int,
    max_log2_cells: float,
    arrow_scale: float,
) -> float:
    """Map cell support to an arrow length using the paper's log2 scale."""

    log2_cells = math.log2(max(int(n_cells), 1))
    support = log2_cells / max(max_log2_cells, 1.0)
    return min(0.95, 0.15 + 0.62 * float(arrow_scale) * support)


def _encoding_legend_svg(
    summary: pd.DataFrame,
    *,
    arrow_scale: float,
) -> str:
    """Return the paper-style arrow length and quarter-fan legend as SVG."""

    max_cells = max(int(summary["n_cells"].max()), 1)
    powers = [2**power for power in range(int(math.floor(math.log2(max_cells))) + 1)]
    examples = list(dict.fromkeys([*powers, max_cells]))
    if len(examples) > 6:
        indices = np.linspace(0, len(examples) - 1, 6).round().astype(int)
        examples = [examples[index] for index in indices]
    max_log2_cells = max(math.log2(max_cells), 1.0)

    size_items: list[str] = []
    start_x = 18.0
    available_width = 535.0
    spacing = available_width / max(len(examples), 1)
    for index, count in enumerate(examples):
        x = start_x + index * spacing
        length = 18.0 + 58.0 * (
            (_state_arrow_length(count, max_log2_cells, arrow_scale) - 0.15)
            / 0.80
        )
        size_items.extend(
            [
                (
                    f'<line x1="{x:.1f}" y1="69" x2="{x + length:.1f}" y2="69" '
                    'stroke="#77736B" stroke-width="2.4" marker-end="url(#grey-arrow)"/>'
                ),
                (
                    f'<text x="{x + length / 2:.1f}" y="94" text-anchor="middle" '
                    f'class="value">{count}</text>'
                ),
            ]
        )

    origin_x, origin_y, radius = 690.0, 112.0, 68.0
    probabilities = np.linspace(0.0, 1.0, 6)
    angles = [math.atan2(probability, 1.0 - probability) for probability in probabilities]
    fan_items: list[str] = []
    for band_index, (lower, upper) in enumerate(
        zip(angles[:-1], angles[1:], strict=True)
    ):
        lower_x = origin_x + radius * math.cos(lower)
        lower_y = origin_y - radius * math.sin(lower)
        upper_x = origin_x + radius * math.cos(upper)
        upper_y = origin_y - radius * math.sin(upper)
        midpoint_probability = float(probabilities[band_index] + 0.1)
        colour = _probability_colour(midpoint_probability)
        fan_items.append(
            (
                f'<path d="M {origin_x:.1f},{origin_y:.1f} '
                f'L {lower_x:.1f},{lower_y:.1f} '
                f'A {radius:.1f},{radius:.1f} 0 0,0 {upper_x:.1f},{upper_y:.1f} Z" '
                f'fill="{escape(colour)}" fill-opacity="0.9" stroke="#25231F" stroke-width="0.7"/>'
            )
        )
    for probability in (0.0, 0.25, 0.5, 0.75, 1.0):
        angle = math.atan2(probability, 1.0 - probability)
        end_x = origin_x + (radius - 3) * math.cos(angle)
        end_y = origin_y - (radius - 3) * math.sin(angle)
        fan_items.append(
            (
                f'<line x1="{origin_x:.1f}" y1="{origin_y:.1f}" '
                f'x2="{end_x:.1f}" y2="{end_y:.1f}" stroke="#25231F" '
                'stroke-width="1.7" marker-end="url(#black-arrow)"/>'
            )
        )

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="900" height="132" viewBox="0 0 900 132">
      <defs>
        <marker id="grey-arrow" markerWidth="7" markerHeight="7" refX="5.8" refY="3.5" orient="auto">
          <path d="M0,0 L7,3.5 L0,7 Z" fill="#77736B"/>
        </marker>
        <marker id="black-arrow" markerWidth="7" markerHeight="7" refX="5.8" refY="3.5" orient="auto">
          <path d="M0,0 L7,3.5 L0,7 Z" fill="#25231F"/>
        </marker>
        <style>
          .title {{ font: 18px 'Iowan Old Style', Baskerville, Georgia, serif; fill: #25231F; }}
          .value {{ font: 15px 'Iowan Old Style', Baskerville, Georgia, serif; fill: #25231F; }}
          .small {{ font: 13px 'Iowan Old Style', Baskerville, Georgia, serif; fill: #25231F; }}
        </style>
      </defs>
      <text x="18" y="25" class="title">Cells reaching the state (arrow length increases with log₂ n)</text>
      {''.join(size_items)}
      <text x="610" y="25" class="title">Empirical killing probability</text>
      {''.join(fan_items)}
      <text x="766" y="119" class="value">p = 0</text>
      <text x="741" y="62" class="value">0.5</text>
      <text x="666" y="34" class="value">p = 1</text>
      <text x="602" y="128" class="small">horizontal = non-lethal · vertical = lethal</text>
    </svg>
    """
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def empirical_state_encoding_legend(
    frame: pd.DataFrame,
    *,
    arrow_scale: float = 1.0,
) -> html.Figure:
    """Render the Figure 5 direction and log2 cell-support legend."""

    summary = empirical_state_summary(frame)
    if summary.empty:
        raise ValueError("at least one observed contact is required")
    return html.Figure(
        [
            html.Img(
                src=_encoding_legend_svg(summary, arrow_scale=arrow_scale),
                alt=(
                    "Trajectory arrow legend. Arrow length increases with the log base 2 "
                    "number of cells reaching a state. Horizontal arrows represent zero "
                    "empirical killing probability, vertical arrows represent probability "
                    "one, and intermediate directions represent intermediate probabilities."
                ),
            ),
            html.Figcaption(
                "Direction encodes the empirical probability that the next contact is lethal; length encodes how many cells reached the state on a log₂ scale."
            ),
        ],
        className="barracuda-trajectory-encoding-legend",
    )


def empirical_state_arrow_figure(
    frame: pd.DataFrame,
    condition_colours: Mapping[str, str] | None = None,
    *,
    arrow_scale: float = 1.0,
    figure_height: int = 700,
) -> go.Figure:
    """Plot empirical state arrows, faceted by experimental condition.

    Arrow direction encodes the lethal/nonlethal probability balance, colour
    encodes the numerical lethal probability, and arrow length is proportional
    to ``log2(n_cells + 1)`` within a condition.
    """

    summary = empirical_state_summary(frame)
    if summary.empty:
        raise ValueError("at least one observed contact is required")
    conditions = list(dict.fromkeys(summary["condition"].astype(str)))
    columns = 1 if len(conditions) == 1 else 2
    rows = int(math.ceil(len(conditions) / columns))
    figure = make_subplots(
        rows=rows,
        cols=columns,
        subplot_titles=conditions,
        horizontal_spacing=0.12,
        vertical_spacing=0.16,
    )
    condition_colours = dict(condition_colours or {})
    del condition_colours  # condition names are already the subplot titles
    arrow_scale = float(arrow_scale)
    if not 0.4 <= arrow_scale <= 1.6:
        raise ValueError("arrow scale must be between 0.4 and 1.6")
    figure_height = int(figure_height)
    if not 480 <= figure_height <= 1200:
        raise ValueError("figure height must be between 480 and 1200 pixels")
    global_max_weight = max(math.log2(max(int(summary["n_cells"].max()), 1)), 1.0)

    for condition_number, condition in enumerate(conditions):
        row = condition_number // columns + 1
        column = condition_number % columns + 1
        condition_frame = summary.loc[
            summary["condition"].astype(str) == condition
        ]
        for state in condition_frame.itertuples(index=False):
            probability = float(state.empirical_lethal_probability)
            length = _state_arrow_length(
                int(state.n_cells),
                global_max_weight,
                arrow_scale,
            )
            direction = np.asarray([1.0 - probability, probability])
            norm = float(np.linalg.norm(direction)) or 1.0
            delta = length * direction / norm
            start_x = float(state.x_before)
            start_y = float(state.y_before)
            end_x = start_x + float(delta[0])
            end_y = start_y + float(delta[1])
            angle = 90.0 - math.degrees(math.atan2(delta[1], delta[0]))
            colour = _probability_colour(probability)
            hover = (
                f"{condition}<br>Before contact: "
                f"{int(state.x_before)} non-lethal, {int(state.y_before)} lethal"
                f"<br>Empirical next-contact killing probability = {probability:.3f}"
                f"<br>Cells at state = {int(state.n_cells)}"
                f"<br>Observed contacts = {int(state.n_contacts)}"
            )
            figure.add_trace(
                go.Scatter(
                    x=[start_x, end_x],
                    y=[start_y, end_y],
                    mode="lines+markers",
                    line={"color": colour, "width": 3.5 * arrow_scale},
                    marker={
                        "color": [colour, colour],
                        "size": [max(3.5, 5 * arrow_scale), max(8, 11 * arrow_scale)],
                        "symbol": ["circle", "arrow"],
                        "angle": [0, angle],
                        "line": {"color": INK, "width": 0.55},
                    },
                    customdata=[hover, hover],
                    hovertemplate="%{customdata}<extra></extra>",
                    showlegend=False,
                ),
                row=row,
                col=column,
            )
    axis_max = int(
        max(summary["x_before"].max(), summary["y_before"].max())
    ) + 1
    figure.update_xaxes(
        title_text="Previous non-lethal contacts, f",
        range=[-0.35, axis_max + 0.45],
        dtick=1,
        gridcolor=GRID,
        zeroline=False,
        constrain="domain",
    )
    figure.update_yaxes(
        title_text="Previous lethal contacts, s",
        range=[-0.35, axis_max + 0.45],
        dtick=1,
        gridcolor=GRID,
        zeroline=False,
    )
    for subplot_number in range(1, len(conditions) + 1):
        suffix = "" if subplot_number == 1 else str(subplot_number)
        yaxis = getattr(figure.layout, f"yaxis{suffix}")
        yaxis.update(
            scaleanchor=f"x{suffix}",
            scaleratio=1,
            constrain="domain",
        )
    figure.update_layout(
        template="none",
        height=max(figure_height, 350 * rows + 300),
        paper_bgcolor=SHEET,
        plot_bgcolor=PAPER,
        font={"family": SERIF, "color": INK, "size": 13},
        margin={"l": 82, "r": 42, "t": 70, "b": 76},
        showlegend=False,
    )
    return figure


def _normalise_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(evidence, pd.DataFrame) or evidence.empty:
        raise ValueError("model evidence must contain at least one row")
    output = evidence.copy()
    if "condition" not in output:
        output["condition"] = "Condition 1"
    if "model_key" not in output:
        if "model" not in output:
            raise ValueError("model evidence needs a model_key or model column")
        output["model_key"] = output["model"].astype(str)
    if "model" not in output:
        output["model"] = output["model_key"].map(TRAJECTORY_MODEL_LABELS).fillna(
            output["model_key"].astype(str)
        )

    if "log10_bf_model_vs_best" in output:
        values = pd.to_numeric(
            output["log10_bf_model_vs_best"], errors="raise"
        )
    elif "log10_bf_vs_best" in output:
        values = pd.to_numeric(output["log10_bf_vs_best"], errors="raise")
    elif "log10_BF_model_vs_best" in output:
        values = pd.to_numeric(
            output["log10_BF_model_vs_best"],
            errors="raise",
        )
    elif "log10_bf_best_over_model" in output:
        values = -pd.to_numeric(
            output["log10_bf_best_over_model"], errors="raise"
        )
    elif "log10_BF_best_vs_model" in output:
        values = -pd.to_numeric(
            output["log10_BF_best_vs_model"], errors="raise"
        )
    elif "log_evidence" in output:
        log_evidence = pd.to_numeric(output["log_evidence"], errors="raise")
        best = log_evidence.groupby(output["condition"], sort=False).transform(
            "max"
        )
        values = (log_evidence - best) / np.log(10.0)
    else:
        raise ValueError("model evidence needs log evidence or a log10 BF column")

    numeric = values.to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("model evidence contains non-finite values")
    if (numeric > 1e-9).any():
        raise ValueError("log10 BF(model/best) cannot be positive")
    numeric[np.isclose(numeric, 0.0, atol=1e-10)] = 0.0
    output["log10_bf_model_vs_best"] = np.minimum(numeric, 0.0)
    output["is_best"] = np.isclose(
        output["log10_bf_model_vs_best"],
        0.0,
    )
    return output


def _truth_model_for_condition(
    truth_model: str | Mapping[str, object] | None,
    condition: str,
) -> str | None:
    if truth_model is None:
        return None
    if isinstance(truth_model, Mapping):
        value = truth_model.get(condition)
        if isinstance(value, Mapping):
            value = value.get("true_model_key")
        return None if value is None else str(value)
    return str(truth_model)


def trajectory_bayes_factor_figure(
    evidence: pd.DataFrame,
    truth_model: str | Mapping[str, object] | None = None,
    condition_colours: Mapping[str, str] | None = None,
) -> go.Figure:
    """Plot exact, continuous ``log10 BF(model / best)`` model evidence."""

    table = _normalise_evidence(evidence)
    condition_colours = dict(condition_colours or {})
    table = table.sort_values(
        ["condition", "log10_bf_model_vs_best"],
        ascending=[True, True],
        kind="stable",
    ).reset_index(drop=True)
    raw = table["log10_bf_model_vs_best"].to_numpy(dtype=float)
    lower = min(-3.0, math.floor(float(raw.min()) * 2.0) / 2.0 - 0.25)

    labels: list[str] = []
    for row in table.itertuples(index=False):
        condition = str(row.condition)
        model_key = str(row.model_key)
        suffixes: list[str] = []
        if bool(row.is_best):
            suffixes.append("Best model")
        if model_key == _truth_model_for_condition(truth_model, condition):
            suffixes.append("Ground truth")
        label = f"{condition} · {row.model}"
        if suffixes:
            label += " · " + " · ".join(suffixes)
        labels.append(label)
    hover_evidence = pd.to_numeric(
        table.get("log_evidence", pd.Series(np.nan, index=table.index)),
        errors="coerce",
    ).to_numpy(dtype=float)

    figure = go.Figure()
    for band_name, start, end, colour in BF_BANDS:
        x0 = lower if start is None else start
        figure.add_vrect(
            x0=x0,
            x1=end,
            fillcolor=colour,
            opacity=0.5,
            line_width=0,
            layer="below",
        )
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "symbol": "square",
                    "size": 11,
                    "color": colour,
                    "line": {"color": RULE, "width": 0.8},
                },
                name=band_name,
                hoverinfo="skip",
            )
        )

    figure.add_trace(
        go.Bar(
            x=raw,
            y=labels,
            orientation="h",
            marker={
                "color": [
                    condition_colours.get(str(condition), "#1F78B4")
                    for condition in table["condition"]
                ],
                "line": {"color": INK, "width": 1},
            },
            text=["" if best else f"{value:.2f}" for value, best in zip(raw, table["is_best"])],
            textposition="outside",
            cliponaxis=False,
            customdata=[
                [
                    str(row.condition),
                    str(row.model_key),
                    (
                        float(log_evidence)
                        if np.isfinite(log_evidence)
                        else None
                    ),
                ]
                for row, log_evidence in zip(
                    table.itertuples(index=False),
                    hover_evidence,
                )
            ],
            hovertemplate=(
                "%{y}<br>log10 BF(model/best) = %{x:.4g}"
                "<br>log evidence = %{customdata[2]:.4g}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    best_rows = table.loc[table["is_best"]]
    best_labels = [labels[index] for index in best_rows.index]
    figure.add_trace(
        go.Scatter(
            x=np.zeros(len(best_rows)),
            y=best_labels,
            mode="markers+text",
            marker={
                "symbol": "diamond",
                "size": 11,
                "color": INK,
            },
            text=["Best model"] * len(best_rows),
            textposition="middle right",
            textfont={"family": SERIF, "size": 12, "color": INK},
            name="Best model",
            hovertemplate="%{y}<br>Best model · log10 BF = 0<extra></extra>",
            showlegend=False,
            cliponaxis=False,
        )
    )
    tick_values = sorted(
        set([lower, -3.0, -2.0, -1.0, -BF3_LOG10, 0.0])
    )
    tick_values = [value for value in tick_values if lower <= value <= 0.0]
    tick_text = [
        "−log₁₀3" if np.isclose(value, -BF3_LOG10) else f"{value:g}"
        for value in tick_values
    ]
    figure.update_layout(
        template="none",
        height=max(430, 54 * len(table) + 180),
        paper_bgcolor=SHEET,
        plot_bgcolor=PAPER,
        font={"family": SERIF, "color": INK, "size": 13},
        margin={"l": 235, "r": 120, "t": 92, "b": 82},
        xaxis_title="log₁₀ BF(𝓜 / 𝓜<sub>best</sub>)",
        yaxis_title="Condition and candidate model",
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.04,
            "xanchor": "left",
            "yanchor": "bottom",
        },
    )
    figure.update_xaxes(
        range=[lower, 0.16],
        tickmode="array",
        tickvals=tick_values,
        ticktext=tick_text,
        gridcolor=GRID,
        zeroline=True,
        zerolinecolor=INK,
        automargin=True,
    )
    figure.update_yaxes(gridcolor="rgba(0,0,0,0)", automargin=True)
    return figure


def _paired_wide_draws(draws: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(draws, pd.DataFrame) or draws.empty:
        raise ValueError("posterior draws must contain at least one row")
    frame = draws.copy()
    if "condition" not in frame:
        frame["condition"] = "Condition 1"
    if "model_key" not in frame:
        if "model" not in frame:
            raise ValueError("posterior draws need model_key or model")
        frame["model_key"] = frame["model"].astype(str)
    if {"parameter", "value"}.issubset(frame.columns):
        frame["parameter"] = frame["parameter"].replace(
            {"beta_x": "beta_f", "beta_y": "beta_s"}
        )
        identifiers = [
            column for column in _DRAW_ID_COLUMNS if column in frame.columns
        ]
        if "posterior_draw" not in identifiers and not {
            "chain",
            "draw",
        }.issubset(identifiers):
            frame["posterior_draw"] = frame.groupby(
                ["condition", "model_key", "parameter"],
                sort=False,
            ).cumcount()
            identifiers.append("posterior_draw")
        frame = (
            frame.pivot_table(
                index=identifiers,
                columns="parameter",
                values="value",
                aggfunc="first",
                sort=False,
            )
            .reset_index()
            .rename_axis(columns=None)
        )
    else:
        for backend_name, public_name in (
            ("beta_x", "beta_f"),
            ("beta_y", "beta_s"),
        ):
            if public_name not in frame and backend_name in frame:
                frame[public_name] = frame[backend_name]
    return frame


def _available_parameters(
    draws: pd.DataFrame,
    model_key: str,
    requested: Sequence[str] | None = None,
) -> list[str]:
    available = set(draws.columns)
    if requested is not None:
        candidates = [str(parameter) for parameter in requested]
    elif model_key in MODEL_PARAMETERS:
        candidates = list(MODEL_PARAMETERS[model_key])
    else:
        candidates = list(PARAMETER_LABELS)
    if "mu_eta" not in available:
        baseline_probability = next(
            (
                parameter
                for parameter in ("mu_p0", "p0_centre")
                if parameter in available
            ),
            None,
        )
        if baseline_probability is not None:
            candidates = [
                baseline_probability if parameter == "mu_eta" else parameter
                for parameter in candidates
            ]
    return list(dict.fromkeys(parameter for parameter in candidates if parameter in available))


def _truth_value(
    truth: Mapping[str, object] | None,
    parameter: str,
    condition: str | None = None,
) -> float | None:
    if not truth:
        return None
    selected: Mapping[str, object] = truth
    if condition is not None and isinstance(truth.get(condition), Mapping):
        selected = truth[condition]  # type: ignore[assignment]
    elif any(isinstance(value, Mapping) for value in truth.values()):
        nested = [value for value in truth.values() if isinstance(value, Mapping)]
        if condition is None and len(nested) == 1:
            selected = nested[0]  # type: ignore[assignment]
        else:
            return None
    aliases = {
        "mu_eta": ("mu_eta",),
        "mu_p0": ("mu_p0", "p0_centre", "p0"),
        "p0_centre": ("p0_centre", "mu_p0", "p0"),
        "beta_f": ("beta_f", "beta_x"),
        "beta_s": ("beta_s", "beta_y"),
        "beta_x": ("beta_x", "beta_f"),
        "beta_y": ("beta_y", "beta_s"),
    }
    for key in aliases.get(parameter, (parameter,)):
        if key not in selected:
            continue
        try:
            value = float(selected[key])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    if parameter == "mu_eta":
        for key in ("p0", "mu_p0", "p0_centre"):
            if key not in selected:
                continue
            try:
                probability = float(selected[key])
            except (TypeError, ValueError):
                continue
            if 0.0 < probability < 1.0:
                return float(np.log(probability / (1.0 - probability)))
    if parameter in {"mu_p0", "p0_centre"} and "mu_eta" in selected:
        try:
            eta = float(selected["mu_eta"])
        except (TypeError, ValueError):
            return None
        if np.isfinite(eta):
            return float(1.0 / (1.0 + np.exp(-eta)))
    return None


def _truth_condition_keys(
    truth: Mapping[str, object] | None,
    conditions: Sequence[str],
) -> list[str | None]:
    if truth and any(isinstance(truth.get(condition), Mapping) for condition in conditions):
        return [condition for condition in conditions if isinstance(truth.get(condition), Mapping)]
    return [None]


def _hdi_interval(
    values: Sequence[float] | np.ndarray | pd.Series,
    probability: float = 0.95,
) -> tuple[float, float]:
    samples = np.sort(np.asarray(values, dtype=float).reshape(-1))
    samples = samples[np.isfinite(samples)]
    if samples.size == 0:
        raise ValueError("an HDI needs at least one finite posterior draw")
    if samples.size == 1:
        value = float(samples[0])
        return value, value
    included = min(
        samples.size - 1,
        max(1, int(math.floor(float(probability) * samples.size))),
    )
    widths = samples[included:] - samples[: samples.size - included]
    start = int(np.argmin(widths))
    return float(samples[start]), float(samples[start + included])


def _shared_histogram_bins(
    values: Sequence[np.ndarray],
    bin_count: int = 30,
) -> dict[str, float]:
    finite_parts: list[np.ndarray] = []
    for part in values:
        array = np.asarray(part, dtype=float).reshape(-1)
        finite = array[np.isfinite(array)]
        if finite.size:
            finite_parts.append(finite)
    if not finite_parts:
        raise ValueError("histogram bins need at least one finite posterior draw")
    combined = np.concatenate(finite_parts)
    lower = float(np.min(combined))
    upper = float(np.max(combined))
    if math.isclose(lower, upper):
        padding = max(abs(lower) * 0.05, 0.5)
        lower -= padding
        upper += padding
    width = (upper - lower) / max(1, int(bin_count))
    return {"start": lower, "end": upper + width * 1e-9, "size": width}


def posterior_marginal_figure(
    draws: pd.DataFrame,
    parameter: str,
    condition_colours: Mapping[str, str],
    truth: Mapping[str, object] | None = None,
    model_key: str | None = None,
) -> go.Figure:
    """Overlay condition-specific marginal posteriors without KDE."""

    frame = _paired_wide_draws(draws)
    if model_key is not None:
        frame = frame.loc[frame["model_key"].astype(str) == str(model_key)]
    if parameter not in frame:
        raise ValueError(f"posterior parameter {parameter!r} is unavailable")
    figure = go.Figure()
    conditions = list(dict.fromkeys(frame["condition"].astype(str)))
    condition_values: dict[str, np.ndarray] = {}
    for condition in conditions:
        condition_values[condition] = (
            pd.to_numeric(
                frame.loc[
                    frame["condition"].astype(str) == condition,
                    parameter,
                ],
                errors="coerce",
            )
            .dropna()
            .to_numpy(dtype=float)
        )
    bins = _shared_histogram_bins(list(condition_values.values()))
    for condition in conditions:
        values = condition_values[condition]
        if values.size == 0:
            continue
        colour = condition_colours.get(condition, "#007AFF")
        hdi_low, hdi_high = _hdi_interval(values)
        figure.add_vrect(
            x0=hdi_low,
            x1=hdi_high,
            fillcolor=colour,
            opacity=0.09,
            line={"color": colour, "width": 1, "dash": "dot"},
            layer="below",
        )
        figure.add_trace(
            go.Histogram(
                x=values,
                histnorm="probability density",
                xbins=bins,
                bingroup=parameter,
                name=condition,
                opacity=0.28,
                marker={
                    "color": colour,
                    "line": {"color": colour, "width": 2},
                },
                hovertemplate=(
                    f"{condition}<br>%{{x:.4g}}"
                    "<br>Posterior density %{y:.4g}<extra></extra>"
                ),
            )
        )
    truth_keys = _truth_condition_keys(truth, conditions)
    truth_entries = [
        (condition, _truth_value(truth, parameter, condition))
        for condition in truth_keys
    ]
    truth_entries = [entry for entry in truth_entries if entry[1] is not None]
    for truth_number, (condition, truth_value) in enumerate(truth_entries):
        single = len(truth_entries) == 1
        label = "Ground truth" if single else f"{condition} ground truth"
        colour = (
            TRUTH
            if single or condition is None
            else condition_colours.get(str(condition), TRUTH)
        )
        figure.add_vline(
            x=truth_value,
            line={"color": colour, "width": 2, "dash": "dash"},
            annotation_text=label,
            annotation_position=(
                "top right" if truth_number % 2 == 0 else "top left"
            ),
        )
    figure.update_layout(
        template="none",
        barmode="overlay",
        height=420,
        paper_bgcolor=SHEET,
        plot_bgcolor=PAPER,
        font={"family": SERIF, "color": INK, "size": 13},
        margin={"l": 78, "r": 30, "t": 72, "b": 70},
        xaxis_title=PARAMETER_LABELS.get(parameter, parameter),
        yaxis_title="Posterior density",
        legend={"orientation": "h", "x": 0, "y": 1.12},
    )
    figure.update_xaxes(gridcolor=GRID, zeroline=False, automargin=True)
    figure.update_yaxes(gridcolor=GRID, zeroline=False, automargin=True)
    figure.add_annotation(
        text="Translucent bands show 95% HDIs",
        x=1,
        y=1.13,
        xref="paper",
        yref="paper",
        xanchor="right",
        showarrow=False,
        font={"family": SERIF, "size": 11, "color": RULE},
    )
    return figure


def joint_posterior_figure(
    draws: pd.DataFrame,
    model_key: str,
    condition_colours: Mapping[str, str],
    truth: Mapping[str, object] | None = None,
    parameters: Sequence[str] | None = None,
    *,
    max_draws_per_condition: int = 5_000,
) -> go.Figure:
    """Render the full marginal diagonal and paired lower posterior triangle."""

    frame = _paired_wide_draws(draws)
    frame = frame.loc[frame["model_key"].astype(str) == str(model_key)].copy()
    if frame.empty:
        raise ValueError(f"no posterior draws are available for {model_key!r}")
    selected = _available_parameters(frame, str(model_key), parameters)
    if not selected:
        raise ValueError("no reportable trajectory parameters are available")
    conditions = list(dict.fromkeys(frame["condition"].astype(str)))
    sampled: dict[str, pd.DataFrame] = {}
    for condition in conditions:
        group = frame.loc[frame["condition"].astype(str) == condition]
        if len(group) > max_draws_per_condition:
            group = group.sample(
                n=max_draws_per_condition,
                random_state=20260811,
                replace=False,
            ).sort_index()
        sampled[condition] = group

    size = len(selected)
    figure = make_subplots(
        rows=size,
        cols=size,
        horizontal_spacing=min(0.08, 0.25 / max(size, 1)),
        vertical_spacing=min(0.08, 0.25 / max(size, 1)),
    )
    legend_drawn: set[str] = set()
    truth_legend_drawn: set[str] = set()
    truth_keys = _truth_condition_keys(truth, conditions)
    for row_number, y_parameter in enumerate(selected, start=1):
        for column_number, x_parameter in enumerate(selected, start=1):
            if column_number > row_number:
                figure.update_xaxes(
                    visible=False,
                    row=row_number,
                    col=column_number,
                )
                figure.update_yaxes(
                    visible=False,
                    row=row_number,
                    col=column_number,
                )
                continue
            diagonal_bins: dict[str, float] | None = None
            if row_number == column_number:
                diagonal_values = [
                    pd.to_numeric(
                        condition_frame[x_parameter],
                        errors="coerce",
                    )
                    .dropna()
                    .to_numpy(dtype=float)
                    for condition_frame in sampled.values()
                ]
                diagonal_bins = _shared_histogram_bins(diagonal_values)
            for condition, condition_frame in sampled.items():
                colour = condition_colours.get(condition, "#007AFF")
                if row_number == column_number:
                    values = pd.to_numeric(
                        condition_frame[x_parameter], errors="coerce"
                    ).dropna()
                    if values.empty:
                        continue
                    hdi_low, hdi_high = _hdi_interval(values)
                    figure.add_vrect(
                        x0=hdi_low,
                        x1=hdi_high,
                        fillcolor=colour,
                        opacity=0.08,
                        line={
                            "color": colour,
                            "width": 0.8,
                            "dash": "dot",
                        },
                        layer="below",
                        row=row_number,
                        col=column_number,
                    )
                    figure.add_trace(
                        go.Histogram(
                            x=values,
                            histnorm="probability density",
                            xbins=diagonal_bins,
                            bingroup=f"{model_key}-{x_parameter}",
                            name=condition,
                            legendgroup=condition,
                            showlegend=condition not in legend_drawn,
                            opacity=0.24,
                            marker={
                                "color": colour,
                                "line": {"color": colour, "width": 1.8},
                            },
                            hovertemplate=(
                                f"{condition}<br>%{{x:.4g}}"
                                "<br>Posterior density %{y:.4g}<extra></extra>"
                            ),
                        ),
                        row=row_number,
                        col=column_number,
                    )
                    legend_drawn.add(condition)
                else:
                    paired = condition_frame[
                        [x_parameter, y_parameter]
                    ].apply(pd.to_numeric, errors="coerce").dropna()
                    if len(paired) < 4:
                        continue
                    figure.add_trace(
                        go.Histogram2dContour(
                            x=paired[x_parameter],
                            y=paired[y_parameter],
                            name=condition,
                            legendgroup=condition,
                            showlegend=False,
                            showscale=False,
                            ncontours=6,
                            contours={"coloring": "none", "showlabels": False},
                            line={"color": colour, "width": 2},
                            hoverinfo="skip",
                        ),
                        row=row_number,
                        col=column_number,
                    )

            for truth_condition in truth_keys:
                x_truth = _truth_value(
                    truth,
                    x_parameter,
                    truth_condition,
                )
                y_truth = _truth_value(
                    truth,
                    y_parameter,
                    truth_condition,
                )
                if x_truth is None:
                    continue
                single_truth = len(truth_keys) == 1
                truth_name = (
                    "Ground truth"
                    if single_truth or truth_condition is None
                    else f"{truth_condition} ground truth"
                )
                truth_colour = (
                    TRUTH
                    if single_truth or truth_condition is None
                    else condition_colours.get(str(truth_condition), TRUTH)
                )
                if row_number == column_number:
                    figure.add_vline(
                        x=x_truth,
                        line={
                            "color": truth_colour,
                            "width": 1.7,
                            "dash": "dash",
                        },
                        row=row_number,
                        col=column_number,
                    )
                elif y_truth is not None:
                    figure.add_trace(
                        go.Scatter(
                            x=[x_truth],
                            y=[y_truth],
                            mode="markers",
                            marker={
                                "symbol": "star",
                                "size": 12,
                                "color": truth_colour,
                                "line": {"color": INK, "width": 0.8},
                            },
                            name=truth_name,
                            legendgroup=f"truth-{truth_condition}",
                            showlegend=truth_name not in truth_legend_drawn,
                            hovertemplate=f"{truth_name}<extra></extra>",
                        ),
                        row=row_number,
                        col=column_number,
                    )
                    truth_legend_drawn.add(truth_name)
            if row_number == size:
                figure.update_xaxes(
                    title_text=PARAMETER_AXIS_LABELS.get(
                        x_parameter, x_parameter
                    ),
                    row=row_number,
                    col=column_number,
                )
            if column_number == 1:
                figure.update_yaxes(
                    title_text=(
                        "Posterior density"
                        if row_number == 1
                        else PARAMETER_AXIS_LABELS.get(y_parameter, y_parameter)
                    ),
                    row=row_number,
                    col=column_number,
                )
    figure.update_layout(
        template="none",
        barmode="overlay",
        height=max(430, 245 * size),
        paper_bgcolor=SHEET,
        plot_bgcolor=PAPER,
        font={"family": SERIF, "color": INK, "size": 12},
        title={
            "text": TRAJECTORY_MODEL_LABELS.get(model_key, model_key),
            "x": 0.01,
            "xanchor": "left",
            "font": {"size": 16},
        },
        margin={"l": 106, "r": 32, "t": 108, "b": 102},
        legend={"orientation": "h", "x": 0, "y": 1.04},
    )
    figure.update_xaxes(
        showline=True,
        linecolor=RULE,
        gridcolor=GRID,
        ticks="outside",
        zeroline=False,
        automargin=True,
    )
    figure.update_yaxes(
        showline=True,
        linecolor=RULE,
        gridcolor=GRID,
        ticks="outside",
        zeroline=False,
        automargin=True,
    )
    return figure


def trajectory_posterior_payload(
    draws: pd.DataFrame,
    model_keys: Sequence[str] | None = None,
    truth: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Serialise paired posterior rows for lightweight Dash interactions."""

    frame = _paired_wide_draws(draws)
    selected = (
        list(dict.fromkeys(map(str, model_keys)))
        if model_keys is not None
        else list(dict.fromkeys(frame["model_key"].astype(str)))
    )
    frame = frame.loc[frame["model_key"].astype(str).isin(selected)].copy()
    clean = frame.astype(object).where(pd.notna(frame), None)
    return {
        "schema_version": 1,
        "models": selected,
        "conditions": list(dict.fromkeys(frame["condition"].astype(str))),
        "parameters": {
            model_key: _available_parameters(frame, model_key)
            for model_key in selected
        },
        "truth": dict(truth or {}),
        "records": clean.to_dict("records"),
    }


def _csv_link(frame: pd.DataFrame, filename: str, label: str) -> html.A:
    encoded = base64.b64encode(frame.to_csv(index=False).encode("utf-8")).decode(
        "ascii"
    )
    return html.A(
        label,
        href=f"data:text/csv;base64,{encoded}",
        download=filename,
        className="barracuda-button secondary download",
    )


def _download_component(download: object | None) -> object | None:
    if download is None:
        return None
    if isinstance(download, (bytes, bytearray)):
        encoded = base64.b64encode(bytes(download)).decode("ascii")
        return html.A(
            "Download complete trajectory analysis",
            href=f"data:application/zip;base64,{encoded}",
            download="barracuda_trajectory_analysis.zip",
            className="barracuda-button primary download",
        )
    return download


def model_panel_styles(
    selected: Sequence[str] | None,
    panel_ids: Sequence[Mapping[str, object]] | None,
) -> list[dict[str, str]]:
    chosen = {str(value) for value in (selected or [])}
    return [
        {} if str(panel_id.get("index")) in chosen else {"display": "none"}
        for panel_id in (panel_ids or [])
    ]


def render_trajectory_results(
    *,
    evidence: pd.DataFrame,
    posterior_draws: pd.DataFrame,
    condition_colours: Mapping[str, str],
    truth: Mapping[str, object] | None = None,
    truth_model: str | Mapping[str, object] | None = None,
    download: object | None = None,
    prefix: str = "trajectory",
) -> tuple[html.Div, html.Div]:
    """Render model evidence first, followed by selectable posterior results."""

    evidence_table = _normalise_evidence(evidence)
    draws = _paired_wide_draws(posterior_draws)
    evidence_models = list(
        dict.fromkeys(evidence_table["model_key"].astype(str))
    )
    draw_models = set(draws["model_key"].astype(str))
    models = [model for model in evidence_models if model in draw_models]
    if not models:
        raise ValueError("no posterior draws match the model evidence")
    comparison_counts = evidence_table.groupby(
        "condition",
        sort=False,
    )["model_key"].nunique()
    comparison_conditions = set(
        comparison_counts.loc[comparison_counts >= 2].index.astype(str)
    )
    comparison_evidence = evidence_table.loc[
        evidence_table["condition"].astype(str).isin(comparison_conditions)
    ]
    bayes_figure = (
        trajectory_bayes_factor_figure(
            comparison_evidence,
            truth_model=truth_model,
            condition_colours=condition_colours,
        )
        if not comparison_evidence.empty
        else None
    )
    payload = trajectory_posterior_payload(draws, models, truth)
    payload["condition_colours"] = dict(condition_colours)

    panels: list[html.Div] = []
    for model_key in models:
        parameters = payload["parameters"][model_key]  # type: ignore[index]
        if not parameters:
            continue
        marginal_parameter = str(parameters[0])
        marginal = posterior_marginal_figure(
            draws,
            marginal_parameter,
            condition_colours,
            truth,
            model_key,
        )
        joint = joint_posterior_figure(
            draws,
            model_key,
            condition_colours,
            truth,
        )
        panels.append(
            html.Div(
                [
                    html.H4(TRAJECTORY_MODEL_LABELS.get(model_key, model_key)),
                    html.Label(
                        [
                            html.Span(
                                "Marginal posterior parameter",
                                className="barracuda-field-label",
                            ),
                            dcc.Dropdown(
                                id={
                                    "type": f"{prefix}-posterior-parameter",
                                    "index": model_key,
                                },
                                options=[
                                    {
                                        "label": PARAMETER_LABELS.get(
                                            str(parameter), str(parameter)
                                        ),
                                        "value": parameter,
                                    }
                                    for parameter in parameters
                                ],
                                value=marginal_parameter,
                                clearable=False,
                            ),
                        ],
                        className="barracuda-field",
                    ),
                    dcc.Graph(
                        id={
                            "type": f"{prefix}-posterior-marginal",
                            "index": model_key,
                        },
                        figure=marginal,
                        config={
                            "displaylogo": False,
                            "responsive": True,
                            "toImageButtonOptions": {
                                "format": "png",
                                "filename": f"barracuda_trajectory_{model_key}_marginal",
                                "scale": 2,
                            },
                        },
                        responsive=True,
                    ),
                    html.H5("Full joint posterior"),
                    html.P(
                        "Diagonal panels use shared bins and show 95% HDIs. Lower panels retain paired particle dependence between parameters.",
                        className="barracuda-help",
                    ),
                    html.Div(
                        dcc.Graph(
                            id={
                                "type": f"{prefix}-posterior-joint",
                                "index": model_key,
                            },
                            figure=joint,
                            config={
                                "displaylogo": False,
                                "responsive": True,
                                "toImageButtonOptions": {
                                    "format": "png",
                                    "filename": f"barracuda_trajectory_{model_key}_joint",
                                    "scale": 2,
                                },
                            },
                            responsive=True,
                            className="barracuda-joint-posterior-plot",
                            style={
                                "height": f"{int(joint.layout.height)}px",
                                "minWidth": f"{max(760, 220 * len(parameters))}px",
                            },
                        ),
                        className="barracuda-joint-plot-scroll",
                    ),
                ],
                id={"type": f"{prefix}-model-panel", "index": model_key},
                className="barracuda-model-result-panel",
            )
        )

    bayes_result: object
    if bayes_figure is None:
        bayes_result = html.Div(
            [
                html.Strong("Model comparison requires at least two models."),
                html.P(
                    "The posterior below is valid for the selected model. Run inference with two or more candidate models to calculate Bayes factors.",
                ),
            ],
            id=f"{prefix}-bayes-factor-unavailable",
            className="barracuda-results-placeholder",
        )
    else:
        bayes_result = dcc.Graph(
            id=f"{prefix}-bayes-factor-figure",
            figure=bayes_figure,
            config={
                "displaylogo": False,
                "responsive": True,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": "barracuda_trajectory_bayes_factors",
                    "scale": 2,
                },
            },
            responsive=True,
            className="barracuda-bayes-factor-plot",
        )

    content = html.Div(
        [
            html.Section(
                [
                    html.Span("Model evidence", className="barracuda-section-label"),
                    html.H3("Bayes factors by experimental condition"),
                    html.P(
                        "The axis shows log₁₀ BF(candidate model / best model) on a linear scale. The best model is at zero; evidence against alternatives extends left across the exact BF boundaries 3, 10, and 100.",
                        className="barracuda-help",
                    ),
                    bayes_result,
                    _csv_link(
                        evidence_table,
                        "barracuda_trajectory_model_evidence.csv",
                        "Download Bayes factor CSV",
                    ),
                ],
                className="barracuda-result-section barracuda-figure-result",
            ),
            html.Section(
                [
                    html.Span("Posterior results", className="barracuda-section-label"),
                    html.H3("Choose inference results to visualise"),
                    dcc.Store(
                        id=f"{prefix}-posterior-data",
                        data=payload,
                        storage_type="memory",
                    ),
                    dcc.Checklist(
                        id=f"{prefix}-model-view",
                        options=[
                            {
                                "label": TRAJECTORY_MODEL_LABELS.get(
                                    model, model
                                ),
                                "value": model,
                            }
                            for model in models
                        ],
                        value=models,
                        inline=True,
                        className="barracuda-posterior-model-options",
                        inputClassName="barracuda-check-input",
                        labelClassName="barracuda-posterior-model-option",
                    ),
                    html.P(
                        "Condition colours are retained across posterior plots. Parameters fixed by a candidate model are omitted rather than plotted as zero-width distributions.",
                        className="barracuda-help",
                    ),
                    html.Div(panels, className="barracuda-condition-model-panels"),
                ],
                className="barracuda-result-section",
            ),
        ],
        className="barracuda-results barracuda-trajectory-results",
    )

    supplied_download = _download_component(download)
    download_children: list[object] = []
    if supplied_download is not None:
        download_children.append(supplied_download)
    download_children.extend(
        [
            html.Details(
                [
                    html.Summary("Download individual CSV tables"),
                    html.Div(
                        [
                            _csv_link(
                                evidence_table,
                                "barracuda_trajectory_model_evidence.csv",
                                "Model evidence CSV",
                            ),
                            _csv_link(
                                draws,
                                "barracuda_trajectory_posterior_samples.csv",
                                "Paired posterior samples CSV",
                            ),
                        ],
                        className="barracuda-download-grid",
                    ),
                ],
                className="barracuda-details",
            )
        ]
    )
    downloads = html.Div(
        download_children,
        className="barracuda-analysis-downloads",
    )
    return content, downloads


__all__ = [
    "BF3_LOG10",
    "MODEL_PARAMETERS",
    "PARAMETER_LABELS",
    "TRAJECTORY_MODEL_LABELS",
    "empirical_state_arrow_figure",
    "empirical_state_encoding_legend",
    "empirical_state_summary",
    "expanded_history_frame",
    "joint_posterior_figure",
    "model_panel_styles",
    "posterior_marginal_figure",
    "render_trajectory_results",
    "trajectory_bayes_factor_figure",
    "trajectory_posterior_payload",
]
