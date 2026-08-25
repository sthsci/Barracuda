"""Shared PyMC Sequential Monte Carlo progress components for Barracuda."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias

from dash import html


ChainState: TypeAlias = tuple[int, float]
ProgressPayload: TypeAlias = tuple[float, str, str, list[html.Div]]


def _positive_int(value: int, name: str) -> int:
    converted = int(value)
    if converted < 1:
        raise ValueError(f"{name} must be at least one")
    return converted


def _bounded_beta(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def pymc_progress(prefix: str) -> html.Div:
    """Build the shared, initially hidden PyMC progress region.

    The rapidly changing per-chain rows are intentionally outside the live
    region. Screen readers receive one concise, atomic summary while sighted
    users can follow every chain directly.
    """

    component_prefix = str(prefix).strip()
    if not component_prefix:
        raise ValueError("prefix must not be blank")

    return html.Div(
        [
            html.Div(
                html.Progress(
                    id=f"{component_prefix}-pymc-progress-bar",
                    max=1,
                    value=0,
                    **{
                        "aria-label": (
                            "Overall PyMC Sequential Monte Carlo inference progress"
                        )
                    },
                ),
                className="barracuda-pymc-progress-overall",
            ),
            html.Div(
                [
                    html.Strong(
                        "PyMC SMC sampler",
                        id=f"{component_prefix}-pymc-progress-label",
                    ),
                    html.P(
                        (
                            "Start inference to view each chain's SMC stage and "
                            "tempering value β."
                        ),
                        id=f"{component_prefix}-pymc-progress-meta",
                    ),
                ],
                className="barracuda-pymc-progress-summary",
                role="status",
                **{"aria-live": "polite", "aria-atomic": "true"},
            ),
            html.Div(
                id=f"{component_prefix}-chain-progress",
                className="barracuda-pymc-chain-list",
                role="list",
                **{"aria-label": "PyMC SMC progress by chain"},
            ),
        ],
        id=f"{component_prefix}-pymc-progress",
        className="barracuda-pymc-progress is-hidden",
        role="region",
        **{"aria-label": "Live PyMC Sequential Monte Carlo inference progress"},
    )


def chain_progress_rows(
    chain_states: Mapping[int, ChainState] | None,
    *,
    chains: int,
) -> list[html.Div]:
    """Render every configured chain, including chains awaiting an update."""

    chain_count = _positive_int(chains, "chains")
    states = dict(chain_states or {})
    rows: list[html.Div] = []
    for chain_index in range(chain_count):
        stage_raw, beta_raw = states.get(chain_index, (0, 0.0))
        stage = max(0, int(stage_raw))
        beta = _bounded_beta(beta_raw)
        display_index = chain_index + 1
        rows.append(
            html.Div(
                [
                    html.Span(
                        f"Chain {display_index}",
                        className="barracuda-pymc-chain-name",
                    ),
                    html.Progress(
                        max=1,
                        value=beta,
                        **{
                            "aria-label": (
                                f"PyMC SMC chain {display_index} tempering progress"
                            )
                        },
                    ),
                    html.Span(
                        f"Stage {stage} · β = {beta:.3f}",
                        className="barracuda-pymc-chain-state",
                    ),
                ],
                className="barracuda-pymc-chain-row",
                role="listitem",
            )
        )
    return rows


def sampling_progress_payload(
    *,
    condition_index: int = 1,
    total_conditions: int = 1,
    condition_label: str | None = None,
    model_index: int,
    total_models: int,
    model_label: str,
    chains: int,
    particles: int,
    chain_states: Mapping[int, ChainState] | None,
) -> ProgressPayload:
    """Return the four values consumed by a Dash progress update.

    PyMC does not know the number of SMC stages in advance. The overall bar
    therefore uses the mean of the chains' native tempering values ``beta``
    within the current condition/model, rather than inventing a stage count.
    """

    condition_number = _positive_int(condition_index, "condition_index")
    condition_count = _positive_int(total_conditions, "total_conditions")
    model_number = _positive_int(model_index, "model_index")
    model_count = _positive_int(total_models, "total_models")
    chain_count = _positive_int(chains, "chains")
    particle_count = _positive_int(particles, "particles")
    if condition_number > condition_count:
        raise ValueError("condition_index cannot exceed total_conditions")
    if model_number > model_count:
        raise ValueError("model_index cannot exceed total_models")

    states = dict(chain_states or {})
    betas = [
        _bounded_beta(states.get(chain_index, (0, 0.0))[1])
        for chain_index in range(chain_count)
    ]
    current_model_fraction = sum(betas) / chain_count
    completed_models = (condition_number - 1) * model_count + model_number - 1
    total_work = condition_count * model_count
    overall = min(
        0.999,
        max(0.0, (completed_models + current_model_fraction) / total_work),
    )

    model_text = str(model_label).strip() or "Selected model"
    if condition_label is None:
        label = f"Model {model_number} of {model_count}: {model_text}"
    else:
        condition_text = str(condition_label).strip() or "Unnamed condition"
        label = (
            f"Condition {condition_number} of {condition_count}: {condition_text}"
            f" · Model {model_number} of {model_count}: {model_text}"
        )
    chain_word = "chain" if chain_count == 1 else "chains"
    meta = (
        "Direct PyMC SMC updates"
        f" · {chain_count} independent {chain_word}"
        f" · {particle_count:,} particles per chain"
        " · β moves from the prior at 0 to the posterior at 1"
    )
    return (
        overall,
        label,
        meta,
        chain_progress_rows(states, chains=chain_count),
    )


def sampling_complete_payload(
    *,
    chains: int,
    particles: int,
    chain_states: Mapping[int, ChainState] | None = None,
) -> ProgressPayload:
    """Return a terminal sampling state while reports are being prepared."""

    chain_count = _positive_int(chains, "chains")
    particle_count = _positive_int(particles, "particles")
    states = dict(chain_states or {})
    completed = {
        chain_index: (states.get(chain_index, (0, 0.0))[0], 1.0)
        for chain_index in range(chain_count)
    }
    return (
        1.0,
        "PyMC sampling complete",
        (
            f"{particle_count:,} particles per chain · Preparing posterior "
            "and Bayes factor plots."
        ),
        chain_progress_rows(completed, chains=chain_count),
    )
