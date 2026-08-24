"""Posterior, SMC, and trajectory diagnostic summaries.

All helpers return NumPy, pandas, or xarray-compatible objects and never draw
figures.  They are therefore suitable for notebooks, batch validation jobs,
and downstream plotting libraries.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


def _positive_int(value: Any, name: str, *, allow_none: bool = False) -> int | None:
    if allow_none and value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    converted = int(value)
    if converted <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return converted


def _probability(value: Any, name: str, *, inclusive: bool = False) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a probability") from exc
    valid = 0 <= converted <= 1 if inclusive else 0 < converted < 1
    if not np.isfinite(converted) or not valid:
        interval = "[0, 1]" if inclusive else "(0, 1)"
        raise ValueError(f"{name} must lie in {interval}")
    return converted


def smc_log_evidence_by_chain(idata: Any) -> pd.DataFrame:
    """Extract the final finite SMC log marginal likelihood for every chain.

    BARRACUDA stores one final value per chain, but this parser also accepts older
    inference files containing a stage dimension or an attribute fallback.
    Chains with no finite value are retained with ``NaN`` so incomplete output
    cannot silently masquerade as a lower-chain run.
    """

    sample_stats = getattr(idata, "sample_stats", None)
    if sample_stats is not None and "log_marginal_likelihood" in sample_stats:
        variable = sample_stats["log_marginal_likelihood"]
        values = np.asarray(variable, dtype=float)
        dims = tuple(getattr(variable, "dims", ()))
        if "chain" in dims:
            chain_axis = dims.index("chain")
            values = np.moveaxis(values, chain_axis, 0)
        elif values.ndim == 0:
            values = values.reshape(1)
        elif dims:
            # A labelled stage/draw axis without an explicit chain axis is one
            # historical chain, not one chain per stage.
            values = values.reshape(1, -1)
        elif values.ndim > 0:
            # Historical files normally put chain first when dimension labels
            # are unavailable.
            values = values.reshape(values.shape[0], -1)
    else:
        attrs = getattr(idata, "attrs", {})
        if "log_marginal_likelihood" not in attrs:
            raise RuntimeError(
                "SMC evidence is missing from sample_stats and idata.attrs"
            )
        values = np.asarray(attrs["log_marginal_likelihood"], dtype=float)
        if values.ndim == 0:
            values = values.reshape(1)

    if values.ndim == 1:
        matrix = values.reshape(values.size, 1)
    else:
        matrix = values.reshape(values.shape[0], -1)
    rows: list[dict[str, float | int | bool]] = []
    for chain, row in enumerate(matrix):
        finite = row[np.isfinite(row)]
        log_evidence = float(finite[-1]) if finite.size else np.nan
        rows.append(
            {
                "chain": int(chain),
                "log_evidence": log_evidence,
                "n_finite_stages": int(finite.size),
                "is_finite": bool(np.isfinite(log_evidence)),
            }
        )
    return pd.DataFrame(rows)


def smc_evidence_summary(idata: Any) -> pd.Series:
    """Summarize between-chain stability of the SMC evidence estimate."""

    chains = smc_log_evidence_by_chain(idata)
    finite = chains.loc[chains["is_finite"], "log_evidence"].to_numpy(float)
    if not finite.size:
        raise RuntimeError("SMC evidence contains no finite chain estimates")
    return pd.Series(
        {
            "n_chains": int(len(chains)),
            "n_finite_chains": int(finite.size),
            "mean_log_evidence": float(np.mean(finite)),
            "sd_log_evidence": (
                float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan
            ),
            "min_log_evidence": float(np.min(finite)),
            "max_log_evidence": float(np.max(finite)),
            "range_log_evidence": float(np.ptp(finite)),
        },
        name="smc_evidence",
    )


def posterior_diagnostics(
    idata: Any,
    *,
    var_names: Sequence[str] | None = None,
    hdi_prob: float = 0.95,
) -> pd.DataFrame:
    """Return an ArviZ posterior summary with stable parameter columns.

    R-hat is undefined for one chain and remains ``NaN``.  Effective sample
    sizes should be interpreted cautiously for weighted/resampled SMC draws;
    the table is a diagnostic aid, not an automatic validity certificate.
    """

    probability = _probability(hdi_prob, "hdi_prob")
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        raise ValueError("idata must contain a posterior group")
    if var_names is not None:
        names = [str(name) for name in var_names]
        missing = [name for name in names if name not in posterior]
        if missing:
            raise ValueError("posterior variables not found: " + ", ".join(missing))
    else:
        names = None

    import arviz as az

    table = az.summary(
        idata,
        var_names=names,
        kind="all",
        hdi_prob=probability,
        round_to=None,
    ).rename_axis("parameter").reset_index()
    table.insert(1, "hdi_probability", probability)
    return table


def diagnostic_flags(
    diagnostics: pd.DataFrame,
    *,
    min_ess_bulk: float = 100.0,
    min_ess_tail: float = 100.0,
    max_r_hat: float = 1.01,
) -> pd.DataFrame:
    """Add transparent ESS/R-hat flags to a posterior diagnostic table.

    Missing R-hat values (for example one-chain SMC output) are marked
    ``r_hat_available=False`` and are not treated as a pass.
    """

    if not isinstance(diagnostics, pd.DataFrame):
        raise TypeError("diagnostics must be a pandas DataFrame")
    required = {"parameter", "ess_bulk", "ess_tail", "r_hat"}
    missing = sorted(required.difference(diagnostics.columns))
    if missing:
        raise ValueError("diagnostics is missing columns: " + ", ".join(missing))
    thresholds = {
        "min_ess_bulk": float(min_ess_bulk),
        "min_ess_tail": float(min_ess_tail),
        "max_r_hat": float(max_r_hat),
    }
    if not all(np.isfinite(list(thresholds.values()))):
        raise ValueError("diagnostic thresholds must be finite")
    if min_ess_bulk <= 0 or min_ess_tail <= 0 or max_r_hat < 1:
        raise ValueError("ESS thresholds must be positive and max_r_hat at least one")

    output = diagnostics.copy()
    ess_bulk = pd.to_numeric(output["ess_bulk"], errors="coerce")
    ess_tail = pd.to_numeric(output["ess_tail"], errors="coerce")
    r_hat = pd.to_numeric(output["r_hat"], errors="coerce")
    output["ess_bulk_ok"] = ess_bulk >= float(min_ess_bulk)
    output["ess_tail_ok"] = ess_tail >= float(min_ess_tail)
    output["r_hat_available"] = r_hat.notna()
    output["r_hat_ok"] = r_hat.notna() & (r_hat <= float(max_r_hat))
    output["all_available_checks_ok"] = (
        output["ess_bulk_ok"]
        & output["ess_tail_ok"]
        & (~output["r_hat_available"] | output["r_hat_ok"])
    )
    output["all_checks_available"] = output["r_hat_available"]
    output["diagnostic_status"] = np.select(
        [
            ~output["all_available_checks_ok"],
            ~output["all_checks_available"],
        ],
        ["review", "limited"],
        default="ok",
    )
    return output


def population_p0_draws(
    idata: Any,
    *,
    n_parameter_draws: int | None = 1_000,
    n_population_draws: int = 1_000,
    seed: int | None = None,
) -> np.ndarray:
    """Draw baseline lethal probabilities from a trajectory population.

    For every retained posterior pair ``(mu_eta, sigma_eta)``, the function
    samples latent cell propensities and applies the logistic transform.  The
    returned two-dimensional array preserves parameter-draw rows.
    """

    parameter_limit = _positive_int(
        n_parameter_draws, "n_parameter_draws", allow_none=True
    )
    cells = _positive_int(n_population_draws, "n_population_draws")
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        raise ValueError("idata must contain a posterior group")
    missing = [name for name in ("mu_eta", "sigma_eta") if name not in posterior]
    if missing:
        raise ValueError("posterior variables not found: " + ", ".join(missing))
    mu_eta = np.asarray(posterior["mu_eta"], dtype=float).reshape(-1)
    sigma_eta = np.asarray(posterior["sigma_eta"], dtype=float).reshape(-1)
    finite = np.isfinite(mu_eta) & np.isfinite(sigma_eta) & (sigma_eta >= 0)
    mu_eta = mu_eta[finite]
    sigma_eta = sigma_eta[finite]
    if not mu_eta.size:
        raise ValueError("posterior contains no finite valid eta parameters")
    rng = np.random.default_rng(seed)
    if parameter_limit is not None and mu_eta.size > parameter_limit:
        indices = np.sort(
            rng.choice(mu_eta.size, size=int(parameter_limit), replace=False)
        )
        mu_eta = mu_eta[indices]
        sigma_eta = sigma_eta[indices]
    eta = rng.normal(
        loc=mu_eta[:, None],
        scale=sigma_eta[:, None],
        size=(mu_eta.size, int(cells)),
    )
    # Stable logistic transformation for large absolute eta.
    positive = eta >= 0
    probabilities = np.empty_like(eta, dtype=float)
    probabilities[positive] = 1.0 / (1.0 + np.exp(-eta[positive]))
    exp_eta = np.exp(eta[~positive])
    probabilities[~positive] = exp_eta / (1.0 + exp_eta)
    return probabilities


def population_p0_summary(
    idata: Any,
    *,
    n_parameter_draws: int | None = 1_000,
    n_population_draws: int = 1_000,
    seed: int | None = None,
) -> pd.Series:
    """Summarize simulated baseline lethal probabilities."""

    values = population_p0_draws(
        idata,
        n_parameter_draws=n_parameter_draws,
        n_population_draws=n_population_draws,
        seed=seed,
    ).reshape(-1)
    return pd.Series(
        {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=0)),
            "q025": float(np.quantile(values, 0.025)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "q975": float(np.quantile(values, 0.975)),
            "n_values": int(values.size),
        },
        name="population_p0",
    )


def trajectory_state_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate observed lethal decisions at every pre-contact state.

    ``frame`` may be any compact/wide/long trajectory format accepted by
    :func:`barracuda.trajectories.normalize_trajectory_frame`, or an already
    expanded frame returned by ``expanded_trajectory_frame``.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    canonical_expanded = {
        "cell_id",
        "condition",
        "previous_nonlethal_contacts",
        "previous_lethal_contacts",
        "outcome",
    }
    legacy_expanded = {
        "cell_id",
        "condition",
        "x_before",
        "y_before",
        "outcome",
    }
    if canonical_expanded.issubset(frame.columns):
        expanded = frame.rename(
            columns={
                "previous_nonlethal_contacts": "x_before",
                "previous_lethal_contacts": "y_before",
            }
        ).copy()
    elif legacy_expanded.issubset(frame.columns):
        expanded = frame.copy()
    else:
        from .trajectories import expanded_trajectory_frame

        expanded = expanded_trajectory_frame(frame).rename(
            columns={
                "previous_nonlethal_contacts": "x_before",
                "previous_lethal_contacts": "y_before",
            }
        )
    columns = [
        "condition",
        "x_before",
        "y_before",
        "n_cells",
        "n_contacts",
        "n_lethal",
        "n_nonlethal",
        "empirical_lethal_probability",
    ]
    if expanded.empty:
        return pd.DataFrame(columns=columns)
    outcome = pd.to_numeric(expanded["outcome"], errors="raise")
    if not outcome.isin([0, 1]).all():
        raise ValueError("outcome must contain only 0 and 1")
    expanded = expanded.assign(outcome=outcome.astype(int))
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
    return grouped.loc[:, columns]


__all__ = [
    "diagnostic_flags",
    "population_p0_draws",
    "population_p0_summary",
    "posterior_diagnostics",
    "smc_evidence_summary",
    "smc_log_evidence_by_chain",
    "trajectory_state_summary",
]
