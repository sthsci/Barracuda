#!/usr/bin/env python3
"""Donor-aware count models with deviations on unconstrained scales.

This module is an experimental alternative to ``inference_donor.py``. It
exposes the same four public inference functions:

* ``inference_homo``: donor-specific Poisson rates;
* ``inference_Z2P``: donor-specific rates and structural-zero fractions;
* ``inference_Dis2P``: donor-specific active-rate means and SDs; and
* ``inference_hetero3``: all three donor-specific parameters.

The hierarchy uses shared reference priors followed by zero-centred Normal
donor deviations. Positive donor parameters are shifted on a base-10 log
scale, while donor zero fractions are shifted on the logit scale. These
transformations enforce the parameter domains without truncating the donor
deviations. Population parameters are deterministic moments of the weighted
donor mixture.

These population moments are comparable to the parameters of the corresponding
donor-ignorant model. A mixture of donor Gamma distributions is not generally
itself Gamma, so this construction does not force the two inferred posteriors to
be identical.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

try:
    from barracuda._backends.event_counts.smc_progress import (
        SMCProgressCallback,
        run_with_smc_progress,
    )
except ImportError:  # Support the research source tree without installation.
    from section_1.src.smc_progress import (
        SMCProgressCallback,
        run_with_smc_progress,
    )


@dataclass(frozen=True)
class _DonorData:
    counts: np.ndarray
    donor_index: np.ndarray
    donor_weights: np.ndarray
    observation_time: float
    donor_num: int

    @property
    def max_count(self) -> int:
        return int(self.counts.max(initial=0))


@dataclass(frozen=True)
class _DonorParameters:
    mu_lambda_donor: Any
    mu_lambda_population: Any
    sigma_lambda_donor: Any | None = None
    sigma_lambda_population: Any | None = None
    phi_0_donor: Any | None = None
    phi_0_population: Any | None = None


# Data preparation -----------------------------------------------------------


def _prepare_data(
    contacts_per_cell,
    donor_idx,
    obs_time: float,
    donor_num: int,
) -> _DonorData:
    counts_raw = np.asarray(contacts_per_cell)
    donor_raw = np.asarray(donor_idx)
    observation_time = float(obs_time)
    donor_num = int(donor_num)

    if counts_raw.ndim != 1 or counts_raw.size == 0:
        raise ValueError("contacts_per_cell must be a non-empty 1D array")
    if not np.issubdtype(counts_raw.dtype, np.number):
        raise ValueError("contacts_per_cell must contain numeric counts")
    if not np.all(np.isfinite(counts_raw)):
        raise ValueError("contacts_per_cell must contain finite values")
    if np.any(counts_raw < 0) or np.any(counts_raw != np.floor(counts_raw)):
        raise ValueError("contacts_per_cell must contain integers >= 0")

    if donor_raw.shape != counts_raw.shape:
        raise ValueError("donor_idx must contain one index per cell")
    if not np.issubdtype(donor_raw.dtype, np.number):
        raise ValueError("donor_idx must contain numeric indices")
    if not np.all(np.isfinite(donor_raw)):
        raise ValueError("donor_idx must contain finite values")
    if np.any(donor_raw != np.floor(donor_raw)):
        raise ValueError("donor_idx must contain integer values")

    if donor_num <= 0:
        raise ValueError("donor_num must be > 0")
    if not np.isfinite(observation_time) or observation_time <= 0:
        raise ValueError("obs_time must be finite and > 0")

    counts = counts_raw.astype(int)
    donor_index = donor_raw.astype(int)
    if np.any(donor_index < 0) or np.any(donor_index >= donor_num):
        raise ValueError(
            f"donor_idx values must be between 0 and {donor_num - 1}"
        )

    donor_cell_counts = np.bincount(
        donor_index,
        minlength=donor_num,
    ).astype(float)
    if np.any(donor_cell_counts == 0):
        raise ValueError("every donor must have at least one observed cell")

    return _DonorData(
        counts=counts,
        donor_index=donor_index,
        donor_weights=donor_cell_counts / donor_cell_counts.sum(),
        observation_time=observation_time,
        donor_num=donor_num,
    )


def _validate_hierarchy_inputs(
    *,
    lambda_prior_bounds: tuple[float, float],
    mu_log_deviation_prior: float,
    sigma_lambda_prior: float | None,
    sigma_log_deviation_prior: float | None,
    phi_0_prior: tuple[float, float] | None,
    phi_logit_deviation_prior: float | None,
) -> None:
    lower, upper = map(float, lambda_prior_bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("lambda_prior_bounds must be finite and increasing")
    if (
        not np.isfinite(mu_log_deviation_prior)
        or float(mu_log_deviation_prior) <= 0
    ):
        raise ValueError("mu_log_deviation_prior must be finite and > 0")

    include_sigma = sigma_lambda_prior is not None
    include_phi = phi_0_prior is not None
    if include_sigma != (sigma_log_deviation_prior is not None):
        raise ValueError(
            "sigma_lambda_prior and sigma_log_deviation_prior "
            "must be supplied together"
        )
    if include_phi != (phi_logit_deviation_prior is not None):
        raise ValueError(
            "phi_0_prior and phi_logit_deviation_prior "
            "must be supplied together"
        )
    if include_sigma:
        if not np.isfinite(sigma_lambda_prior) or sigma_lambda_prior <= 0:
            raise ValueError("sigma_lambda_prior must be finite and > 0")
        if (
            not np.isfinite(sigma_log_deviation_prior)
            or sigma_log_deviation_prior <= 0
        ):
            raise ValueError(
                "sigma_log_deviation_prior must be finite and > 0"
            )
    if include_phi:
        if len(phi_0_prior) != 2 or np.any(np.asarray(phi_0_prior) <= 0):
            raise ValueError("phi_0_prior must contain two positive values")
        if (
            not np.isfinite(phi_logit_deviation_prior)
            or phi_logit_deviation_prior <= 0
        ):
            raise ValueError(
                "phi_logit_deviation_prior must be finite and > 0"
            )


# Gamma-Poisson likelihood ---------------------------------------------------


def _gamma_shape_rate_from_mean_sd(mu, sigma):
    variance = pt.maximum(sigma, 1e-12) ** 2
    return mu**2 / variance, mu / variance


def _gamma_poisson_logp(
    counts,
    obs_time: float,
    mu_lambda,
    sigma_lambda,
    *,
    max_count: int,
):
    """Gamma-marginalized Poisson log probability for integer counts."""
    shape, rate = _gamma_shape_rate_from_mean_sd(
        mu_lambda,
        sigma_lambda,
    )

    log_rising = pt.zeros_like(counts)
    for k in range(int(max_count)):
        log_rising = pt.switch(
            counts > float(k),
            log_rising + pt.log(shape + float(k)),
            log_rising,
        )

    return (
        counts * pt.log(float(obs_time))
        - pt.gammaln(counts + 1.0)
        + log_rising
        - counts * pt.log(rate)
        - (shape + counts) * pt.log1p(float(obs_time) / rate)
    )


# Donor hierarchy on unconstrained scales -----------------------------------


def _build_log_hierarchy(
    data: _DonorData,
    *,
    lambda_prior_bounds: tuple[float, float],
    mu_log_deviation_prior: float,
    sigma_lambda_prior: float | None = None,
    sigma_log_deviation_prior: float | None = None,
    phi_0_prior: tuple[float, float] | None = None,
    phi_logit_deviation_prior: float | None = None,
) -> _DonorParameters:
    _validate_hierarchy_inputs(
        lambda_prior_bounds=lambda_prior_bounds,
        mu_log_deviation_prior=mu_log_deviation_prior,
        sigma_lambda_prior=sigma_lambda_prior,
        sigma_log_deviation_prior=sigma_log_deviation_prior,
        phi_0_prior=phi_0_prior,
        phi_logit_deviation_prior=phi_logit_deviation_prior,
    )

    include_sigma = sigma_lambda_prior is not None
    include_phi = phi_0_prior is not None
    weights_t = pt.as_tensor_variable(data.donor_weights)

    eta_ref = pm.Uniform(
        "eta_ref",
        lower=float(lambda_prior_bounds[0]),
        upper=float(lambda_prior_bounds[1]),
    )
    mu_lambda_ref = pm.Deterministic(
        "mu_lambda_ref",
        10.0**eta_ref,
    )
    eta_mu_donor = pm.Normal(
        "eta_mu_donor",
        mu=0.0,
        sigma=float(mu_log_deviation_prior),
        dims="donor",
    )
    mu_lambda_donor = pm.Deterministic(
        "mu_lambda_donor",
        10.0 ** (eta_ref + eta_mu_donor),
        dims="donor",
    )

    sigma_lambda_donor = None
    if include_sigma:
        sigma_lambda_ref = pm.HalfNormal(
            "sigma_lambda_ref",
            sigma=float(sigma_lambda_prior),
        )
        eta_sigma_donor = pm.Normal(
            "eta_sigma_donor",
            mu=0.0,
            sigma=float(sigma_log_deviation_prior),
            dims="donor",
        )
        sigma_lambda_donor = pm.Deterministic(
            "sigma_lambda_donor",
            sigma_lambda_ref * 10.0**eta_sigma_donor,
            dims="donor",
        )

    phi_0_donor = None
    phi_0_population = None
    if include_phi:
        phi_0_ref = pm.Beta(
            "phi_0_ref",
            alpha=float(phi_0_prior[0]),
            beta=float(phi_0_prior[1]),
        )
        eta_phi_donor = pm.Normal(
            "eta_phi_donor",
            mu=0.0,
            sigma=float(phi_logit_deviation_prior),
            dims="donor",
        )
        phi_ref_safe = pt.clip(phi_0_ref, 1e-12, 1.0 - 1e-12)
        phi_ref_logit = pt.log(phi_ref_safe) - pt.log1p(-phi_ref_safe)
        phi_0_donor = pm.Deterministic(
            "phi_0_donor",
            pm.math.sigmoid(phi_ref_logit + eta_phi_donor),
            dims="donor",
        )
        phi_0_population = pm.Deterministic(
            "phi_0_population",
            pt.sum(weights_t * phi_0_donor),
        )
        pm.Deterministic(
            "phi_0_avg",
            phi_0_population,
        )
        active_mass = weights_t * (1.0 - phi_0_donor)
        parameter_weights = active_mass / pt.maximum(
            pt.sum(active_mass),
            1e-12,
        )
    else:
        parameter_weights = weights_t

    active_donor_weights = pm.Deterministic(
        "active_donor_weights",
        parameter_weights,
        dims="donor",
    )
    mu_lambda_population = pm.Deterministic(
        "mu_lambda_population",
        pt.sum(active_donor_weights * mu_lambda_donor),
    )
    pm.Deterministic(
        "mu_lambda_avg",
        mu_lambda_population,
    )

    sigma_lambda_population = None
    if include_sigma:
        lambda_second_moment = pt.sum(
            active_donor_weights
            * (sigma_lambda_donor**2 + mu_lambda_donor**2)
        )
        sigma_lambda_population = pm.Deterministic(
            "sigma_lambda_population",
            pt.sqrt(
                pt.maximum(
                    lambda_second_moment - mu_lambda_population**2,
                    0.0,
                )
            ),
        )
        pm.Deterministic(
            "sigma_lambda_avg",
            sigma_lambda_population,
        )

    return _DonorParameters(
        mu_lambda_donor=mu_lambda_donor,
        sigma_lambda_donor=sigma_lambda_donor,
        phi_0_donor=phi_0_donor,
        mu_lambda_population=mu_lambda_population,
        sigma_lambda_population=sigma_lambda_population,
        phi_0_population=phi_0_population,
    )


# Model builders -------------------------------------------------------------


def _build_model(
    contacts_per_cell,
    donor_idx,
    obs_time: float,
    *,
    donor_num: int,
    lambda_prior_bounds: tuple[float, float],
    mu_log_deviation_prior: float,
    sigma_lambda_prior: float | None = None,
    sigma_log_deviation_prior: float | None = None,
    phi_0_prior: tuple[float, float] | None = None,
    phi_logit_deviation_prior: float | None = None,
) -> pm.Model:
    data = _prepare_data(
        contacts_per_cell,
        donor_idx,
        obs_time,
        donor_num,
    )
    counts_t = pt.as_tensor_variable(data.counts.astype(float))
    donor_index_t = pt.as_tensor_variable(data.donor_index)
    include_sigma = sigma_lambda_prior is not None
    include_phi = phi_0_prior is not None

    with pm.Model(coords={"donor": np.arange(data.donor_num)}) as model:
        parameters = _build_log_hierarchy(
            data,
            lambda_prior_bounds=lambda_prior_bounds,
            mu_log_deviation_prior=mu_log_deviation_prior,
            sigma_lambda_prior=sigma_lambda_prior,
            sigma_log_deviation_prior=sigma_log_deviation_prior,
            phi_0_prior=phi_0_prior,
            phi_logit_deviation_prior=phi_logit_deviation_prior,
        )

        mu_cell = parameters.mu_lambda_donor[donor_index_t]

        if not include_sigma and not include_phi:
            pm.Poisson(
                "contacts",
                mu=mu_cell * data.observation_time,
                observed=data.counts,
            )
        elif not include_sigma:
            pm.ZeroInflatedPoisson(
                "contacts",
                psi=1.0 - parameters.phi_0_donor[donor_index_t],
                mu=mu_cell * data.observation_time,
                observed=data.counts,
            )
        else:
            sigma_cell = parameters.sigma_lambda_donor[donor_index_t]
            gamma_poisson_logp = _gamma_poisson_logp(
                counts_t,
                data.observation_time,
                mu_cell,
                sigma_cell,
                max_count=data.max_count,
            )

            if not include_phi:
                cell_logp = gamma_poisson_logp
            else:
                phi_cell = pt.clip(
                    parameters.phi_0_donor[donor_index_t],
                    1e-12,
                    1.0 - 1e-12,
                )
                active_zero_logp = _gamma_poisson_logp(
                    pt.zeros_like(counts_t),
                    data.observation_time,
                    mu_cell,
                    sigma_cell,
                    max_count=0,
                )
                log_active = pt.log1p(-phi_cell) + gamma_poisson_logp
                log_zero = pt.logaddexp(
                    pt.log(phi_cell),
                    pt.log1p(-phi_cell) + active_zero_logp,
                )
                cell_logp = pt.switch(
                    pt.eq(counts_t, 0.0),
                    log_zero,
                    log_active,
                )

            pm.Potential(
                "donor_count_loglik",
                pt.sum(cell_logp),
            )

    return model


def build_model(
    contacts_per_cell,
    donor_idx,
    obs_time: float,
    *,
    donor_num: int = 4,
    lambda_prior_bounds: tuple[float, float] = (-1.0, 2.0),
    sigma_lambda_prior: float = 1.0,
    phi_0_prior: tuple[float, float] = (1.0, 1.0),
    deviation_prior: tuple[float, float, float] = (0.2, 0.2, 0.5),
) -> pm.Model:
    """Build the full zero-inflated Gamma-Poisson donor-aware model.

    ``deviation_prior`` contains the standard deviations of the donor
    deviations for log10(mean), log10(SD), and logit(zero fraction).
    """
    if len(deviation_prior) != 3:
        raise ValueError("deviation_prior must contain three scales")
    return _build_model(
        contacts_per_cell,
        donor_idx,
        obs_time,
        donor_num=donor_num,
        lambda_prior_bounds=lambda_prior_bounds,
        mu_log_deviation_prior=deviation_prior[0],
        sigma_lambda_prior=sigma_lambda_prior,
        sigma_log_deviation_prior=deviation_prior[1],
        phi_0_prior=phi_0_prior,
        phi_logit_deviation_prior=deviation_prior[2],
    )


def math_model(
    contacts_per_cell,
    donor_idx,
    obs_time: float,
    eta_prior_bounds: tuple[float, float] = (-1.0, 2.0),
    sigma_lambda_prior: float = 1.0,
    phi_0_prior: tuple[float, float] = (1.0, 1.0),
    deviation_prior: tuple[float, float, float] = (0.2, 0.2, 0.5),
    donor_num: int = 4,
) -> pm.Model:
    """Compatibility wrapper for the full hetero3 model."""
    return build_model(
        contacts_per_cell,
        donor_idx,
        obs_time,
        donor_num=donor_num,
        lambda_prior_bounds=eta_prior_bounds,
        sigma_lambda_prior=sigma_lambda_prior,
        phi_0_prior=phi_0_prior,
        deviation_prior=deviation_prior,
    )


# SMC sampling and evidence --------------------------------------------------


def _resolve_cores(cores) -> int:
    available = os.cpu_count() or 1
    if cores is None:
        return available
    try:
        requested = int(cores)
    except (TypeError, ValueError):
        return available
    return available if requested <= 0 else min(requested, available)


def _parse_log_marginal_likelihood(
    raw,
    *,
    chains: int,
) -> Optional[np.ndarray]:
    if not isinstance(raw, (list, tuple, np.ndarray)):
        try:
            return np.full(chains, float(raw), dtype=float)
        except (TypeError, ValueError):
            return None

    if all(
        not isinstance(item, (list, tuple, np.ndarray))
        for item in raw
    ):
        values = np.asarray(raw, dtype=float)
        values = values[np.isfinite(values)]
        if values.size:
            return np.full(chains, float(values[-1]), dtype=float)
        return None

    final_values: list[float] = []
    for item in raw:
        if isinstance(item, (list, tuple, np.ndarray)):
            values = np.asarray(item, dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                final_values.append(float(values[-1]))
            continue
        try:
            value = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            final_values.append(value)

    if not final_values:
        return None
    logml = np.asarray(final_values, dtype=float)
    if logml.size < chains:
        logml = np.concatenate(
            [logml, np.full(chains - logml.size, np.nan)]
        )
    return logml[:chains]


def _store_smc_log_marginal_likelihood(
    idata,
    trace,
    *,
    chains: int,
) -> None:
    report = getattr(trace, "report", None)
    if report is None or not hasattr(report, "log_marginal_likelihood"):
        return
    logml = _parse_log_marginal_likelihood(
        report.log_marginal_likelihood,
        chains=chains,
    )
    if logml is None:
        return
    try:
        idata.sample_stats["log_marginal_likelihood"] = (("chain",), logml)
    except Exception:
        idata.attrs["log_marginal_likelihood"] = logml.tolist()


def _sample_smc(
    *,
    draws: int,
    chains: int,
    cores=None,
    random_seed: Optional[int] = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
    progress_callback: SMCProgressCallback | None = None,
):
    trace = run_with_smc_progress(
        progress_callback,
        lambda: pm.sample_smc(
            draws=int(draws),
            chains=int(chains),
            cores=_resolve_cores(cores),
            random_seed=random_seed,
            progressbar=True,
            return_inferencedata=False,
            threshold=float(threshold),
            correlation_threshold=float(correlation_threshold),
        ),
    )
    to_idata = getattr(pm, "to_inferencedata", None) or getattr(
        pm,
        "to_inference_data",
        None,
    )
    if to_idata is None:
        raise RuntimeError(
            "PyMC does not expose to_inferencedata/to_inference_data"
        )
    idata = to_idata(trace, log_likelihood=True)
    _store_smc_log_marginal_likelihood(
        idata,
        trace,
        chains=int(chains),
    )
    return idata


def smc_log_evidence(idata) -> float:
    sample_stats = getattr(idata, "sample_stats", None)
    if (
        sample_stats is not None
        and "log_marginal_likelihood"
        in getattr(sample_stats, "data_vars", {})
    ):
        values = np.asarray(
            sample_stats["log_marginal_likelihood"].values,
            dtype=float,
        )
    else:
        attrs = getattr(idata, "attrs", {})
        if "log_marginal_likelihood" not in attrs:
            raise RuntimeError(
                "SMC evidence missing: "
                "sample_stats['log_marginal_likelihood']"
            )
        values = np.asarray(
            attrs["log_marginal_likelihood"],
            dtype=float,
        )

    values = values[np.isfinite(values)]
    if not values.size:
        raise RuntimeError("Could not parse finite SMC log-evidence values")
    return float(np.mean(values))


def _infer_model(
    model: pm.Model,
    *,
    draws: int,
    chains: int,
    cores=None,
    random_seed: Optional[int] = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
    progress_callback: SMCProgressCallback | None = None,
):
    with model:
        idata = _sample_smc(
            draws=draws,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            threshold=threshold,
            correlation_threshold=correlation_threshold,
            progress_callback=progress_callback,
        )
    print(az.summary(idata, hdi_prob=0.95))
    return {"idata": idata, "model": model}


# Public inference API -------------------------------------------------------


def inference_homo(
    contacts_per_cell,
    donor_idx,
    obs_time: float,
    *,
    donor_num: int = 4,
    draws: int = 3000,
    chains: int = 4,
    cores=None,
    lambda_prior_bounds=(-1.0, 1.0),
    deviation_prior: float = 0.2,
    random_seed: Optional[int] = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
    progress_callback: SMCProgressCallback | None = None,
):
    model = _build_model(
        contacts_per_cell,
        donor_idx,
        obs_time,
        donor_num=donor_num,
        lambda_prior_bounds=lambda_prior_bounds,
        mu_log_deviation_prior=float(deviation_prior),
    )
    return _infer_model(
        model,
        draws=draws,
        chains=chains,
        cores=cores,
        random_seed=random_seed,
        threshold=threshold,
        correlation_threshold=correlation_threshold,
        progress_callback=progress_callback,
    )


def inference_Z2P(
    contacts_per_cell,
    donor_idx,
    obs_time: float,
    *,
    donor_num: int = 4,
    draws: int = 3000,
    chains: int = 4,
    cores=None,
    lambda_prior_bounds=(-1.0, 2.0),
    p_prior_bounds=(1.0, 1.0),
    deviation_prior: tuple[float, float] = (0.2, 0.5),
    random_seed: Optional[int] = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
    progress_callback: SMCProgressCallback | None = None,
):
    if len(deviation_prior) != 2:
        raise ValueError("deviation_prior must contain two scales")
    model = _build_model(
        contacts_per_cell,
        donor_idx,
        obs_time,
        donor_num=donor_num,
        lambda_prior_bounds=lambda_prior_bounds,
        mu_log_deviation_prior=deviation_prior[0],
        phi_0_prior=p_prior_bounds,
        phi_logit_deviation_prior=deviation_prior[1],
    )
    return _infer_model(
        model,
        draws=draws,
        chains=chains,
        cores=cores,
        random_seed=random_seed,
        threshold=threshold,
        correlation_threshold=correlation_threshold,
        progress_callback=progress_callback,
    )


def inference_Dis2P(
    contacts_per_cell,
    donor_idx,
    obs_time: float,
    *,
    donor_num: int = 4,
    draws: int = 3000,
    chains: int = 4,
    cores=None,
    lambda_prior_bounds=(-1.0, 2.0),
    std_prior_factor: float = 1.0,
    deviation_prior: tuple[float, float] = (0.2, 0.2),
    random_seed: Optional[int] = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
    progress_callback: SMCProgressCallback | None = None,
):
    if len(deviation_prior) != 2:
        raise ValueError("deviation_prior must contain two scales")
    model = _build_model(
        contacts_per_cell,
        donor_idx,
        obs_time,
        donor_num=donor_num,
        lambda_prior_bounds=lambda_prior_bounds,
        mu_log_deviation_prior=deviation_prior[0],
        sigma_lambda_prior=std_prior_factor,
        sigma_log_deviation_prior=deviation_prior[1],
    )
    return _infer_model(
        model,
        draws=draws,
        chains=chains,
        cores=cores,
        random_seed=random_seed,
        threshold=threshold,
        correlation_threshold=correlation_threshold,
        progress_callback=progress_callback,
    )


def inference_hetero3(
    contacts_per_cell,
    donor_idx,
    obs_time: float,
    *,
    donor_num: int = 4,
    draws: int = 3000,
    chains: int = 4,
    cores=None,
    lambda_prior_bounds=(-1.0, 2.0),
    p_prior_bounds=(1.0, 1.0),
    std_prior_factor: float = 1.0,
    deviation_prior: tuple[float, float, float] = (0.2, 0.2, 0.5),
    random_seed: Optional[int] = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
    progress_callback: SMCProgressCallback | None = None,
):
    model = build_model(
        contacts_per_cell,
        donor_idx,
        obs_time,
        donor_num=donor_num,
        lambda_prior_bounds=lambda_prior_bounds,
        sigma_lambda_prior=std_prior_factor,
        phi_0_prior=p_prior_bounds,
        deviation_prior=deviation_prior,
    )
    return _infer_model(
        model,
        draws=draws,
        chains=chains,
        cores=cores,
        random_seed=random_seed,
        threshold=threshold,
        correlation_threshold=correlation_threshold,
        progress_callback=progress_callback,
    )
