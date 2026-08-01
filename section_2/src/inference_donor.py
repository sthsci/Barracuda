#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt


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
class _DonorHierarchy:
    mu_lambda_donor: Any
    mu_lambda_avg: Any
    sigma_lambda_donor: Any | None = None
    phi_0_donor: Any | None = None
    sigma_lambda_avg: Any | None = None
    phi_0_avg: Any | None = None


# Input validation -----------------------------------------------------------


def _validate_counts(
    contacts_per_cell,
    donor_idx,
    obs_time: float,
    donor_num: int,
) -> Tuple[np.ndarray, np.ndarray, float]:
    T = float(obs_time)
    if not np.isfinite(T) or T <= 0:
        raise ValueError("obs_time must be finite and > 0")

    raw_counts = np.asarray(contacts_per_cell)
    if raw_counts.ndim != 1:
        raise ValueError("contacts_per_cell must be 1D")
    if raw_counts.size == 0:
        raise ValueError("contacts_per_cell must contain at least one count")
    if not np.issubdtype(raw_counts.dtype, np.number):
        raise ValueError("contacts_per_cell must contain numeric counts")
    if not np.all(np.isfinite(raw_counts)):
        raise ValueError("contacts_per_cell must contain finite values")
    if np.any(raw_counts < 0) or np.any(raw_counts != np.floor(raw_counts)):
        raise ValueError("contacts_per_cell must contain integer values >= 0")

    counts = raw_counts.astype(int)

    donor_num = int(donor_num)
    if donor_num <= 0:
        raise ValueError("donor_num must be > 0")

    raw_donor_idx = np.asarray(donor_idx)
    if raw_donor_idx.shape != counts.shape:
        raise ValueError("donor_idx must have one integer value per cell")
    if not np.issubdtype(raw_donor_idx.dtype, np.number):
        raise ValueError("donor_idx must contain numeric donor indices")
    if not np.all(np.isfinite(raw_donor_idx)):
        raise ValueError("donor_idx must contain finite values")
    if np.any(raw_donor_idx != np.floor(raw_donor_idx)):
        raise ValueError("donor_idx must contain integer values")

    donor_index = raw_donor_idx.astype(int)
    if np.any(donor_index < 0) or np.any(donor_index >= donor_num):
        raise ValueError(
            f"donor_idx values must be between 0 and {donor_num - 1}"
        )

    return counts, donor_index, T


def _prepare_donor_data(
    contacts_per_cell,
    donor_idx,
    obs_time: float,
    donor_num: int,
) -> _DonorData:
    counts, donor_index, observation_time = _validate_counts(
        contacts_per_cell,
        donor_idx,
        obs_time,
        donor_num,
    )

    donor_cell_counts = np.bincount(
        donor_index,
        minlength=int(donor_num),
    ).astype(float)

    donor_weights = donor_cell_counts / donor_cell_counts.sum()

    return _DonorData(
        counts=counts,
        donor_index=donor_index,
        donor_weights=donor_weights,
        observation_time=observation_time,
        donor_num=int(donor_num),
    )


# SMC sampling and evidence --------------------------------------------------


def _resolve_cores(cores) -> int:
    cpu = os.cpu_count() or 1
    if cores is None:
        return cpu
    try:
        cores = int(cores)
    except Exception:
        return cpu
    if cores <= 0:
        return cpu
    return min(cores, cpu)


def _sample_smc(
    *,
    draws: int,
    chains: int,
    cores=None,
    random_seed: Optional[int] = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
):
    trace = pm.sample_smc(
        draws=int(draws),
        chains=int(chains),
        cores=_resolve_cores(cores),
        random_seed=random_seed,
        progressbar=True,
        return_inferencedata=False,
        threshold=float(threshold),
        correlation_threshold=float(correlation_threshold),
    )

    to_idata = getattr(pm, "to_inferencedata", None) or getattr(
        pm, "to_inference_data", None
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


def _store_smc_log_marginal_likelihood(idata, trace, *, chains: int) -> None:
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
        idata.sample_stats["log_marginal_likelihood"] = (
            ("chain",),
            logml,
        )
    except Exception:
        idata.attrs["log_marginal_likelihood"] = logml.tolist()


def _parse_log_marginal_likelihood(
    raw,
    *,
    chains: int,
) -> Optional[np.ndarray]:
    if not isinstance(raw, (list, tuple, np.ndarray)):
        try:
            return np.full(chains, float(raw), dtype=float)
        except Exception:
            return None

    if all(
        not isinstance(item, (list, tuple, np.ndarray))
        for item in raw
    ):
        arr = np.asarray(raw, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            return np.full(chains, float(arr[-1]), dtype=float)
        return None

    values: list[float] = []

    for item in raw:
        if isinstance(item, (list, tuple, np.ndarray)):
            arr = np.asarray(item, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                values.append(float(arr[-1]))
            continue

        try:
            values.append(float(item))
        except Exception:
            continue

    values = [value for value in values if np.isfinite(value)]
    if not values:
        return None

    logml = np.asarray(values, dtype=float)

    if logml.size < chains:
        logml = np.concatenate(
            [
                logml,
                np.full(chains - logml.size, np.nan),
            ]
        )

    return logml[:chains]


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
                "sample_stats['log_marginal_likelihood']."
            )

        values = np.asarray(
            attrs["log_marginal_likelihood"],
            dtype=float,
        )

    values = values[np.isfinite(values)]

    if not values.size:
        raise RuntimeError("Could not parse SMC log evidence values.")

    return float(np.mean(values))


def _summarize(idata) -> None:
    print(az.summary(idata, hdi_prob=0.95))


def _fit_model(
    model: pm.Model,
    *,
    draws: int,
    chains: int,
    cores=None,
    random_seed: Optional[int] = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
):
    with model:
        idata = _sample_smc(
            draws=draws,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            threshold=threshold,
            correlation_threshold=correlation_threshold,
        )
        _summarize(idata)

    return {
        "idata": idata,
        "model": model,
    }


# Likelihood mathematics -----------------------------------------------------


def _gamma_shape_rate_from_mean_sd(mu, sd):
    sd = pt.maximum(sd, 1e-12)
    variance = sd**2
    return mu**2 / variance, mu / variance


def _gamma_marginal_logI(
    n_t,
    T: float,
    mu_lam,
    sig_lam,
    *,
    max_count: int,
):
    shape, rate = _gamma_shape_rate_from_mean_sd(
        mu_lam,
        sig_lam,
    )

    max_count = int(max_count)
    if max_count < 0:
        raise ValueError("max_count must be >= 0")

    log_rising = pt.zeros_like(n_t)

    for k in range(max_count):
        log_rising = pt.switch(
            n_t > float(k),
            log_rising + pt.log(shape + float(k)),
            log_rising,
        )

    return (
        log_rising
        - n_t * pt.log(rate)
        - (shape + n_t) * pt.log1p(float(T) / rate)
    )


def _poisson_count_constant(n_t, T: float):
    return (
        n_t * pt.log(float(T))
        - pt.gammaln(n_t + 1.0)
    )


# Shared donor hierarchy -----------------------------------------------------


def _build_donor_hierarchy(
    donor_num: int,
    donor_weights: np.ndarray,
    *,
    lambda_prior_bounds: tuple[float, float],
    mu_zeta_prior: float,
    sigma_lambda_prior: float | None = None,
    sigma_zeta_prior: float | None = None,
    phi_0_prior: tuple[float, float] | None = None,
    phi_zeta_prior: float | None = None,
) -> _DonorHierarchy:
    """Create population centres, deviations, and donor parameters."""
    include_sigma = sigma_lambda_prior is not None
    include_phi = phi_0_prior is not None

    if include_sigma != (sigma_zeta_prior is not None):
        raise ValueError(
            "sigma_lambda_prior and sigma_zeta_prior "
            "must be provided together"
        )

    if include_phi != (phi_zeta_prior is not None):
        raise ValueError(
            "phi_0_prior and phi_zeta_prior "
            "must be provided together"
        )

    donor_weights = np.asarray(
        donor_weights,
        dtype=float,
    )

    if donor_weights.shape != (donor_num,):
        raise ValueError(
            "donor_weights must have shape (donor_num,)"
        )

    if not np.all(np.isfinite(donor_weights)):
        raise ValueError(
            "donor_weights must contain finite values"
        )

    if np.any(donor_weights < 0):
        raise ValueError(
            "donor_weights must be non-negative"
        )

    weight_total = donor_weights.sum()
    if weight_total <= 0:
        raise ValueError(
            "donor_weights must have a positive sum"
        )

    donor_weights = donor_weights / weight_total
    weights_t = pt.as_tensor_variable(donor_weights)

    eta = pm.Uniform(
        "eta",
        lower=float(lambda_prior_bounds[0]),
        upper=float(lambda_prior_bounds[1]),
    )

    mu_lambda_ref = pm.Deterministic(
        "mu_lambda_ref",
        10.0**eta,
    )

    sigma_lambda_ref = (
        pm.HalfNormal(
            "sigma_lambda_ref",
            sigma=float(sigma_lambda_prior),
        )
        if include_sigma
        else None
    )

    phi_0_ref = (
        pm.Beta(
            "phi_0_ref",
            alpha=float(phi_0_prior[0]),
            beta=float(phi_0_prior[1]),
        )
        if include_phi
        else None
    )

    zeta_mu_lambda = pm.Normal(
        "zeta_mu_lambda",
        mu=0.0,
        sigma=float(mu_zeta_prior),
        shape=donor_num,
    )

    zeta_sigma_lambda = (
        pm.Normal(
            "zeta_sigma_lambda",
            mu=0.0,
            sigma=float(sigma_zeta_prior),
            shape=donor_num,
        )
        if include_sigma
        else None
    )

    zeta_phi_0 = (
        pm.Normal(
            "zeta_phi_0",
            mu=0.0,
            sigma=float(phi_zeta_prior),
            shape=donor_num,
        )
        if include_phi
        else None
    )

    mu_lambda_donor = pm.Deterministic(
        "mu_lambda_donor",
        mu_lambda_bar * pt.exp(zeta_mu_lambda),
    )

    sigma_lambda_donor = (
        pm.Deterministic(
            "sigma_lambda_donor",
            sigma_lambda_bar * pt.exp(zeta_sigma_lambda),
        )
        if include_sigma
        else None
    )

    phi_0_donor = None

    if include_phi:
        phi_safe = pt.clip(
            phi_0_bar,
            1e-12,
            1.0 - 1e-12,
        )

        phi_logit = (
            pt.log(phi_safe)
            - pt.log1p(-phi_safe)
        )

        phi_0_donor = pm.Deterministic(
            "phi_0_donor",
            pm.math.sigmoid(
                phi_logit + zeta_phi_0
            ),
        )

    phi_0_avg = (
        pm.Deterministic(
            "phi_0_avg",
            pt.sum(
                weights_t * phi_0_donor
            ),
        )
        if include_phi
        else None
    )

    if include_phi:
        active_mass = (
            weights_t
            * (1.0 - phi_0_donor)
        )

        active_mass_total = pt.sum(active_mass)

        parameter_weights = (
            active_mass
            / pt.maximum(
                active_mass_total,
                1e-12,
            )
        )
    else:
        parameter_weights = weights_t

    mu_lambda_avg = pm.Deterministic(
        "mu_lambda_avg",
        pt.sum(
            parameter_weights
            * mu_lambda_donor
        ),
    )

    sigma_lambda_avg = None

    if include_sigma:
        lambda_second_moment = pt.sum(
            parameter_weights
            * (
                sigma_lambda_donor**2
                + mu_lambda_donor**2
            )
        )

        lambda_variance = pt.maximum(
            lambda_second_moment
            - mu_lambda_avg**2,
            0.0,
        )

        sigma_lambda_avg = pm.Deterministic(
            "sigma_lambda_avg",
            pt.sqrt(lambda_variance),
        )

    return _DonorHierarchy(
        mu_lambda_donor=mu_lambda_donor,
        sigma_lambda_donor=sigma_lambda_donor,
        phi_0_donor=phi_0_donor,
        mu_lambda_avg=mu_lambda_avg,
        sigma_lambda_avg=sigma_lambda_avg,
        phi_0_avg=phi_0_avg,
    )


# Model builders -------------------------------------------------------------


def _build_homo_model(
    contacts_per_cell,
    donor_idx,
    obs_time: float,
    *,
    donor_num: int,
    lambda_prior_bounds: tuple[float, float],
    zeta_prior: float,
) -> pm.Model:
    data = _prepare_donor_data(
        contacts_per_cell,
        donor_idx,
        obs_time,
        donor_num,
    )

    donor_idx_t = pt.as_tensor_variable(
        data.donor_index
    )

    with pm.Model() as model:
        hierarchy = _build_donor_hierarchy(
            data.donor_num,
            data.donor_weights,
            lambda_prior_bounds=lambda_prior_bounds,
            mu_zeta_prior=zeta_prior,
        )

        pm.Poisson(
            "contacts",
            mu=(
                hierarchy.mu_lambda_donor[donor_idx_t]
                * data.observation_time
            ),
            observed=data.counts,
        )

    return model


def _build_z2p_model(
    contacts_per_cell,
    donor_idx,
    obs_time: float,
    *,
    donor_num: int,
    lambda_prior_bounds: tuple[float, float],
    p_prior_bounds: tuple[float, float],
    zeta_prior: tuple[float, float],
) -> pm.Model:
    data = _prepare_donor_data(
        contacts_per_cell,
        donor_idx,
        obs_time,
        donor_num,
    )

    donor_idx_t = pt.as_tensor_variable(
        data.donor_index
    )

    with pm.Model() as model:
        hierarchy = _build_donor_hierarchy(
            data.donor_num,
            data.donor_weights,
            lambda_prior_bounds=lambda_prior_bounds,
            mu_zeta_prior=zeta_prior[0],
            phi_0_prior=p_prior_bounds,
            phi_zeta_prior=zeta_prior[1],
        )

        pm.ZeroInflatedPoisson(
            "contacts",
            psi=(
                1.0
                - hierarchy.phi_0_donor[donor_idx_t]
            ),
            mu=(
                hierarchy.mu_lambda_donor[donor_idx_t]
                * data.observation_time
            ),
            observed=data.counts,
        )

    return model


def _build_dis2p_model(
    contacts_per_cell,
    donor_idx,
    obs_time: float,
    *,
    donor_num: int,
    lambda_prior_bounds: tuple[float, float],
    std_prior_factor: float,
    zeta_prior: tuple[float, float],
) -> pm.Model:
    data = _prepare_donor_data(
        contacts_per_cell,
        donor_idx,
        obs_time,
        donor_num,
    )

    n_t = pt.as_tensor_variable(
        data.counts.astype(float)
    )

    donor_idx_t = pt.as_tensor_variable(
        data.donor_index
    )

    with pm.Model() as model:
        hierarchy = _build_donor_hierarchy(
            data.donor_num,
            data.donor_weights,
            lambda_prior_bounds=lambda_prior_bounds,
            mu_zeta_prior=zeta_prior[0],
            sigma_lambda_prior=std_prior_factor,
            sigma_zeta_prior=zeta_prior[1],
        )

        cell_logp = (
            _poisson_count_constant(
                n_t,
                data.observation_time,
            )
            + _gamma_marginal_logI(
                n_t,
                data.observation_time,
                hierarchy.mu_lambda_donor[donor_idx_t],
                hierarchy.sigma_lambda_donor[donor_idx_t],
                max_count=data.max_count,
            )
        )

        pm.Potential(
            "gamma_donor_loglik",
            pt.sum(cell_logp),
        )

    return model


def _build_hetero3_model(
    contacts_per_cell,
    donor_idx,
    obs_time: float,
    *,
    donor_num: int,
    eta_prior_bounds: tuple[float, float],
    sigma_lambda_prior: float,
    phi_0_prior: tuple[float, float],
    zeta_prior: tuple[float, float, float],
) -> pm.Model:
    data = _prepare_donor_data(
        contacts_per_cell,
        donor_idx,
        obs_time,
        donor_num,
    )

    n_t = pt.as_tensor_variable(
        data.counts.astype(float)
    )

    donor_idx_t = pt.as_tensor_variable(
        data.donor_index
    )

    with pm.Model() as model:
        hierarchy = _build_donor_hierarchy(
            data.donor_num,
            data.donor_weights,
            lambda_prior_bounds=eta_prior_bounds,
            mu_zeta_prior=zeta_prior[0],
            sigma_lambda_prior=sigma_lambda_prior,
            sigma_zeta_prior=zeta_prior[1],
            phi_0_prior=phi_0_prior,
            phi_zeta_prior=zeta_prior[2],
        )

        mu_cell = (
            hierarchy.mu_lambda_donor[donor_idx_t]
        )

        sigma_cell = (
            hierarchy.sigma_lambda_donor[donor_idx_t]
        )

        phi_cell = pt.clip(
            hierarchy.phi_0_donor[donor_idx_t],
            1e-12,
            1.0 - 1e-12,
        )

        gamma_logp = (
            _poisson_count_constant(
                n_t,
                data.observation_time,
            )
            + _gamma_marginal_logI(
                n_t,
                data.observation_time,
                mu_cell,
                sigma_cell,
                max_count=data.max_count,
            )
        )

        log_active = (
            pt.log1p(-phi_cell)
            + gamma_logp
        )

        active_zero_logp = _gamma_marginal_logI(
            pt.zeros_like(n_t),
            data.observation_time,
            mu_cell,
            sigma_cell,
            max_count=0,
        )

        log_zero = pt.logaddexp(
            pt.log(phi_cell),
            (
                pt.log1p(-phi_cell)
                + active_zero_logp
            ),
        )

        cell_logp = pt.switch(
            pt.eq(n_t, 0.0),
            log_zero,
            log_active,
        )

        pm.Potential(
            "zi_gamma_donor_loglik",
            pt.sum(cell_logp),
        )

    return model


# Public API -----------------------------------------------------------------


def math_model(
    contacts_per_cell,
    donor_idx,
    obs_time: float,
    eta_prior_bounds: tuple[float, float] = (-1.0, 2.0),
    sigma_lambda_prior: float = 1.0,
    phi_0_prior: tuple[float, float] = (1.0, 1.0),
    zeta_prior: tuple[float, float, float] = (
        0.2,
        0.2,
        0.35,
    ),
    donor_num: int = 4,
):
    """Build the donor-aware zero-inflated Gamma-Poisson model."""
    return _build_hetero3_model(
        contacts_per_cell,
        donor_idx,
        obs_time,
        donor_num=donor_num,
        eta_prior_bounds=eta_prior_bounds,
        sigma_lambda_prior=sigma_lambda_prior,
        phi_0_prior=phi_0_prior,
        zeta_prior=zeta_prior,
    )


def inference_homo(
    contacts_per_cell,
    donor_idx,
    obs_time: float,
    *,
    donor_num: int = 4,
    draws: int = 3000,
    chains: int = 4,
    cores=None,
    lambda_prior_bounds=(-1.0, 2.0),
    zeta_prior: float = 0.2,
    random_seed: Optional[int] = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
):
    """Fit the donor-aware homogeneous Poisson model."""
    model = _build_homo_model(
        contacts_per_cell,
        donor_idx,
        obs_time,
        donor_num=donor_num,
        lambda_prior_bounds=lambda_prior_bounds,
        zeta_prior=zeta_prior,
    )

    return _fit_model(
        model,
        draws=draws,
        chains=chains,
        cores=cores,
        random_seed=random_seed,
        threshold=threshold,
        correlation_threshold=correlation_threshold,
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
    zeta_prior: tuple[float, float] = (
        0.2,
        0.35,
    ),
    random_seed: Optional[int] = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
):
    """Fit the donor-aware zero-inflated Poisson model."""
    model = _build_z2p_model(
        contacts_per_cell,
        donor_idx,
        obs_time,
        donor_num=donor_num,
        lambda_prior_bounds=lambda_prior_bounds,
        p_prior_bounds=p_prior_bounds,
        zeta_prior=zeta_prior,
    )

    return _fit_model(
        model,
        draws=draws,
        chains=chains,
        cores=cores,
        random_seed=random_seed,
        threshold=threshold,
        correlation_threshold=correlation_threshold,
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
    zeta_prior: tuple[float, float] = (
        0.2,
        0.2,
    ),
    random_seed: Optional[int] = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
):
    """Fit the donor-aware Gamma-Poisson model."""
    model = _build_dis2p_model(
        contacts_per_cell,
        donor_idx,
        obs_time,
        donor_num=donor_num,
        lambda_prior_bounds=lambda_prior_bounds,
        std_prior_factor=std_prior_factor,
        zeta_prior=zeta_prior,
    )

    return _fit_model(
        model,
        draws=draws,
        chains=chains,
        cores=cores,
        random_seed=random_seed,
        threshold=threshold,
        correlation_threshold=correlation_threshold,
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
    zeta_prior: tuple[float, float, float] = (
        0.2,
        0.2,
        0.35,
    ),
    random_seed: Optional[int] = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
):
    """Fit the donor-aware zero-inflated Gamma-Poisson model."""
    model = math_model(
        contacts_per_cell,
        donor_idx,
        obs_time,
        eta_prior_bounds=lambda_prior_bounds,
        sigma_lambda_prior=std_prior_factor,
        phi_0_prior=p_prior_bounds,
        zeta_prior=zeta_prior,
        donor_num=donor_num,
    )

    return _fit_model(
        model,
        draws=draws,
        chains=chains,
        cores=cores,
        random_seed=random_seed,
        threshold=threshold,
        correlation_threshold=correlation_threshold,
    )