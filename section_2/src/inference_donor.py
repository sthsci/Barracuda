#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Optional, Tuple

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt


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
    if not np.issubdtype(raw_counts.dtype, np.number):
        raise ValueError("contacts_per_cell must contain numeric counts")
    if not np.all(np.isfinite(raw_counts)):
        raise ValueError("contacts_per_cell must contain finite values")
    if np.any(raw_counts < 0) or np.any(raw_counts != np.floor(raw_counts)):
        raise ValueError("contacts_per_cell must contain integer values >= 0")
    N = raw_counts.astype(int)

    donor_num = int(donor_num)
    if donor_num <= 0:
        raise ValueError("donor_num must be > 0")

    raw_donor_idx = np.asarray(donor_idx)
    if raw_donor_idx.shape != N.shape:
        raise ValueError("donor_idx must have one integer value per cell")
    if not np.issubdtype(raw_donor_idx.dtype, np.number):
        raise ValueError("donor_idx must contain numeric donor indices")
    if not np.all(np.isfinite(raw_donor_idx)):
        raise ValueError("donor_idx must contain finite values")
    if np.any(raw_donor_idx != np.floor(raw_donor_idx)):
        raise ValueError("donor_idx must contain integer values")

    donor_idx_array = raw_donor_idx.astype(int)
    if np.any(donor_idx_array < 0) or np.any(donor_idx_array >= donor_num):
        raise ValueError(f"donor_idx values must be between 0 and {donor_num - 1}")

    return N, donor_idx_array, T


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
        raise RuntimeError("PyMC does not expose to_inferencedata/to_inference_data")

    idata = to_idata(trace, log_likelihood=True)
    _store_smc_log_marginal_likelihood(idata, trace, chains=int(chains))
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
        idata.sample_stats["log_marginal_likelihood"] = (("chain",), logml)
    except Exception:
        idata.attrs["log_marginal_likelihood"] = logml.tolist()


def _parse_log_marginal_likelihood(raw, *, chains: int) -> Optional[np.ndarray]:
    if not isinstance(raw, (list, tuple, np.ndarray)):
        try:
            return np.full(chains, float(raw), dtype=float)
        except Exception:
            return None

    if all(not isinstance(item, (list, tuple, np.ndarray)) for item in raw):
        arr = np.asarray(raw, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            return np.full(chains, float(arr[-1]), dtype=float)
        return None

    vals: list[float] = []
    for item in raw:
        if isinstance(item, (list, tuple, np.ndarray)):
            arr = np.asarray(item, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                vals.append(float(arr[-1]))
            continue
        try:
            vals.append(float(item))
        except Exception:
            continue

    vals = [val for val in vals if np.isfinite(val)]
    if not vals:
        return None

    logml = np.asarray(vals, dtype=float)
    if logml.size < chains:
        logml = np.concatenate([logml, np.full(chains - logml.size, np.nan)])
    return logml[:chains]


def smc_log_evidence(idata) -> float:
    sample_stats = getattr(idata, "sample_stats", None)
    if sample_stats is not None and "log_marginal_likelihood" in getattr(
        sample_stats, "data_vars", {}
    ):
        vals = np.asarray(sample_stats["log_marginal_likelihood"].values, dtype=float)
    else:
        attrs = getattr(idata, "attrs", {})
        if "log_marginal_likelihood" not in attrs:
            raise RuntimeError(
                "SMC evidence missing: sample_stats['log_marginal_likelihood']."
            )
        vals = np.asarray(attrs["log_marginal_likelihood"], dtype=float)

    vals = vals[np.isfinite(vals)]
    if not vals.size:
        raise RuntimeError("Could not parse SMC log evidence values.")
    return float(np.mean(vals))


def _summarize(idata) -> None:
    print(az.summary(idata, hdi_prob=0.95))


def _gamma_shape_rate_from_mean_sd(mu, sd):
    sd = pt.maximum(sd, 1e-12)
    variance = sd**2
    shape = mu**2 / variance
    rate = mu / variance
    return shape, rate


def _gamma_marginal_logI(n_t, T: float, mu_lam, sig_lam, *, max_count: int):
    shape, rate = _gamma_shape_rate_from_mean_sd(mu_lam, sig_lam)
    max_count = int(max_count)
    if max_count < 0:
        raise ValueError("max_count must be >= 0")

    # Stable evaluation of log Gamma(shape + n) - log Gamma(shape).
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
    return n_t * pt.log(float(T)) - pt.gammaln(n_t + 1.0)


def math_model(
    contacts_per_cell,
    donor_idx,
    obs_time: float,
    eta_prior_bounds: tuple[float, float] = (-1.0, 2.0),
    sigma_lambda_prior: float = 1.0,
    phi_0_prior: tuple[float, float] = (1.0, 1.0),
    zeta_prior: tuple[float, float, float] = (0.2, 0.2, 0.35),
    donor_num: int = 4,
):
    """Build the donor-aware zero-inflated Gamma-Poisson model."""
    N, donor_idx_array, T = _validate_counts(
        contacts_per_cell,
        donor_idx,
        obs_time,
        donor_num,
    )
    n_t = pt.as_tensor_variable(N.astype(float))
    donor_idx_t = pt.as_tensor_variable(donor_idx_array)
    max_count = int(N.max(initial=0))

    with pm.Model() as model:
        # Population centres across donors.
        eta = pm.Uniform(
            "eta",
            lower=float(eta_prior_bounds[0]),
            upper=float(eta_prior_bounds[1]),
        )
        mu_lambda_bar = pm.Deterministic("mu_lambda_bar", 10.0**eta)
        sigma_lambda_bar = pm.HalfNormal(
            "sigma_lambda_bar",
            sigma=float(sigma_lambda_prior),
        )
        phi_0_bar = pm.Beta(
            "phi_0_bar",
            alpha=float(phi_0_prior[0]),
            beta=float(phi_0_prior[1]),
        )

        # Donor deviations. The small prior scales enforce partial pooling.
        zeta_mu_lambda = pm.Normal(
            "zeta_mu_lambda",
            mu=0.0,
            sigma=float(zeta_prior[0]),
            shape=donor_num,
        )
        zeta_sigma_lambda = pm.Normal(
            "zeta_sigma_lambda",
            mu=0.0,
            sigma=float(zeta_prior[1]),
            shape=donor_num,
        )
        zeta_phi_0 = pm.Normal(
            "zeta_phi_0",
            mu=0.0,
            sigma=float(zeta_prior[2]),
            shape=donor_num,
        )

        # Transformations keep rates positive and zero fractions in (0, 1).
        mu_lambda_donor = pm.Deterministic(
            "mu_lambda_donor",
            mu_lambda_bar * pt.exp(zeta_mu_lambda),
        )
        sigma_lambda_donor = pm.Deterministic(
            "sigma_lambda_donor",
            sigma_lambda_bar * pt.exp(zeta_sigma_lambda),
        )
        phi_0_bar_safe = pt.clip(phi_0_bar, 1e-12, 1.0 - 1e-12)
        phi_0_bar_logit = pt.log(phi_0_bar_safe) - pt.log1p(-phi_0_bar_safe)
        phi_0_donor = pm.Deterministic(
            "phi_0_donor",
            pm.math.sigmoid(phi_0_bar_logit + zeta_phi_0),
        )

        mu_lambda_cell = mu_lambda_donor[donor_idx_t]
        sigma_lambda_cell = sigma_lambda_donor[donor_idx_t]
        phi_0_cell = pt.clip(phi_0_donor[donor_idx_t], 1e-12, 1.0 - 1e-12)

        gamma_logp = _poisson_count_constant(n_t, T) + _gamma_marginal_logI(
            n_t,
            T,
            mu_lambda_cell,
            sigma_lambda_cell,
            max_count=max_count,
        )
        log_active = pt.log1p(-phi_0_cell) + gamma_logp
        active_zero_logp = _gamma_marginal_logI(
            pt.zeros_like(n_t),
            T,
            mu_lambda_cell,
            sigma_lambda_cell,
            max_count=0,
        )
        log_zero = pt.logaddexp(
            pt.log(phi_0_cell),
            pt.log1p(-phi_0_cell) + active_zero_logp,
        )
        cell_logp = pt.switch(pt.eq(n_t, 0.0), log_zero, log_active)
        pm.Potential("zi_gamma_donor_loglik", pt.sum(cell_logp))

    return model


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
    N, donor_idx_array, T = _validate_counts(
        contacts_per_cell,
        donor_idx,
        obs_time,
        donor_num,
    )
    donor_idx_t = pt.as_tensor_variable(donor_idx_array)

    with pm.Model() as model:
        eta = pm.Uniform(
            "eta",
            lower=float(lambda_prior_bounds[0]),
            upper=float(lambda_prior_bounds[1]),
        )
        mu_lambda_bar = pm.Deterministic("mu_lambda_bar", 10.0**eta)
        zeta_mu_lambda = pm.Normal(
            "zeta_mu_lambda",
            mu=0.0,
            sigma=float(zeta_prior),
            shape=donor_num,
        )
        mu_lambda_donor = pm.Deterministic(
            "mu_lambda_donor",
            mu_lambda_bar * pt.exp(zeta_mu_lambda),
        )
        pm.Poisson(
            "contacts",
            mu=mu_lambda_donor[donor_idx_t] * T,
            observed=N,
        )

        idata = _sample_smc(
            draws=draws,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            threshold=threshold,
            correlation_threshold=correlation_threshold,
        )
        _summarize(idata)

    return {"idata": idata, "model": model}


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
    zeta_prior: tuple[float, float] = (0.2, 0.35),
    random_seed: Optional[int] = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
):
    N, donor_idx_array, T = _validate_counts(
        contacts_per_cell,
        donor_idx,
        obs_time,
        donor_num,
    )
    donor_idx_t = pt.as_tensor_variable(donor_idx_array)

    with pm.Model() as model:
        eta = pm.Uniform(
            "eta",
            lower=float(lambda_prior_bounds[0]),
            upper=float(lambda_prior_bounds[1]),
        )
        mu_lambda_bar = pm.Deterministic("mu_lambda_bar", 10.0**eta)
        phi_0_bar = pm.Beta(
            "phi_0_bar",
            alpha=float(p_prior_bounds[0]),
            beta=float(p_prior_bounds[1]),
        )
        zeta_mu_lambda = pm.Normal(
            "zeta_mu_lambda",
            mu=0.0,
            sigma=float(zeta_prior[0]),
            shape=donor_num,
        )
        zeta_phi_0 = pm.Normal(
            "zeta_phi_0",
            mu=0.0,
            sigma=float(zeta_prior[1]),
            shape=donor_num,
        )
        mu_lambda_donor = pm.Deterministic(
            "mu_lambda_donor",
            mu_lambda_bar * pt.exp(zeta_mu_lambda),
        )
        phi_0_bar_safe = pt.clip(phi_0_bar, 1e-12, 1.0 - 1e-12)
        phi_0_bar_logit = pt.log(phi_0_bar_safe) - pt.log1p(-phi_0_bar_safe)
        phi_0_donor = pm.Deterministic(
            "phi_0_donor",
            pm.math.sigmoid(phi_0_bar_logit + zeta_phi_0),
        )
        pm.ZeroInflatedPoisson(
            "contacts",
            psi=1.0 - phi_0_donor[donor_idx_t],
            mu=mu_lambda_donor[donor_idx_t] * T,
            observed=N,
        )

        idata = _sample_smc(
            draws=draws,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            threshold=threshold,
            correlation_threshold=correlation_threshold,
        )
        _summarize(idata)

    return {"idata": idata, "model": model}


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
    zeta_prior: tuple[float, float] = (0.2, 0.2),
    random_seed: Optional[int] = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
):
    N, donor_idx_array, T = _validate_counts(
        contacts_per_cell,
        donor_idx,
        obs_time,
        donor_num,
    )
    n_t = pt.as_tensor_variable(N.astype(float))
    donor_idx_t = pt.as_tensor_variable(donor_idx_array)
    max_count = int(N.max(initial=0))

    with pm.Model() as model:
        eta = pm.Uniform(
            "eta",
            lower=float(lambda_prior_bounds[0]),
            upper=float(lambda_prior_bounds[1]),
        )
        mu_lambda_bar = pm.Deterministic("mu_lambda_bar", 10.0**eta)
        sigma_lambda_bar = pm.HalfNormal(
            "sigma_lambda_bar",
            sigma=float(std_prior_factor),
        )
        zeta_mu_lambda = pm.Normal(
            "zeta_mu_lambda",
            mu=0.0,
            sigma=float(zeta_prior[0]),
            shape=donor_num,
        )
        zeta_sigma_lambda = pm.Normal(
            "zeta_sigma_lambda",
            mu=0.0,
            sigma=float(zeta_prior[1]),
            shape=donor_num,
        )
        mu_lambda_donor = pm.Deterministic(
            "mu_lambda_donor",
            mu_lambda_bar * pt.exp(zeta_mu_lambda),
        )
        sigma_lambda_donor = pm.Deterministic(
            "sigma_lambda_donor",
            sigma_lambda_bar * pt.exp(zeta_sigma_lambda),
        )
        cell_logp = _poisson_count_constant(n_t, T) + _gamma_marginal_logI(
            n_t,
            T,
            mu_lambda_donor[donor_idx_t],
            sigma_lambda_donor[donor_idx_t],
            max_count=max_count,
        )
        pm.Potential("gamma_donor_loglik", pt.sum(cell_logp))

        idata = _sample_smc(
            draws=draws,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            threshold=threshold,
            correlation_threshold=correlation_threshold,
        )
        _summarize(idata)

    return {"idata": idata, "model": model}


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
    zeta_prior: tuple[float, float, float] = (0.2, 0.2, 0.35),
    random_seed: Optional[int] = None,
    threshold: float = 0.5,
    correlation_threshold: float = 0.01,
):
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

    return {"idata": idata, "model": model}
