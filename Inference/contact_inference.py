#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any, Literal, Optional, Tuple

import numpy as np
import pymc as pm
import pytensor.tensor as pt

DistName = Literal["gamma", "lognormal", "truncnorm"]
SamplerName = Literal["nuts", "smc"]


def _resolve_cores(cores, *, chains: int) -> int:
    cpu = os.cpu_count() or 1
    if cores is None:
        return int(cpu)
    try:
        c = int(cores)
    except Exception:
        return int(cpu)
    if c <= 0:
        return int(cpu)
    return int(min(c, cpu))


def _gamma_shape_rate_from_mean_sd(
    mean: pm.TensorVariable, sd: pm.TensorVariable
) -> Tuple[pm.TensorVariable, pm.TensorVariable]:
    shape = (mean / sd) ** 2
    rate = mean / (sd * sd)
    return shape, rate


def _lognormal_mu_sigma_from_mean_sd(
    mean: pm.TensorVariable, sd: pm.TensorVariable
) -> Tuple[pm.TensorVariable, pm.TensorVariable]:
    sigma2 = pm.math.log(1.0 + (sd / mean) ** 2)
    mu = pm.math.log(mean) - 0.5 * sigma2
    return mu, pm.math.sqrt(sigma2)


def _stack_dt_list(dt_list) -> Tuple[np.ndarray, np.ndarray]:
    values = []
    idx = []
    for i, dt in enumerate(dt_list):
        arr = np.asarray(dt, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size:
            values.append(arr)
            idx.append(np.full(arr.size, int(i), dtype=np.int64))
    if not values:
        return np.array([], dtype=float), np.array([], dtype=np.int64)
    return np.concatenate(values).astype(float), np.concatenate(idx).astype(np.int64)


def _sample_backend(
    *,
    sampler: SamplerName,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    cores,
    random_seed: Optional[int],
    smc_draws: Optional[int] = None,
    smc_cores=None,
    smc_threshold: float = 0.5,
    smc_correlation_threshold: float = 0.01,
):
    if sampler == "smc":
        draws_smc = int(draws if smc_draws is None else smc_draws)
        cores_smc = smc_cores if smc_cores is not None else cores
        idata = pm.sample_smc(
            draws=int(draws_smc),
            chains=int(chains),
            cores=int(_resolve_cores(cores_smc, chains=int(chains))),
            random_seed=random_seed,
            progressbar=True,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
            threshold=float(smc_threshold),
            correlation_threshold=float(smc_correlation_threshold),
        )
        return idata

    cores_to_use = _resolve_cores(cores, chains=int(chains))
    idata = pm.sample(
        draws=int(draws),
        tune=int(tune),
        chains=int(chains),
        cores=int(cores_to_use),
        target_accept=float(target_accept),
        random_seed=random_seed,
        progressbar=True,
        idata_kwargs={"log_likelihood": True},
    )
    return idata


def smc_log_evidence(idata) -> float:
    ss = getattr(idata, "sample_stats", None)
    if ss is None or "log_marginal_likelihood" not in getattr(ss, "data_vars", {}):
        raise RuntimeError("SMC evidence missing: sample_stats['log_marginal_likelihood'].")

    v = ss["log_marginal_likelihood"].values
    flat = np.asarray(v, dtype=object).ravel()

    per_chain = []
    for x in flat:
        if x is None:
            continue
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                per_chain.append(float(arr[-1]))
            continue
        try:
            fx = float(x)
        except Exception:
            continue
        if np.isfinite(fx):
            per_chain.append(float(fx))

    if not per_chain:
        raise RuntimeError("Could not parse SMC log evidence values.")
    return float(np.mean(per_chain))


def inference_homo(
    kills_per_cell,
    obs_time: float,
    dt_data=None,
    mode: str = "counts+gaps",
    sampler: SamplerName = "nuts",
    draws: int = 3000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    cores=None,
    lambda_prior_bounds=(-5.0, 2.0),
    random_seed: Optional[int] = None,
    smc_draws: Optional[int] = None,
    smc_cores=None,
    smc_threshold: float = 0.5,
    smc_correlation_threshold: float = 0.01,
):
    mode_s = str(mode).strip().lower()
    if mode_s not in {"counts", "counts+gaps"}:
        raise ValueError("mode must be 'counts' or 'counts+gaps'")

    T = float(obs_time)
    if T <= 0:
        raise ValueError("obs_time must be > 0")

    N = np.asarray(kills_per_cell, dtype=int)
    if N.ndim != 1 or np.any(N < 0):
        raise ValueError("kills_per_cell must be 1D with values >= 0")

    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
        lam = pm.Deterministic("lambda", 10.0 ** eta)

        pm.Poisson("kills", mu=lam * T, observed=N)

        if mode_s == "counts+gaps":
            if dt_data is None:
                raise ValueError("dt_data must be provided when mode='counts+gaps'")
            if not isinstance(dt_data, (list, tuple, np.ndarray)):
                raise ValueError("dt_data must be list/tuple/ndarray of per-cell arrays")
            if len(dt_data) != N.size:
                raise ValueError("dt_data must have the same length as kills_per_cell")

            dt_sizes = np.array([np.asarray(d, dtype=float).size for d in dt_data], dtype=int)
            dt_sums = np.array([float(np.sum(np.asarray(d, dtype=float))) for d in dt_data], dtype=float)

            expected_sizes = np.where(N >= 2, N - 1, 0)
            if np.any(dt_sizes != expected_sizes):
                bad = np.where(dt_sizes != expected_sizes)[0][:10]
                raise ValueError(
                    "Inconsistent dt_data: for each cell i, len(dt_i) must equal N_i-1 if N_i>=2, else 0. "
                    f"First mismatches at indices: {bad.tolist()}"
                )
            if np.any(dt_sums > T + 1e-12):
                bad = np.where(dt_sums > T + 1e-12)[0][:10]
                raise ValueError(
                    "Inconsistent dt_data: sum(dt_i) must be <= obs_time for every cell. "
                    f"First violations at indices: {bad.tolist()}"
                )

            mask = N >= 2
            if bool(np.any(mask)):
                n_pos = N[mask]
                S_pos = dt_sums[mask]
                log_nf = pt.gammaln(n_pos + 1.0)
                log_term = pm.math.sum(log_nf - n_pos * pm.math.log(T) + pm.math.log(T - S_pos))
                pm.Potential("gaps_given_counts", log_term)

        idata = _sample_backend(
            sampler=sampler,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            cores=cores,
            random_seed=random_seed,
            smc_draws=smc_draws,
            smc_cores=smc_cores,
            smc_threshold=smc_threshold,
            smc_correlation_threshold=smc_correlation_threshold,
        )

    return {"idata": idata, "model": model}


def inference_Z2P(
    kills_per_cell,
    obs_time: float,
    dt_data=None,
    mode: str = "counts+gaps",
    sampler: SamplerName = "nuts",
    draws: int = 3000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    cores=None,
    lambda_prior_bounds=(-5.0, 2.0),
    p_prior_bounds=(1.0, 1.0),
    random_seed: Optional[int] = None,
    smc_draws: Optional[int] = None,
    smc_cores=None,
    smc_threshold: float = 0.5,
    smc_correlation_threshold: float = 0.01,
):
    mode_s = str(mode).strip().lower()
    if mode_s not in {"counts", "counts+gaps"}:
        raise ValueError("mode must be 'counts' or 'counts+gaps'")

    T = float(obs_time)
    if T <= 0:
        raise ValueError("obs_time must be > 0")

    N = np.asarray(kills_per_cell, dtype=int)
    if N.ndim != 1 or np.any(N < 0):
        raise ValueError("kills_per_cell must be 1D with values >= 0")

    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
        p0 = pm.Beta("p_zero", alpha=float(p_prior_bounds[0]), beta=float(p_prior_bounds[1]))
        lam = pm.Deterministic("lambda", 10.0 ** eta)

        pm.ZeroInflatedPoisson("kills", psi=1.0 - p0, mu=lam * T, observed=N)

        if mode_s == "counts+gaps":
            if dt_data is None:
                raise ValueError("dt_data must be provided when mode='counts+gaps'")
            if not isinstance(dt_data, (list, tuple, np.ndarray)):
                raise ValueError("dt_data must be list/tuple/ndarray of per-cell arrays")
            if len(dt_data) != N.size:
                raise ValueError("dt_data must have the same length as kills_per_cell")

            dt_sizes = np.array([np.asarray(d, dtype=float).size for d in dt_data], dtype=int)
            dt_sums = np.array([float(np.sum(np.asarray(d, dtype=float))) for d in dt_data], dtype=float)

            expected_sizes = np.where(N >= 2, N - 1, 0)
            if np.any(dt_sizes != expected_sizes):
                bad = np.where(dt_sizes != expected_sizes)[0][:10]
                raise ValueError(
                    "Inconsistent dt_data: for each cell i, len(dt_i) must equal N_i-1 if N_i>=2, else 0. "
                    f"First mismatches at indices: {bad.tolist()}"
                )
            if np.any(dt_sums > T + 1e-12):
                bad = np.where(dt_sums > T + 1e-12)[0][:10]
                raise ValueError(
                    "Inconsistent dt_data: sum(dt_i) must be <= obs_time for every cell. "
                    f"First violations at indices: {bad.tolist()}"
                )

            mask = N >= 2
            if bool(np.any(mask)):
                n_pos = N[mask]
                S_pos = dt_sums[mask]
                log_nf = pt.gammaln(n_pos + 1.0)
                log_term = pm.math.sum(log_nf - n_pos * pm.math.log(T) + pm.math.log(T - S_pos))
                pm.Potential("gaps_given_counts", log_term)

        idata = _sample_backend(
            sampler=sampler,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            cores=cores,
            random_seed=random_seed,
            smc_draws=smc_draws,
            smc_cores=smc_cores,
            smc_threshold=smc_threshold,
            smc_correlation_threshold=smc_correlation_threshold,
        )

    return {"idata": idata, "model": model}



def inference_Dis2P(
    kills_per_cell,
    obs_time: float,
    dt_data=None,
    mode: str = "counts",
    dis_mode: str = "gamma",
    sampler: SamplerName = "nuts",
    draws: int = 3000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    cores=None,
    lambda_prior_bounds=(-5.0, 2.0),
    std_prior_factor: float = 1.0,
    random_seed: Optional[int] = None,
    smc_draws: Optional[int] = None,
    smc_cores=None,
    marginalized: bool = False,
    smc_threshold: float = 0.5,
    smc_correlation_threshold: float = 0.01,
):
    dis_mode_s = str(dis_mode).strip().lower()
    if dis_mode_s not in {"gamma", "lognormal", "truncnorm"}:
        raise ValueError("dis_mode must be one of: gamma, lognormal, truncnorm")

    mode_s = str(mode).strip().lower()
    if mode_s not in {"counts", "counts+gaps"}:
        raise ValueError("mode must be 'counts' or 'counts+gaps'")

    T = float(obs_time)
    if T <= 0:
        raise ValueError("obs_time must be > 0")

    N = np.asarray(kills_per_cell, dtype=int)
    if N.ndim != 1 or np.any(N < 0):
        raise ValueError("kills_per_cell must be 1D with values >= 0")

    if mode_s == "counts+gaps":
        if dt_data is None:
            raise ValueError("dt_data must be provided when mode='counts+gaps'")
        if not isinstance(dt_data, (list, tuple, np.ndarray)):
            raise ValueError("dt_data must be list/tuple/ndarray of per-cell arrays")
        if len(dt_data) != N.size:
            raise ValueError("dt_data must have the same length as kills_per_cell")

        dt_sizes = np.array([np.asarray(d, dtype=float).size for d in dt_data], dtype=int)
        dt_sums = np.array([float(np.sum(np.asarray(d, dtype=float))) for d in dt_data], dtype=float)

        expected_sizes = np.where(N >= 2, N - 1, 0)
        if np.any(dt_sizes != expected_sizes):
            bad = np.where(dt_sizes != expected_sizes)[0][:10]
            raise ValueError(
                "Inconsistent dt_data: for each cell i, len(dt_i) must equal N_i-1 if N_i>=2, else 0. "
                f"First mismatches at indices: {bad.tolist()}"
            )
        if np.any(dt_sums > T + 1e-12):
            bad = np.where(dt_sums > T + 1e-12)[0][:10]
            raise ValueError(
                "Inconsistent dt_data: sum(dt_i) must be <= obs_time for every cell. "
                f"First violations at indices: {bad.tolist()}"
            )
    else:
        dt_sums = None

    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
        sig_lam = pm.HalfNormal("sigma_lambda", sigma=float(std_prior_factor))
        mu_lam = pm.Deterministic("mu_lambda", 10.0 ** eta)

        if marginalized and dis_mode_s == "gamma":
            alpha_nb = (sig_lam / mu_lam) ** 2
            pm.NegativeBinomial("kills", mu=mu_lam * T, alpha=alpha_nb, observed=N)
        else:
            if dis_mode_s == "gamma":
                lam_shape, lam_rate = _gamma_shape_rate_from_mean_sd(mu_lam, sig_lam)
                lam = pm.Gamma("lambda_pos", alpha=lam_shape, beta=lam_rate, shape=N.size)
            elif dis_mode_s == "lognormal":
                lam_mu, lam_sigma = _lognormal_mu_sigma_from_mean_sd(mu_lam, sig_lam)
                lam = pm.LogNormal("lambda_pos", mu=lam_mu, sigma=lam_sigma, shape=N.size)
            else:
                lam = pm.TruncatedNormal("lambda_pos", mu=mu_lam, sigma=sig_lam, lower=0.0, shape=N.size)

            pm.Poisson("kills", mu=lam * T, observed=N)

        if mode_s == "counts+gaps":
            mask = N >= 2
            if bool(np.any(mask)):
                n_pos = N[mask]
                S_pos = dt_sums[mask]
                log_nf = pt.gammaln(n_pos + 1.0)
                log_term = pm.math.sum(log_nf - n_pos * pm.math.log(T) + pm.math.log(T - S_pos))
                pm.Potential("gaps_given_counts", log_term)

        idata = _sample_backend(
            sampler=sampler,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            cores=cores,
            random_seed=random_seed,
            smc_draws=smc_draws,
            smc_cores=smc_cores,
            smc_threshold=smc_threshold,
            smc_correlation_threshold=smc_correlation_threshold,
        )

    return {"idata": idata, "model": model}


def inference_hetero3(
    kills_per_cell,
    obs_time: float,
    dt_data=None,
    mode: str = "counts",
    dis_mode: str = "gamma",
    sampler: SamplerName = "nuts",
    draws: int = 3000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    cores=None,
    lambda_prior_bounds=(-5.0, 2.0),
    p_prior_bounds=(1.0, 1.0),
    std_prior_factor: float = 1.0,
    random_seed: Optional[int] = None,
    smc_draws: Optional[int] = None,
    smc_cores=None,
    marginalized: bool = False,
    smc_threshold: float = 0.5,
    smc_correlation_threshold: float = 0.01,
):
    dis_mode_s = str(dis_mode).strip().lower()
    if dis_mode_s not in {"gamma", "lognormal", "truncnorm"}:
        raise ValueError("dis_mode must be one of: gamma, lognormal, truncnorm")

    mode_s = str(mode).strip().lower()
    if mode_s not in {"counts", "counts+gaps"}:
        raise ValueError("mode must be 'counts' or 'counts+gaps'")

    T = float(obs_time)
    if T <= 0:
        raise ValueError("obs_time must be > 0")

    N = np.asarray(kills_per_cell, dtype=int)
    if N.ndim != 1 or np.any(N < 0):
        raise ValueError("kills_per_cell must be 1D with values >= 0")

    if mode_s == "counts+gaps":
        if dt_data is None:
            raise ValueError("dt_data must be provided when mode='counts+gaps'")
        if not isinstance(dt_data, (list, tuple, np.ndarray)):
            raise ValueError("dt_data must be list/tuple/ndarray of per-cell arrays")
        if len(dt_data) != N.size:
            raise ValueError("dt_data must have the same length as kills_per_cell")

        dt_sizes = np.array([np.asarray(d, dtype=float).size for d in dt_data], dtype=int)
        dt_sums = np.array([float(np.sum(np.asarray(d, dtype=float))) for d in dt_data], dtype=float)

        expected_sizes = np.where(N >= 2, N - 1, 0)
        if np.any(dt_sizes != expected_sizes):
            bad = np.where(dt_sizes != expected_sizes)[0][:10]
            raise ValueError(
                "Inconsistent dt_data: for each cell i, len(dt_i) must equal N_i-1 if N_i>=2, else 0. "
                f"First mismatches at indices: {bad.tolist()}"
            )
        if np.any(dt_sums > T + 1e-12):
            bad = np.where(dt_sums > T + 1e-12)[0][:10]
            raise ValueError(
                "Inconsistent dt_data: sum(dt_i) must be <= obs_time for every cell. "
                f"First violations at indices: {bad.tolist()}"
            )
    else:
        dt_sums = None

    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
        p0 = pm.Beta("p_zero", alpha=float(p_prior_bounds[0]), beta=float(p_prior_bounds[1]))
        sig_lam = pm.HalfNormal("sigma_lambda", sigma=float(std_prior_factor))
        mu_lam = pm.Deterministic("mu_lambda", 10.0 ** eta)

        if marginalized and dis_mode_s == "gamma":
            alpha_nb = (sig_lam / mu_lam) ** 2
            pm.ZeroInflatedNegativeBinomial(
                "kills", psi=1.0 - p0, mu=mu_lam * T, alpha=alpha_nb, observed=N
            )
        else:
            if dis_mode_s == "gamma":
                lam_shape, lam_rate = _gamma_shape_rate_from_mean_sd(mu_lam, sig_lam)
                lam = pm.Gamma("lambda_pos", alpha=lam_shape, beta=lam_rate, shape=N.size)
            elif dis_mode_s == "lognormal":
                lam_mu, lam_sigma = _lognormal_mu_sigma_from_mean_sd(mu_lam, sig_lam)
                lam = pm.LogNormal("lambda_pos", mu=lam_mu, sigma=lam_sigma, shape=N.size)
            else:
                lam = pm.TruncatedNormal("lambda_pos", mu=mu_lam, sigma=sig_lam, lower=0.0, shape=N.size)

            pm.ZeroInflatedPoisson("kills", psi=1.0 - p0, mu=lam * T, observed=N)

        if mode_s == "counts+gaps":
            mask = N >= 2
            if bool(np.any(mask)):
                n_pos = N[mask]
                S_pos = dt_sums[mask]
                log_nf = pt.gammaln(n_pos + 1.0)
                log_term = pm.math.sum(log_nf - n_pos * pm.math.log(T) + pm.math.log(T - S_pos))
                pm.Potential("gaps_given_counts", log_term)

        idata = _sample_backend(
            sampler=sampler,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            cores=cores,
            random_seed=random_seed,
            smc_draws=smc_draws,
            smc_cores=smc_cores,
            smc_threshold=smc_threshold,
            smc_correlation_threshold=smc_correlation_threshold,
        )

    return {"idata": idata, "model": model}
