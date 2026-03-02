#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Any, Literal, Optional, Tuple

import numpy as np
import pymc as pm
import pytensor.tensor as pt

DistName = Literal["gamma", "lognormal", "truncnorm"]
SamplerName = Literal["nuts", "smc"]
GapsScheme = Literal["full", "no_beginning", "no_tail"]


def _prep_counts_gaps_sufficient_stats(
    N: np.ndarray,
    dt_data,
    T: float,
    *,
    dt_scheme: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-cell (m, S) for the gaps-based likelihood.

    Schemes:
    - full: include initial gap and tail (event gaps + tail)
            dt lengths: N+1 if N>=1 else 1
            m = N, S = sum(dt) (== per-cell observation time)
    - no_beginning: exclude initial gap, include tail (inter-contact gaps + tail)
            dt lengths: N if N>=1 else 1
            m = max(N-1, 0), S = sum(dt) (== T - t1 when N>=1)
    - no_tail: include initial gap, exclude tail (event gaps only)
            dt lengths: N (including N=0 -> 0)
            m = N, S = sum(dt) (== last event time when N>=1)
    """
    scheme = str(dt_scheme).strip().lower()
    if scheme not in {"full", "no_beginning", "no_tail"}:
        raise ValueError("dt_scheme must be one of: full, no_beginning, no_tail")

    M = int(N.size)
    if dt_data is None:
        raise ValueError("dt_data must be provided when mode='counts+gaps'")
    if not isinstance(dt_data, (list, tuple, np.ndarray)):
        raise ValueError("dt_data must be list/tuple/ndarray of per-cell arrays")
    if len(dt_data) != M:
        raise ValueError("dt_data must have the same length as contacts_per_cell")

    dt_sizes = np.fromiter((np.asarray(d, dtype=float).size for d in dt_data), dtype=int, count=M)
    dt_sums = np.fromiter((float(np.sum(np.asarray(d, dtype=float))) for d in dt_data), dtype=float, count=M)

    if scheme == "full":
        expected_sizes = np.where(N == 0, 1, N + 1)
        m = N.astype(float)
    elif scheme == "no_beginning":
        expected_sizes = np.where(N == 0, 1, N)
        m = np.maximum(N - 1, 0).astype(float)
    else:  # no_tail
        expected_sizes = N
        m = N.astype(float)

    if np.any(dt_sizes != expected_sizes):
        bad = np.where(dt_sizes != expected_sizes)[0][:10]
        raise ValueError(
            "Inconsistent dt_data lengths for counts+gaps. "
            f"dt_scheme={scheme!r}. First mismatches at indices: {bad.tolist()}"
        )

    if np.any(dt_sums < -1e-12):
        bad = np.where(dt_sums < -1e-12)[0][:10]
        raise ValueError(f"Negative durations in dt_data at indices: {bad.tolist()}")

    if np.any(dt_sums > float(T) + 1e-12):
        bad = np.where(dt_sums > float(T) + 1e-12)[0][:10]
        raise ValueError(
            "Inconsistent dt_data: sum(dt_i) must be <= obs_time for every cell. "
            f"First violations at indices: {bad.tolist()}"
        )

    S = dt_sums.astype(float)
    return m, S


def _safe_m_log_lam(m, lam):
    """Compute m * log(lam) safely when m can be 0."""
    return pt.switch(pt.gt(m, 0), m * pt.log(lam), 0.0)


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
        trace = pm.sample_smc(
            draws=int(draws_smc),
            chains=int(chains),
            cores=int(_resolve_cores(cores_smc, chains=int(chains))),
            random_seed=random_seed,
            progressbar=True,
            return_inferencedata=False,
            threshold=float(smc_threshold),
            correlation_threshold=float(smc_correlation_threshold),
        )
        to_idata = getattr(pm, "to_inferencedata", None) or getattr(pm, "to_inference_data", None)
        if to_idata is None:
            raise RuntimeError("PyMC does not expose to_inferencedata/to_inference_data")
        idata = to_idata(trace, log_likelihood=True)

        report = getattr(trace, "report", None)
        logml = None
        if report is not None and hasattr(report, "log_marginal_likelihood"):
            raw = report.log_marginal_likelihood
            if isinstance(raw, (list, tuple, np.ndarray)):
                vals: list[float] = []
                for item in raw:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        arr = np.asarray(item, dtype=float)
                        arr = arr[np.isfinite(arr)]
                        if arr.size:
                            vals.append(float(arr[-1]))
                    else:
                        try:
                            vals.append(float(item))
                        except Exception:
                            continue
                logml = np.asarray(vals, dtype=float)
            else:
                try:
                    logml = np.full(int(chains), float(raw))
                except Exception:
                    logml = None

            if logml is not None:
                if logml.size != int(chains):
                    if logml.size < int(chains):
                        pad = np.full(int(chains) - logml.size, np.nan)
                        logml = np.concatenate([logml, pad])
                    else:
                        logml = logml[: int(chains)]

        if logml is not None:
            stored = False
            try:
                idata.sample_stats["log_marginal_likelihood"] = (("chain",), logml)
                stored = True
            except Exception:
                stored = False
            if not stored:
                try:
                    idata.attrs["log_marginal_likelihood"] = logml.tolist()
                except Exception:
                    pass

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
    contacts_per_cell,
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
    dt_scheme: GapsScheme = "no_beginning",
):
    mode_s = str(mode).strip().lower()
    if mode_s not in {"counts", "counts+gaps"}:
        raise ValueError("mode must be 'counts' or 'counts+gaps'")

    T = float(obs_time)
    if T <= 0:
        raise ValueError("obs_time must be > 0")

    N = np.asarray(contacts_per_cell, dtype=int)
    if N.ndim != 1 or np.any(N < 0):
        raise ValueError("contacts_per_cell must be 1D with values >= 0")

    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
        lam = pm.Deterministic("lambda", 10.0 ** eta)
        
        if mode_s == "counts":
            pm.Poisson("contacts", mu=lam * T, observed=N)

        if mode_s == "counts+gaps":
            m, S = _prep_counts_gaps_sufficient_stats(N, dt_data, T, dt_scheme=dt_scheme)

            pm.Potential("gaps_tail", pm.math.sum(_safe_m_log_lam(m, lam) - lam * S))


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
    contacts_per_cell,
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
    dt_scheme: GapsScheme = "no_beginning",
):
    mode_s = str(mode).strip().lower()
    if mode_s not in {"counts", "counts+gaps"}:
        raise ValueError("mode must be 'counts' or 'counts+gaps'")

    T = float(obs_time)
    if T <= 0:
        raise ValueError("obs_time must be > 0")

    N = np.asarray(contacts_per_cell, dtype=int)
    if N.ndim != 1 or np.any(N < 0):
        raise ValueError("contacts_per_cell must be 1D with values >= 0")

    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
        p0 = pm.Beta("p_zero", alpha=float(p_prior_bounds[0]), beta=float(p_prior_bounds[1]))
        lam = pm.Deterministic("lambda", 10.0 ** eta)
        
        if mode_s == "counts":
            pm.ZeroInflatedPoisson("contacts", psi=1.0 - p0, mu=lam * T, observed=N)

        if mode_s == "counts+gaps":
            m, S = _prep_counts_gaps_sufficient_stats(N, dt_data, T, dt_scheme=dt_scheme)

            log_active = pt.log1p(-p0) + _safe_m_log_lam(m, lam) - lam * S

            log_zero = pt.logaddexp(
                pt.log(p0),
                pt.log1p(-p0) - lam * S
            )

            ll = pt.sum(pt.switch(pt.eq(N, 0), log_zero, log_active))
            pm.Potential("gaps_tail", ll)

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



def _gamma_shape_rate_from_mean_sd(mu, sd):
    mu = pt.as_tensor_variable(mu)
    sd = pt.as_tensor_variable(sd)
    var = sd**2
    shape = mu**2 / var
    rate = mu / var
    return shape, rate


def _lognormal_mu_sigma_from_mean_sd(mu, sd):
    mu = pt.as_tensor_variable(mu)
    sd = pt.as_tensor_variable(sd)
    var = sd**2
    sigma2 = pt.log1p(var / (mu**2))
    sigma = pt.sqrt(sigma2)
    m = pt.log(mu) - 0.5 * sigma2
    return m, sigma


def _normal_logpdf(z):
    return -0.5 * z**2 - 0.5 * pt.log(2.0 * np.pi)


def _normal_cdf(z):
    return 0.5 * (1.0 + pt.erf(z / pt.sqrt(2.0)))


def _gh_nodes_weights(K: int):
    x, w = np.polynomial.hermite.hermgauss(int(K))
    return x.astype("float64"), w.astype("float64")


def _gl01_nodes_weights(K: int):
    x, w = np.polynomial.legendre.leggauss(int(K))
    t = 0.5 * (x + 1.0)
    wt = 0.5 * w
    return t.astype("float64"), wt.astype("float64")


def _logI_lognormal(m_t, S_t, mu_u, sig_u, *, K: int = 24):
    x, w = _gh_nodes_weights(K)
    x_t = pt.as_tensor_variable(x)
    w_t = pt.as_tensor_variable(w)

    u = mu_u + pt.sqrt(2.0) * sig_u * x_t
    log_term = (
        m_t[..., None] * u[None, ...]
        - S_t[..., None] * pt.exp(u)[None, ...]
        + pt.log(w_t)[None, ...]
        - 0.5 * pt.log(np.pi)
    )
    return pm.math.logsumexp(log_term, axis=-1)


def _logI_truncnorm_pos(m_t, S_t, mu, sig, *, K: int = 48):
    t, w = _gl01_nodes_weights(K)
    t_t = pt.as_tensor_variable(t)
    w_t = pt.as_tensor_variable(w)

    lam = t_t / (1.0 - t_t)
    log_jac = -2.0 * pt.log1p(-t_t)

    z = (lam - mu) / sig
    logpdf = _normal_logpdf(z) - pt.log(sig)

    z0 = (-mu) / sig
    Z = 1.0 - _normal_cdf(z0)
    logZ = pt.log(Z)

    log_integrand = (
        m_t[..., None] * pt.log(lam)[None, ...]
        - S_t[..., None] * lam[None, ...]
        + logpdf[None, ...]
        - logZ
        + log_jac[None, ...]
        + pt.log(w_t)[None, ...]
    )
    return pm.math.logsumexp(log_integrand, axis=-1)


def _prep_gaps_tail_sufficient_stats(
    N: np.ndarray, dt_data, T: float, *, expect_tail_for_zero: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    M = int(N.size)
    if dt_data is None:
        raise ValueError("dt_data must be provided when mode='counts+gaps'")
    if not isinstance(dt_data, (list, tuple, np.ndarray)):
        raise ValueError("dt_data must be list/tuple/ndarray of per-cell arrays")
    if len(dt_data) != M:
        raise ValueError("dt_data must have the same length as contacts_per_cell")

    dt_sizes = np.fromiter((np.asarray(d, dtype=float).size for d in dt_data), dtype=int, count=M)
    dt_sums = np.fromiter((float(np.sum(np.asarray(d, dtype=float))) for d in dt_data), dtype=float, count=M)

    if expect_tail_for_zero:
        expected_sizes = np.where(N >= 1, N, 1)
    else:
        expected_sizes = np.where(N >= 1, N, 0)

    if np.any(dt_sizes != expected_sizes):
        bad = np.where(dt_sizes != expected_sizes)[0][:10]
        raise ValueError(
            "Inconsistent dt_data: expected len(dt_i)=N_i if N_i>=1, else 1 (tail only). "
            f"First mismatches at indices: {bad.tolist()}"
        )

    if np.any(dt_sums < -1e-12):
        bad = np.where(dt_sums < -1e-12)[0][:10]
        raise ValueError(f"Negative durations in dt_data at indices: {bad.tolist()}")

    if np.any(dt_sums > T + 1e-12):
        bad = np.where(dt_sums > T + 1e-12)[0][:10]
        raise ValueError(
            "Inconsistent dt_data: sum(dt_i) must be <= obs_time for every cell. "
            f"First violations at indices: {bad.tolist()}"
        )

    # Backward-compatible helper used by Route-B code.
    # This function retains its name but delegates the actual convention to the
    # new scheme-based helper. Historically, this code used the "no_beginning"
    # scheme (inter-contact gaps + tail).
    dt_scheme: str = "no_beginning" if expect_tail_for_zero else "no_tail"
    return _prep_counts_gaps_sufficient_stats(N, dt_data, float(T), dt_scheme=dt_scheme)


def inference_Dis2P(
    contacts_per_cell,
    obs_time: float,
    dt_data=None,
    mode: str = "counts",
    dis_mode: str = "gamma",
    sampler: str = "nuts",
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
    smc_threshold: float = 0.5,
    smc_correlation_threshold: float = 0.01,
    dt_scheme: GapsScheme = "no_beginning",
):
    dis_mode_s = str(dis_mode).strip().lower()
    if dis_mode_s not in {"gamma", "lognormal", "truncnorm"}:
        raise ValueError("dis_mode must be one of: gamma, lognormal, truncnorm")

    mode_s = str(mode).strip().lower()
    if mode_s not in {"counts", "counts+gaps"}:
        raise ValueError("mode must be 'counts' or 'counts+gaps'")

    if str(sampler).strip().lower() == "smc":
        return inference_Dis2P_routeB_smc(
            contacts_per_cell,
            obs_time,
            dt_data=dt_data,
            mode=mode,
            dis_mode=dis_mode,
            sampler="smc",
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            cores=cores,
            lambda_prior_bounds=lambda_prior_bounds,
            std_prior_factor=std_prior_factor,
            random_seed=random_seed,
            smc_draws=smc_draws,
            smc_cores=smc_cores,
            smc_threshold=smc_threshold,
            smc_correlation_threshold=smc_correlation_threshold,
            dt_scheme=dt_scheme,
        )

    T = float(obs_time)
    if T <= 0:
        raise ValueError("obs_time must be > 0")

    N = np.asarray(contacts_per_cell, dtype=int)
    if N.ndim != 1 or np.any(N < 0):
        raise ValueError("contacts_per_cell must be 1D with values >= 0")
    M = N.size

    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
        mu_lam = pm.Deterministic("mu_lambda", 10.0 ** eta)
        sig_lam = pm.HalfNormal("sigma_lambda", sigma=float(std_prior_factor))

        if dis_mode_s == "gamma":
            shape, rate = _gamma_shape_rate_from_mean_sd(mu_lam, sig_lam)
            lam = pm.Gamma("lambda_i", alpha=shape, beta=rate, shape=M)
        elif dis_mode_s == "lognormal":
            m_ln, s_ln = _lognormal_mu_sigma_from_mean_sd(mu_lam, sig_lam)
            lam = pm.LogNormal("lambda_i", mu=m_ln, sigma=s_ln, shape=M)
        elif dis_mode_s == "truncnorm":
            lam = pm.TruncatedNormal("lambda_i", mu=mu_lam, sigma=sig_lam, lower=0.0, shape=M)

        if mode_s == "counts":
            pm.Poisson("contacts", mu=lam * T, observed=N)

        elif mode_s == "counts+gaps":
            m_obs, S_obs = _prep_counts_gaps_sufficient_stats(N, dt_data, T, dt_scheme=dt_scheme)

            m_t = pt.as_tensor_variable(m_obs)
            S_t = pt.as_tensor_variable(S_obs)
            pm.Potential("gaps_tail", pt.sum(_safe_m_log_lam(m_t, lam) - lam * S_t))

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
    contacts_per_cell,
    obs_time: float,
    dt_data=None,
    mode: str = "counts",
    dis_mode: str = "gamma",
    sampler: str = "nuts",
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
    smc_threshold: float = 0.5,
    smc_correlation_threshold: float = 0.01,
    dt_scheme: GapsScheme = "no_beginning",
):
    dis_mode_s = str(dis_mode).strip().lower()
    if dis_mode_s not in {"gamma", "lognormal", "truncnorm"}:
        raise ValueError("dis_mode must be one of: gamma, lognormal, truncnorm")

    mode_s = str(mode).strip().lower()
    if mode_s not in {"counts", "counts+gaps"}:
        raise ValueError("mode must be 'counts' or 'counts+gaps'")

    if str(sampler).strip().lower() == "smc":
        return inference_hetero3_routeB_smc(
            contacts_per_cell,
            obs_time,
            dt_data=dt_data,
            mode=mode,
            dis_mode=dis_mode,
            sampler="smc",
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            cores=cores,
            lambda_prior_bounds=lambda_prior_bounds,
            p_prior_bounds=p_prior_bounds,
            std_prior_factor=std_prior_factor,
            random_seed=random_seed,
            smc_draws=smc_draws,
            smc_cores=smc_cores,
            smc_threshold=smc_threshold,
            smc_correlation_threshold=smc_correlation_threshold,
            dt_scheme=dt_scheme,
        )

    T = float(obs_time)
    if T <= 0:
        raise ValueError("obs_time must be > 0")

    N = np.asarray(contacts_per_cell, dtype=int)
    if N.ndim != 1 or np.any(N < 0):
        raise ValueError("contacts_per_cell must be 1D with values >= 0")
    M = N.size

    if mode_s == "counts+gaps":
        m_obs, S_obs = _prep_counts_gaps_sufficient_stats(N, dt_data, T, dt_scheme=dt_scheme)
        is_zero = (N == 0)
    else:
        m_obs = S_obs = is_zero = None

    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
        mu_lam = pm.Deterministic("mu_lambda", 10.0 ** eta)
        sig_lam = pm.HalfNormal("sigma_lambda", sigma=float(std_prior_factor))

        p_zero = pm.Beta("p_zero", alpha=float(p_prior_bounds[0]), beta=float(p_prior_bounds[1]))

        if dis_mode_s == "gamma":
            shape, rate = _gamma_shape_rate_from_mean_sd(mu_lam, sig_lam)
            lam = pm.Gamma("lambda_i", alpha=shape, beta=rate, shape=M)
        elif dis_mode_s == "lognormal":
            m_ln, s_ln = _lognormal_mu_sigma_from_mean_sd(mu_lam, sig_lam)
            lam = pm.LogNormal("lambda_i", mu=m_ln, sigma=s_ln, shape=M)
        else:
            lam = pm.TruncatedNormal("lambda_i", mu=mu_lam, sigma=sig_lam, lower=0.0, shape=M)

        if mode_s == "counts":
            pm.ZeroInflatedPoisson("contacts", psi=1.0 - p_zero, mu=lam * T, observed=N)

        else:
            m_t = pt.as_tensor_variable(m_obs)
            S_t = pt.as_tensor_variable(S_obs)

            idx0 = np.where(is_zero)[0]
            idx1 = np.where(~is_zero)[0]

            ll = pt.as_tensor_variable(0.0)

            if idx1.size:
                lam1 = lam[idx1]
                m1 = m_t[idx1]
                S1 = S_t[idx1]
                ll = ll + pt.sum(pt.log1p(-p_zero) + _safe_m_log_lam(m1, lam1) - lam1 * S1)

            if idx0.size:
                lam0 = lam[idx0]
                S0 = S_t[idx0]
                ll = ll + pt.sum(pt.logaddexp(pt.log(p_zero), pt.log1p(-p_zero) - lam0 * S0))

            pm.Potential("gaps_tail", ll)

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


def inference_Dis2P_routeB_smc(
    contacts_per_cell,
    obs_time: float,
    dt_data=None,
    mode: str = "counts",
    dis_mode: DistName = "gamma",
    sampler: SamplerName = "smc",
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
    smc_threshold: float = 0.5,
    smc_correlation_threshold: float = 0.01,
    quad_K_lognormal: int = 24,
    quad_K_truncnorm: int = 48,
    expect_tail_for_zero: bool = True,
    dt_scheme: GapsScheme = "no_beginning",
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

    N = np.asarray(contacts_per_cell, dtype=int)
    if N.ndim != 1 or np.any(N < 0):
        raise ValueError("contacts_per_cell must be 1D with values >= 0")

    if mode_s == "counts+gaps":
        m_obs, S_obs = _prep_counts_gaps_sufficient_stats(N, dt_data, T, dt_scheme=dt_scheme)
    else:
        m_obs = S_obs = None

    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
        mu_lam = pm.Deterministic("mu_lambda", 10.0 ** eta)
        sig_lam = pm.HalfNormal("sigma_lambda", sigma=float(std_prior_factor))

        if sampler != "smc":
            raise ValueError("Route B here is intended for SMC evidence (marginalised). Use NUTS functions for posterior.")

        if mode_s == "counts":
            n_t = pt.as_tensor_variable(N.astype(float))
            log_fact = pt.gammaln(n_t + 1.0)
            const = n_t * pt.log(T) - log_fact

            if dis_mode_s == "gamma":
                shape, rate = _gamma_shape_rate_from_mean_sd(mu_lam, sig_lam)
                logI = (
                    pt.gammaln(shape + n_t)
                    - pt.gammaln(shape)
                    + shape * pt.log(rate)
                    - (shape + n_t) * pt.log(rate + T)
                )
            elif dis_mode_s == "lognormal":
                mu_u, sig_u = _lognormal_mu_sigma_from_mean_sd(mu_lam, sig_lam)
                logI = _logI_lognormal(n_t, pt.full_like(n_t, T), mu_u, sig_u, K=int(quad_K_lognormal))
            else:
                logI = _logI_truncnorm_pos(n_t, pt.full_like(n_t, T), mu_lam, sig_lam, K=int(quad_K_truncnorm))

            pm.Potential("marg_counts_ll", pt.sum(const + logI))

        else:
            m_t = pt.as_tensor_variable(m_obs)
            S_t = pt.as_tensor_variable(S_obs)

            if dis_mode_s == "gamma":
                shape, rate = _gamma_shape_rate_from_mean_sd(mu_lam, sig_lam)
                logI = (
                    pt.gammaln(shape + m_t)
                    - pt.gammaln(shape)
                    + shape * pt.log(rate)
                    - (shape + m_t) * pt.log(rate + S_t)
                )
            elif dis_mode_s == "lognormal":
                mu_u, sig_u = _lognormal_mu_sigma_from_mean_sd(mu_lam, sig_lam)
                logI = _logI_lognormal(m_t, S_t, mu_u, sig_u, K=int(quad_K_lognormal))
            else:
                logI = _logI_truncnorm_pos(m_t, S_t, mu_lam, sig_lam, K=int(quad_K_truncnorm))

            pm.Potential("marg_gaps_tail_ll", pt.sum(logI))

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


def inference_hetero3_routeB_smc(
    contacts_per_cell,
    obs_time: float,
    dt_data=None,
    mode: str = "counts",
    dis_mode: DistName = "gamma",
    sampler: SamplerName = "smc",
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
    smc_threshold: float = 0.5,
    smc_correlation_threshold: float = 0.01,
    quad_K_lognormal: int = 24,
    quad_K_truncnorm: int = 48,
    expect_tail_for_zero: bool = True,
    dt_scheme: GapsScheme = "no_beginning",
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

    N = np.asarray(contacts_per_cell, dtype=int)
    if N.ndim != 1 or np.any(N < 0):
        raise ValueError("contacts_per_cell must be 1D with values >= 0")

    is_zero = (N == 0)

    if mode_s == "counts+gaps":
        m_obs, S_obs = _prep_counts_gaps_sufficient_stats(N, dt_data, T, dt_scheme=dt_scheme)
    else:
        m_obs = S_obs = None

    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
        mu_lam = pm.Deterministic("mu_lambda", 10.0 ** eta)
        sig_lam = pm.HalfNormal("sigma_lambda", sigma=float(std_prior_factor))
        p_zero = pm.Beta("p_zero", alpha=float(p_prior_bounds[0]), beta=float(p_prior_bounds[1]))

        if sampler != "smc":
            raise ValueError("Route B here is intended for SMC evidence (marginalised). Use NUTS functions for posterior.")

        if dis_mode_s == "lognormal":
            mu_u, sig_u = _lognormal_mu_sigma_from_mean_sd(mu_lam, sig_lam)
        else:
            mu_u = sig_u = None

        def _logI(m_t, S_t):
            if dis_mode_s == "gamma":
                shape, rate = _gamma_shape_rate_from_mean_sd(mu_lam, sig_lam)
                return (
                    pt.gammaln(shape + m_t)
                    - pt.gammaln(shape)
                    + shape * pt.log(rate)
                    - (shape + m_t) * pt.log(rate + S_t)
                )
            if dis_mode_s == "lognormal":
                return _logI_lognormal(m_t, S_t, mu_u, sig_u, K=int(quad_K_lognormal))
            return _logI_truncnorm_pos(m_t, S_t, mu_lam, sig_lam, K=int(quad_K_truncnorm))

        if mode_s == "counts":
            n_t = pt.as_tensor_variable(N.astype(float))
            log_fact = pt.gammaln(n_t + 1.0)
            const = n_t * pt.log(T) - log_fact

            logI_n = _logI(n_t, pt.full_like(n_t, T))
            logI_0 = _logI(pt.zeros_like(n_t), pt.full_like(n_t, T))

            log_active = pt.log1p(-p_zero) + const + logI_n
            log_zero = pt.logaddexp(pt.log(p_zero), pt.log1p(-p_zero) + logI_0)

            N_t_int = pt.as_tensor_variable(N.astype("int64"))
            ll = pt.sum(pt.switch(pt.eq(N_t_int, 0), log_zero, log_active))
            pm.Potential("zi_marg_counts_ll", ll)

        else:
            m_t = pt.as_tensor_variable(m_obs)
            S_t = pt.as_tensor_variable(S_obs)
            iz_t = pt.as_tensor_variable(is_zero.astype(bool))

            logI_active = _logI(m_t, S_t)
            logI_zero = _logI(pt.zeros_like(m_t), pt.full_like(S_t, T))

            log_active = pt.log1p(-p_zero) + logI_active
            log_zero = pt.logaddexp(pt.log(p_zero), pt.log1p(-p_zero) + logI_zero)

            ll = pt.sum(pt.switch(iz_t, log_zero, log_active))
            pm.Potential("zi_marg_gaps_tail_ll", ll)

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
