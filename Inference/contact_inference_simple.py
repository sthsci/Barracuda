#!/usr/bin/env python3
from __future__ import annotations

import os
import numpy as np
import pymc as pm
from tqdm.auto import tqdm


def _resolve_cores(cores, *, chains: int) -> int:
    """Pick a safe `cores` value for pm.sample.

    PyMC parallelizes primarily over chains, so the effective parallelism is
    `min(chains, cores)`. We default to all available CPU cores.
    """

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


def inference_counts(
    kills_per_cell,
    obs_time,
    draws=3000,
    tune=2000,
    chains=4,
    target_accept=0.9,
    cores=None,
    log10_bounds=(-6.0, 2.0),
):
    N = np.asarray(kills_per_cell, dtype=int)
    T = float(obs_time)
    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=float(log10_bounds[0]), upper=float(log10_bounds[1]))
        lambda_rate = pm.Deterministic("mu_lambda", 10.0 ** eta)
        pm.Poisson("kills", mu=lambda_rate * T, observed=N)
        cores_to_use = _resolve_cores(cores, chains=int(chains))
        idata = pm.sample(
            draws=int(draws),
            tune=int(tune),
            chains=int(chains),
            cores=int(cores_to_use),
            target_accept=float(target_accept),
            random_seed=None,
            progressbar=False,
        )
    return idata


def inference_durations(
    durations,
    draws=3000,
    tune=2000,
    chains=4,
    target_accept=0.9,
    cores=None,
    log10_bounds=(-6.0, 2.0),
):
    dt = np.asarray(durations, dtype=float)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("durations is empty after filtering non-finite/<=0 values")

    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=float(log10_bounds[0]), upper=float(log10_bounds[1]))
        mu_lambda = pm.Deterministic("mu_lambda", 10.0 ** eta)
        pm.Exponential("dt_obs", lam=mu_lambda, observed=dt)
        cores_to_use = _resolve_cores(cores, chains=int(chains))
        idata = pm.sample(
            draws=int(draws),
            tune=int(tune),
            chains=int(chains),
            cores=int(cores_to_use),
            target_accept=float(target_accept),
            random_seed=None,
            progressbar=False,
        )
    return idata


def inference_both(
    kills_per_cell,
    durations,
    obs_time,
    draws=3000,
    tune=2000,
    chains=4,
    target_accept=0.9,
    cores=None,
    log10_bounds=(-6.0, 2.0),
):
    N = np.asarray(kills_per_cell, dtype=int)
    T = float(obs_time)
    dt = np.asarray(durations, dtype=float)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("durations is empty after filtering non-finite/<=0 values")

    with pm.Model() as model:
        eta = pm.Uniform("eta", lower=float(log10_bounds[0]), upper=float(log10_bounds[1]))
        lambda_rate = pm.Deterministic("mu_lambda", 10.0 ** eta)
        pm.Poisson("kills", mu=lambda_rate * T, observed=N)
        pm.Exponential("dt_obs", lam=lambda_rate, observed=dt)
        cores_to_use = _resolve_cores(cores, chains=int(chains))
        idata = pm.sample(
            draws=int(draws),
            tune=int(tune),
            chains=int(chains),
            cores=int(cores_to_use),
            target_accept=float(target_accept),
            random_seed=None,
            progressbar=False,
        )
    return idata



def inference_all(experiment, labels=None, mode="counts", draws=3000, tune=2000, chains=4, target_accept=0.9, cores=None, obs_time=None):
    if labels is None:
        labels = [f"cond_{i}" for i in range(len(experiment))]

    out = []
    for i, exp in enumerate(tqdm(experiment)):
        if mode == "counts":
            kills = exp.get_summary()["kills"] if hasattr(exp, "get_summary") else exp
            T = obs_time if obs_time is not None else (float(exp.max_time) if hasattr(exp, "max_time") else 1.0)
            idata = inference_counts(
                kills_per_cell=kills,
                obs_time=T,
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                cores=cores,
            )
        elif mode == "dt":
            dts = exp.get_summary()["dt_all"] if hasattr(exp, "get_summary") else exp
            idata = inference_durations(
                durations=dts,
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                cores=cores,
            )
        elif mode == "both":
            kills = exp.get_summary()["kills"] if hasattr(exp, "get_summary") else exp
            dts = exp.get_summary()["dt_all"] if hasattr(exp, "get_summary") else exp
            T = obs_time if obs_time is not None else (float(exp.max_time) if hasattr(exp, "max_time") else 1.0)
            idata = inference_both(
                kills_per_cell=kills,
                durations=dts,
                obs_time=T,
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                cores=cores,
            )
        else:
            raise ValueError("mode must be 'counts', 'dt', or 'both'")
        out.append((labels[i], idata))
    return out
