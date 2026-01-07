#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Literal, Optional, Tuple

import numpy as np
import pymc as pm


DistName = Literal["gamma", "lognormal", "truncnorm"]

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


def _gamma_shape_rate_from_mean_sd(mean: pm.TensorVariable, sd: pm.TensorVariable) -> Tuple[pm.TensorVariable, pm.TensorVariable]:
    shape = (mean / sd) ** 2
    rate = mean / (sd * sd)
    return shape, rate


def _lognormal_mu_sigma_from_mean_sd(mean: pm.TensorVariable, sd: pm.TensorVariable) -> Tuple[pm.TensorVariable, pm.TensorVariable]:
    # Some PyMC versions don't expose `pm.math.log1p`; use log(1 + x) instead.
    sigma2 = pm.math.log(1.0 + (sd / mean) ** 2)
    mu = pm.math.log(mean) - 0.5 * sigma2
    return mu, pm.math.sqrt(sigma2)


def _stack_dt_list(dt_list) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dt_values, cell_index) from per-cell dt_list.

    dt_list is expected to be a list/array of arrays, length n_cells.
    """

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


def inference_homo(
    kills_per_cell,
    obs_time: float,
    dt_data=None,
    mode: str = "counts",  # "counts" or "duration"
    draws: int = 3000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    cores=None,
    lambda_prior_bounds=(-5.0, 2.0),
    initial_duration: bool = True,
):
    mode_s = str(mode).strip().lower()
    if mode_s not in {"counts", "duration"}:
        raise ValueError("mode must be 'counts' or 'duration'")
    if mode_s == "counts":
        N = np.asarray(kills_per_cell, dtype=int)
        if N.ndim != 1:
            raise ValueError("kills_per_cell must be a 1D array")
        if np.any(N < 0):
            raise ValueError("kills_per_cell must be >= 0")
        T = float(obs_time)
        if T <= 0:
            raise ValueError("obs_time must be > 0")
        with pm.Model() as model:
            eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
            lam = pm.Deterministic("lambda", 10.0 ** eta)
            pm.Poisson("kills", mu=lam * T, observed=N)
            cores_to_use = _resolve_cores(cores, chains=int(chains))
            idata = pm.sample(
                draws=int(draws),
                tune=int(tune),
                chains=int(chains),
                cores=int(cores_to_use),
                target_accept=float(target_accept),
                random_seed=None,
                progressbar=True,
            )
        return idata
    if mode_s == "duration" and dt_data is None:
        raise ValueError("dt_data must be provided when mode='duration'")
    if mode_s == "duration":
        if not isinstance(dt_data, (list, tuple, np.ndarray)):
            raise ValueError("dt_data must be a list/tuple/ndarray of per-cell arrays")
        n_cells = int(len(dt_data))
        if n_cells <= 0:
            raise ValueError("dt_data must have length > 0")
        T = float(obs_time)
        if T <= 0:
            raise ValueError("obs_time must be > 0")
        dt_vals, _dt_cell = _stack_dt_list(dt_data)
        has_dt = np.array([int(np.asarray(d, dtype=float).size > 0) for d in dt_data], dtype=np.int8)
        with pm.Model() as model:
            eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
            lam = pm.Deterministic("lambda", 10.0 ** eta)
            if bool(initial_duration):
                p_has_dt = 1.0 - pm.math.exp(-lam * T) * (1.0 + lam * T)
            else:
                p_has_dt = 1.0 - pm.math.exp(-lam * T)
            p_has_dt = pm.math.clip(p_has_dt, 1e-12, 1.0 - 1e-12)
            pm.Bernoulli("has_dt", p=p_has_dt, observed=has_dt)
            if np.asarray(dt_vals, dtype=float).size:
                pm.Exponential("dt_obs", lam=lam, observed=np.asarray(dt_vals, dtype=float))
            cores_to_use = _resolve_cores(cores, chains=int(chains))
            idata = pm.sample(
                draws=int(draws),
                tune=int(tune),
                chains=int(chains),
                cores=int(cores_to_use),
                target_accept=float(target_accept),
                random_seed=None,
                progressbar=True,
            )
        return idata



def inference_Z2P(
    kills_per_cell,
    obs_time: float,
    dt_data=None,
    mode: str = "counts",  # "counts" or "duration"
    draws: int = 3000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    cores=None,
    lambda_prior_bounds=(-5.0, 2.0),
    p_prior_bounds=(1.0, 1.0),
    initial_duration: bool = True,
):
    mode_s = str(mode).strip().lower()
    if mode_s not in {"counts", "duration"}:
        raise ValueError("mode must be 'counts' or 'duration'")
    if mode_s == "counts":
        N = np.asarray(kills_per_cell, dtype=int)
        if N.ndim != 1:
            raise ValueError("kills_per_cell must be a 1D array")
        if np.any(N < 0):
            raise ValueError("kills_per_cell must be >= 0")
        T = float(obs_time)
        if T <= 0:
            raise ValueError("obs_time must be > 0")
        with pm.Model() as model:
            eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
            p0 = pm.Beta("p_zero", alpha=float(p_prior_bounds[0]), beta=float(p_prior_bounds[1]))
            lam = pm.Deterministic("lambda", 10.0 ** eta)
            pm.ZeroInflatedPoisson("kills", psi=1-p0, mu=lam * T, observed=N)
            cores_to_use = _resolve_cores(cores, chains=int(chains))
            idata = pm.sample(
                draws=int(draws), 
                tune=int(tune),
                chains=int(chains),
                cores=int(cores_to_use),
                target_accept=float(target_accept),
                random_seed=None,
                progressbar=True,
            )
        return idata
    if mode_s == "duration" and dt_data is None:
        raise ValueError("dt_data must be provided when mode='duration'")
    if mode_s == "duration":
        if not isinstance(dt_data, (list, tuple, np.ndarray)):
            raise ValueError("dt_data must be a list/tuple/ndarray of per-cell arrays")
        n_cells = int(len(dt_data))
        if n_cells <= 0:
            raise ValueError("dt_data must have length > 0")
        T = float(obs_time)
        if T <= 0:
            raise ValueError("obs_time must be > 0")
        dt_vals, _dt_cell = _stack_dt_list(dt_data)
        has_dt = np.array([int(np.asarray(d, dtype=float).size > 0) for d in dt_data], dtype=np.int8)
        with pm.Model() as model:
            eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
            p0 = pm.Beta("p_zero", alpha=float(p_prior_bounds[0]), beta=float(p_prior_bounds[1]))
            lam = pm.Deterministic("lambda", 10.0 ** eta)
            if bool(initial_duration):
                g = 1.0 - pm.math.exp(-lam * T) * (1.0 + lam * T)
            else:
                g = 1.0 - pm.math.exp(-lam * T)
            p_has_dt = (1.0 - p0) * g
            p_has_dt = pm.math.clip(p_has_dt, 1e-12, 1.0 - 1e-12)
            pm.Bernoulli("has_dt", p=p_has_dt, observed=has_dt)
            if np.asarray(dt_vals, dtype=float).size:
                pm.Exponential("dt_obs", lam=lam, observed=np.asarray(dt_vals, dtype=float))
            cores_to_use = _resolve_cores(cores, chains=int(chains))
            idata = pm.sample(
                draws=int(draws),
                tune=int(tune),
                chains=int(chains),
                cores=int(cores_to_use),
                target_accept=float(target_accept),
                random_seed=None,
                progressbar=True,
            )
        return idata



def inference_Dis2P(
    kills_per_cell,
    obs_time: float,
    dt_data=None,
    mode: str = "counts",  # "counts" or "duration"
    dis_mode: str = "gamma",  # "gamma", "lognormal", "truncnorm", "HalfNormal"
    draws: int = 3000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    cores=None,
    lambda_prior_bounds=(-5.0, 2.0),
    std_prior_factor = 1.0,
    initial_duration: bool = True,
):
    dis_mode_s = str(dis_mode).strip().lower()
    if dis_mode_s not in {"gamma", "lognormal", "truncnorm"}:
        raise ValueError("dis_mode must be one of: gamma, lognormal, truncnorm")

    mode_s = str(mode).strip().lower()
    if mode_s not in {"counts", "duration"}:
        raise ValueError("mode must be 'counts' or 'duration'")
    if mode_s == "counts":
        N = np.asarray(kills_per_cell, dtype=int)
        if N.ndim != 1:
            raise ValueError("kills_per_cell must be a 1D array")
        if np.any(N < 0):
            raise ValueError("kills_per_cell must be >= 0")
        T = float(obs_time)
        if T <= 0:
            raise ValueError("obs_time must be > 0")
        with pm.Model() as model:
            eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
            sig_lam = pm.HalfNormal("sigma_lambda", sigma=std_prior_factor)
            mu_lam = pm.Deterministic("mu_lambda", 10.0 ** eta)
            if dis_mode_s == "gamma":
                lam_shape, lam_rate = _gamma_shape_rate_from_mean_sd(mu_lam, sig_lam)
                lam = pm.Gamma("lambda_pos", alpha=lam_shape, beta=lam_rate, shape=N.size)
            elif dis_mode_s == "lognormal":
                lam_mu, lam_sigma = _lognormal_mu_sigma_from_mean_sd(mu_lam, sig_lam)
                lam = pm.LogNormal("lambda_pos", mu=lam_mu, sigma=lam_sigma, shape=N.size)
            else:
                lam = pm.TruncatedNormal("lambda_pos", mu=mu_lam, sigma=sig_lam, lower=0.0, shape=N.size)
            pm.Poisson("kills", mu=lam * T, observed=N)
            cores_to_use = _resolve_cores(cores, chains=int(chains))
            idata = pm.sample(
                draws=int(draws), 
                tune=int(tune),
                chains=int(chains),
                cores=int(cores_to_use),
                target_accept=float(target_accept),
                random_seed=None,
                progressbar=True,
            )
        return idata
    if mode_s == "duration" and dt_data is None:
        raise ValueError("dt_data must be provided when mode='duration'")
    if mode_s == "duration":
        if not isinstance(dt_data, (list, tuple, np.ndarray)):
            raise ValueError("dt_data must be a list/tuple/ndarray of per-cell arrays")
        n_cells = int(len(dt_data))
        if n_cells <= 0:
            raise ValueError("dt_data must have length > 0")
        T = float(obs_time)
        if T <= 0:
            raise ValueError("obs_time must be > 0")
        dt_vals, dt_cell = _stack_dt_list(dt_data)
        has_dt = np.array([int(np.asarray(d, dtype=float).size > 0) for d in dt_data], dtype=np.int8)
        with pm.Model() as model:
            eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
            sig_lam = pm.HalfNormal("sigma_lambda", sigma=std_prior_factor)
            mu_lam = pm.Deterministic("mu_lambda", 10.0 ** eta)
            if dis_mode_s == "gamma":
                lam_shape, lam_rate = _gamma_shape_rate_from_mean_sd(mu_lam, sig_lam)
                lam = pm.Gamma("lambda_pos", alpha=lam_shape, beta=lam_rate, shape=n_cells)
            elif dis_mode_s == "lognormal":
                lam_mu, lam_sigma = _lognormal_mu_sigma_from_mean_sd(mu_lam, sig_lam)
                lam = pm.LogNormal("lambda_pos", mu=lam_mu, sigma=lam_sigma, shape=n_cells)
            else:
                lam = pm.TruncatedNormal("lambda_pos", mu=mu_lam, sigma=sig_lam, lower=0.0, shape=n_cells)
            
            if bool(initial_duration):
                p_has_dt = 1.0 - pm.math.exp(-lam * T) * (1.0 + lam * T)
            else:
                p_has_dt = 1.0 - pm.math.exp(-lam * T)
            p_has_dt = pm.math.clip(p_has_dt, 1e-12, 1.0 - 1e-12)
            pm.Bernoulli("has_dt", p=p_has_dt, observed=has_dt)
            if np.asarray(dt_vals, dtype=float).size:
                dt_cell_data = pm.Data("dt_cell", dt_cell)
                lam_dt = lam[dt_cell_data]
                pm.Exponential("dt_obs", lam=lam_dt, observed=np.asarray(dt_vals, dtype=float))
            cores_to_use = _resolve_cores(cores, chains=int(chains))
            idata = pm.sample(
                draws=int(draws),
                tune=int(tune),
                chains=int(chains),
                cores=int(cores_to_use),
                target_accept=float(target_accept),
                random_seed=None,
                progressbar=True,
            )
        return idata



def inference_hetero3(
    kills_per_cell,
    obs_time: float,
    dt_data=None,
    mode: str = "counts",  # "counts" or "duration"
    dis_mode: str = "gamma",  # "gamma", "lognormal", "truncnorm", "HalfNormal"
    draws: int = 3000,
    tune: int = 2000,
    chains: int = 4,
    target_accept: float = 0.9,
    cores=None,
    lambda_prior_bounds=(-5.0, 2.0),
    p_prior_bounds=(1.0, 1.0),
    std_prior_factor = 1.0,
    initial_duration: bool = True,
):
    dis_mode_s = str(dis_mode).strip().lower()
    if dis_mode_s not in {"gamma", "lognormal", "truncnorm"}:
        raise ValueError("dis_mode must be one of: gamma, lognormal, truncnorm")

    mode_s = str(mode).strip().lower()
    if mode_s not in {"counts", "duration"}:
        raise ValueError("mode must be 'counts' or 'duration'")
    if mode_s == "counts":
        N = np.asarray(kills_per_cell, dtype=int)
        if N.ndim != 1:
            raise ValueError("kills_per_cell must be a 1D array")
        if np.any(N < 0):
            raise ValueError("kills_per_cell must be >= 0")
        T = float(obs_time)
        if T <= 0:
            raise ValueError("obs_time must be > 0")
        with pm.Model() as model:
            eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
            p0 = pm.Beta("p_zero", alpha=float(p_prior_bounds[0]), beta=float(p_prior_bounds[1]))
            sig_lam = pm.HalfNormal("sigma_lambda", sigma=std_prior_factor)
            mu_lam = pm.Deterministic("mu_lambda", 10.0 ** eta)
            if dis_mode_s == "gamma":
                lam_shape, lam_rate = _gamma_shape_rate_from_mean_sd(mu_lam, sig_lam)
                lam = pm.Gamma("lambda_pos", alpha=lam_shape, beta=lam_rate, shape=N.size)
            elif dis_mode_s == "lognormal":
                lam_mu, lam_sigma = _lognormal_mu_sigma_from_mean_sd(mu_lam, sig_lam)
                lam = pm.LogNormal("lambda_pos", mu=lam_mu, sigma=lam_sigma, shape=N.size)
            else:
                lam = pm.TruncatedNormal("lambda_pos", mu=mu_lam, sigma=sig_lam, lower=0.0, shape=N.size)
            pm.ZeroInflatedPoisson("kills", psi=(1.0 - p0), mu=lam * T, observed=N)
            cores_to_use = _resolve_cores(cores, chains=int(chains))
            idata = pm.sample(
                draws=int(draws), 
                tune=int(tune),
                chains=int(chains),
                cores=int(cores_to_use),
                target_accept=float(target_accept),
                random_seed=None,
                progressbar=True,
            )
        return idata
    if mode_s == "duration" and dt_data is None:
        raise ValueError("dt_data must be provided when mode='duration'")
    if mode_s == "duration":
        if not isinstance(dt_data, (list, tuple, np.ndarray)):
            raise ValueError("dt_data must be a list/tuple/ndarray of per-cell arrays")
        n_cells = int(len(dt_data))
        if n_cells <= 0:
            raise ValueError("dt_data must have length > 0")
        T = float(obs_time)
        if T <= 0:
            raise ValueError("obs_time must be > 0")
        dt_vals, dt_cell = _stack_dt_list(dt_data)
        has_dt = np.array([int(np.asarray(d, dtype=float).size > 0) for d in dt_data], dtype=np.int8)
        with pm.Model() as model:
            eta = pm.Uniform("eta", lower=float(lambda_prior_bounds[0]), upper=float(lambda_prior_bounds[1]))
            p0 = pm.Beta("p_zero", alpha=float(p_prior_bounds[0]), beta=float(p_prior_bounds[1]))
            sig_lam = pm.HalfNormal("sigma_lambda", sigma=std_prior_factor)
            mu_lam = pm.Deterministic("mu_lambda", 10.0 ** eta)
            if dis_mode_s == "gamma":
                lam_shape, lam_rate = _gamma_shape_rate_from_mean_sd(mu_lam, sig_lam)
                lam = pm.Gamma("lambda_pos", alpha=lam_shape, beta=lam_rate, shape=n_cells)
            elif dis_mode_s == "lognormal":
                lam_mu, lam_sigma = _lognormal_mu_sigma_from_mean_sd(mu_lam, sig_lam)
                lam = pm.LogNormal("lambda_pos", mu=lam_mu, sigma=lam_sigma, shape=n_cells)
            else:
                lam = pm.TruncatedNormal("lambda_pos", mu=mu_lam, sigma=sig_lam, lower=0.0, shape=n_cells)
            
            if bool(initial_duration):
                g = 1.0 - pm.math.exp(-lam * T) * (1.0 + lam * T)
            else:
                g = 1.0 - pm.math.exp(-lam * T)
            p_has_dt = (1.0 - p0) * g
            p_has_dt = pm.math.clip(p_has_dt, 1e-12, 1.0 - 1e-12)
            pm.Bernoulli("has_dt", p=p_has_dt, observed=has_dt)
            if np.asarray(dt_vals, dtype=float).size:
                dt_cell_data = pm.Data("dt_cell", dt_cell)
                lam_dt = lam[dt_cell_data]
                pm.Exponential("dt_obs", lam=lam_dt, observed=np.asarray(dt_vals, dtype=float))
            cores_to_use = _resolve_cores(cores, chains=int(chains))
            idata = pm.sample(
                draws=int(draws),
                tune=int(tune),
                chains=int(chains),
                cores=int(cores_to_use),
                target_accept=float(target_accept),
                random_seed=None,
                progressbar=True,
            )
        return idata