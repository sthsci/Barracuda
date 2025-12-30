#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytensor.tensor.slinalg as slinalg


def _ensure_object_list(arr):
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return [np.asarray(x, dtype=float) for x in arr.tolist()]
    return [np.asarray(x, dtype=float) for x in arr]


@dataclass(frozen=True)
class Prepared:
    dts: np.ndarray
    nrel: np.ndarray
    events: np.ndarray
    mask: np.ndarray
    T_obs: np.ndarray
    K_by_cell: np.ndarray
    n_cells: int
    L: int


def prepare(dt_list, T_obs, include_censor: bool = True) -> Prepared:
    dt_list = _ensure_object_list(dt_list)
    T_obs = np.asarray(T_obs, dtype=float)
    if T_obs.ndim != 1 or T_obs.size != len(dt_list):
        raise ValueError("T_obs must be length n_cells")

    n_cells = len(dt_list)
    K_by_cell = np.array([int(len(x)) for x in dt_list], dtype=int)

    per_dts, per_nrel, per_events = [], [], []
    Lmax = 0

    for i, dti in enumerate(dt_list):
        dti = np.asarray(dti, dtype=float)
        if np.any(~np.isfinite(dti)) or np.any(dti < 0):
            raise ValueError("dt_list contains invalid dt")

        Ki = int(dti.size)
        t_last = float(dti.sum()) if Ki else 0.0
        cens = float(max(T_obs[i] - t_last, 0.0))

        if include_censor:
            dts_i = np.concatenate([dti, np.array([cens], dtype=float)])
            nrel_i = np.arange(Ki + 1, dtype=int)
            e_i = np.concatenate([np.ones(Ki, dtype=int), np.array([0], dtype=int)])
        else:
            dts_i = dti.copy()
            nrel_i = np.arange(Ki, dtype=int)
            e_i = np.ones(Ki, dtype=int)

        per_dts.append(dts_i)
        per_nrel.append(nrel_i)
        per_events.append(e_i)
        Lmax = max(Lmax, dts_i.size)

    dts = np.zeros((n_cells, Lmax), dtype=float)
    nrel = np.zeros((n_cells, Lmax), dtype=int)
    events = np.zeros((n_cells, Lmax), dtype=int)
    mask = np.zeros((n_cells, Lmax), dtype=float)

    for i in range(n_cells):
        Li = per_dts[i].size
        dts[i, :Li] = per_dts[i]
        nrel[i, :Li] = per_nrel[i]
        events[i, :Li] = per_events[i]
        mask[i, :Li] = 1.0

    return Prepared(
        dts=dts,
        nrel=nrel,
        events=events,
        mask=mask,
        T_obs=T_obs.astype(float),
        K_by_cell=K_by_cell,
        n_cells=n_cells,
        L=Lmax,
    )


def contact_rate(n, lambda0, n50, s):
    n = pt.as_tensor_variable(n, dtype="float64")
    lambda0 = pt.as_tensor_variable(lambda0, dtype="float64")
    n50 = pt.as_tensor_variable(n50, dtype="float64")
    s = pt.as_tensor_variable(s, dtype="float64")
    return lambda0 / (1.0 + pt.exp((n - n50) / s))


def _build_generator(lam_vec):
    N = lam_vec.shape[0] - 1
    Q = pt.zeros((N + 1, N + 1), dtype="float64")
    Q = pt.set_subtensor(Q[pt.arange(N), pt.arange(N)], -lam_vec[:-1])
    Q = pt.set_subtensor(Q[pt.arange(N), pt.arange(N) + 1], lam_vec[:-1])
    return Q


def _tree_sum(xs):
    xs = list(xs)
    if not xs:
        return pt.constant(0.0, dtype="float64")
    while len(xs) > 1:
        xs = [xs[i] + xs[i + 1] if i + 1 < len(xs) else xs[i] for i in range(0, len(xs), 2)]
    return xs[0]


def _lognorm_mu_sigma_from_mean_sd(mean, sd):
    mean = pt.as_tensor_variable(mean, dtype="float64")
    sd = pt.as_tensor_variable(sd, dtype="float64")
    cv2 = (sd / (mean + 1e-18)) ** 2
    sigma = pt.sqrt(pt.log1p(cv2))
    mu = pt.log(mean + 1e-18) - 0.5 * sigma**2
    return mu, sigma


def _as_cell_param(x, n_cells: int):
    x = pt.as_tensor_variable(x, dtype="float64")
    if x.ndim == 0:
        return x[None, None, None]
    if x.ndim == 1:
        return x[None, :, None]
    raise ValueError("Parameter must be scalar or shape (n_cells,)")


def _logL_paths_all_cells_all_m(
    prep: Prepared,
    m_grid: np.ndarray,
    lambda0,
    n50,
    s,
):
    m = pt.as_tensor_variable(m_grid, dtype="int64")[:, None, None]
    n_abs = pt.cast(prep.nrel[None, :, :], "float64") + pt.cast(m, "float64")

    lam = contact_rate(
        n_abs,
        lambda0=_as_cell_param(lambda0, prep.n_cells),
        n50=_as_cell_param(n50, prep.n_cells),
        s=_as_cell_param(s, prep.n_cells),
    )

    dts = pt.as_tensor_variable(prep.dts[None, :, :], dtype="float64")
    events = pt.as_tensor_variable(prep.events[None, :, :], dtype="float64")
    mask = pt.as_tensor_variable(prep.mask[None, :, :], dtype="float64")

    loglam = pt.log(lam + 1e-30)
    term = mask * (events * loglam - lam * dts)
    return pt.sum(term, axis=2)  # (M, n_cells)


def _logp_count_homogeneous_vectorised(
    prep: Prepared,
    m_grid: np.ndarray,
    lambda0,
    n50,
    s,
    pad: int,
):
    m_grid_t = pt.as_tensor_variable(m_grid, dtype="int64")
    M = int(m_grid.shape[0])
    n_cells = prep.n_cells
    Kmax = int(prep.K_by_cell.max()) if n_cells else 0

    Nmax = int(m_grid.max()) + Kmax + int(pad)
    n_grid = pt.arange(Nmax + 1, dtype="float64")

    lam_vec = contact_rate(n_grid, lambda0=lambda0, n50=n50, s=s)
    lam_vec = pt.set_subtensor(lam_vec[Nmax], 0.0)

    Q = _build_generator(lam_vec)

    # NOTE: if T_obs differs per cell, count-only + vectorised expm is not valid.
    # In your synthetic pipeline T_obs is usually constant; we enforce it here.
    T0 = float(np.asarray(prep.T_obs).max())
    if not np.allclose(prep.T_obs, T0):
        raise ValueError("count-only homogeneous vectorised assumes all T_obs equal; use obs_mode='both' or per-cell method.")

    P = slinalg.expm(Q * T0)

    rows = m_grid_t[:, None]  # (M,1)
    cols = m_grid_t[:, None] + pt.as_tensor_variable(prep.K_by_cell[None, :], dtype="int64")  # (M,n_cells)

    probs = P[rows, cols]  # (M,n_cells)
    return pt.log(probs + 1e-30)


def _logp_count_one_cell_expm(
    m_grid: np.ndarray,
    K: int,
    T: float,
    lambda0,
    n50,
    s,
    pad: int,
):
    m = pt.as_tensor_variable(m_grid, dtype="int64")
    Nmax = int(m_grid.max()) + int(K) + int(pad)

    n_grid = pt.arange(Nmax + 1, dtype="float64")
    lam_vec = contact_rate(n_grid, lambda0=lambda0, n50=n50, s=s)
    lam_vec = pt.set_subtensor(lam_vec[Nmax], 0.0)

    Q = _build_generator(lam_vec)
    P = slinalg.expm(Q * float(T))

    probs = P[m, m + int(K)]
    return pt.log(probs + 1e-30)  # (M,)


def build_model_homogeneous(
    *,
    prep: Prepared,
    obs_mode: str = "both",
    offset_max: int = 10,
    m_prior: str = "uniform",
):
    obs_mode = str(obs_mode).lower()
    if obs_mode not in {"both", "count"}:
        raise ValueError("obs_mode must be 'both' or 'count'")
    if offset_max < 0:
        raise ValueError("offset_max must be >= 0")

    m_grid = np.arange(offset_max + 1, dtype=int)
    if m_prior == "uniform":
        logw = -pt.log(offset_max + 1.0)
    else:
        raise ValueError("Only m_prior='uniform' implemented")

    n_cells = prep.n_cells
    Kmax = int(prep.K_by_cell.max()) if n_cells else 0
    pad = max(10, 2 * Kmax + 5)

    with pm.Model() as model:
        lambda0 = pm.LogNormal("lambda0", mu=pt.log(1.0), sigma=1.0)
        n50 = pm.HalfNormal("n50", sigma=30.0)
        s = pm.LogNormal("s", mu=pt.log(2.0), sigma=1.0)

        if obs_mode == "both":
            logL = _logL_paths_all_cells_all_m(prep, m_grid=m_grid, lambda0=lambda0, n50=n50, s=s)  # (M,n_cells)
            logp_cells = pm.math.logsumexp(logL + logw, axis=0)  # (n_cells,)
            pm.Potential("likelihood", pt.sum(logp_cells))
            return model

        # count-only (homogeneous): vectorised expm if all T_obs equal
        logpK = _logp_count_homogeneous_vectorised(prep, m_grid=m_grid, lambda0=lambda0, n50=n50, s=s, pad=pad)  # (M,n_cells)
        logp_cells = pm.math.logsumexp(logpK + logw, axis=0)
        pm.Potential("likelihood", pt.sum(logp_cells))
        return model


def build_model_heterogeneous(
    *,
    prep: Prepared,
    obs_mode: str = "both",
    offset_max: int = 10,
    m_prior: str = "uniform",
):
    obs_mode = str(obs_mode).lower()
    if obs_mode not in {"both", "count"}:
        raise ValueError("obs_mode must be 'both' or 'count'")
    if offset_max < 0:
        raise ValueError("offset_max must be >= 0")

    m_grid = np.arange(offset_max + 1, dtype=int)
    if m_prior == "uniform":
        logw = -pt.log(offset_max + 1.0)
    else:
        raise ValueError("Only m_prior='uniform' implemented")

    n_cells = prep.n_cells
    Kmax = int(prep.K_by_cell.max()) if n_cells else 0
    pad = max(10, 2 * Kmax + 5)

    with pm.Model() as model:
        # Hyperpriors on mean & sd (natural space)
        lambda0_mean = pm.LogNormal("lambda0_mean", mu=pt.log(1.0), sigma=1.0)
        lambda0_sd = pm.HalfNormal("lambda0_sd", sigma=1.0)

        n50_mean = pm.HalfNormal("n50_mean", sigma=30.0)
        n50_sd = pm.HalfNormal("n50_sd", sigma=30.0)

        s_mean = pm.LogNormal("s_mean", mu=pt.log(2.0), sigma=1.0)
        s_sd = pm.HalfNormal("s_sd", sigma=2.0)

        mu_l0, sig_l0 = _lognorm_mu_sigma_from_mean_sd(lambda0_mean, lambda0_sd + 1e-12)
        mu_n50, sig_n50 = _lognorm_mu_sigma_from_mean_sd(n50_mean + 1e-9, n50_sd + 1e-9)
        mu_s, sig_s = _lognorm_mu_sigma_from_mean_sd(s_mean, s_sd + 1e-12)

        lambda0 = pm.LogNormal("lambda0", mu=mu_l0, sigma=sig_l0, shape=n_cells)
        n50 = pm.LogNormal("n50", mu=mu_n50, sigma=sig_n50, shape=n_cells)
        s = pm.LogNormal("s", mu=mu_s, sigma=sig_s, shape=n_cells)

        if obs_mode == "both":
            logL = _logL_paths_all_cells_all_m(prep, m_grid=m_grid, lambda0=lambda0, n50=n50, s=s)  # (M,n_cells)
            logp_cells = pm.math.logsumexp(logL + logw, axis=0)
            pm.Potential("likelihood", pt.sum(logp_cells))
            return model

        # count-only heterogeneous: correct but slow (expm per cell)
        logps = []
        for i in range(n_cells):
            logpK_m = _logp_count_one_cell_expm(
                m_grid=m_grid,
                K=int(prep.K_by_cell[i]),
                T=float(prep.T_obs[i]),
                lambda0=lambda0[i],
                n50=n50[i],
                s=s[i],
                pad=pad,
            )
            logps.append(pm.math.logsumexp(logpK_m + logw))
        pm.Potential("likelihood", _tree_sum(logps))
        return model
