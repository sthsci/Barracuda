#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
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
    dt_list: list[np.ndarray]
    T_obs: np.ndarray
    K_by_cell: np.ndarray
    dts: list[np.ndarray]
    nrel: list[np.ndarray]
    events: list[np.ndarray]
    n_cells: int


def prepare(dt_list, T_obs, include_censor: bool = True) -> Prepared:
    dt_list = _ensure_object_list(dt_list)
    T_obs = np.asarray(T_obs, dtype=float)
    if T_obs.ndim != 1 or T_obs.size != len(dt_list):
        raise ValueError("T_obs must be length n_cells")

    n_cells = len(dt_list)
    K_by_cell = np.array([int(len(x)) for x in dt_list], dtype=int)

    dts_all, nrel_all, events_all = [], [], []
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

        dts_all.append(dts_i.astype(float))
        nrel_all.append(nrel_i.astype(int))
        events_all.append(e_i.astype(int))

    return Prepared(
        dt_list=dt_list,
        T_obs=T_obs.astype(float),
        K_by_cell=K_by_cell,
        dts=dts_all,
        nrel=nrel_all,
        events=events_all,
        n_cells=n_cells,
    )


def contact_rate(n, lambda0, n50, s):
    n = pt.as_tensor_variable(n, dtype="float64")
    return lambda0 / (1.0 + pt.exp((n - n50) / s))


def _build_generator(lam_vec):
    N = lam_vec.shape[0] - 1
    Q = pt.zeros((N + 1, N + 1), dtype="float64")
    Q = pt.set_subtensor(Q[pt.arange(N), pt.arange(N)], -lam_vec[:-1])
    Q = pt.set_subtensor(Q[pt.arange(N), pt.arange(N) + 1], lam_vec[:-1])
    return Q


def _logp_event_path_for_cell(dts_i, nrel_i, e_i, m_grid, lambda0, n50, s):
    m_grid = pt.as_tensor_variable(m_grid, dtype="int64")
    n_abs = pt.cast(nrel_i, "float64")[None, :] + pt.cast(m_grid, "float64")[:, None]
    lam = contact_rate(n_abs, lambda0=lambda0, n50=n50, s=s)
    dts_t = pt.as_tensor_variable(dts_i, dtype="float64")[None, :]
    e_t = pt.as_tensor_variable(e_i, dtype="float64")[None, :]
    return pt.sum(e_t * pt.log(lam) - lam * dts_t, axis=1)


def _logp_count_for_cell(m_grid, K, T, lambda0, n50, s, pad=30):
    m_grid = pt.as_tensor_variable(m_grid, dtype="int64")
    Nmax = int(K) + int(pad) + int(m_grid.shape[0] - 1)
    n_grid = pt.arange(Nmax + 1, dtype="float64")
    lam_vec = contact_rate(n_grid, lambda0=lambda0, n50=n50, s=s)
    lam_vec = pt.set_subtensor(lam_vec[Nmax], 0.0)
    Q = _build_generator(lam_vec)
    P = slinalg.expm(Q * float(T))
    probs = P[m_grid, m_grid + int(K)]
    return pt.log(probs + 1e-30)


def build_model_homogeneous(
    prep: Prepared,
    obs_mode: str = "both",
    truncation: str = "unknown",
    offset_obs: np.ndarray | None = None,
    offset_max: int = 10,
):
    obs_mode = str(obs_mode).lower()
    if obs_mode not in {"both", "count"}:
        raise ValueError("obs_mode must be 'both' or 'count'")
    truncation = str(truncation).lower()
    if truncation not in {"known", "unknown"}:
        raise ValueError("truncation must be 'known' or 'unknown'")
    if offset_max < 0:
        raise ValueError("offset_max must be >= 0")

    n_cells = prep.n_cells
    K_by_cell = prep.K_by_cell
    T_obs = prep.T_obs
    K_max = int(K_by_cell.max()) if n_cells else 0
    pad = max(30, 2 * K_max + 10)

    with pm.Model() as model:
        lambda0 = pm.LogNormal("lambda0", mu=pt.log(1.0), sigma=1.0)
        n50 = pm.HalfNormal("n50", sigma=30.0)
        s = pm.LogNormal("s", mu=pt.log(2.0), sigma=1.0)

        logps = []

        if truncation == "known":
            if offset_obs is None:
                raise ValueError("offset_obs required for truncation='known'")
            offset_obs = np.asarray(offset_obs, dtype=int)
            if offset_obs.shape != (n_cells,):
                raise ValueError("offset_obs must have shape (n_cells,)")

            for i in range(n_cells):
                m_grid = np.array([int(offset_obs[i])], dtype=int)
                if obs_mode == "both":
                    logp = _logp_event_path_for_cell(
                        dts_i=prep.dts[i],
                        nrel_i=prep.nrel[i],
                        e_i=prep.events[i],
                        m_grid=m_grid,
                        lambda0=lambda0,
                        n50=n50,
                        s=s,
                    )[0]
                else:
                    logp = _logp_count_for_cell(
                        m_grid=m_grid,
                        K=int(K_by_cell[i]),
                        T=float(T_obs[i]),
                        lambda0=lambda0,
                        n50=n50,
                        s=s,
                        pad=pad,
                    )[0]
                logps.append(logp)

        else:
            m_grid = np.arange(offset_max + 1, dtype=int)
            logw = -pt.log(offset_max + 1.0)

            for i in range(n_cells):
                if obs_mode == "both":
                    logL_m = _logp_event_path_for_cell(
                        dts_i=prep.dts[i],
                        nrel_i=prep.nrel[i],
                        e_i=prep.events[i],
                        m_grid=m_grid,
                        lambda0=lambda0,
                        n50=n50,
                        s=s,
                    )
                    logp = pm.math.logsumexp(logL_m + logw)
                else:
                    logpK_m = _logp_count_for_cell(
                        m_grid=m_grid,
                        K=int(K_by_cell[i]),
                        T=float(T_obs[i]),
                        lambda0=lambda0,
                        n50=n50,
                        s=s,
                        pad=pad,
                    )
                    logp = pm.math.logsumexp(logpK_m + logw)
                logps.append(logp)

        pm.Potential("likelihood", pt.sum(pt.stack(logps)))

    return model


def _build_parser():
    p = argparse.ArgumentParser(description="Infer sigmoid-rate CTMC parameters (both/count modes).")
    p.add_argument("input_npz", type=str, help="Filtered .npz from filter_contacts.py")
    p.add_argument("--obs-mode", choices=["both", "count"], default="both")
    p.add_argument("--truncation", choices=["unknown", "known"], default="unknown")
    p.add_argument("--offset-max", type=int, default=10, help="Only used when truncation=unknown")
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--tune", type=int, default=2000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--target-accept", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-idata", type=str, default="idata.nc")
    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)
    data = np.load(args.input_npz, allow_pickle=True)

    dt_list = _ensure_object_list(data["dt_list"])
    T_obs = np.asarray(data["T_obs"], dtype=float)
    prep = prepare(dt_list=dt_list, T_obs=T_obs, include_censor=True)

    offset_obs = None
    if args.truncation == "known":
        if "dropped" not in data:
            raise ValueError("Need 'dropped' in npz for truncation=known")
        offset_obs = np.asarray(data["dropped"], dtype=int)[:, 0]

    model = build_model_homogeneous(
        prep=prep,
        obs_mode=args.obs_mode,
        truncation=args.truncation,
        offset_obs=offset_obs,
        offset_max=int(args.offset_max),
    )

    with model:
        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept,
            random_seed=args.seed,
        )

    out_path = os.path.abspath(args.out_idata)
    idata.to_netcdf(out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
