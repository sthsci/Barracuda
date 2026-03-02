#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, LogLocator, ScalarFormatter, NullFormatter, FuncFormatter

plt.rcParams.update({
    "font.family": ["Monaco", "DejaVu Sans Mono", "monospace"],
    "mathtext.fontset": "stix",
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})



def _gamma_shape_rate_from_mean_sd(mean: float, sd: float) -> Tuple[float, float]:
    mean = float(mean)
    sd = float(sd)
    if mean <= 0:
        raise ValueError("gamma mean must be > 0")
    if sd < 0:
        raise ValueError("gamma sd must be >= 0")
    if sd == 0:
        return np.inf, np.inf
    shape = (mean / sd) ** 2
    rate = mean / (sd * sd)
    return float(shape), float(rate)


def _lognormal_mu_sigma_from_mean_sd(mean: float, sd: float) -> Tuple[float, float]:
    mean = float(mean)
    sd = float(sd)
    if mean <= 0:
        raise ValueError("lognormal mean must be > 0")
    if sd < 0:
        raise ValueError("lognormal sd must be >= 0")
    if sd == 0:
        return float(np.log(mean)), 0.0
    sigma2 = float(np.log1p((sd / mean) ** 2))
    mu = float(np.log(mean) - 0.5 * sigma2)
    return mu, float(np.sqrt(sigma2))


def _halfnormal_sigma_from_mean_sd(mean: float, sd: float, tol: float = 1e-2) -> float:
    mean = float(mean)
    sd = float(sd)
    if mean < 0:
        raise ValueError("halfnormal mean must be >= 0")
    if sd < 0:
        raise ValueError("halfnormal sd must be >= 0")
    if mean == 0 and sd == 0:
        return 0.0

    ratio_expected = float(np.sqrt(np.pi / 2.0 - 1.0))
    if mean > 0:
        ratio = sd / mean
        if not np.isclose(ratio, ratio_expected, rtol=tol, atol=0.0):
            raise ValueError(
                f"HalfNormal cannot match arbitrary (mean, sd). Need sd/mean ≈ {ratio_expected:.6f}, got {ratio:.6f}. "
                f"Use gamma/lognormal if you want free mean+sd."
            )

    sigma = mean / float(np.sqrt(2.0 / np.pi)) if mean > 0 else sd / float(np.sqrt(1.0 - 2.0 / np.pi))
    return float(sigma)

DistMode = Literal["gamma", "lognormal", "truncnorm"]
Mode = Literal["homogeneous", "heterogeneous"]


def _sample_truncnorm_positive(
    rng: np.random.Generator,
    mean: float,
    sd: float,
    size: int,
    *,
    max_rounds: int = 50,
) -> np.ndarray:
    mean = float(mean)
    sd = float(sd)
    size = int(size)
    if size < 0:
        raise ValueError("size must be >= 0")
    if size == 0:
        return np.array([], dtype=float)
    if sd < 0:
        raise ValueError("truncnorm sd must be >= 0")
    if sd == 0:
        if mean <= 0:
            raise ValueError("truncnorm with sd=0 requires mean > 0 for positive support")
        return np.full(size, mean, dtype=float)

    out = np.empty(size, dtype=float)
    filled = 0

    for _ in range(int(max_rounds)):
        need = size - filled
        if need <= 0:
            break
        draw_n = int(max(need, 32))
        if mean < 0:
            draw_n = int(draw_n * 5)
        samples = rng.normal(loc=mean, scale=sd, size=draw_n).astype(float)
        samples = samples[samples > 0]
        if samples.size == 0:
            continue
        take = min(need, int(samples.size))
        out[filled : filled + take] = samples[:take]
        filled += take

    if filled != size:
        raise RuntimeError(
            "Failed to sample enough positive values for truncnorm; "
            "try increasing mean relative to sd (or use gamma/lognormal)."
        )
    return out

def sample_lambda(
    n_cells: int,
    mode: Mode = "homogeneous",
    seed: Optional[int] = None,
    *,
    mu_lambda: Optional[float] = None,
    p0_lambda: Optional[float] = None,
    sd_lambda: Optional[float] = None,
    Dist_mode: DistMode = "gamma"
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_cells = int(n_cells)
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")

    mode_s = str(mode).strip().lower()
    dist_mode_s = str(Dist_mode).strip().lower()
    
    if mode_s not in {"homogeneous", "heterogeneous"}:
        raise ValueError(
            "mode must be one of: homogeneous, heterogeneous"
        )
    
    if mode_s == "homogeneous":
        if mu_lambda is None:
            raise ValueError("homogeneous requires mu_lambda")
        lam = np.full(n_cells, float(mu_lambda), dtype=float)
    
    if mode_s == "heterogeneous":
        if mu_lambda is None or sd_lambda is None or p0_lambda is None:
            raise ValueError("heterogeneous sampling requires mu_lambda, sd_lambda, and p0_lambda")
        mu_lambda, sd_lambda = float(mu_lambda), float(sd_lambda)

        if dist_mode_s == "gamma":
            if sd_lambda == 0:
                lam = np.full(n_cells, mu_lambda, dtype=float)
            else:
                shape, rate = _gamma_shape_rate_from_mean_sd(mu_lambda, sd_lambda)
                lam = rng.gamma(shape=shape, scale=1.0 / rate, size=n_cells).astype(float)
        
        elif dist_mode_s == "lognormal":
            mu, sigma = _lognormal_mu_sigma_from_mean_sd(mu_lambda, sd_lambda)
            lam = rng.lognormal(mean=mu, sigma=sigma, size=n_cells).astype(float)
        
        elif dist_mode_s == "truncnorm":
            lam = _sample_truncnorm_positive(rng=rng, mean=mu_lambda, sd=sd_lambda, size=n_cells)
        
        if sd_lambda == 0:
            lam = np.full(n_cells, float(mu_lambda), dtype=float)
        if p0_lambda > 0:
            is_zero = rng.uniform(0.0, 1.0, size=n_cells) < p0_lambda
            lam[is_zero] = 0.0
        
    return lam


def process_times(rate: float, T: float, rng: np.random.Generator) -> np.ndarray:
    rate = float(rate)
    T = float(T)
    if T < 0:
        raise ValueError("T must be >= 0")
    if not np.isfinite(rate) or rate < 0:
        raise ValueError("rate must be finite and >= 0")
    if rate == 0 or T == 0:
        return np.array([], dtype=float)

    t = 0.0
    ts: list[float] = []
    while True:
        dt = float(rng.exponential(1.0 / rate))
        if t + dt > T:
            break
        t += dt
        ts.append(t)
    return np.asarray(ts, dtype=float)

def simulate_SingleCell(
    lambda_rate: float,
    T: float,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    times = process_times(rate=lambda_rate, T=T, rng=rng)
    
    # Between-contact gaps only (excludes the initial waiting time 0 -> first contact).
    dt = np.diff(times) if times.size >= 2 else np.array([], dtype=float)

    # Event gaps including the initial waiting time 0 -> first contact.
    dt_full = np.diff(np.concatenate(([0.0], times))) if times.size >= 1 else np.array([], dtype=float)
    
    n_contacts = int(times.size)
    
    return {
        "times": times,
        "dt": dt,
        "dt_full": dt_full,
        "n_contacts": n_contacts,
    }

OBS_MODE = Literal["Complete", "Truncated"]

def simulate_Population(
    n_cells: int,
    T: float,
    truncation_noise: Optional[float] = None,
    obs_mode: OBS_MODE = "Complete",
    *,
    rates: Optional[np.ndarray] = None,
    mode: Mode = "homogeneous",
    seed: Optional[int] = None,
    mu_lambda: Optional[float] = None,
    sd_lambda: Optional[float] = None,
    p0_lambda: Optional[float] = None,
    Dist_mode: DistMode = "gamma",
) -> dict[str, Any]:
    n_cells = int(n_cells)
    T = float(T)
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    if T < 0:
        raise ValueError("T must be >= 0")
    
    rng = np.random.default_rng(seed)
    
    if rates is None:
        rates = sample_lambda(
            n_cells=n_cells,
            mode=mode,
            seed=seed,
            mu_lambda=mu_lambda,
            sd_lambda=sd_lambda,
            p0_lambda=p0_lambda,
            Dist_mode=Dist_mode,
        )
    rates = np.asarray(rates, dtype=float)
    if rates.shape != (n_cells,):
        raise ValueError(f"rates must have shape ({n_cells},), got {rates.shape}")
    
    cell_seeds = rng.integers(0, 2**32 - 1, size=n_cells, dtype=np.uint32)
    
    ###simulate each cell####
    times_list: list[np.ndarray] = []
    dt_list: list[np.ndarray] = []
    dt_list_full: list[np.ndarray] = []
    dt_list_plus_final: list[np.ndarray] = []
    dt_list_full_plus_final: list[np.ndarray] = []
    n_contacts = np.zeros(n_cells, dtype=int)
    
    if obs_mode not in ("Complete", "Truncated"):
        raise ValueError("obs_mode must be one of: Complete, Truncated")
    
    if obs_mode == "Complete":
        for i in range(n_cells):
            simulation_singlecell = simulate_SingleCell(
                lambda_rate=float(rates[i]),
                T=T,
                seed=int(cell_seeds[i]),
            )
            times_list.append(simulation_singlecell["times"])
            dt_list.append(simulation_singlecell["dt"])
            dt_list_full.append(simulation_singlecell["dt_full"])
            n_contacts[i] = simulation_singlecell["n_contacts"]
            if simulation_singlecell["times"].size:
                final_dt = float(T - simulation_singlecell["times"][-1])
            else:
                final_dt = float(T)
            final_dt = max(final_dt, 0.0)
            dt_list_plus_final.append(
                np.concatenate([simulation_singlecell["dt"], np.array([final_dt], dtype=float)])
            )
            dt_list_full_plus_final.append(
                np.concatenate([simulation_singlecell["dt_full"], np.array([final_dt], dtype=float)])
            )
    elif obs_mode == "Truncated":
        if truncation_noise is None:
            raise ValueError("truncation_noise must be provided for Truncated obs_mode")
        trunc_sd = float(truncation_noise)
        if trunc_sd < 0:
            raise ValueError("truncation_noise must be >= 0 for Truncated obs_mode")
        trunc_noise = np.abs(rng.normal(loc=0.0, scale=trunc_sd, size=n_cells)).astype(float)
        T_list = T - trunc_noise
        for i in range(n_cells):
            Ti = min(max(float(T_list[i]), 0.0), T)
            simulation_singlecell = simulate_SingleCell(
                lambda_rate=float(rates[i]),
                T=Ti,
                seed=int(cell_seeds[i]),
            )
            times_list.append(simulation_singlecell["times"])
            dt_list.append(simulation_singlecell["dt"])
            dt_list_full.append(simulation_singlecell["dt_full"])
            n_contacts[i] = simulation_singlecell["n_contacts"]
            if simulation_singlecell["times"].size:
                final_dt = float(Ti - simulation_singlecell["times"][-1])
            else:
                final_dt = float(Ti)
            final_dt = max(final_dt, 0.0)
            dt_list_plus_final.append(
                np.concatenate([simulation_singlecell["dt"], np.array([final_dt], dtype=float)])
            )
            dt_list_full_plus_final.append(
                np.concatenate([simulation_singlecell["dt_full"], np.array([final_dt], dtype=float)])
            )

    return {
        "n_cells": n_cells,
        "max_time": T,
        "rates": rates,
        "times_list": np.asarray(times_list, dtype=object),
        "dt_list": np.asarray(dt_list, dtype=object),
        "dt_list_full": np.asarray(dt_list_full, dtype=object),
        "dt_list+final": np.asarray(dt_list_plus_final, dtype=object),
        "dt_list_full+final": np.asarray(dt_list_full_plus_final, dtype=object),
        "n_contacts": n_contacts,
        "obs_mode": obs_mode,
        "truncation_noise": truncation_noise,
    }



def main():
    pass

if __name__ == "__main__":
    main()