#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Literal, Optional, Tuple
import numpy as np

"""
Synthetic event-count simulator for NK-cell interaction data.

This module generates cell-specific event rates and event counts under
homogeneous and heterogeneous population models. In the heterogeneous case,
rates are sampled from a zero-inflated Gamma distribution.
"""


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


DistMode = Literal["gamma"]
Mode = Literal["homogeneous", "heterogeneous"]


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
        
        if sd_lambda == 0:
            lam = np.full(n_cells, float(mu_lambda), dtype=float)
        
        if p0_lambda > 0:
            is_zero = rng.uniform(0.0, 1.0, size=n_cells) < p0_lambda
            lam[is_zero] = 0.0
            
        
        if dist_mode_s != "gamma":
            raise ValueError("Dist_mode must be one of: gamma (we currently only support gamma distribution for heterogeneous lambda sampling)")
        
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
    T: float = 1.0,
    seed: Optional[int] = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)

    times = process_times(rate=lambda_rate, T=T, rng=rng)
    
    # Between-contact gaps only (excludes the initial waiting time 0 -> first contact).
    dt = np.diff(times) if times.size >= 2 else np.array([], dtype=float)

    # Event gaps including the initial waiting time 0 -> first contact.
    dt_full = np.diff(np.concatenate(([0.0], times))) if times.size >= 1 else np.array([], dtype=float)
    
    n_events = int(times.size)
    
    return {
        "times": times,
        "dt": dt,
        "dt_full": dt_full,
        "n_events": n_events,
    }


def simulate_Population(
    n_cells: int,
    T: float = 1.0,
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
    n_events = np.zeros(n_cells, dtype=int)
    
    for i in range(n_cells):
        simulation_singlecell = simulate_SingleCell(
            lambda_rate=float(rates[i]),
            T=T,
            seed=int(cell_seeds[i]),
        )
        n_events[i] = simulation_singlecell["n_events"]
        if simulation_singlecell["times"].size:
            final_dt = float(T - simulation_singlecell["times"][-1])
        else:
            final_dt = float(T)
        final_dt = max(final_dt, 0.0)


    return {
        "n_cells": n_cells,
        "max_time": T,
        "rates": rates,
        "n_events": n_events,
        
    }


def main():
    pass

if __name__ == "__main__":
    main()