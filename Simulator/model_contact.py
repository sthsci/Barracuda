#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, LogLocator, ScalarFormatter, LogLocator, ScalarFormatter, NullFormatter
plt.rcParams.update({
    "font.family": ["Monaco", "DejaVu Sans Mono", "monospace"],
    "mathtext.fontset": "stix",
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

# -----------------------------
# Core utilities
# -----------------------------
def _ensure_object_list(arr):
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return [np.asarray(x, dtype=float) for x in arr.tolist()]
    return [np.asarray(x, dtype=float) for x in arr]


def _safe_expit(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def contact_rate(n, lambda0, n50, s):
    n = np.asarray(n, dtype=float)
    lambda0 = np.asarray(lambda0, dtype=float)
    n50 = np.asarray(n50, dtype=float)
    s = np.asarray(s, dtype=float)

    if np.any(~np.isfinite(lambda0)) or np.any(lambda0 < 0):
        raise ValueError("lambda0 must be finite and >= 0")
    if np.any(~np.isfinite(n50)):
        raise ValueError("n50 must be finite")
    if np.any(~np.isfinite(s)) or np.any(s <= 0):
        raise ValueError("s must be finite and > 0")

    z = (n - n50) / s
    ez = np.exp(-np.abs(z))
    sigmoid = np.where(z >= 0, ez / (1.0 + ez), 1.0 / (1.0 + ez))
    return lambda0 * sigmoid


# -----------------------------
# Heterogeneity sampler
# -----------------------------
def sample_contact_parameters_hierarchical(
    n_cells,
    dist_mode="lognormal",
    seed=None,
    min_n50=0.0,
    min_lambda0=0.0,
    min_s=1e-6,
    mean_lambda0=1.0,
    sd_lambda0=0.0,
    mean_n50=5.0,
    sd_n50=0.0,
    mean_s=1.0,
    sd_s=0.0,
    mu_log_lambda0=None,
    sigma_log_lambda0=None,
    mu_log_s=None,
    sigma_log_s=None,
):
    rng = np.random.default_rng(seed)
    n_cells = int(n_cells)
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")

    dist_mode = str(dist_mode).lower()
    if dist_mode not in {"lognormal", "normal", "trunc_normal", "truncated_normal", "halfnormal", "half_normal", "gamma"}:
        raise ValueError(f"Unknown dist_mode: {dist_mode}")

    def _lognormal_from_mean_sd(mean, sd):
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

    def _gamma_from_mean_sd(mean, sd):
        mean = float(mean)
        sd = float(sd)
        if mean <= 0:
            raise ValueError("gamma mean must be > 0")
        if sd < 0:
            raise ValueError("gamma sd must be >= 0")
        if sd == 0:
            return np.inf, 0.0
        shape = (mean / sd) ** 2
        scale = (sd * sd) / mean
        return float(shape), float(scale)

    mean_lambda0 = float(mean_lambda0)
    sd_lambda0 = float(sd_lambda0)
    mean_n50 = float(mean_n50)
    sd_n50 = float(sd_n50)
    mean_s = float(mean_s)
    sd_s = float(sd_s)

    if dist_mode == "lognormal":
        if (mu_log_lambda0 is not None) and (sigma_log_lambda0 is not None):
            mu = float(mu_log_lambda0)
            sig = float(sigma_log_lambda0)
            mean_lambda0 = float(np.exp(mu + 0.5 * sig * sig))
            sd_lambda0 = float(mean_lambda0 * np.sqrt(np.exp(sig * sig) - 1.0))
        if (mu_log_s is not None) and (sigma_log_s is not None):
            mu = float(mu_log_s)
            sig = float(sigma_log_s)
            mean_s = float(np.exp(mu + 0.5 * sig * sig))
            sd_s = float(mean_s * np.sqrt(np.exp(sig * sig) - 1.0))

    if sd_lambda0 < 0 or sd_n50 < 0 or sd_s < 0:
        raise ValueError("sd values must be >= 0")

    if dist_mode == "lognormal":
        mu0, sig0 = _lognormal_from_mean_sd(mean_lambda0, sd_lambda0)
        mu2, sig2 = _lognormal_from_mean_sd(mean_s, sd_s)
        lambda0_vals = rng.lognormal(mean=mu0, sigma=sig0, size=n_cells)
        n50_vals = rng.normal(loc=mean_n50, scale=sd_n50, size=n_cells)
        s_vals = rng.lognormal(mean=mu2, sigma=sig2, size=n_cells)

    elif dist_mode == "gamma":
        if sd_lambda0 == 0:
            lambda0_vals = np.full(n_cells, mean_lambda0, dtype=float)
        else:
            a0, b0 = _gamma_from_mean_sd(mean_lambda0, sd_lambda0)
            lambda0_vals = rng.gamma(shape=a0, scale=b0, size=n_cells)

        if sd_n50 == 0:
            n50_vals = np.full(n_cells, mean_n50, dtype=float)
        else:
            a1, b1 = _gamma_from_mean_sd(mean_n50, sd_n50)
            n50_vals = rng.gamma(shape=a1, scale=b1, size=n_cells)

        if sd_s == 0:
            s_vals = np.full(n_cells, mean_s, dtype=float)
        else:
            a2, b2 = _gamma_from_mean_sd(mean_s, sd_s)
            s_vals = rng.gamma(shape=a2, scale=b2, size=n_cells)

    elif dist_mode == "normal":
        lambda0_vals = rng.normal(loc=mean_lambda0, scale=sd_lambda0, size=n_cells)
        n50_vals = rng.normal(loc=mean_n50, scale=sd_n50, size=n_cells)
        s_vals = rng.normal(loc=mean_s, scale=sd_s, size=n_cells)

    elif dist_mode in {"trunc_normal", "truncated_normal"}:
        def _sample_trunc_normal(mu, sigma, low):
            mu = float(mu)
            sigma = float(sigma)
            low = float(low)
            if sigma < 0:
                raise ValueError("sd values must be >= 0")
            if sigma == 0:
                return np.full(n_cells, max(mu, low), dtype=float)
            vals = rng.normal(loc=mu, scale=sigma, size=n_cells)
            mask = vals < low
            tries = 0
            while np.any(mask) and tries < 300:
                vals[mask] = rng.normal(loc=mu, scale=sigma, size=int(np.sum(mask)))
                mask = vals < low
                tries += 1
            if np.any(mask):
                vals[mask] = low
            return vals

        lambda0_vals = _sample_trunc_normal(mean_lambda0, sd_lambda0, min_lambda0 if min_lambda0 is not None else 0.0)
        n50_vals = _sample_trunc_normal(mean_n50, sd_n50, min_n50 if min_n50 is not None else 0.0)
        s_vals = _sample_trunc_normal(mean_s, sd_s, min_s if min_s is not None else 0.0)

    else:
        lambda0_vals = float(mean_lambda0) + np.abs(rng.normal(loc=0.0, scale=sd_lambda0, size=n_cells))
        n50_vals = float(mean_n50) + np.abs(rng.normal(loc=0.0, scale=sd_n50, size=n_cells))
        s_vals = float(mean_s) + np.abs(rng.normal(loc=0.0, scale=sd_s, size=n_cells))

    if min_lambda0 is not None:
        lambda0_vals = np.maximum(np.asarray(lambda0_vals, dtype=float), float(min_lambda0))
    if min_n50 is not None:
        n50_vals = np.maximum(np.asarray(n50_vals, dtype=float), float(min_n50))
    if min_s is not None:
        s_vals = np.maximum(np.asarray(s_vals, dtype=float), float(min_s))

    return np.column_stack([lambda0_vals, n50_vals, s_vals]).astype(float)


# -----------------------------
# Simulation modes
# -----------------------------
def simulate_population_individual(n_cells, max_time, thetas, seed=None, max_events=None):
    rng = np.random.default_rng(seed)
    max_time = float(max_time)
    n_cells = int(n_cells)
    thetas = np.asarray(thetas, dtype=float)
    if thetas.shape != (n_cells, 3):
        raise ValueError("thetas must have shape (n_cells, 3)")

    cell_seeds = rng.integers(0, 2**32 - 1, size=n_cells, dtype=np.uint32)
    times_list, dt_list = [], []
    n_end = np.zeros(n_cells, dtype=int)

    for i in range(n_cells):
        lam0, n50, s = map(float, thetas[i])
        rrng = np.random.default_rng(int(cell_seeds[i]))
        t = 0.0
        n = 0
        times = []
        while True:
            if (max_events is not None) and (len(times) >= int(max_events)):
                break
            lam = float(contact_rate(n, lam0, n50, s))
            if not (np.isfinite(lam) and lam > 0):
                break
            dt = float(rrng.exponential(1.0 / lam))
            if t + dt > max_time:
                break
            t += dt
            n += 1
            times.append(t)
        ts = np.asarray(times, dtype=float)
        dts = np.diff(np.concatenate(([0.0], ts))) if ts.size else np.array([], dtype=float)
        times_list.append(ts)
        dt_list.append(dts)
        n_end[i] = int(ts.size)

    return dict(
        times_list=np.array(times_list, dtype=object),
        dt_list=np.array(dt_list, dtype=object),
        thetas=thetas,
        n_end=n_end,
        max_time=float(max_time),
        n_cells=int(n_cells),
        sim_mode="individual",
    )


def simulate_population_global(n_cells, max_time, thetas, seed=None, max_events=None):
    rng = np.random.default_rng(seed)
    max_time = float(max_time)
    n_cells = int(n_cells)
    thetas = np.asarray(thetas, dtype=float)
    if thetas.shape != (n_cells, 3):
        raise ValueError("thetas must have shape (n_cells, 3)")

    t = 0.0
    n = np.zeros(n_cells, dtype=int)
    times_list = [[] for _ in range(n_cells)]
    total_events = 0

    while True:
        if (max_events is not None) and (total_events >= int(max_events)):
            break
        lam = contact_rate(n, thetas[:, 0], thetas[:, 1], thetas[:, 2]).astype(float)
        lam = np.where(np.isfinite(lam) & (lam > 0), lam, 0.0)
        total_rate = float(np.sum(lam))
        if total_rate <= 0:
            break
        dt = float(rng.exponential(1.0 / total_rate))
        if t + dt > max_time:
            break
        t += dt
        probs = lam / total_rate
        i = int(rng.choice(n_cells, p=probs))
        n[i] += 1
        times_list[i].append(t)
        total_events += 1

    times_list = [np.asarray(ts, dtype=float) for ts in times_list]
    dt_list = [np.diff(np.concatenate(([0.0], ts))) if ts.size else np.array([], dtype=float) for ts in times_list]
    n_end = np.array([int(ts.size) for ts in times_list], dtype=int)

    return dict(
        times_list=np.array(times_list, dtype=object),
        dt_list=np.array(dt_list, dtype=object),
        thetas=thetas,
        n_end=n_end,
        max_time=float(max_time),
        n_cells=int(n_cells),
        sim_mode="global",
    )


def simulate_population(
    n_cells,
    max_time,
    theta=None,
    thetas=None,
    hierarchical=None,
    seed=None,
    max_events=None,
    sim_mode="global",
):
    if sum(x is not None for x in [theta, thetas, hierarchical]) != 1:
        raise ValueError("Provide exactly one of theta, thetas, or hierarchical")

    n_cells = int(n_cells)
    if theta is not None:
        theta = tuple(theta)
        if len(theta) != 3:
            raise ValueError("theta must be (lambda0, n50, s)")
        thetas_used = np.tile(np.array(theta, dtype=float)[None, :], (n_cells, 1))
    elif thetas is not None:
        thetas_used = np.asarray(thetas, dtype=float)
        if thetas_used.shape != (n_cells, 3):
            raise ValueError("thetas must have shape (n_cells, 3)")
    else:
        if not isinstance(hierarchical, dict):
            raise ValueError("hierarchical must be a dict")
        thetas_used = sample_contact_parameters_hierarchical(n_cells=n_cells, seed=seed, **hierarchical)

    sim_mode = str(sim_mode).lower()
    if sim_mode == "global":
        return simulate_population_global(n_cells, max_time, thetas_used, seed=seed, max_events=max_events)
    if sim_mode == "individual":
        return simulate_population_individual(n_cells, max_time, thetas_used, seed=seed, max_events=max_events)
    raise ValueError("sim_mode must be 'global' or 'individual'")


# -----------------------------
# Filtering: mimic incomplete experimental observation
# -----------------------------
def filter_cell_times(times, T, rng, drop_start_max=0, drop_end_max=0, max_censor_extra=None):
    times = np.asarray(times, dtype=float)
    T = float(T)
    if times.size and (np.any(times < 0) or np.any(times > T)):
        raise ValueError("times must lie within [0, T]")

    K = int(times.size)
    m = int(rng.integers(0, min(int(drop_start_max), K) + 1)) if drop_start_max else 0
    r = int(rng.integers(0, min(int(drop_end_max), K - m) + 1)) if drop_end_max else 0

    start_time = float(times[m - 1]) if m > 0 else 0.0
    kept = times[m: K - r] if (K - r) >= m else np.array([], dtype=float)

    last_kept = float(kept[-1]) if kept.size else start_time

    if r > 0:
        next_hidden = float(times[K - r])
        end_limit = min(next_hidden, T)
    else:
        end_limit = T

    slack = max(end_limit - last_kept, 0.0)
    cap = slack if max_censor_extra is None else min(float(max_censor_extra), slack)
    censor_extra = float(rng.uniform(0.0, cap)) if cap > 0 else 0.0

    end_time = min(max(last_kept + censor_extra, start_time), end_limit)

    kept = kept[(kept >= start_time) & (kept <= end_time)]
    times_obs = kept - start_time
    T_obs = float(end_time - start_time)

    if times_obs.size:
        dt = np.diff(np.concatenate(([0.0], times_obs)))
        t_last = float(times_obs[-1])
        K_obs = int(times_obs.size)
    else:
        dt = np.array([], dtype=float)
        t_last = 0.0
        K_obs = 0

    censor_dt = float(max(T_obs - t_last, 0.0))

    meta = dict(
        dropped_start=m,
        dropped_end=r,
        start_time=start_time,
        end_time=end_time,
        T_obs=T_obs,
        K_obs=K_obs,
        censor_dt=censor_dt,
    )
    return times_obs.astype(float), dt.astype(float), meta


def filter_dataset_in_memory(times_list, T, seed=0, drop_start_max=0, drop_end_max=0, max_censor_extra=None):
    times_list = _ensure_object_list(times_list)
    T = float(T)
    n_cells = len(times_list)

    rng = np.random.default_rng(seed)
    times_f, dt_f = [], []
    T_obs = np.zeros(n_cells, dtype=float)
    censor_dt = np.zeros(n_cells, dtype=float)
    dropped = np.zeros((n_cells, 2), dtype=int)
    start_end = np.zeros((n_cells, 2), dtype=float)

    for i, times in enumerate(times_list):
        t_obs, dt_obs, meta = filter_cell_times(
            times=times,
            T=T,
            rng=rng,
            drop_start_max=drop_start_max,
            drop_end_max=drop_end_max,
            max_censor_extra=max_censor_extra,
        )
        times_f.append(t_obs)
        dt_f.append(dt_obs)
        T_obs[i] = meta["T_obs"]
        censor_dt[i] = meta["censor_dt"]
        dropped[i, 0] = meta["dropped_start"]
        dropped[i, 1] = meta["dropped_end"]
        start_end[i, 0] = meta["start_time"]
        start_end[i, 1] = meta["end_time"]

    return dict(
        max_time=float(T),
        times_list=np.array(times_f, dtype=object),
        dt_list=np.array(dt_f, dtype=object),
        T_obs=T_obs,
        censor_dt=censor_dt,
        dropped=dropped,
        start_end=start_end,
    )


def save_simulation_npz(out, path):
    path = os.path.abspath(str(path))
    np.savez_compressed(
        path,
        thetas=np.asarray(out["thetas"], dtype=float),
        n_end=np.asarray(out["n_end"], dtype=int),
        max_time=float(out["max_time"]),
        times_list=np.asarray(out["times_list"], dtype=object),
        dt_list=np.asarray(out["dt_list"], dtype=object),
        n_cells=int(out["n_cells"]),
        sim_mode=str(out.get("sim_mode", "")),
    )
    return path


def save_filtered_npz(filtered, path):
    path = os.path.abspath(str(path))
    np.savez_compressed(
        path,
        max_time=float(filtered["max_time"]),
        times_list=np.asarray(filtered["times_list"], dtype=object),
        dt_list=np.asarray(filtered["dt_list"], dtype=object),
        T_obs=np.asarray(filtered["T_obs"], dtype=float),
        censor_dt=np.asarray(filtered["censor_dt"], dtype=float),
        dropped=np.asarray(filtered["dropped"], dtype=int),
        start_end=np.asarray(filtered["start_end"], dtype=float),
    )
    return path


def load_sim_or_filtered_npz(path):
    data = np.load(str(path), allow_pickle=True)
    out = {k: data[k] for k in data.files}
    if "times_list" in out:
        out["times_list"] = np.asarray(out["times_list"], dtype=object)
    if "dt_list" in out:
        out["dt_list"] = np.asarray(out["dt_list"], dtype=object)
    return out


# -----------------------------
# Visualisation: 2x2 before/after
# -----------------------------
def _apply_transparent_axes(ax):
    ax.set_facecolor("none")


def _legend_white(ax, loc="best"):
    leg = ax.legend(loc=loc, frameon=True, fontsize=10, edgecolor="black")
    if leg is not None:
        fr = leg.get_frame()
        fr.set_facecolor("white")
        fr.set_alpha(1.0)


def _pooled_dt(dt_list):
    parts = []
    for dt in _ensure_object_list(dt_list):
        dt = np.asarray(dt, dtype=float)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            parts.append(dt)
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def _counts_from_times_list(times_list):
    return np.array([int(np.asarray(t, dtype=float).size) for t in _ensure_object_list(times_list)], dtype=int)

def tune_log_xticks(ax, num_major=6, minor_subs=(2, 5)):
    ax.set_xscale("log")

    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=num_major))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=minor_subs, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())  # <- key: no minor labels

    ax.tick_params(axis="x", which="major", labelsize=10, length=6)
    ax.tick_params(axis="x", which="minor", length=3)

def plot_before_after_2x2(
    raw_groups,
    filt_groups,
    labels,
    T,
    save_path,
    cmap_name="YlGnBu",
    dpi=300,
):
    if len(raw_groups) != len(filt_groups) or len(labels) != len(raw_groups):
        raise ValueError("raw_groups, filt_groups, labels must have same length")

    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(x) for x in np.linspace(0.35, 0.95, num=len(labels))]

    raw_counts = [np.asarray(g["n_end"], dtype=int) if "n_end" in g else _counts_from_times_list(g["times_list"]) for g in raw_groups]
    filt_counts = [_counts_from_times_list(g["times_list"]) for g in filt_groups]

    all_counts = np.concatenate([*raw_counts, *filt_counts]) if (raw_counts or filt_counts) else np.array([], dtype=int)
    max_n = int(np.max(all_counts)) if all_counts.size else 0
    count_bins = np.arange(-0.5, max_n + 1.5, 1.0)

    raw_dt = [_pooled_dt(g["dt_list"]) for g in raw_groups]
    filt_dt = [_pooled_dt(g["dt_list"]) for g in filt_groups]

    pooled_dt = np.concatenate([*raw_dt, *filt_dt]) if (raw_dt or filt_dt) else np.array([], dtype=float)
    pooled_dt = pooled_dt[np.isfinite(pooled_dt) & (pooled_dt > 0)]
    if pooled_dt.size:
        lo = max(float(np.min(pooled_dt)), 1e-8)
        hi = float(np.max(pooled_dt))
        dt_bins = np.logspace(np.log10(lo), np.log10(hi), num=24)
    else:
        dt_bins = "auto"

    fig, axs = plt.subplots(2, 2, figsize=(12.8, 8.4), dpi=dpi)
    fig.patch.set_alpha(0.0)
    for ax in axs.ravel():
        _apply_transparent_axes(ax)

    ax00, ax01 = axs[0, 0], axs[0, 1]
    ax10, ax11 = axs[1, 0], axs[1, 1]

    for counts, lab, col in zip(raw_counts, labels, colors):
        ax00.hist(
            counts,
            bins=count_bins,
            density=True,
            histtype="stepfilled",
            alpha=0.5,
            edgecolor=col,
            facecolor=col,
            lw=1.5,
            label=lab,
        )
    ax00.set_title(f"Contact number (raw)  T={float(T):g}")
    ax00.set_xlabel("Contacts by T")
    ax00.set_ylabel("Density")
    ax00.xaxis.set_major_locator(MultipleLocator(1))
    ax00.grid(True, alpha=0.25)
    ax00.set_xlim(left=0)
    _legend_white(ax00, loc="best")

    for counts, lab, col in zip(filt_counts, labels, colors):
        ax01.hist(
            counts,
            bins=count_bins,
            density=True,
            histtype="stepfilled",
            alpha=0.5,
            edgecolor=col,
            facecolor=col,
            lw=1.5,
            label=lab,
        )
    ax01.set_title("Contact number (filtered)")
    ax01.set_xlabel("Observed contacts")
    ax01.set_ylabel("Density")
    ax01.xaxis.set_major_locator(MultipleLocator(1))
    ax01.grid(True, alpha=0.25)
    ax01.set_xlim(left=0)
    _legend_white(ax01, loc="best")

    for dt, lab, col in zip(raw_dt, labels, colors):
        if dt.size == 0:
            continue
        ax10.hist(
            dt,
            bins=dt_bins,
            density=True,
            histtype="stepfilled",
            alpha=0.5,
            edgecolor=col,
            facecolor=col,
            lw=1.5,
            label=lab,
        )
    ax10.set_title("Inter-contact Δt (raw)")
    ax10.set_xlabel("Δt")
    ax10.set_ylabel("Density")
    ax10.set_xscale("log")
    ax10.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    # ax10.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax10.xaxis.set_major_formatter(ScalarFormatter())
    # ax10.xaxis.set_minor_formatter(ScalarFormatter())
    ax10.tick_params(axis="x", which="minor", labelsize=8)
    ax10.grid(True, which="both", alpha=0.25)
    _legend_white(ax10, loc="best")

    for dt, lab, col in zip(filt_dt, labels, colors):
        if dt.size == 0:
            continue
        ax11.hist(
            dt,
            bins=dt_bins,
            density=True,
            histtype="stepfilled",
            alpha=0.5,
            edgecolor=col,
            facecolor=col,
            lw=1.5,
            label=lab,
        )
    ax11.set_title("Inter-contact Δt (filtered)")
    ax11.set_xlabel("Observed Δt")
    ax11.set_ylabel("Density")
    ax11.set_xscale("log")
    ax11.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    # ax11.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax11.xaxis.set_major_formatter(ScalarFormatter())
    # ax11.xaxis.set_minor_formatter(ScalarFormatter())
    ax11.tick_params(axis="x", which="minor", labelsize=8)
    ax11.grid(True, which="both", alpha=0.25)
    _legend_white(ax11, loc="best")
    
    axs[1, 0].set_xlim(1e-8, 1e1)
    axs[1, 1].set_xlim(1e-8, 1e1)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)


# -----------------------------
# Demo scenarios (like your reference script)
# -----------------------------
def run_demo(outdir, T=180.0, n_cells=500, seed=990, sim_mode="global", drop_start_max=2, drop_end_max=2, max_censor_extra=None, cmap_name="YlGnBu"):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        dict(
            name="homo_lambda0=1.0__n50=5.0__s=0.3",
            kind="homogeneous",
            theta=(1.0, 5.0, 0.3),
        ),
        dict(
            name="homo_lambda0=1.0__n50=5.0__s=1.5",
            kind="homogeneous",
            theta=(1.0, 5.0, 1.5),
        ),
        dict(
            name="hete_lambda0~LogN(2.0)__n50~N(8.0)__s~LogN(0.6)",
            kind="heterogeneous",
            hierarchical=dict(
                dist_mode="lognormal",
                mu_log_lambda0=float(np.log(2.0)),
                sigma_log_lambda0=0.35,
                mean_n50=8.0,
                sd_n50=1.0,
                mu_log_s=float(np.log(0.6)),
                sigma_log_s=0.25,
            ),
        ),
    ]

    raw_groups, filt_groups, labels = [], [], []

    for j, sc in enumerate(scenarios):
        if sc["kind"] == "homogeneous":
            raw = simulate_population(
                n_cells=n_cells,
                max_time=T,
                theta=sc["theta"],
                seed=seed + j,
                sim_mode=sim_mode,
            )
        else:
            raw = simulate_population(
                n_cells=n_cells,
                max_time=T,
                hierarchical=sc["hierarchical"],
                seed=seed + j,
                sim_mode=sim_mode,
            )

        filt = filter_dataset_in_memory(
            times_list=raw["times_list"],
            T=float(raw["max_time"]),
            seed=seed + j,
            drop_start_max=drop_start_max,
            drop_end_max=drop_end_max,
            max_censor_extra=max_censor_extra,
        )

        raw_groups.append(raw)
        filt_groups.append(filt)
        labels.append(sc["name"].replace("homo_", "").replace("hete_", ""))

        raw_mean = float(np.mean(raw["n_end"]))
        kept_counts = _counts_from_times_list(filt["times_list"])
        kept_mean = float(np.mean(kept_counts))
        print(sc["name"], "| raw mean=", raw_mean, "| kept mean=", kept_mean)

    plot_before_after_2x2(
        raw_groups=raw_groups,
        filt_groups=filt_groups,
        labels=labels,
        T=T,
        save_path=outdir / "compare_before_after_2x2.png",
        cmap_name=cmap_name,
        dpi=300,
    )
    print("Saved:", str(outdir / "compare_before_after_2x2.png"))


# -----------------------------
# CLI
# -----------------------------
def _build_parser():
    p = argparse.ArgumentParser(description="Contact simulator (global default) + filtering + 2x2 visualisation.")

    sub = p.add_subparsers(dest="cmd", required=True)

    simf = sub.add_parser("simulate_and_filter", help="Simulate then filter, write .npz outputs.")
    simf.add_argument("--T", type=float, default=120.0)
    simf.add_argument("--n_cells", type=int, default=600)
    simf.add_argument("--seed", type=int, default=2025)
    simf.add_argument("--sim_mode", choices=["global", "individual"], default="global")
    simf.add_argument("--max_events", type=int, default=None)

    simf.add_argument("--mode", choices=["homogeneous", "hierarchical"], default="homogeneous")
    simf.add_argument("--lambda0", type=float, default=1.0)
    simf.add_argument("--n50", type=float, default=5.0)
    simf.add_argument("--s", type=float, default=1.0)

    simf.add_argument("--dist_mode", type=str, default="lognormal")
    simf.add_argument("--mean_lambda0", type=float, default=1.0)
    simf.add_argument("--sd_lambda0", type=float, default=0.0)
    simf.add_argument("--mean_n50", type=float, default=5.0)
    simf.add_argument("--sd_n50", type=float, default=0.0)
    simf.add_argument("--mean_s", type=float, default=1.0)
    simf.add_argument("--sd_s", type=float, default=0.0)
    simf.add_argument("--mu_log_lambda0", type=float, default=None)
    simf.add_argument("--sigma_log_lambda0", type=float, default=None)
    simf.add_argument("--mu_log_s", type=float, default=None)
    simf.add_argument("--sigma_log_s", type=float, default=None)

    simf.add_argument("--drop_start_max", type=int, default=2)
    simf.add_argument("--drop_end_max", type=int, default=2)
    simf.add_argument("--max_censor_extra", type=float, default=None)

    simf.add_argument("--out_sim_npz", type=str, required=True)
    simf.add_argument("--out_filtered_npz", type=str, required=True)

    plot = sub.add_parser("plot_before_after", help="Make the 2x2 plot from (raw.npz, filtered.npz).")
    plot.add_argument("raw_npz", type=str)
    plot.add_argument("filtered_npz", type=str)
    plot.add_argument("--label", type=str, default="run")
    plot.add_argument("--T", type=float, default=None, help="Override T shown in the plot title")
    plot.add_argument("--out_png", type=str, required=True)
    plot.add_argument("--cmap", type=str, default="YlGnBu")
    plot.add_argument("--dpi", type=int, default=300)

    demo = sub.add_parser("demo", help="Run built-in scenarios and save the 2x2 plot.")
    demo.add_argument("--outdir", type=str, default="demo_plots_contact")
    demo.add_argument("--T", type=float, default=500.0)
    demo.add_argument("--n_cells", type=int, default=5000)
    demo.add_argument("--seed", type=int, default=990)
    demo.add_argument("--sim_mode", choices=["global", "individual"], default="global")
    demo.add_argument("--drop_start_max", type=int, default=2)
    demo.add_argument("--drop_end_max", type=int, default=2)
    demo.add_argument("--max_censor_extra", type=float, default=None)
    demo.add_argument("--cmap", type=str, default="YlGnBu")

    return p


def main(argv=None):
    args = _build_parser().parse_args(argv)

    if args.cmd == "simulate_and_filter":
        if args.mode == "homogeneous":
            raw = simulate_population(
                n_cells=args.n_cells,
                max_time=args.T,
                theta=(args.lambda0, args.n50, args.s),
                seed=args.seed,
                max_events=args.max_events,
                sim_mode=args.sim_mode,
            )
        else:
            raw = simulate_population(
                n_cells=args.n_cells,
                max_time=args.T,
                hierarchical=dict(
                    dist_mode=args.dist_mode,
                    mean_lambda0=args.mean_lambda0,
                    sd_lambda0=args.sd_lambda0,
                    mean_n50=args.mean_n50,
                    sd_n50=args.sd_n50,
                    mean_s=args.mean_s,
                    sd_s=args.sd_s,
                    mu_log_lambda0=args.mu_log_lambda0,
                    sigma_log_lambda0=args.sigma_log_lambda0,
                    mu_log_s=args.mu_log_s,
                    sigma_log_s=args.sigma_log_s,
                ),
                seed=args.seed,
                max_events=args.max_events,
                sim_mode=args.sim_mode,
            )

        filtered = filter_dataset_in_memory(
            times_list=raw["times_list"],
            T=float(raw["max_time"]),
            seed=args.seed,
            drop_start_max=args.drop_start_max,
            drop_end_max=args.drop_end_max,
            max_censor_extra=args.max_censor_extra,
        )

        sim_path = save_simulation_npz(raw, args.out_sim_npz)
        filt_path = save_filtered_npz(filtered, args.out_filtered_npz)

        kept_counts = _counts_from_times_list(filtered["times_list"])
        print("Saved simulation:", sim_path)
        print("Saved filtered:", filt_path)
        print("sim_mode:", raw["sim_mode"], "n_cells:", raw["n_cells"], "T:", raw["max_time"], "seed:", args.seed)
        print("contacts per cell (raw):  mean=", float(np.mean(raw["n_end"])), "std=", float(np.std(raw["n_end"])))
        print("contacts per cell (kept): mean=", float(np.mean(kept_counts)), "std=", float(np.std(kept_counts)))

    elif args.cmd == "plot_before_after":
        raw = load_sim_or_filtered_npz(args.raw_npz)
        filt = load_sim_or_filtered_npz(args.filtered_npz)
        T = float(args.T) if args.T is not None else float(raw.get("max_time", np.nan))
        plot_before_after_2x2(
            raw_groups=[raw],
            filt_groups=[filt],
            labels=[args.label],
            T=T,
            save_path=args.out_png,
            cmap_name=args.cmap,
            dpi=int(args.dpi),
        )
        print("Saved:", os.path.abspath(args.out_png))

    else:
        run_demo(
            outdir=args.outdir,
            T=args.T,
            n_cells=args.n_cells,
            seed=args.seed,
            sim_mode=args.sim_mode,
            drop_start_max=args.drop_start_max,
            drop_end_max=args.drop_end_max,
            max_censor_extra=args.max_censor_extra,
            cmap_name=args.cmap,
        )


if __name__ == "__main__":
    main()
