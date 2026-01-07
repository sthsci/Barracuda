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
    lambda0: Optional[float] = None,
    p0: Optional[float] = None,
    mean: Optional[float] = None,
    sd: Optional[float] = None,
    Dist_mode: DistMode = "gamma"
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_cells = int(n_cells)
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")

    mode_s = str(mode).strip().lower()
    dist_mode_s = str(Dist_mode).strip().lower()
    
    if mode_s not in {"homogeneous", "heterogeneous", "gamma", "lognormal", "truncnorm"}:
        raise ValueError(
            "mode must be one of: homogeneous, heterogeneous, gamma, lognormal, truncnorm"
        )
    
    if mode_s == "homogeneous":
        if lambda0 is None:
            raise ValueError("homogeneous requires lambda0")
        lam = np.full(n_cells, float(lambda0), dtype=float)
    
    else:
        # Heterogeneous families.
        if mode_s != "heterogeneous":
            dist_mode_s = mode_s

        if mean is None or sd is None:
            raise ValueError("heterogeneous sampling requires mean and sd")
        mean, sd = float(mean), float(sd)

        if dist_mode_s == "gamma":
            if sd == 0:
                lam = np.full(n_cells, mean, dtype=float)
            else:
                shape, rate = _gamma_shape_rate_from_mean_sd(mean, sd)
                lam = rng.gamma(shape=shape, scale=1.0 / rate, size=n_cells).astype(float)
        
        elif dist_mode_s == "lognormal":
            mu, sigma = _lognormal_mu_sigma_from_mean_sd(mean, sd)
            lam = rng.lognormal(mean=mu, sigma=sigma, size=n_cells).astype(float)
        
        elif dist_mode_s == "truncnorm":
            lam = _sample_truncnorm_positive(rng=rng, mean=mean, sd=sd, size=n_cells)
        
        else:
            raise ValueError("Dist_mode must be one of: gamma, lognormal, truncnorm")
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
    
    dt = np.diff(times) if times.size >= 2 else np.array([], dtype=float)
    
    n_contacts = int(times.size)
    
    return {
        "times": times,
        "dt": dt,
        "n_contacts": n_contacts,
    }


def simulate_Population(
    n_cells: int,
    T: float,
    *,
    rates: Optional[np.ndarray] = None,
    mode: Mode = "homogeneous",
    seed: Optional[int] = None,
    lambda0: Optional[float] = None,
    mean: Optional[float] = None,
    sd: Optional[float] = None,
    Dist_mode: DistMode = "gamma",
) -> dict[str, Any]:
    n_cells = int(n_cells)
    T = float(T)
    if n_cells <= 0:
        raise ValueError("n_cells must be positive")
    if T < 0:
        raise ValueError("T must be >= 0")
    
    rng = np.random.default_rng(seed)
    
    # Sample or validate rates
    if rates is None:
        rates = sample_lambda(
            n_cells=n_cells,
            mode=mode,
            seed=seed,
            lambda0=lambda0,
            mean=mean,
            sd=sd,
            Dist_mode=Dist_mode,
        )
    rates = np.asarray(rates, dtype=float)
    if rates.shape != (n_cells,):
        raise ValueError(f"rates must have shape ({n_cells},), got {rates.shape}")
    
    # Generate independent seeds for each cell
    cell_seeds = rng.integers(0, 2**32 - 1, size=n_cells, dtype=np.uint32)
    
    # Simulate each cell
    times_list: list[np.ndarray] = []
    dt_list: list[np.ndarray] = []
    n_contacts = np.zeros(n_cells, dtype=int)
    
    for i in range(n_cells):
        cell_rng = np.random.default_rng(int(cell_seeds[i]))
        times = process_times(rate=float(rates[i]), T=T, rng=cell_rng)
        dt = np.diff(times) if times.size >= 2 else np.array([], dtype=float)
        
        times_list.append(times)
        dt_list.append(dt)
        n_contacts[i] = int(times.size)
    
    return {
        "n_cells": n_cells,
        "max_time": T,
        "rates": rates,
        "times_list": np.asarray(times_list, dtype=object),
        "dt_list": np.asarray(dt_list, dtype=object),
        "n_contacts": n_contacts,
    }





def save_npz(path: str | Path, **payload: Any) -> str:
    path = os.path.abspath(str(path))
    np.savez_compressed(path, **payload)
    return path


def load_npz(path: str | Path) -> Dict[str, Any]:
    data = np.load(str(path), allow_pickle=True)
    out = {k: data[k] for k in data.files}
    if "times_list" in out:
        out["times_list"] = np.asarray(out["times_list"], dtype=object)
    if "dt_list" in out:
        out["dt_list"] = np.asarray(out["dt_list"], dtype=object)
    return out


def _parse_weights_string(s: str) -> Dict[float, float]:
    s = s.strip()
    if not s:
        raise ValueError("empty weights string")
    items = [x.strip() for x in s.split(",") if x.strip()]
    weights: Dict[float, float] = {}
    for it in items:
        if ":" not in it:
            raise ValueError(f"bad weight item '{it}', expected 'rate:prop'")
        a, b = it.split(":", 1)
        rate = float(a.strip())
        prop = float(b.strip())
        if rate < 0 or not np.isfinite(rate):
            raise ValueError("rates in weights must be finite and >= 0")
        if prop < 0 or not np.isfinite(prop):
            raise ValueError("proportions in weights must be finite and >= 0")
        weights[rate] = weights.get(rate, 0.0) + prop
    if not weights:
        raise ValueError("no valid weights parsed")
    return weights


def _parse_weight_items(items: Optional[list[str]]) -> Optional[Dict[float, float]]:
    if not items:
        return None
    weights: Dict[float, float] = {}
    for it in items:
        it = it.strip()
        if ":" not in it:
            raise ValueError(f"bad --weight '{it}', expected 'rate:prop'")
        a, b = it.split(":", 1)
        rate = float(a.strip())
        prop = float(b.strip())
        if rate < 0 or not np.isfinite(rate):
            raise ValueError("rates in weights must be finite and >= 0")
        if prop < 0 or not np.isfinite(prop):
            raise ValueError("proportions in weights must be finite and >= 0")
        weights[rate] = weights.get(rate, 0.0) + prop
    return weights


def _apply_transparent_axes(ax):
    ax.set_facecolor("none")


def _legend_white(ax, loc: str = "best"):
    leg = ax.legend(loc=loc, frameon=True, fontsize=10, edgecolor="black")
    if leg is not None:
        fr = leg.get_frame()
        fr.set_facecolor("white")
        fr.set_alpha(1.0)


def _pooled_dt(dt_list: Any) -> np.ndarray:
    parts: list[np.ndarray] = []
    if dt_list is None:
        return np.array([], dtype=float)
    for dt in dt_list if isinstance(dt_list, (list, tuple, np.ndarray)) else []:
        arr = np.asarray(dt, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size:
            parts.append(arr)
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def tune_log_xticks(ax, num_major: int = 8, minor_subs: tuple[int, ...] = (2, 5)):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.00, numticks=num_major))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.{2}f}"))
    # ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=minor_subs, numticks=12))
    # ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="major", labelsize=10, length=6)
    # ax.tick_params(axis="x", which="minor", length=3)


def plot_default_histograms(
    sim: Dict[str, Any],
    save_path: str | Path,
    *,
    dpi: int = 300,
    cmap_name: str = "inferno",
) -> None:
    T_show = float(sim.get("max_time", np.nan))
    counts = np.asarray(sim.get("n_end", []), dtype=int)
    dt = _pooled_dt(sim.get("dt_list", []))

    cmap = plt.get_cmap(cmap_name)
    col = cmap(0.75)

    max_n = int(np.max(counts)) if counts.size else 0
    count_bins = np.arange(-0.5, max_n + 1.5, 1.0)

    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size:
        lo = max(float(np.min(dt)), 1e-8)
        hi = float(np.max(dt))
        dt_bins = np.logspace(np.log10(lo), np.log10(hi), num=28)
    else:
        dt_bins = "auto"

    fig, axs = plt.subplots(1, 2, figsize=(12.4, 4.2), dpi=dpi)
    fig.patch.set_alpha(0.0)
    for ax in axs:
        _apply_transparent_axes(ax)

    axL, axR = axs[0], axs[1]

    axL.hist(
        counts,
        bins=count_bins,
        density=True,
        histtype="stepfilled",
        alpha=0.55,
        edgecolor=col,
        facecolor=col,
        lw=1.5,
        label="Counts",
    )
    axL.set_title(f"Contact number  (T={T_show:g})")
    axL.set_xlabel("Contacts by T")
    axL.set_ylabel("Density")
    axL.xaxis.set_major_locator(MultipleLocator(1))
    axL.grid(True, alpha=0.25)
    axL.set_xlim(left=0)
    _legend_white(axL, )

    if dt.size:
        axR.hist(
            dt,
            bins=dt_bins,
            density=True,
            histtype="stepfilled",
            alpha=0.55,
            edgecolor=col,
            facecolor=col,
            lw=1.5,
            label="Δt between contacts",
        )
    axR.set_title("Inter-contact Δt")
    axR.set_xlabel("Δt")
    axR.set_ylabel("Density")
    tune_log_xticks(axR)
    axR.grid(True, which="both", alpha=0.25)
    _legend_white(axR)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Constant-rate contact model: simulate + visualise (raw only).")
    p.add_argument("--n_cells", type=int, default=500)
    p.add_argument("--T", type=float, default=180.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["homogeneous", "discrete", "gamma", "lognormal", "halfnormal"], default="homogeneous")

    p.add_argument("--lambda0", type=float, default=1.0)

    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--weight", action="append", default=None)

    p.add_argument("--mean", type=float, default=None)
    p.add_argument("--sd", type=float, default=None)

    p.add_argument("--out_npz", type=str, default="sim_raw.npz")
    p.add_argument("--out_png", type=str, default="sim_raw_hist.png")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--cmap", type=str, default="YlGnBu")
    return p


def _resolve_mode_args(args) -> tuple[DistMode, Optional[Dict[float, float]], Optional[float], Optional[float], Optional[float]]:
    mode: DistMode = str(args.mode).lower()  # type: ignore[assignment]

    weights = _parse_weight_items(args.weight)
    if weights is None and args.weights is not None:
        weights = _parse_weights_string(args.weights)

    if mode == "discrete" and not weights:
        raise ValueError("mode=discrete requires --weights 'rate:prop,...' or repeated --weight rate:prop")

    if mode in {"gamma", "lognormal", "halfnormal"} and (args.mean is None or args.sd is None):
        raise ValueError(f"mode={mode} requires --mean and --sd")

    lam0 = float(args.lambda0) if mode == "homogeneous" else None
    mean = float(args.mean) if args.mean is not None else None
    sd = float(args.sd) if args.sd is not None else None
    return mode, weights, lam0, mean, sd

def plot_three_conditions_overlay(
    sims: list[Dict[str, Any]],
    labels: list[str],
    save_path: str | Path,
    *,
    dpi: int = 300,
    cmap_name: str = "YlGnBu",
) -> None:
    if len(sims) != len(labels):
        raise ValueError("sims and labels must have the same length")
    if len(sims) < 1:
        raise ValueError("Need at least one sim")

    counts_list = [np.asarray(sim.get("n_end", []), dtype=int) for sim in sims]
    dt_list = [_pooled_dt(sim.get("dt_list", [])) for sim in sims]

    all_counts = np.concatenate([c for c in counts_list if c.size]) if any(c.size for c in counts_list) else np.array([], dtype=int)
    max_n = int(np.max(all_counts)) if all_counts.size else 0
    count_bins = np.arange(-0.5, max_n + 1.5, 1.0)

    all_dt = np.concatenate([d[(np.isfinite(d)) & (d > 0)] for d in dt_list if d.size]) if any(d.size for d in dt_list) else np.array([], dtype=float)
    if all_dt.size:
        lo = max(float(np.min(all_dt)), 1e-8)
        hi = float(np.max(all_dt))
        dt_bins = np.logspace(np.log10(lo), np.log10(hi), num=32)
    else:
        lo, hi, dt_bins = 1e-8, 1.0, "auto"

    cmap = plt.get_cmap(cmap_name)
    colors = [cmap(x) for x in np.linspace(0.35, 0.95, num=len(sims))]

    fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.4), dpi=dpi)
    fig.patch.set_alpha(0.0)
    for ax in axs:
        _apply_transparent_axes(ax)

    axL, axR = axs[0], axs[1]

    for counts, lab, col in zip(counts_list, labels, colors):
        axL.hist(
            counts,
            bins=count_bins,
            density=True,
            histtype="stepfilled",
            alpha=0.40,
            edgecolor=col,
            facecolor=col,
            lw=1.5,
            label=lab,
        )

    T_show = float(sims[0].get("max_time", np.nan))
    axL.set_title(f"Contact number  (T={T_show:g})")
    axL.set_xlabel("Contacts by T")
    axL.set_ylabel("Density")
    axL.xaxis.set_major_locator(MultipleLocator(1))
    axL.grid(True, alpha=0.25)
    axL.set_xlim(left=0)
    _legend_white(axL)

    for dt, lab, col in zip(dt_list, labels, colors):
        dt = np.asarray(dt, dtype=float)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size == 0:
            continue
        axR.hist(
            dt,
            bins=dt_bins,
            density=True,
            histtype="stepfilled",
            alpha=0.40,
            edgecolor=col,
            facecolor=col,
            lw=1.5,
            label=lab,
        )

    axR.set_title("Inter-contact Δt")
    axR.set_xlabel("Δt")
    axR.set_ylabel("Density")
    tune_log_xticks(axR)
    axR.set_xlim(lo, hi)
    axR.grid(True, which="both", alpha=0.25)
    _legend_white(axR)

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)

def _run_default() -> None:
    outdir = Path("demo_contact_simple")
    outdir.mkdir(parents=True, exist_ok=True)

    sims: list[Dict[str, Any]] = []

    sim1 = simulate_population_simple(
        n_cells=5000,
        T=100.0,
        mode="homogeneous",
        seed=0,
        lambda0=0.05,
    )
    sims.append(sim1)

    sim2 = simulate_population_simple(
        n_cells=5000,
        T=100.0,
        mode="homogeneous",
        seed=1,
        lambda0=0.10,
    )
    sims.append(sim2)

    sim3 = simulate_population_simple(
        n_cells=5000,
        T=100.0,
        mode="discrete",
        seed=2,
        weights={0.0: 0.3, 0.10: 0.7},
    )
    sims.append(sim3)

    labels = [
        "hom λ=0.05",
        "hom λ=0.10",
        "disc {0:0.3, 0.10:0.7}",
    ]

    save_png = outdir / "sim_three_conditions_hist.png"
    plot_three_conditions_overlay(
        sims,
        labels,
        save_png,
        dpi=300,
        cmap_name="YlGnBu",
    )

    print("Default run complete:")
    print("  png:", os.path.abspath(str(save_png)))



def main(argv: Optional[Iterable[str]] = None) -> None:
    if argv is None and len(sys.argv) == 1:
        _run_default()
        return

    args = _build_parser().parse_args(argv)
    mode, weights, lam0, mean, sd = _resolve_mode_args(args)

    out = simulate_population_simple(
        n_cells=args.n_cells,
        T=args.T,
        mode=mode,
        seed=args.seed,
        lambda0=lam0,
        weights=weights,
        mean=mean,
        sd=sd,
    )

    npz_path = save_npz(
        args.out_npz,
        n_cells=out["n_cells"],
        max_time=out["max_time"],
        rates=out["rates"],
        times_list=out["times_list"],
        dt_list=out["dt_list"],
        n_end=out["n_end"],
        model=out["model"],
        heterogeneity_mode=out["heterogeneity_mode"],
    )
    plot_default_histograms(out, args.out_png, dpi=int(args.dpi), cmap_name=str(args.cmap))

    print("Saved simulation:", os.path.abspath(npz_path))
    print("Saved plot:", os.path.abspath(str(args.out_png)))


if __name__ == "__main__":
    main()