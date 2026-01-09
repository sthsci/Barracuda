#!/usr/bin/env python3
from __future__ import annotations

import os

os.environ.setdefault("PYTENSOR_FLAGS", "optimizer_excluding=fusion")

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter, FuncFormatter
from matplotlib.patches import Patch
import arviz as az


plt.rcParams.update(
    {
        "font.family": ["Monaco", "DejaVu Sans Mono", "monospace"],
        "mathtext.fontset": "stix",
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    }
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAP_ROOT = Path(__file__).resolve().parents[1]

import sys  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CHAP_ROOT))

from Simulator.model_contact_I import *  # noqa: F401,F403,E402
from Inference import contact_inference_simple as cis  # type: ignore  # noqa: E402


def _legend_white(ax, loc: str = "best"):
    leg = ax.legend(loc=loc, frameon=True, fontsize=10, edgecolor="black")
    if leg is not None:
        fr = leg.get_frame()
        fr.set_facecolor("white")
        fr.set_alpha(1.0)

def _legend_patches(lambda_list, colors, *, alpha=0.25, lw=2.0):
    return [
        Patch(
            facecolor=col,
            edgecolor=col,
            linewidth=lw,
            alpha=alpha,
            label=fr"$\lambda={float(lam):g}$",
        )
        for lam, col in zip(lambda_list, colors)
    ]



def _apply_transparent_axes(ax):
    ax.set_facecolor("none")


def _pooled_dt(dt_list: Any) -> np.ndarray:
    if dt_list is None:
        return np.array([], dtype=float)
    parts: List[np.ndarray] = []
    for dt in dt_list if isinstance(dt_list, (list, tuple, np.ndarray)) else []:
        arr = np.asarray(dt, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size:
            parts.append(arr)
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def _counts_from_sim(sim: Dict[str, Any]) -> np.ndarray:
    if "n_end" in sim:
        return np.asarray(sim["n_end"], dtype=int)
    times_list = sim.get("times_list", [])
    return np.array([int(np.asarray(t, dtype=float).size) for t in times_list], dtype=int)


def _get_rgba_colors(cmap_name: str, n: int) -> List[Tuple[float, float, float, float]]:
    cmap = plt.colormaps.get_cmap(str(cmap_name))
    vals = np.linspace(0.30, 0.90, int(n), dtype=float)
    arr = np.asarray(cmap(vals), dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        arr = np.tile(np.array([0.2, 0.2, 0.2, 1.0], dtype=float), (int(n), 1))
    return [tuple(map(float, row)) for row in arr]


def tune_log_xticks(ax, *, num_major: int = 8, minor_subs: Tuple[int, ...] = (2, 5), decimals: int = 2):
    def fmt(x, _):
        x = float(x)
        axx = abs(x)
        if axx == 0:
            return "0"
        if axx < 1e-2 or axx >= 1e3:
            return f"{x:.{int(decimals)}e}"
        return f"{x:.{int(decimals)}f}"

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=int(num_major)))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt))
    # ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=minor_subs, numticks=12))
    # ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="major", labelsize=10, length=6)
    ax.tick_params(axis="x", which="minor", length=3)


# def _gamma_posterior_lambda_samples(
#     counts: np.ndarray,
#     T: float,
#     *,
#     prior_shape: float = 1.0,
#     prior_rate: float = 1.0,
#     draws: int = 4000,
#     seed: Optional[int] = None,
# ) -> np.ndarray:
#     counts = np.asarray(counts, dtype=int)
#     T = float(T)
#     n = int(counts.size)
#     k = int(np.sum(counts)) if n else 0
#     a_post = float(prior_shape + k)
#     b_post = float(prior_rate + n * T)
#     rng = np.random.default_rng(seed)
#     return rng.gamma(shape=a_post, scale=1.0 / b_post, size=int(draws)).astype(float)


# def _to_idata_mu_lambda(samples: np.ndarray) -> az.InferenceData:
#     samples = np.asarray(samples, dtype=float)
#     samples = samples[np.isfinite(samples)]
#     if samples.size == 0:
#         samples = np.array([np.nan], dtype=float)
#     return az.from_dict(posterior={"mu_lambda": samples[None, :]})


def _extract_posterior_1d(idata: az.InferenceData, var: str) -> np.ndarray:
    post = idata.posterior
    if var not in post:
        return np.array([], dtype=float)
    vals = post[var].stack(sample=("chain", "draw")).values.ravel()
    return vals[np.isfinite(vals)]


def plot(
    *,
    sims: Sequence[Dict[str, Any]],
    lambda_list: Sequence[float],
    idatas: Sequence[Tuple[str, az.InferenceData]],
    save_path: str | Path,
    cmap_name: str = "inferno",
    dpi: int = 400,
    hdi_prob: float = 0.95,
    posterior_var: str = "mu_lambda",
    posterior_plot_samples: Optional[int] = 3000,
    tick_decimals_log: int = 2,
    dt_xlim: Optional[Tuple[float, float]] = (1e-3, 1e2),
    lambda_xlim: Optional[Tuple[float, float]] = None,
    mode: str = "dt",
) -> None:
    from matplotlib.patches import Patch

    def _legend_patches(alpha: float = 0.35, lw: float = 2.0):
        return [
            Patch(
                facecolor=col,
                edgecolor=col,
                linewidth=lw,
                alpha=alpha,
                label=fr"$\lambda={float(lam):g}$",
            )
            for lam, col in zip(lambda_list, colors)
        ]

    if len(sims) != len(lambda_list):
        raise ValueError("sims and lambda_list must have the same length")
    if len(idatas) != len(lambda_list):
        raise ValueError("idatas and lambda_list must have the same length (one idata per condition)")

    colors = _get_rgba_colors(str(cmap_name), len(lambda_list))

    counts_all = [_counts_from_sim(sim) for sim in sims]
    all_counts_cat = np.concatenate(counts_all) if counts_all else np.array([], dtype=int)
    max_n = int(np.max(all_counts_cat)) if all_counts_cat.size else 0
    count_bins = np.arange(-0.5, max_n + 1.5, 1.0)

    use_counts = mode in ("counts", "both")
    use_dt = mode in ("dt", "both")

    dt_all = []
    dt_bins = "auto"
    if use_dt:
        dt_all = [_pooled_dt(sim.get("dt_list", [])) for sim in sims]
        pooled_dt = np.concatenate([d for d in dt_all if np.asarray(d).size]) if dt_all else np.array([], dtype=float)
        pooled_dt = np.asarray(pooled_dt, dtype=float)
        pooled_dt = pooled_dt[np.isfinite(pooled_dt) & (pooled_dt > 0)]
        if pooled_dt.size:
            lo = max(float(np.min(pooled_dt)), 1e-12)
            hi = float(np.max(pooled_dt))
            dt_bins = np.logspace(np.log10(lo), np.log10(hi), num=50)

    if mode == "both":
        fig = plt.figure(figsize=(18, 5.2), dpi=int(dpi))
        gs = gridspec.GridSpec(1, 3, width_ratios=[1.0, 1.0, 1.05], wspace=0.28)
        counts_ax = fig.add_subplot(gs[0, 0])
        dt_ax = fig.add_subplot(gs[0, 1])
        post_ax = fig.add_subplot(gs[0, 2])
    elif mode == "counts":
        fig = plt.figure(figsize=(12.0, 5.0), dpi=int(dpi))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.05], wspace=0.25)
        counts_ax = fig.add_subplot(gs[0, 0])
        dt_ax = None
        post_ax = fig.add_subplot(gs[0, 1])
    elif mode == "dt":
        fig = plt.figure(figsize=(12.0, 5.0), dpi=int(dpi))
        gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.05], wspace=0.25)
        counts_ax = None
        dt_ax = fig.add_subplot(gs[0, 0])
        post_ax = fig.add_subplot(gs[0, 1])
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if counts_ax is not None:
        for counts, lam, col in zip(counts_all, lambda_list, colors):
            counts_ax.hist(
                counts,
                bins=count_bins,
                density=True,
                histtype="stepfilled",
                alpha=0.18,
                color=col,
                linewidth=1.8,
                label=fr"$\\lambda={float(lam):g}$",
            )
            counts_ax.hist(
                counts,
                bins=count_bins,
                density=True,
                histtype="step",
                alpha=1.0,
                color=col,
                linewidth=1.8,
            )

        counts_ax.set_title("Contact number", fontweight="bold")
        counts_ax.set_xlabel("Contacts per killer cell")
        counts_ax.set_ylabel("Density")
        counts_ax.xaxis.set_major_locator(MultipleLocator(1))
        counts_ax.grid(True, alpha=0.25)
        counts_ax.set_xlim(left=0)

        leg = counts_ax.legend(handles=_legend_patches(), loc="best", frameon=True, fontsize=10, edgecolor="black")
        if leg is not None:
            fr = leg.get_frame()
            fr.set_facecolor("white")
            fr.set_alpha(1.0)

    if dt_ax is not None:
        for dt, lam, col in zip(dt_all, lambda_list, colors):
            dt = np.asarray(dt, dtype=float)
            dt = dt[np.isfinite(dt) & (dt > 0)]
            if dt.size == 0:
                continue
            dt_ax.hist(
                dt,
                bins=dt_bins,
                density=True,
                histtype="stepfilled",
                alpha=0.18,
                color=col,
                linewidth=1.8,
                label=fr"$\\lambda={float(lam):g}$",
            )
            dt_ax.hist(
                dt,
                bins=dt_bins,
                density=True,
                histtype="step",
                alpha=1.0,
                color=col,
                linewidth=1.8,
            )

        dt_ax.set_title("Inter-contact Δt", fontweight="bold")
        dt_ax.set_xlabel("log(Δt)")
        dt_ax.set_ylabel("Density")
        tune_log_xticks(dt_ax, decimals=int(tick_decimals_log))
        dt_ax.grid(True, which="both", alpha=0.25)
        if dt_xlim is not None:
            dt_ax.set_xlim(float(dt_xlim[0]), float(dt_xlim[1]))

        leg = dt_ax.legend(handles=_legend_patches(), loc="best", frameon=True, fontsize=10, edgecolor="black")
        if leg is not None:
            fr = leg.get_frame()
            fr.set_facecolor("white")
            fr.set_alpha(1.0)

    rng = np.random.default_rng(0)
    all_post_for_xlim: List[np.ndarray] = []

    for (label, idata), lam_true, col in zip(idatas, lambda_list, colors):
        vals_full = _extract_posterior_1d(idata, str(posterior_var))
        if vals_full.size == 0:
            continue

        if (posterior_plot_samples is not None) and (vals_full.size > int(posterior_plot_samples)):
            idx = rng.choice(vals_full.size, int(posterior_plot_samples), replace=False)
            vals_plot = vals_full[idx]
        else:
            vals_plot = vals_full

        all_post_for_xlim.append(vals_full)

        post_ax.hist(
            vals_plot,
            bins=30,
            density=True,
            histtype="stepfilled",
            alpha=0.18,
            color=col,
            linewidth=1.8,
            label=label,
        )
        post_ax.hist(
            vals_plot,
            bins=30,
            density=True,
            histtype="step",
            alpha=1.0,
            color=col,
            linewidth=1.8,
        )

        lo, hi = az.hdi(vals_full, hdi_prob=float(hdi_prob))
        post_ax.axvspan(float(lo), float(hi), color=col, alpha=0.10, linewidth=0)
        post_ax.axvline(float(lam_true), color=col, linestyle="-", linewidth=2.0)

    post_ax.set_title("Posterior of λ", fontweight="bold")
    post_ax.set_xlabel(r"$\lambda$")
    post_ax.set_ylabel("Density")
    post_ax.xaxis.set_major_locator(MultipleLocator(0.01)) 
    post_ax.grid(True, alpha=0.25)

    leg = post_ax.legend(handles=_legend_patches(), loc="best", frameon=True, fontsize=10, edgecolor="black")
    if leg is not None:
        fr = leg.get_frame()
        fr.set_facecolor("white")
        fr.set_alpha(1.0)

    if lambda_xlim is not None:
        post_ax.set_xlim(float(lambda_xlim[0]), float(lambda_xlim[1]))
    else:
        if all_post_for_xlim:
            all_post = np.concatenate(all_post_for_xlim)
            if all_post.size:
                lo = max(0.0, float(np.quantile(all_post, 0.001)))
                hi = float(np.quantile(all_post, 0.999))
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    post_ax.set_xlim(lo, hi)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path.with_suffix(".png"), dpi=int(dpi), bbox_inches="tight", transparent=True)
    # fig.savefig(save_path.with_suffix(".pdf"), dpi=int(dpi), bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("Saved:", str(save_path.with_suffix(".png")))
    # print("Saved:", str(save_path.with_suffix(".pdf")))



def main(argv: Optional[Iterable[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Simulate 3 homogeneous Poisson-contact conditions and plot + infer λ.")
    p.add_argument("--n_cells", type=int, default=200)
    p.add_argument("--T", type=float, default=40.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--lambdas", type=str, default="0.02,0.04,0.06")
    # p.add_argument("--out", type=str, default="results_counts/cell200")
    p.add_argument("--cmap", type=str, default="inferno")
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--hdi", type=float, default=0.95)
    p.add_argument("--posterior_var", type=str, default="mu_lambda")
    p.add_argument("--posterior_samples", type=int, default=50000)
    p.add_argument("--posterior_tune", type=int, default=3000)
    p.add_argument("--posterior_chains", type=int, default=8)
    p.add_argument(
        "--cores",
        type=int,
        default=0,
        help="Number of CPU processes for PyMC sampling (0 = all available cores; effective use is min(chains, cores))",
    )
    p.add_argument("--posterior_target", type=float, default=0.98)
    p.add_argument("--posterior_plot_samples", type=int, default=5000)
    p.add_argument("--tick_decimals_log", type=int, default=2)
    p.add_argument("--dt_xlim", type=str, default="1e-3,1e2")
    p.add_argument("--lambda_xlim", type=str, default="")
    p.add_argument(
        "--infer_mode",
        type=str,
        choices=["counts", "dt", "both"],
        default="dt",
        help="Use contact counts, inter-contact durations, or both for inference",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    lambda_list = [float(x.strip()) for x in str(args.lambdas).split(",") if x.strip()]
    if len(lambda_list) != 3:
        raise ValueError("--lambdas must contain exactly 3 comma-separated values (e.g. 0.03,0.05,0.08)")

    dt_xlim = None
    if str(args.dt_xlim).strip():
        a, b = [float(x.strip()) for x in str(args.dt_xlim).split(",")]
        dt_xlim = (a, b)

    lambda_xlim = None
    if str(args.lambda_xlim).strip():
        a, b = [float(x.strip()) for x in str(args.lambda_xlim).split(",")]
        lambda_xlim = (a, b)

    sims: List[Dict[str, Any]] = []
    idatas: List[Tuple[str, az.InferenceData]] = []

    for j, lam in enumerate(lambda_list):
        sim = simulate_population_simple(  # noqa: F405
            n_cells=int(args.n_cells),
            T=float(args.T),
            mode="homogeneous",
            seed=int(args.seed) + 2026 if args.seed is not None else None,
            lambda0=float(lam),
        )
        sims.append(sim)

        counts = _counts_from_sim(sim)
        label = fr"$\lambda={float(lam):g}$"
        obs_time = float(sim.get("max_time", args.T))
        dt_flat = _pooled_dt(sim.get("dt_list", []))

        cores = None if int(args.cores) == 0 else int(args.cores)

        if args.infer_mode == "counts":
            idata = cis.inference_counts(
                kills_per_cell=counts,
                obs_time=obs_time,
                draws=int(args.posterior_samples),
                tune=int(args.posterior_tune),
                chains=int(args.posterior_chains),
                cores=cores,
                target_accept=float(args.posterior_target),
                log10_bounds=(-3, 0)
            )
        elif args.infer_mode == "dt":
            idata = cis.inference_durations(
                durations=dt_flat,
                draws=int(args.posterior_samples),
                tune=int(args.posterior_tune),
                chains=int(args.posterior_chains),
                cores=cores,
                target_accept=float(args.posterior_target),
                log10_bounds=(-3, 0)
            )
        else:
            idata = cis.inference_both(
                kills_per_cell=counts,
                durations=dt_flat,
                obs_time=obs_time,
                draws=int(args.posterior_samples),
                tune=int(args.posterior_tune),
                chains=int(args.posterior_chains),
                cores=cores,
                target_accept=float(args.posterior_target),
                log10_bounds=(-3, 0)
            )

        idatas.append((label, idata))

    # out_base = Path(str(args.out))
    out_base = Path(f"results_{args.infer_mode}/{args.n_cells}__{args.T}")
    plot(
        sims=sims,
        lambda_list=lambda_list,
        idatas=idatas,
        save_path=out_base,
        cmap_name=str(args.cmap),
        dpi=int(args.dpi),
        hdi_prob=float(args.hdi),
        posterior_var=str(args.posterior_var),
        posterior_plot_samples=int(args.posterior_plot_samples) if args.posterior_plot_samples is not None else None,
        tick_decimals_log=int(args.tick_decimals_log),
        dt_xlim=dt_xlim,
        lambda_xlim=lambda_xlim,
        mode=str(args.infer_mode),
    )


if __name__ == "__main__":
    main()
