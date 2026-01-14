#!/usr/bin/env python3
from __future__ import annotations

import os

os.environ.setdefault("PYTENSOR_FLAGS", "optimizer_excluding=fusion")

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
from matplotlib import gridspec
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHAP_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CHAP_ROOT))

from Simulator import model_contact_I as model  # noqa: E402
from Inference import contact_inference as cih  # type: ignore  # noqa: E402


class scenario:
    def __init__(
        self,
        name: str,
        mode: str,
        n_cell: int,
        T: float,
        obs_mode: str,
        T_sd: Optional[float] = None,
        mu_lambda: float = 0.05,
        sd_lambda: Optional[float] = None,
        p0_lambda: Optional[float] = None,
        dis_mode: Optional[str] = None,
    ):
        self.name = str(name)
        self.mode = str(mode)
        self.n_cell = int(n_cell)
        self.T = float(T)
        self.T_sd = None if T_sd is None else float(T_sd)
        self.obs_mode = str(obs_mode)
        self.mu_lambda = float(mu_lambda) if mu_lambda is not None else None
        self.sd_lambda = None if sd_lambda is None else float(sd_lambda)
        self.p0_lambda = None if p0_lambda is None else float(p0_lambda)
        self.dis_mode = None if dis_mode is None else str(dis_mode)
        self.results_dir: Dict[str, Any] = {}
        self.auto_convert()

    def auto_convert(self) -> None:
        if self.mode not in ("homogeneous", "heterogeneous"):
            raise ValueError("mode must be 'homogeneous' or 'heterogeneous'")
        if self.obs_mode not in ("Complete", "Truncated"):
            raise ValueError("obs_mode must be 'Complete' or 'Truncated'")
        if self.n_cell <= 0:
            raise ValueError("n_cell must be positive")
        if self.T <= 0:
            raise ValueError("T must be positive")
        if self.mu_lambda is None:
            raise ValueError("mu_lambda must be provided")

        if self.mode == "homogeneous":
            self.sd_lambda = None
            self.p0_lambda = None
            self.dis_mode = None
        else:
            if self.sd_lambda is None:
                raise ValueError("heterogeneous mode requires sd_lambda (use 0.0 if desired)")
            if self.p0_lambda is None:
                raise ValueError("heterogeneous mode requires p0_lambda (use 0.0 if desired)")
            if float(self.sd_lambda) == 0.0:
                self.dis_mode = None
            else:
                if self.dis_mode is None:
                    raise ValueError("heterogeneous mode with sd_lambda > 0 requires dis_mode")

        if self.obs_mode == "Truncated":
            if self.T_sd is None or self.T_sd <= 0:
                raise ValueError("Truncated obs_mode requires T_sd > 0")

    def take_results(self, results_dir: Dict[str, Any]) -> None:
        self.results_dir = dict(results_dir)


@dataclass(frozen=True)
class GroundTruth:
    mu_lambda: Optional[float] = None
    sigma_lambda: float = 0.0
    p_zero: float = 0.0


MODEL_PARAM_MAP: Dict[str, Dict[str, str]] = {
    "homo": {"mu_lambda": "lambda"},
    "Z2P": {"mu_lambda": "lambda", "p_zero": "p_zero"},
    "Dis2P": {"mu_lambda": "mu_lambda", "sigma_lambda": "sigma_lambda"},
    "hetero3": {"mu_lambda": "mu_lambda", "sigma_lambda": "sigma_lambda", "p_zero": "p_zero"},
}


def scenario_ground_truth(sc: scenario) -> GroundTruth:
    mu = None if sc.mu_lambda is None else float(sc.mu_lambda)
    sigma = 0.0 if sc.sd_lambda is None else float(sc.sd_lambda)
    p0 = 0.0 if sc.p0_lambda is None else float(sc.p0_lambda)
    return GroundTruth(mu_lambda=mu, sigma_lambda=sigma, p_zero=p0)


def ground_truth_for_model(scenarios: Sequence[scenario], model_name: str) -> Dict[str, Dict[str, float]]:
    mapping = MODEL_PARAM_MAP.get(model_name, {})
    out: Dict[str, Dict[str, float]] = {}
    for sc in scenarios:
        gt = scenario_ground_truth(sc)
        per: Dict[str, float] = {}
        for canonical_key, param_name in mapping.items():
            val = getattr(gt, canonical_key)
            if val is None:
                continue
            per[str(param_name)] = float(val)
        if per:
            out[sc.name] = per
    return out


def _legend_white(ax, loc: str = "best") -> None:
    leg = ax.legend(loc=loc, frameon=True, fontsize=10, edgecolor="black")
    if leg is not None:
        fr = leg.get_frame()
        fr.set_facecolor("white")
        fr.set_alpha(1.0)


def tune_log_xticks(ax, *, num_major: int = 8, decimals: int = 2) -> None:
    def fmt(x, _):
        if x <= 0 or not np.isfinite(x):
            return ""
        p = float(np.log10(x))
        if np.isclose(p, round(p), atol=1e-10):
            return fr"$10^{{{int(round(p))}}}$"
        return f"{x:.{int(decimals)}g}"

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=int(num_major)))
    ax.xaxis.set_major_formatter(FuncFormatter(fmt))
    ax.tick_params(axis="x", which="major", labelsize=10, length=6)
    ax.tick_params(axis="x", which="minor", length=3)


def _pooled_dt(dt_list: Any) -> np.ndarray:
    if dt_list is None:
        return np.array([], dtype=float)
    parts: List[np.ndarray] = []
    if isinstance(dt_list, (list, tuple, np.ndarray)):
        for dt in dt_list:
            arr = np.asarray(dt, dtype=float)
            arr = arr[np.isfinite(arr) & (arr > 0)]
            if arr.size:
                parts.append(arr)
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def plot_counts_and_dt(
    *,
    size: Sequence[float] = (14, 6),
    sims: Sequence[Dict[str, Any]],
    labels: Sequence[str],
    save_path: str | Path,
    cmap_name: str = "inferno",
    dpi: int = 350,
    counts_max: Optional[int] = None,
    dt_xlim: Optional[Tuple[float, float]] = (1e-3, 1e2),
) -> None:
    if len(sims) != len(labels):
        raise ValueError("sims and labels must have the same length")

    cmap = plt.colormaps.get_cmap(str(cmap_name))
    colors = cmap(np.linspace(0.3, 0.9, len(labels)))

    fig = plt.figure(figsize=(float(size[0]), float(size[1])), dpi=int(dpi))
    fig.patch.set_alpha(0.0)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 1.0], wspace=0.2)
    ax_counts = fig.add_subplot(gs[0, 0])
    ax_dt = fig.add_subplot(gs[0, 1])
    ax_counts.set_facecolor("none")
    ax_dt.set_facecolor("none")

    for col, sim, lab in zip(colors, sims, labels):
        counts = np.asarray(sim.get("n_contacts", []), dtype=int)
        if counts.size:
            bins = np.arange(int(counts.max()) + 2) - 0.5
            w = np.ones_like(counts, dtype=float) / float(counts.size)
            ax_counts.hist(counts, bins=bins, weights=w, histtype="step", linewidth=2.0, color=col, label=lab)
            ax_counts.hist(counts, bins=bins, weights=w, histtype="stepfilled", alpha=0.15, color=col)

    ax_counts.set_title("Contact number distribution", fontweight="bold")
    ax_counts.set_xlabel("Contacts per cell")
    ax_counts.set_ylabel("Frequency")
    if counts_max is not None:
        ax_counts.set_xlim(right=float(counts_max))
    ax_counts.set_xlim(left=-0.5)
    ax_counts.xaxis.set_major_locator(MultipleLocator(1))
    ax_counts.grid(True, alpha=0.30)
    _legend_white(ax_counts, loc="upper right")

    pooled_dt = np.concatenate([_pooled_dt(sim.get("dt_list", [])) for sim in sims]) if sims else np.array([], dtype=float)
    pooled_dt = pooled_dt[np.isfinite(pooled_dt) & (pooled_dt > 0)]
    if pooled_dt.size:
        lo = max(float(np.min(pooled_dt)), 1e-12)
        hi = float(np.max(pooled_dt))
        dt_bins = np.logspace(np.log10(lo), np.log10(hi), num=22)

        for col, sim, lab in zip(colors, sims, labels):
            dt = _pooled_dt(sim.get("dt_list", []))
            dt = dt[np.isfinite(dt) & (dt > 0)]
            if dt.size == 0:
                continue
            ax_dt.hist(dt, bins=dt_bins, density=True, histtype="stepfilled", alpha=0.15, color=col, linewidth=1.8)
            ax_dt.hist(dt, bins=dt_bins, density=True, histtype="step", alpha=1.0, color=col, linewidth=1.8, label=lab)

        if dt_xlim is not None:
            ax_dt.set_xlim(float(dt_xlim[0]), float(dt_xlim[1]))
        tune_log_xticks(ax_dt, num_major=7, decimals=2)
        ax_dt.set_title("Inter-contact Δt", fontweight="bold")
        ax_dt.set_xlabel("Δt")
        ax_dt.set_ylabel("Density")
        ax_dt.grid(True, which="both", alpha=0.30)

    _legend_white(ax_dt, loc="best")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path.with_suffix(".png"), dpi=int(dpi), bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("Saved:", str(save_path.with_suffix(".png")))


def plot_posteriors(
    idatas: Sequence[Tuple[str, az.InferenceData]],
    *,
    ground_truth: Optional[Dict[str, Dict[str, float]]] = None,
    parameters: Sequence[str],
    parameter_display: Optional[Dict[str, str]] = None,
    hdi_prob: float = 0.95,
    sample_size: int = 200000,
    save_path: str | Path = "posteriors",
    cmap_name: str = "inferno",
    font_scale: float = 0.7,
    diagonal_style: str = "hist",
    marginal_style: str = "circle",
    seed: Optional[int] = None,
    dpi: int = 300,
    xlims: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    sns.set_context("talk", font_scale=float(font_scale))
    cmap = plt.colormaps.get_cmap(str(cmap_name))
    colors = cmap(np.linspace(0.3, 0.9, len(idatas)))
    rng = np.random.default_rng(seed)

    params = [str(p) for p in parameters]
    if parameter_display is None:
        parameter_display = {p: p for p in params}

    def _posterior_vals(posterior, name: str) -> np.ndarray:
        if name in posterior:
            vals = posterior[name].stack(sample=("chain", "draw")).values.ravel()
            vals = np.asarray(vals, dtype=float)
            return vals[np.isfinite(vals)]
        return np.array([], dtype=float)

    label_to_df: Dict[str, pd.DataFrame] = {}
    for label, idata in idatas:
        posterior = idata.posterior
        df = pd.DataFrame()
        for p in params:
            vals = _posterior_vals(posterior, p)
            if vals.size == 0:
                continue
            if vals.size > int(sample_size):
                idx = rng.choice(vals.size, int(sample_size), replace=False)
                vals = vals[idx]
            df[p] = vals
        if not df.empty:
            df["label"] = label
            label_to_df[label] = df

    def _robust_limits(all_vals: np.ndarray) -> Optional[Tuple[float, float]]:
        all_vals = np.asarray(all_vals, dtype=float)
        all_vals = all_vals[np.isfinite(all_vals)]
        if all_vals.size == 0:
            return None
        lo = float(np.quantile(all_vals, 0.005))
        hi = float(np.quantile(all_vals, 0.995))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = float(np.min(all_vals))
            hi = float(np.max(all_vals))
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            return None
        pad = 0.03 * (hi - lo)
        return lo - pad, hi + pad

    param_xlims: Dict[str, Tuple[float, float]] = {}
    for p in params:
        cols: List[np.ndarray] = []
        for _lab, _df in label_to_df.items():
            if p in _df.columns:
                arr = np.asarray(_df[p].values, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size:
                    cols.append(arr)
        if cols:
            lims = _robust_limits(np.concatenate(cols))
            if lims is not None:
                param_xlims[p] = lims

    if xlims:
        for k, v in xlims.items():
            if k in params and v is not None:
                param_xlims[k] = (float(v[0]), float(v[1]))

    npar = len(params)
    fig = plt.figure(figsize=(5 * npar, 5 * npar), dpi=int(dpi))
    fig.patch.set_alpha(0.0)
    gs = gridspec.GridSpec(npar, npar, wspace=0.35, hspace=0.35)
    gaxes = np.empty((npar, npar), dtype=object)

    for irow, rowpar in enumerate(params):
        for icol, colpar in enumerate(params):
            ax = plt.subplot(gs[irow, icol])
            gaxes[irow, icol] = ax
            ax.set_facecolor("none")

            if icol > irow:
                ax.axis("off")
                continue

            for color, (label, df) in zip(colors, label_to_df.items()):
                if icol == irow:
                    if rowpar not in df.columns:
                        continue
                    vals = df[rowpar].dropna().values
                    if vals.size == 0:
                        continue

                    if diagonal_style == "kde":
                        sns.kdeplot(
                            vals,
                            ax=ax,
                            fill=True,
                            color=color,
                            alpha=0.2,
                            linewidth=1.5,
                            label=label if irow == 0 else None,
                        )
                    else:
                        sns.histplot(vals, bins=30, stat="density", kde=False, ax=ax, color=color, alpha=0.18, element="step", fill=True)
                        sns.histplot(
                            vals,
                            bins=30,
                            stat="density",
                            kde=False,
                            ax=ax,
                            color=color,
                            alpha=1.0,
                            element="step",
                            fill=False,
                            linewidth=1.8,
                            label=label if irow == 0 else None,
                        )

                    try:
                        lo, hi = az.hdi(vals, hdi_prob=float(hdi_prob))
                        ax.axvspan(float(lo), float(hi), color=color, alpha=0.1, linewidth=0)
                    except Exception:
                        pass

                    if ground_truth is not None and label in ground_truth and rowpar in ground_truth[label]:
                        ax.axvline(float(ground_truth[label][rowpar]), color=color, linestyle="-", linewidth=1.8)

                    ax.grid(alpha=0.2)
                else:
                    if (colpar not in df.columns) or (rowpar not in df.columns):
                        continue
                    if marginal_style == "circle":
                        sns.kdeplot(x=df[colpar], y=df[rowpar], ax=ax, fill=False, color=color, alpha=0.6, levels=7, linewidths=1.0)
                    else:
                        sns.histplot(x=df[colpar], y=df[rowpar], bins=60, pthresh=0.01, cmap=str(cmap_name), cbar=False, ax=ax)

            if icol != irow:
                ax.grid(alpha=0.3)
                if ground_truth is not None:
                    for color, (label, _df) in zip(colors, label_to_df.items()):
                        gt = ground_truth.get(label)
                        if gt and (colpar in gt) and (rowpar in gt):
                            ax.scatter(gt[colpar], gt[rowpar], marker="*", color=color, s=80, linewidths=2.0, zorder=1000)

            if icol == irow:
                ax.set_xlabel(parameter_display.get(rowpar, rowpar))
                ax.set_ylabel("Density")
            else:
                ax.set_xlabel(parameter_display.get(colpar, colpar))
                ax.set_ylabel(parameter_display.get(rowpar, rowpar))

            xpar = rowpar if icol == irow else colpar
            if xpar in param_xlims:
                ax.set_xlim(*param_xlims[xpar])

            ax.margins(0.05)

    handles, leg_labels = gaxes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels, loc="center", bbox_to_anchor=(0.85, 0.85), frameon=True, edgecolor="black", fontsize=10)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=int(dpi), bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("Saved joint posterior plot:", str(save_path.with_suffix(".png")))


def _estimate_logml(pm_model: pm.Model, idata: az.InferenceData) -> float:
    def _robust_mean_logml(obj: Any) -> float:
        try:
            arr = np.asarray(obj, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                return float(np.mean(arr))
        except Exception:
            pass

        values = getattr(obj, "values", obj)
        flat = np.asarray(values, dtype=object).ravel()
        extracted: List[float] = []
        for x in flat:
            if x is None:
                continue
            if isinstance(x, (list, tuple, np.ndarray)):
                try:
                    xa = np.asarray(x, dtype=float)
                except Exception:
                    continue
                xa = xa[np.isfinite(xa)]
                if xa.size:
                    extracted.append(float(xa[-1]))
                continue
            try:
                fx = float(x)
            except Exception:
                continue
            if np.isfinite(fx):
                extracted.append(fx)

        if not extracted:
            raise RuntimeError("Could not extract numeric log_marginal_likelihood values from sample_stats")
        return float(np.mean(extracted))

    sample_stats = getattr(idata, "sample_stats", None)
    if sample_stats is not None:
        data_vars = getattr(sample_stats, "data_vars", None)
        if data_vars is not None and "log_marginal_likelihood" in data_vars:
            return _robust_mean_logml(sample_stats["log_marginal_likelihood"])
        if isinstance(sample_stats, dict) and "log_marginal_likelihood" in sample_stats:
            return _robust_mean_logml(sample_stats["log_marginal_likelihood"])

    raise RuntimeError(
        "Thermodynamic log evidence is unavailable. "
        "Use SMC; it should provide sample_stats['log_marginal_likelihood']."
    )


def _bayes_factor_table(logml_by_model: Dict[str, float], ref: str = "homo") -> pd.DataFrame:
    if ref not in logml_by_model:
        raise ValueError(f"Reference model {ref!r} missing")

    ref_logml = float(logml_by_model[ref])
    rows: List[Dict[str, float | str]] = []

    for m, logml in logml_by_model.items():
        d = float(logml) - ref_logml
        log10_bf = d / np.log(10.0)
        bf = np.inf if d > 700.0 else float(np.exp(d))
        rows.append(
            {
                "model": m,
                "logml": float(logml),
                f"Δlogml_vs_{ref}": float(d),
                f"log10_BF_vs_{ref}": float(log10_bf),
                f"BF_vs_{ref}": float(bf) if np.isfinite(bf) else np.inf,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(f"Δlogml_vs_{ref}", ascending=False)
        .reset_index(drop=True)
    )


def plot_bayes_factor_groups(
    *,
    bf_tables: Dict[str, Dict[str, pd.DataFrame]],
    scenarios_order: Sequence[str],
    model_order: Sequence[str],
    ref: str = "homo",
    save_path: str | Path,
    dpi: int = 350,
    use_log10: bool = True,
) -> None:
    samplers = list(bf_tables.keys())
    ncols = len(samplers)

    fig = plt.figure(figsize=(7.0 * ncols, 5.2), dpi=int(dpi))
    fig.patch.set_alpha(0.0)

    for j, sampler in enumerate(samplers, start=1):
        ax = fig.add_subplot(1, ncols, j)
        ax.set_facecolor("none")

        group_x = np.arange(len(scenarios_order), dtype=float)
        n_models = len(model_order)
        width = 0.82 / max(1, n_models)
        offsets = (np.arange(n_models, dtype=float) - (n_models - 1) / 2.0) * width

        for k, m in enumerate(model_order):
            ys: List[float] = []
            for sc_name in scenarios_order:
                df = bf_tables[sampler][sc_name]
                row = df.loc[df["model"] == m]
                if row.empty:
                    ys.append(np.nan)
                    continue

                if use_log10 and f"log10_BF_vs_{ref}" in df.columns:
                    ys.append(float(row.iloc[0][f"log10_BF_vs_{ref}"]))
                else:
                    bf = float(row.iloc[0][f"BF_vs_{ref}"])
                    ys.append(float(np.log10(max(bf, 1e-300))) if use_log10 else float(bf))

            ax.bar(group_x + offsets[k], ys, width=width, label=m)

        ax.axhline(0.0, linewidth=1.2)
        ax.set_xticks(group_x)
        ax.set_xticklabels(list(scenarios_order))
        ax.set_title(f"Bayes factors ({sampler})", fontweight="bold")
        ax.set_ylabel(r"$\log_{10}\,\mathrm{BF}$ vs " + ref if use_log10 else f"BF vs {ref}")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=True, edgecolor="black", fontsize=9)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path.with_suffix(".png"), dpi=int(dpi), bbox_inches="tight", transparent=True)
    plt.close(fig)
    print("Saved BF comparison plot:", str(save_path.with_suffix(".png")))


def run_all_inference(sc: scenario, args, *, sampler: str) -> Dict[str, Any]:
    def _safe_infer(model_label: str, fn, *fn_args, **fn_kwargs):
        try:
            return fn(*fn_args, **fn_kwargs)
        except Exception as e:
            print(f"[Inference] Failed: scenario={sc.name} sampler={sampler} model={model_label}: {e}")
            return None

    base_kwargs = dict(
        mode=str(args.inference_mode),
        sampler=str(sampler),
        draws=int(args.posterior_samples),
        tune=int(args.posterior_tune),
        chains=int(args.posterior_chains),
        cores=int(args.cores),
        smc_draws=int(args.smc_particles),
        smc_cores=int(args.smc_cores),
        lambda_prior_bounds=(-5.0, 2.0),
        random_seed=(int(args.seed) if args.seed is not None else None),
    )

    pack_homo = _safe_infer(
        "homo",
        cih.inference_homo,
        sc.results_dir["n_contacts"],
        sc.results_dir["max_time"],
        sc.results_dir["dt_list"],
        **base_kwargs,
    )

    pack_Z2P = _safe_infer(
        "Z2P",
        cih.inference_Z2P,
        sc.results_dir["n_contacts"],
        sc.results_dir["max_time"],
        sc.results_dir["dt_list"],
        p_prior_bounds=(1.0, 1.0),
        **base_kwargs,
    )

    pack_Dis2P = _safe_infer(
        "Dis2P",
        cih.inference_Dis2P,
        sc.results_dir["n_contacts"],
        sc.results_dir["max_time"],
        sc.results_dir["dt_list"],
        dis_mode=str(sc.dis_mode or "gamma"),
        std_prior_factor=1.0,
        marginalized=(sampler == "smc"),
        **base_kwargs,
    )

    pack_hetero3 = _safe_infer(
        "hetero3",
        cih.inference_hetero3,
        sc.results_dir["n_contacts"],
        sc.results_dir["max_time"],
        sc.results_dir["dt_list"],
        dis_mode=str(sc.dis_mode or "gamma"),
        p_prior_bounds=(1.0, 1.0),
        std_prior_factor=1.0,
        marginalized=(sampler == "smc"),
        **base_kwargs,
    )

    return {"homo": pack_homo, "Z2P": pack_Z2P, "Dis2P": pack_Dis2P, "hetero3": pack_hetero3}


def _model_plot_settings(model_name: str):
    if model_name == "homo":
        return ("lambda",), {"lambda": r"$\lambda$"}, {"lambda": (0.0, 0.12)}
    if model_name == "Z2P":
        return ("lambda", "p_zero"), {"lambda": r"$\lambda$", "p_zero": r"$\phi_0$"}, {"lambda": (0.0, 0.12), "p_zero": (0.0, 1.0)}
    if model_name == "Dis2P":
        return ("mu_lambda", "sigma_lambda"), {"mu_lambda": r"$\mu_{\lambda}$", "sigma_lambda": r"$\sigma_{\lambda}$"}, {"mu_lambda": (0.0, 0.12), "sigma_lambda": (0.0, 0.12)}
    if model_name == "hetero3":
        return (
            ("mu_lambda", "sigma_lambda", "p_zero"),
            {"mu_lambda": r"$\mu_{\lambda}$", "sigma_lambda": r"$\sigma_{\lambda}$", "p_zero": r"$\phi_0$"},
            {"mu_lambda": (0.0, 0.12), "sigma_lambda": (0.0, 0.12), "p_zero": (0.0, 1.0)},
        )
    raise ValueError(f"Unknown model_name: {model_name!r}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    p = argparse.ArgumentParser(
        description=(
            "Simulate contact data and run inference with multiple models. "
            "Runs NUTS and/or SMC, plots posteriors, and compares models via Bayes factors."
        )
    )

    p.add_argument("--n_cell", type=int, default=5000, help="Number of killer cells in the simulation.")
    p.add_argument("--T", type=float, default=60.0, help="Total simulation time.")
    p.add_argument("--obs_mode", type=str, default="Complete", help="Observation mode: Complete or Truncated.")
    p.add_argument("--T_sd", type=float, default=None, help="Standard deviation of observation time for Truncated mode.")
    p.add_argument("--dis_mode", type=str, default="gamma", help="Distribution mode for heterogeneous lambda.")
    p.add_argument("--seed", type=int, default=None, help="Random seed for simulation.")
    p.add_argument("--out_dir", type=str, default="results_1", help="Directory to save results.")

    p.add_argument("--inference_mode", type=str, default="counts+gaps", help="Inference mode: counts or counts+gaps.")
    p.add_argument("--posterior_samples", type=int, default=5000)
    p.add_argument("--posterior_tune", type=int, default=3000)
    p.add_argument("--posterior_chains", type=int, default=4)

    p.add_argument("--smc_particles", type=int, default=5000, help="Number of SMC particles (draws) per chain.")
    p.add_argument("--smc_cores", type=int, default=0, help="CPU processes for SMC (0 = all available cores).")
    p.add_argument("--cores", type=int, default=0, help="CPU processes for PyMC sampling (0 = all available cores).")

    args = p.parse_args(list(argv) if argv is not None else None)

    if args.inference_mode not in {"counts", "counts+gaps"}:
        raise ValueError("--inference_mode must be 'counts' or 'counts+gaps'")

    if args.obs_mode == "Truncated":
        if args.T_sd is None or args.T_sd <= 0:
            raise ValueError("T_sd must be positive for Truncated observation mode.")

    scenarios: List[scenario] = [
        scenario(name="Homo", mode="homogeneous", n_cell=args.n_cell, T=args.T, T_sd=args.T_sd, obs_mode=args.obs_mode, mu_lambda=0.05),
        scenario(
            name="Hetero",
            mode="heterogeneous",
            n_cell=args.n_cell,
            T=args.T,
            T_sd=args.T_sd,
            obs_mode=args.obs_mode,
            mu_lambda=0.08,
            sd_lambda=0.06,
            p0_lambda=0.2,
            dis_mode=args.dis_mode,
        ),
    ]

    for sc in scenarios:
        print(f"--- Running simulation: {sc.name} ---")
        sim_data = model.simulate_Population(
            n_cells=sc.n_cell,
            T=sc.T,
            truncation_noise=sc.T_sd,
            obs_mode=sc.obs_mode,
            mode=sc.mode,
            mu_lambda=sc.mu_lambda,
            sd_lambda=sc.sd_lambda,
            p0_lambda=sc.p0_lambda,
            Dist_mode=sc.dis_mode,
            seed=(int(args.seed) if args.seed is not None else None),
        )
        sc.take_results(sim_data)
        print(f"Simulation completed for {sc.name}.")

    base_dir = Path(args.out_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    plot_counts_and_dt(
        size=(14, 6),
        sims=[sc.results_dir for sc in scenarios],
        labels=[sc.name for sc in scenarios],
        save_path=base_dir / "simulation",
        cmap_name="gist_earth",
        dpi=400,
        dt_xlim=(1e-2, 1e2),
        counts_max=15,
    )

    model_names = ["homo", "Z2P", "Dis2P", "hetero3"]

    bay_store: Dict[str, Dict[str, Dict[str, Any]]] = {sc.name: {} for sc in scenarios}

    # 1. Run NUTS for Posterior Inference
    for sc in scenarios:
        print(f"\n=== Inference (NUTS - Posterior): scenario={sc.name} ===")
        bay_store[sc.name]["nuts"] = run_all_inference(sc, args, sampler="nuts")

    # 2. Run SMC for Bayes Factors
    for sc in scenarios:
        print(f"\n=== Inference (SMC - Evidence): scenario={sc.name} ===")
        bay_store[sc.name]["smc"] = run_all_inference(sc, args, sampler="smc")

    # 3. Plot Posteriors (Using NUTS results)
    for model_name in model_names:
        idatas: List[Tuple[str, az.InferenceData]] = []
        for sc in scenarios:
            pack = bay_store[sc.name].get("nuts", {}).get(model_name)
            if pack is None:
                continue
            if isinstance(pack, dict) and "idata" in pack:
                idatas.append((sc.name, pack["idata"]))
            else:
                pass

        if not idatas:
            continue

        parameters, parameter_display, xlims = _model_plot_settings(model_name)
        ground_truth = ground_truth_for_model(scenarios, model_name)

        plot_posteriors(
            idatas,
            ground_truth=ground_truth if ground_truth else None,
            parameters=parameters,
            parameter_display=parameter_display,
            save_path=base_dir / f"posterior_triangle_{model_name}_nuts",
            cmap_name="gist_earth",
            sample_size=6000,
            dpi=400,
            diagonal_style="hist",
            marginal_style="circle",
            xlims=xlims,
            seed=(int(args.seed) if args.seed is not None else None),
        )

    print("\n===== Bayes factors per scenario (ref = homo) using SMC =====")
    bf_tables: Dict[str, pd.DataFrame] = {}

    for sc in scenarios:
        logml_by_model: Dict[str, float] = {}
        for model_name in model_names:
            pack = bay_store[sc.name].get("smc", {}).get(model_name)
            if pack is None:
                continue
            try:
                logml_by_model[model_name] = _estimate_logml(pack["model"], pack["idata"])
            except RuntimeError as e:
                print(f"[BayesFactor] Skipping evidence: scenario={sc.name} sampler=smc model={model_name}: {e}")

        if "homo" not in logml_by_model:
            print(f"[BayesFactor] Skipping scenario={sc.name} sampler=smc: no evidence for ref 'homo'.")
            continue

        df_bf = _bayes_factor_table(logml_by_model, ref="homo")
        bf_tables[sc.name] = df_bf

        out_csv = base_dir / f"bayes_factors_{sc.name}_smc.csv"
        df_bf.to_csv(out_csv, index=False)
        print(f"\nScenario: {sc.name} | sampler=smc")
        print(df_bf.to_string(index=False))
        print("Saved:", str(out_csv))

    if bf_tables:
        params_wrapper = {"smc": bf_tables}
        plot_bayes_factor_groups(
            bf_tables=params_wrapper,
            scenarios_order=[sc.name for sc in scenarios],
            model_order=model_names,
            ref="homo",
            save_path=base_dir / "bayes_factor_comparison_grouped",
            dpi=450,
            use_log10=True,
        )
    else:
        print("[BayesFactor] No Bayes factor tables produced (no thermodynamic evidence found).")

    print("\nDone.")


if __name__ == "__main__":
    main()
