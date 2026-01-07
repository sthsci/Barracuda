#!/usr/bin/env python3
from __future__ import annotations

import os
os.environ.setdefault("PYTENSOR_FLAGS", "optimizer_excluding=fusion")

import argparse
from pathlib import Path
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    "font.family": ["Monaco", "DejaVu Sans Mono", "monospace"],
    "mathtext.fontset": "stix",
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAP_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(CHAP_ROOT))


from Simulator.model_contact import (  # noqa: E402
    simulate_population,
    filter_dataset_in_memory,
    save_simulation_npz,
    save_filtered_npz,
)
from Inference import contact_inference  # type: ignore  # noqa: E402


def _counts_from_dt_list(dt_list) -> list[int]:
    dt_list = contact_inference._ensure_object_list(dt_list)
    return [int(np.asarray(dt, dtype=float).size) for dt in dt_list]


def simulate_and_filter_one(
    *,
    theta: tuple[float, float, float],
    n_cells: int,
    max_time: float,
    seed: int,
    sim_mode: str,
    drop_start_max: int,
    drop_end_max: int,
    max_censor_extra: float | None,
) -> tuple[dict, dict]:
    raw = simulate_population(
        n_cells=int(n_cells),
        max_time=float(max_time),
        theta=theta,
        seed=int(seed),
        sim_mode=str(sim_mode),
    )
    filt = filter_dataset_in_memory(
        times_list=raw["times_list"],
        T=float(raw["max_time"]),
        seed=int(seed),
        drop_start_max=int(drop_start_max),
        drop_end_max=int(drop_end_max),
        max_censor_extra=max_censor_extra,
    )
    return raw, filt


def run_inference_from_filtered(
    *,
    filtered: dict | Path | str,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    seed: int,
    obs_mode: str,
    offset_max: int,
):
    if isinstance(filtered, (str, Path)):
        data = np.load(str(filtered), allow_pickle=True)
        dt_list = contact_inference._ensure_object_list(data["dt_list"])
        T_obs = np.asarray(data["T_obs"], dtype=float)
    else:
        dt_list = contact_inference._ensure_object_list(filtered["dt_list"])
        T_obs = np.asarray(filtered["T_obs"], dtype=float)

    prep = contact_inference.prepare(dt_list=dt_list, T_obs=T_obs, include_censor=True)

    model = contact_inference.build_model_homogeneous(
        prep=prep,
        obs_mode=str(obs_mode),
        offset_max=int(offset_max),
    )

    with model:
        idata = contact_inference.pm.sample(
            draws=int(draws),
            tune=int(tune),
            chains=int(chains),
            target_accept=float(target_accept),
            random_seed=int(seed),
            progressbar=False,
        )

    return idata


def plot_contact_count_frequency(
    count_lists,
    labels,
    save_path,
    cmap_name="inferno",
    font_scale=0.9,
    figure_label=None,
    style="hist",          # "hist" (default), "line", "both"
    hist_alpha=0.22,
    line_width=2.0,
    marker_size=3.5,
):
    sns.set_context("talk", font_scale=font_scale)
    cmap = plt.colormaps.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.3, 0.9, len(labels)))

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    for colour, label, counts in zip(colors, labels, count_lists):
        counts = np.asarray(counts, dtype=int)
        n_cells = int(counts.size)
        max_k = int(counts.max()) if n_cells else 0

        binc = np.bincount(counts, minlength=max_k + 1) if n_cells else np.array([0], dtype=int)
        freq = binc / n_cells if n_cells else np.zeros_like(binc, dtype=float)

        x = np.arange(freq.size)

        if style in {"hist", "both"}:
            # Connected-bin step histogram (filled + contour)
            edges = np.arange(freq.size + 1) - 0.5  # bin edges centred on integers
            y_step = np.r_[freq, freq[-1]]          # length N+1 to match edges

            ax.fill_between(
                edges,
                y_step,
                step="post",
                alpha=hist_alpha,
                facecolor=colour,
                edgecolor="none",
                label=(f"{label} (hist)" if style == "both" else label),
            )
            ax.step(
                edges,
                y_step,
                where="post",
                linewidth=line_width,
                color=colour,
            )

        if style in {"line", "both"}:
            ax.plot(
                x,
                freq,
                color=colour,
                linewidth=line_width,
                label=(f"{label} (line)" if style == "both" else label),
                marker="o",
                markersize=marker_size,
            )

    ax.set_title("Contact count frequency", fontweight="bold")
    ax.set_xlabel("Contacts per cell")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    leg = ax.legend(frameon=True, edgecolor="black", loc="upper right", fontsize=10)
    if leg:
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(1.0)

    if figure_label:
        fig.text(0.5, 0.98, figure_label, ha="center", va="top", fontsize=16, fontweight="bold")

    out = Path(save_path)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)



def plot_joint_posteriors_corner(
    idatas,
    parameters=("lambda0", "n50", "s"),
    parameter_display=None,
    ground_truth=None,
    hdi_prob=0.95,
    sample_size=8000,
    save_path="posteriors_corner",
    cmap_name="inferno",
    diagonal_style="hist",
    marginal_style="circle",
    font_scale=0.85,
    seed=None,
    figure_label=None,
):
    sns.set_context("talk", font_scale=font_scale)
    rng = np.random.default_rng(seed)

    if parameter_display is None:
        parameter_display = {"lambda0": r"$\lambda_0$", "n50": r"$n_{50}$", "s": r"$s$"}

    if isinstance(idatas, dict):
        idatas = list(idatas.items())

    parameters = list(parameters)
    npar = len(parameters)

    cmap = plt.colormaps.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.3, 0.9, len(idatas)))

    label_to_df = {}
    for label, idata in idatas:
        post = idata.posterior
        cols = {}
        for p in parameters:
            vals = post[p].stack(sample=("chain", "draw")).values.ravel()
            vals = vals[np.isfinite(vals)]
            if (sample_size is not None) and (vals.size > sample_size):
                vals = vals[rng.choice(vals.size, sample_size, replace=False)]
            cols[p] = vals
        df = pd.DataFrame(cols)
        df["label"] = label
        label_to_df[label] = df

    fig = plt.figure(figsize=(5.0 * npar, 5.0 * npar), dpi=300)
    fig.patch.set_alpha(0.0)
    gs = gridspec.GridSpec(npar, npar, wspace=0.4, hspace=0.4)
    gaxes = np.empty((npar, npar), dtype=object)

    for irow, rowpar in enumerate(parameters):
        for icol, colpar in enumerate(parameters):
            ax = fig.add_subplot(gs[irow, icol])
            ax.set_facecolor("none")
            gaxes[irow, icol] = ax

            if icol > irow:
                ax.axis("off")
                continue

            for colour, (label, df) in zip(colors, label_to_df.items()):
                if irow == icol:
                    vals = df[rowpar].dropna().values

                    if diagonal_style == "kde":
                        sns.kdeplot(vals, ax=ax, fill=True, color=colour, alpha=0.20, linewidth=1.5, label=label if irow == 0 else None)
                    else:
                        sns.histplot(vals, bins=30, stat="density", kde=False, ax=ax, color=colour, alpha=0.18, element="step", fill=True)
                        sns.histplot(vals, bins=30, stat="density", kde=False, ax=ax, color=colour, alpha=1.0, element="step", fill=False, linewidth=1.8, label=label if irow == 0 else None)

                    lo, hi = az.hdi(vals, hdi_prob=hdi_prob)
                    ax.axvspan(lo, hi, color=colour, alpha=0.06, linewidth=0)

                    if ground_truth and label in ground_truth and rowpar in ground_truth[label]:
                        ax.axvline(ground_truth[label][rowpar], color=colour, linestyle="-", linewidth=1.8)

                    ax.set_xlabel(parameter_display.get(rowpar, rowpar))
                    ax.set_ylabel("Density")
                    ax.grid(alpha=0.2)

                else:
                    if marginal_style == "circle":
                        sns.kdeplot(x=df[colpar], y=df[rowpar], ax=ax, fill=False, color=colour, alpha=0.6, levels=7, linewidths=1.0)
                    else:
                        sns.histplot(x=df[colpar], y=df[rowpar], bins=60, pthresh=0.01, cmap=cmap_name, cbar=False, ax=ax)

                    if ground_truth and label in ground_truth and colpar in ground_truth[label] and rowpar in ground_truth[label]:
                        gt = ground_truth[label]
                        ax.scatter(gt[colpar], gt[rowpar], marker="*", color=colour, s=80, linewidths=2.0, zorder=1000)

                    ax.set_xlabel(parameter_display.get(colpar, colpar))
                    ax.set_ylabel(parameter_display.get(rowpar, rowpar))
                    ax.grid(alpha=0.3)

            ax.margins(0.05)

    handles, leg_labels = gaxes[0, 0].get_legend_handles_labels()
    if handles:
        leg = fig.legend(handles, leg_labels, loc="center", bbox_to_anchor=(0.85, 0.85), frameon=True, edgecolor="black", fontsize=10)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(1.0)

    if figure_label:
        fig.text(0.5, 0.99, figure_label, ha="center", va="top", fontsize=18, fontweight="bold")

    plt.tight_layout()
    out = Path(save_path)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="Simulate (new model_contact), filter in-memory, infer (both/count), plot.")
    p.add_argument("--n_cells", type=int, default=10000)
    p.add_argument("--max_time", type=float, default=120.0)
    p.add_argument("--sim-mode", choices=["global", "individual"], default="global")

    p.add_argument("--drop_start_max", type=int, default=0)
    p.add_argument("--drop_end_max", type=int, default=0)
    p.add_argument("--max_censor_extra", type=float, default=None)

    p.add_argument("--seed", type=int, default=2026)

    p.add_argument("--draws", type=int, default=5000)
    p.add_argument("--tune", type=int, default=3000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--target_accept", type=float, default=0.95)

    p.add_argument("--obs-mode", choices=["both", "count"], default="both")
    p.add_argument("--offset-max", type=int, default=10)

    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--save-data", action="store_false", help="Save raw/filtered npz and idata netcdf.")
    args = p.parse_args(argv)

    outdir = Path(args.outdir) if args.outdir else Path(__file__).resolve().parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "idata").mkdir(parents=True, exist_ok=True)
    (outdir / "npz").mkdir(parents=True, exist_ok=True)

    scenarios = [
        {"name": "lam1_n50_2_s0.4", "theta": (1.0, 2.0, 0.4)},
        # {"name": "lam1p4_n50_6_s1p0", "theta": (1.4, 6.0, 1.0)},
        # {"name": "lam0p8_n50_4_s1p6", "theta": (0.8, 4.0, 1.6)},
    ]

    count_lists, labels, idatas, gt_vals = [], [], [], {}

    for j, sc in enumerate(scenarios):
        label = sc["name"]
        theta = sc["theta"]
        seed_j = int(args.seed) + j

        raw, filt = simulate_and_filter_one(
            theta=theta,
            n_cells=args.n_cells,
            max_time=args.max_time,
            seed=seed_j,
            sim_mode=args.sim_mode,
            drop_start_max=args.drop_start_max,
            drop_end_max=args.drop_end_max,
            max_censor_extra=args.max_censor_extra,
        )

        if args.save_data:
            raw_path = outdir / "npz" / f"sim_raw__{label}__mode={args.sim_mode}.npz"
            filt_path = outdir / "npz" / f"sim_filt__{label}__mode={args.sim_mode}__ds={args.drop_start_max}__de={args.drop_end_max}.npz"
            save_simulation_npz(raw, raw_path)
            save_filtered_npz(filt, filt_path)
            filt_for_infer = filt_path
        else:
            filt_for_infer = filt

        counts = _counts_from_dt_list(filt["dt_list"])
        count_lists.append(counts)
        labels.append(label)
        gt_vals[label] = {"lambda0": float(theta[0]), "n50": float(theta[1]), "s": float(theta[2])}

        idata = run_inference_from_filtered(
            filtered=filt_for_infer,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept,
            seed=seed_j,
            obs_mode=args.obs_mode,
            offset_max=args.offset_max,
        )
        idatas.append(idata)

        if args.save_data:
            idata_path = outdir / "idata" / f"idata__{label}__obs={args.obs_mode}__mmax={args.offset_max}__sim={args.sim_mode}.nc"
            idata.to_netcdf(idata_path)

    plot_contact_count_frequency(
        count_lists=count_lists,
        labels=labels,
        save_path=outdir / f"synthetic_counts__obs={args.obs_mode}__mmax={args.offset_max}__sim={args.sim_mode}",
        cmap_name="inferno",
        font_scale=0.85,
        figure_label=f"Counts (obs={args.obs_mode}, mmax={args.offset_max}, sim={args.sim_mode})",
    )

    plot_joint_posteriors_corner(
        idatas=list(zip(labels, idatas)),
        parameters=("lambda0", "n50", "s"),
        parameter_display={"lambda0": r"$\lambda_0$", "n50": r"$n_{50}$", "s": r"$s$"},
        ground_truth=gt_vals,
        sample_size=6000,
        hdi_prob=0.95,
        cmap_name="inferno",
        diagonal_style="hist",
        marginal_style="circle",
        save_path=outdir / f"synthetic_posteriors__obs={args.obs_mode}__mmax={args.offset_max}__sim={args.sim_mode}",
        seed=args.seed,
        font_scale=0.85,
        figure_label=f"Posteriors (obs={args.obs_mode}, mmax={args.offset_max}, sim={args.sim_mode})",
    )

    print("Completed. Outputs in:", str(outdir))


if __name__ == "__main__":
    main()
