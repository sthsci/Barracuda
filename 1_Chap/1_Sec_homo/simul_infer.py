from __future__ import annotations

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

from Simulator.model_contact import population_contact  # noqa: E402
import filter_contacts  # type: ignore  # noqa: E402
from Inference import contac_inference  # type: ignore  # noqa: E402


def simulate_population(theta, n_cells, max_time, seed, outdir, save_data=False):
    out = population_contact(
        n_cells=n_cells,
        max_time=max_time,
        theta=theta,
        seed=seed,
    )

    if save_data:
        raw_path = outdir / f"sim_raw_lambda0={theta[0]:.3f}_n50={theta[1]:.3f}_s={theta[2]:.3f}.npz"
        np.savez_compressed(
            raw_path,
            times_list=np.array(out["times_list"], dtype=object),
            dt_list=np.array(out["dt_list"], dtype=object),
            max_time=float(out["max_time"]),
            n_cells=int(out["n_cells"]),
        )
        return raw_path

    return out


def filter_simulation(raw, drop_start_max, drop_end_max, max_censor_extra, seed, outdir, save_data=False):
    if save_data:
        raw_path = Path(raw)
        filtered_path = outdir / (raw_path.stem + "__filtered.npz")
        filter_contacts.filter_dataset(
            npz_path=str(raw_path),
            out_path=str(filtered_path),
            seed=seed,
            drop_start_max=drop_start_max,
            drop_end_max=drop_end_max,
            max_censor_extra=max_censor_extra,
        )
        return filtered_path

    raise ValueError("This driver expects save_data=True to create filtered npz on disk.")


def run_inference(filtered_npz, draws, tune, chains, target_accept, seed, obs_mode, truncation, offset_max):
    data = np.load(filtered_npz, allow_pickle=True)

    dt_list = contac_inference._ensure_object_list(data["dt_list"])
    T_obs = np.asarray(data["T_obs"], dtype=float)
    prep = contac_inference.prepare(dt_list=dt_list, T_obs=T_obs, include_censor=True)

    offset_obs = None
    if truncation == "known":
        offset_obs = np.asarray(data["dropped"], dtype=int)[:, 0]

    model = contac_inference.build_model_homogeneous(
        prep=prep,
        obs_mode=obs_mode,
        truncation=truncation,
        offset_obs=offset_obs,
        offset_max=int(offset_max),
    )

    with model:
        idata = contac_inference.pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=False,
        )

    return idata


def plot_contact_count_frequency(count_lists, labels, save_path, cmap_name="inferno", font_scale=0.9, figure_label=None):
    sns.set_context("talk", font_scale=font_scale)
    cmap = plt.colormaps.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.3, 0.9, len(labels)))

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    for colour, label, counts in zip(colors, labels, count_lists):
        n_cells = len(counts)
        max_k = int(np.max(counts)) if n_cells else 0
        binc = np.bincount(np.asarray(counts, dtype=int), minlength=max_k + 1) if n_cells else np.array([0])
        freq = binc / n_cells if n_cells else np.zeros_like(binc, dtype=float)
        x = np.arange(binc.size)
        ax.plot(x, freq, color=colour, linewidth=2.0, label=label, marker="o", markersize=3.5)

    ax.set_title("Contact count frequency", fontweight="bold")
    ax.set_xlabel("Contacts per cell")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.grid(True, alpha=0.3)
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
    sample_size=6000,
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
    gs = gridspec.GridSpec(npar, npar, wspace=0.2, hspace=0.2)
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

                    if ground_truth and label in ground_truth:
                        ax.axvline(ground_truth[label][rowpar], color=colour, linestyle="-", linewidth=1.8)

                    ax.set_xlabel(parameter_display.get(rowpar, rowpar))
                    ax.set_ylabel("Density")
                    ax.grid(alpha=0.2)

                else:
                    if marginal_style == "circle":
                        sns.kdeplot(x=df[colpar], y=df[rowpar], ax=ax, fill=False, color=colour, alpha=0.6, levels=7, linewidths=1.0)
                    else:
                        sns.histplot(x=df[colpar], y=df[rowpar], bins=60, pthresh=0.01, cmap=cmap_name, cbar=False, ax=ax)

                    if ground_truth and label in ground_truth:
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
    parser = argparse.ArgumentParser(description="Simulate, filter, infer (both/count), and plot.")
    parser.add_argument("--n_cells", type=int, default=600)
    parser.add_argument("--max_time", type=float, default=120.0)
    parser.add_argument("--drop_start_max", type=int, default=2)
    parser.add_argument("--drop_end_max", type=int, default=2)
    parser.add_argument("--max_censor_extra", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--target_accept", type=float, default=0.9)
    parser.add_argument("--outdir", type=str, default=None)

    parser.add_argument("--obs-mode", choices=["both", "count"], default="both")
    parser.add_argument("--truncation", choices=["unknown", "known"], default="unknown")
    parser.add_argument("--offset-max", type=int, default=10)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--save-data", dest="save_data", action="store_true")
    mode.add_argument("--ram-only", dest="save_data", action="store_false")
    parser.set_defaults(save_data=True)

    args = parser.parse_args(argv)

    outdir = Path(args.outdir) if args.outdir else Path(__file__).resolve().parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        {"name": "lam1_n50_5_s0.4", "theta": (1.0, 5.0, 0.4)},
        {"name": "lam1p4_n50_6_s1p0", "theta": (1.4, 6.0, 1.0)},
        {"name": "lam0p8_n50_4_s1p6", "theta": (0.8, 4.0, 1.6)},
    ]

    count_lists, labels, idatas, gt_vals = [], [], [], {}

    for j, sc in enumerate(scenarios):
        label = sc["name"]
        theta = sc["theta"]
        labels.append(label)
        seed_j = args.seed + j

        raw_path = simulate_population(
            theta=theta,
            n_cells=args.n_cells,
            max_time=args.max_time,
            seed=seed_j,
            outdir=outdir,
            save_data=True,
        )

        filtered_path = filter_simulation(
            raw=raw_path,
            drop_start_max=args.drop_start_max,
            drop_end_max=args.drop_end_max,
            max_censor_extra=args.max_censor_extra,
            seed=seed_j,
            outdir=outdir,
            save_data=True,
        )

        data_f = np.load(filtered_path, allow_pickle=True)
        dt_list = contac_inference._ensure_object_list(data_f["dt_list"])
        counts = [len(dt) for dt in dt_list]
        count_lists.append(counts)

        gt_vals[label] = {"lambda0": float(theta[0]), "n50": float(theta[1]), "s": float(theta[2])}

        idata = run_inference(
            filtered_npz=filtered_path,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept,
            seed=seed_j,
            obs_mode=args.obs_mode,
            truncation=args.truncation,
            offset_max=args.offset_max,
        )
        idatas.append(idata)

        (outdir / "idata").mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(outdir / "idata" / f"idata__{label}__{args.obs_mode}__{args.truncation}.nc")

    plot_contact_count_frequency(
        count_lists=count_lists,
        labels=labels,
        save_path=outdir / f"synthetic_counts__{args.obs_mode}__{args.truncation}",
        cmap_name="inferno",
        font_scale=0.85,
        figure_label=f"Counts (obs={args.obs_mode}, trunc={args.truncation})",
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
        save_path=outdir / f"synthetic_posteriors__{args.obs_mode}__{args.truncation}",
        seed=args.seed,
        font_scale=0.85,
        figure_label=f"Posteriors (obs={args.obs_mode}, trunc={args.truncation})",
    )

    print("Completed. Outputs in:", str(outdir))


if __name__ == "__main__":
    main()
