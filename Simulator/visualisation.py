from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.ticker import MultipleLocator


plt.rcParams.update({
    "font.family": ["Monaco", "DejaVu Sans Mono", "monospace"],
    "mathtext.fontset": "stix",
    "legend.fontsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.edgecolor": "none",
})


__all__ = [
    "scenario_label",
    "decision_map",
    "plot_targets_remaining",
    "plot_targets_remaining_normalised",
    "heatmap_contacts_vs_kills",
    "_pick_colors",
    "_parse_bool",
    "_parse_float_list",
    "_parse_int_list",
]


def _path_from_decisions(decisions):
    x, y = [0], [0]
    for d in decisions:
        if d == 1:
            x.append(x[-1])
            y.append(y[-1] + 1)
        else:
            x.append(x[-1] + 1)
            y.append(y[-1])
    return np.array(x), np.array(y)


def _parse_bool(x: str) -> bool:
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {x}")


def _parse_float_list(x: str):
    s = str(x).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    parts = [p for p in parts if p != ""]
    try:
        return [float(p) for p in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid float list: {x}") from e


def _parse_int_list(x: str):
    s = str(x).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    parts = [p for p in parts if p != ""]
    try:
        return [int(p) for p in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid int list: {x}") from e


def _fmt_float(x, nd=2):
    return f"{float(x):.{nd}f}"


def scenario_label(
    sc,
    n_killers: int | None = None,
    n_targets: int | None = None,
    rate0: float | None = None,
    *,
    show_counts: bool = True,
    show_rate: bool = True,
):
    cap_mode = sc["capacity_mode"]
    kw = sc.get("capacity_kwargs", {}) or {}
    K0 = kw.get("K0", None)
    sigma = kw.get("sigma", None)
    proportions = kw.get("proportions", None)
    community_K = kw.get("community_K", None)
    min_K = kw.get("min_K", None)
    max_K = kw.get("max_K", None)

    cap_bits = [f"K~{cap_mode}"]
    if K0 is not None:
        cap_bits.append(f"K0={_fmt_float(K0,2)}")
    if sigma is not None and cap_mode in {"normal", "lognormal", "gamma"}:
        cap_bits.append(f"s={_fmt_float(sigma,2)}")
    if cap_mode == "discrete" and proportions is not None and community_K is not None:
        try:
            props_str = ",".join(_fmt_float(p, 2) for p in list(proportions))
            ks_str = ",".join(str(int(k)) for k in list(community_K))
            cap_bits.append(f"props=[{props_str}]")
            cap_bits.append(f"Kc=[{ks_str}]")
        except Exception:
            cap_bits.append("(discrete)")
    if min_K is not None:
        cap_bits.append(f"min={int(min_K)}")
    if max_K is not None:
        cap_bits.append(f"max={int(max_K)}")

    p_bits = [f"p:{sc['p_mode']}"]
    if sc["p_mode"] == "deterministic":
        # Deterministic mode uses p_kill_matrix = 1 everywhere (until capacity runs out).
        p_bits.append("p=1")
    elif sc["p_mode"] == "stochastic":
        p_bits.append(f"{sc['p_stochastic_mode']}")
        p_bits.append(f"p0={_fmt_float(sc['p0'],2)}")
        p_bits.append(f"a={_fmt_float(sc['alpha'],2)}")
        p_bits.append(f"b={_fmt_float(sc['beta'],2)}")

    top = " | ".join(p_bits)

    bottom_parts = [" , ".join(cap_bits)]
    if show_counts:
        if n_killers is not None:
            bottom_parts.append(f"Nkill={int(n_killers)}")
        if n_targets is not None:
            bottom_parts.append(f"Ntarg={int(n_targets)}")
    if show_rate and (rate0 is not None):
        bottom_parts.append(f"r0={_fmt_float(rate0,2)}")

    bottom = " | ".join(bottom_parts).strip()
    if not bottom:
        return top
    return top + "\n" + bottom


def _pick_colors(n, cmap_name="YlGnBu"):
    cmap = plt.get_cmap(cmap_name)
    if n <= 1:
        return [cmap(0.75)]
    xs = np.linspace(0.35, 0.9, n)
    return [cmap(x) for x in xs]


def _apply_legend(ax, loc="upper left", show: bool = True, legend_kwargs: dict | None = None):
    if not show:
        return
    defaults = dict(
        loc=loc,
        frameon=True,
        edgecolor="black",
        fontsize=10,
        handlelength=2.0,
        borderpad=0.6,
        labelspacing=0.6,
    )
    if legend_kwargs:
        defaults.update(dict(legend_kwargs))
    ax.legend(**defaults)


def _hdi(samples, cred_mass: float = 0.95):
    s = np.asarray(samples)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return (np.nan, np.nan)
    s = np.sort(s)
    n = int(s.size)
    k = int(np.floor(cred_mass * n))
    if k < 1:
        return (float(s[0]), float(s[-1]))
    widths = s[k:] - s[: n - k]
    i = int(np.argmin(widths))
    return (float(s[i]), float(s[i + k]))


def decision_map(
    decision_data_group,
    figsize=(9.2, 8.6),
    dpi=300,
    max_event: int | None = None,
    x_max: int | None = None,
    y_max: int | None = None,
    offset_scale=4.0,
    alpha=0.5,
    lw=2.0,
    s=18.0,
    edgecolor="black",
    title: str | None = None,
    xlabel: str = "Non-lethal contacts",
    ylabel: str = "Lethal contacts",
    show_legend: bool = True,
    legend_loc: str = "upper left",
    legend_kwargs: dict | None = None,
    save_png=False,
    png_path="trajectories.png",
    save_pdf=False,
    pdf_path="trajectories.pdf",
):
    plt.figure(figsize=figsize, dpi=dpi)

    # Default axis limits: use max_event if provided (so axes match simulation grid).
    if x_max is None and max_event is not None:
        x_max = int(max_event) - 1
    if y_max is None and max_event is not None:
        y_max = int(max_event) - 1

    for decisions_list, label, color in decision_data_group:
        n_cells = len(decisions_list)
        for i, traj in enumerate(decisions_list):
            x, y = _path_from_decisions(traj)
            offset = (i - n_cells / 2) / (offset_scale * n_cells)
            x = x + offset
            y = y + offset
            plt.plot(x, y, alpha=alpha, color=color, lw=lw)
            plt.scatter(x[-1], y[-1], color=color, s=s, edgecolor=edgecolor, zorder=5)
        plt.plot([], [], color=color, label=label, lw=8)

    ax = plt.gca()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    # Axis limits (useful when jitter is on).
    pad = 0.5
    if x_max is not None:
        ax.set_xlim(-pad, float(x_max) + pad)
    if y_max is not None:
        ax.set_ylim(-pad, float(y_max) + pad)

    _apply_legend(ax, loc=legend_loc, show=show_legend, legend_kwargs=legend_kwargs)
    plt.tight_layout()

    if save_png:
        plt.savefig(png_path, dpi=dpi, bbox_inches="tight", transparent=True)
    if save_pdf:
        plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight", transparent=True)


def plot_targets_remaining(
    outs,
    labels,
    colors,
    figsize=(9.2, 4.6),
    dpi=300,
    title: str | None = None,
    xlabel: str = "Time",
    ylabel: str = "Targets remaining",
    show_legend: bool = True,
    legend_loc: str = "upper left",
    legend_kwargs: dict | None = None,
    save_png=False,
    png_path="targets_remaining.png",
    save_pdf=False,
    pdf_path="targets_remaining.pdf",
):
    plt.figure(figsize=figsize, dpi=dpi)

    for out, label, color in zip(outs, labels, colors):
        plt.step(
            out["targets_times"],
            out["targets_trace"],
            where="post",
            label=label,
            color=color,
        )

    ax = plt.gca()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.25, linestyle="--")
    _apply_legend(ax, loc=legend_loc, show=show_legend, legend_kwargs=legend_kwargs)
    plt.tight_layout()

    if save_png:
        plt.savefig(png_path, dpi=dpi, bbox_inches="tight", transparent=True)
    if save_pdf:
        plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight", transparent=True)


def plot_targets_remaining_normalised(
    outs,
    labels,
    colors,
    figsize=(9.2, 4.6),
    dpi=300,
    title: str | None = None,
    xlabel: str = "Time",
    ylabel: str = "Targets remaining (normalised)",
    show_legend: bool = True,
    legend_loc: str = "upper left",
    legend_kwargs: dict | None = None,
    save_png=False,
    png_path="targets_remaining_normalised.png",
    save_pdf=False,
    pdf_path="targets_remaining_normalised.pdf",
):
    plt.figure(figsize=figsize, dpi=dpi)

    for out, label, color in zip(outs, labels, colors):
        n0 = float(out.get("n_targets_init", 0.0))
        if n0 <= 0:
            raise ValueError("n_targets_init must be present and > 0 to normalise targets_trace")
        y = np.asarray(out["targets_trace"], dtype=float) / n0
        plt.step(
            out["targets_times"],
            y,
            where="post",
            label=label,
            color=color,
        )

    ax = plt.gca()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25, linestyle="--")
    _apply_legend(ax, loc=legend_loc, show=show_legend, legend_kwargs=legend_kwargs)
    plt.tight_layout()

    if save_png:
        plt.savefig(png_path, dpi=dpi, bbox_inches="tight", transparent=True)
    if save_pdf:
        plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight", transparent=True)


def heatmap_contacts_vs_kills(
    out,
    *,
    use_total_contacts: bool = True,
    color=None,
    mode: str = "frequency",
    bins_x: int | None = None,
    bins_y: int | None = None,
    x_max: int | None = None,
    y_max: int | None = None,
    square_range: bool = True,
    figsize=(9.2, 7.6),
    dpi=300,
    cmap="YlGnBu",
    cmap_min: float = 0.3,
    cmap_max: float = 0.9,
    density: bool = False,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Kills per killer cell",
    show_hdi_mean: bool = True,
    zero_color: str = "0.9",
    cbar_label_inside: bool = True,
    cbar_label: str | None = None,
    save_png=False,
    png_path="contacts_vs_kills_heatmap.png",
    save_pdf=False,
    pdf_path="contacts_vs_kills_heatmap.pdf",
):
    x_nonlethal = np.asarray(out["x"], dtype=float)
    y_kills = np.asarray(out["y"], dtype=float)
    if use_total_contacts:
        x_contacts = x_nonlethal + y_kills
        if xlabel is None:
            xlabel = "Contacts per killer cell"
    else:
        x_contacts = x_nonlethal
        if xlabel is None:
            xlabel = "Non-lethal contacts per killer cell"

    if x_contacts.size == 0:
        raise ValueError("Empty x/y data: out['x'] and out['y'] must contain per-killer counts")

    x_int = np.asarray(np.round(x_contacts), dtype=int)
    y_int = np.asarray(np.round(y_kills), dtype=int)

    if x_max is None:
        x_max = int(np.max(x_int))
    if y_max is None:
        y_max = int(np.max(y_int))
    x_max = max(0, int(x_max))
    y_max = max(0, int(y_max))

    # Optionally expand the smaller axis so both x/y share the same range.
    # This keeps the heatmap square in data coordinates (useful when comparing marginals).
    if square_range:
        m = max(x_max, y_max)
        x_max = m
        y_max = m

    if bins_x is None:
        bins_x = x_max + 1
    if bins_y is None:
        bins_y = y_max + 1
    bins_x = int(max(1, bins_x))
    bins_y = int(max(1, bins_y))

    x_edges = np.linspace(-0.5, x_max + 0.5, bins_x + 1)
    y_edges = np.linspace(-0.5, y_max + 0.5, bins_y + 1)

    mode = str(mode).strip().lower()
    if density and mode != "count":
        raise ValueError("Use either density=True or mode in {'count','frequency'}, not both")
    if mode not in {"count", "frequency"}:
        raise ValueError("mode must be 'count' or 'frequency'")

    H, xedges, yedges = np.histogram2d(x_int, y_int, bins=[x_edges, y_edges], density=density)
    H = H.T  # (y, x) for imshow
    if mode == "frequency":
        total = float(np.sum(H))
        if total > 0:
            H = H / total

    # Make ALL zero cells fully transparent.
    positives = H[H > 0]
    if positives.size:
        vmin = float(np.min(positives))
        vmax = float(np.max(H))
        if vmax <= 0:
            vmax = vmin
    else:
        vmin = 0.0
        vmax = 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    H_plot = np.ma.masked_where(H == 0, H)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    # Layout columns: [colorbar | blank gap | main heatmap | right marginal]
    # The blank gap keeps things visually clean without letting Matplotlib resize 8ax_main.
    gs = gridspec.GridSpec(
        2,
        4,
        width_ratios=[0.22, 0.38, 4.0, 1.4],
        height_ratios=[1.4, 4.0],
        wspace=0.08,
        hspace=0.08,
    )

    # Reserve a dedicated column for the left colorbar.
    # This prevents Matplotlib from shrinking ax_main (which would desynchronise ax_top vs ax_main widths).
    ax_cbar = fig.add_subplot(gs[1, 0])
    ax_gap = fig.add_subplot(gs[1, 1])
    ax_gap.axis("off")

    # Do NOT share axes here: Matplotlib forbids adjustable='datalim' when axes are shared.
    # We align marginals by explicitly syncing limits instead.
    ax_top = fig.add_subplot(gs[0, 2])
    ax_main = fig.add_subplot(gs[1, 2])
    ax_right = fig.add_subplot(gs[1, 3])

    ax_cbar_top = fig.add_subplot(gs[0, 0])
    ax_cbar_top.axis("off")
    ax_gap_top = fig.add_subplot(gs[0, 1])
    ax_gap_top.axis("off")
    ax_corner = fig.add_subplot(gs[0, 3])
    ax_corner.axis("off")

    # Colormap (optionally truncated to avoid full spectrum).
    cmap_min = float(cmap_min)
    cmap_max = float(cmap_max)
    if not (0.0 <= cmap_min <= 1.0 and 0.0 <= cmap_max <= 1.0 and cmap_min <= cmap_max):
        raise ValueError("cmap_min/cmap_max must satisfy 0<=cmap_min<=cmap_max<=1")

    base_cmap = plt.get_cmap(cmap)
    if (cmap_min, cmap_max) != (0.0, 1.0):
        xs = np.linspace(cmap_min, cmap_max, 256)
        cmap_obj = mcolors.LinearSegmentedColormap.from_list(
            f"{base_cmap.name}_trunc_{cmap_min:.2f}_{cmap_max:.2f}",
            base_cmap(xs),
        )
    else:
        cmap_obj = base_cmap

    try:
        cmap_obj = cmap_obj.copy()
    except Exception:
        # Older Matplotlib: fallback without copying.
        pass
    try:
        # 'bad' covers masked values; 'under' is a safety net for any values < vmin.
        cmap_obj.set_bad(color=(0, 0, 0, 0))
        cmap_obj.set_under(color=(0, 0, 0, 0))
    except Exception:
        pass

    im = ax_main.imshow(
        H_plot,
        origin="lower",
        aspect="auto",
        cmap=cmap_obj,
        norm=norm,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        interpolation="nearest",
    )

    # Colorbar on the left, in a dedicated axis (so the main panel stays aligned with the top marginal).
    cbar = fig.colorbar(im, cax=ax_cbar)
    try:
        cbar.ax.yaxis.set_label_position("left")
        cbar.ax.yaxis.set_ticks_position("left")
    except Exception:
        pass

    if cbar_label is None:
        if density:
            cbar_label = "Density"
        else:
            cbar_label = "Frequency" if mode == "frequency" else "Count"

    if cbar_label_inside:
        # Draw label inside the colorbar axis (avoids outside overlap).
        cbar.set_label("")
        cbar.ax.text(
            0.5,
            0.5,
            str(cbar_label),
            transform=cbar.ax.transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=10,
            color="white",
            bbox=dict(facecolor="none", edgecolor="none", alpha=0.0, pad=2.0),
        )
    else:
        cbar.set_label(str(cbar_label))

    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    # Force integer ticks starting at 0 so we never display negative labels.
    ax_main.set_xticks(np.arange(0, int(x_max) + 1, 1))
    ax_main.set_yticks(np.arange(0, int(y_max) + 1, 1))
    ax_main.grid(True, alpha=0.15, linestyle="--")

    if title:
        ax_top.set_title(title)

    # Marginals (match trajectory colour by default).
    marginal_color = color if color is not None else "0.2"
    ax_top.hist(x_int, bins=x_edges, color=marginal_color, alpha=0.35, edgecolor="none", linewidth=0)
    ax_right.hist(y_int, bins=y_edges, orientation="horizontal", color=marginal_color, alpha=0.35, edgecolor="none", linewidth=0)

    # Keep marginals tightly aligned to the heatmap extents (avoid autoscale padding).
    # Also ensure square bins without resizing the axes box (which would desynchronise ax_top vs ax_main).
    xlim0 = (-0.5, float(x_max) + 0.5)
    ylim0 = (-0.5, float(y_max) + 0.5)
    ax_main.set_xlim(*xlim0)
    ax_main.set_ylim(*ylim0)
    ax_main.set_aspect("equal", adjustable="datalim")
    # Apply aspect adjustment now, then sync the resulting limits to the marginal axes.
    try:
        ax_main.apply_aspect()
    except Exception:
        pass
    xlim = ax_main.get_xlim()
    ylim = ax_main.get_ylim()
    ax_top.set_xlim(*xlim)
    ax_right.set_ylim(*ylim)
    ax_top.margins(x=0)
    ax_right.margins(y=0)

    if show_hdi_mean:
        stat_color = color if color is not None else "tab:red"
        mx = float(np.mean(x_int))
        my = float(np.mean(y_int))
        (lx, ux) = _hdi(x_int, 0.95)
        (ly, uy) = _hdi(y_int, 0.95)

        ax_top.axvspan(lx, ux, color=stat_color, alpha=0.18)
        ax_top.axvline(mx, color=stat_color, lw=2)
        ax_right.axhspan(ly, uy, color=stat_color, alpha=0.18)
        ax_right.axhline(my, color=stat_color, lw=2)

        ax_top.text(
            0.99,
            0.95,
            f"mean={mx:.2f}\nHDI95%=[{lx:.0f},{ux:.0f}]",
            transform=ax_top.transAxes,
            ha="right",
            va="top",
            fontsize=10,
        )
        ax_right.text(
            0.05,
            0.99,
            f"mean={my:.2f}\nHDI95%=[{ly:.0f},{uy:.0f}]",
            transform=ax_right.transAxes,
            ha="left",
            va="top",
            rotation=270,
            fontsize=10,
        )

    # Clean marginal axes
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_top.set_ylabel("Count")
    ax_top.grid(True, alpha=0.15, linestyle="--")

    plt.setp(ax_right.get_yticklabels(), visible=False)
    ax_right.set_xlabel("Count")
    ax_right.grid(True, alpha=0.15, linestyle="--")

    # (Limits already set above to keep everything aligned.)
    # Avoid tight_layout warnings for this multi-axes layout; GridSpec spacing controls layout.

    if save_png:
        plt.savefig(png_path, dpi=dpi, bbox_inches="tight", transparent=True)
    if save_pdf:
        plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight", transparent=True)

    return {
        "x_mean": float(np.mean(x_int)),
        "x_hdi95": _hdi(x_int, 0.95),
        "y_mean": float(np.mean(y_int)),
        "y_hdi95": _hdi(y_int, 0.95),
        "x_max": int(x_max),
        "y_max": int(y_max),
    }