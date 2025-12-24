#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
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


def prob_matrix(
    mode="constant",
    p0=0.5,
    alpha=0.3,
    beta=0.3,
    max_event=20,
    clip=True,
):
    if mode == "constant":
        p = np.full((max_event, max_event), float(p0), dtype=float)
    elif mode == "history_exp":
        i = np.arange(max_event)[:, None]
        j = np.arange(max_event)[None, :]
        p0_safe = np.clip(float(p0), 1e-6, 1.0 - 1e-6)
        logit_p0 = np.log(p0_safe) - np.log1p(-p0_safe)
        lin = logit_p0 + float(alpha) * i + float(beta) * j
        p = 1.0 / (1.0 + np.exp(-lin))
    else:
        raise ValueError(f"Unknown prob mode: {mode}")

    if clip:
        p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return p


def rate_matrix(mode="constant", rate0=1.0, max_event=20):
    if mode != "constant":
        raise ValueError(f"Unknown rate mode: {mode}")
    return np.full((max_event, max_event), float(rate0), dtype=float)


def killing_capacities_matrix(
    n_killers,
    mode="homogeneous",
    K0=3.0,
    sigma=1.0,
    proportions=None,
    community_K=None,
    min_K=0,
    max_K=None,
    seed=None,
):
    rng = np.random.default_rng(seed)
    n = int(n_killers)
    if n <= 0:
        raise ValueError("n_killers must be positive")

    mean = float(K0)
    sigma = float(sigma)

    if mode == "homogeneous":
        K = np.full(n, int(round(mean)), dtype=int)
    elif mode == "normal":
        vals = rng.normal(loc=mean, scale=sigma, size=n)
        K = np.rint(vals).astype(int)
    elif mode == "lognormal":
        mu = np.log(max(mean, 1e-8))
        vals = rng.lognormal(mean=mu, sigma=sigma, size=n)
        K = np.rint(vals).astype(int)
    elif mode == "gamma":
        if mean <= 0:
            raise ValueError("K0 must be > 0 for gamma mode")
        if sigma <= 0:
            raise ValueError("sigma must be > 0 for gamma mode")
        shape = (mean / sigma) ** 2
        scale = (sigma**2) / mean
        vals = rng.gamma(shape=float(shape), scale=float(scale), size=n)
        K = np.rint(vals).astype(int)
    elif mode == "discrete":
        if proportions is None or community_K is None:
            raise ValueError("discrete mode requires proportions and community_K")

        props = np.asarray(proportions, dtype=float)
        commK = np.asarray(community_K, dtype=int)
        if props.ndim != 1 or commK.ndim != 1:
            raise ValueError("proportions and community_K must be 1D sequences")
        if props.size == 0:
            raise ValueError("proportions must be non-empty")
        if props.size != commK.size:
            raise ValueError("proportions and community_K must have the same length")
        if np.any(~np.isfinite(props)):
            raise ValueError("proportions must be finite")
        if np.any(props < 0):
            raise ValueError("proportions must be non-negative")
        s = float(np.sum(props))
        if s <= 0:
            raise ValueError("proportions must sum to a positive value")
        props = props / s

        comm_idx = rng.choice(props.size, size=n, p=props)
        K = commK[comm_idx].astype(int, copy=False)
    else:
        raise ValueError(f"Unknown capacity mode: {mode}")

    if min_K is not None:
        K = np.maximum(K, int(min_K))
    if max_K is not None:
        K = np.minimum(K, int(max_K))
    return K


def sim_traj_global(
    n_killers,
    r_matrix,
    p_kill_matrix=None,
    p_mode="deterministic",
    p_stochastic_mode="constant",
    p0=0.5,
    alpha=0.3,
    beta=0.3,
    capacities=None,
    capacity_mode="homogeneous",
    capacity_kwargs=None,
    max_time=10.0,
    max_event=20,
    seed=None,
    n_targets=None,
    target_multiplier=4.0,
    target_rate_floor=0.0,
):
    rng = np.random.default_rng(seed)
    n_killers = int(n_killers)
    if n_killers <= 0:
        raise ValueError("n_killers must be positive")

    if n_targets is None:
        n_targets = int(round(float(target_multiplier) * n_killers))
    n_targets = int(n_targets)
    if n_targets < 0:
        raise ValueError("n_targets must be non-negative")

    T0 = max(n_targets, 1)
    T = n_targets

    if capacities is None:
        kw = {} if capacity_kwargs is None else dict(capacity_kwargs)
        capacities = killing_capacities_matrix(
            n_killers=n_killers,
            mode=capacity_mode,
            seed=seed,
            **kw,
        )
    else:
        capacities = np.asarray(capacities, dtype=int)
        if capacities.shape != (n_killers,):
            raise ValueError("capacities must have shape (n_killers,)")

    x = np.zeros(n_killers, dtype=int)
    y = np.zeros(n_killers, dtype=int)
    K_rem = capacities.copy()
    t = 0.0

    alive = np.ones(n_killers, dtype=bool)
    decisions_list = [[] for _ in range(n_killers)]
    times_list = [[] for _ in range(n_killers)]

    if p_kill_matrix is None:
        if p_mode == "deterministic":
            p_kill_matrix = np.ones((max_event, max_event), dtype=float)
        elif p_mode == "stochastic":
            p_kill_matrix = prob_matrix(
                mode=p_stochastic_mode,
                p0=p0,
                alpha=alpha,
                beta=beta,
                max_event=max_event,
                clip=True,
            )
        else:
            raise ValueError(f"Unknown p_mode: {p_mode}")
    else:
        p_kill_matrix = np.asarray(p_kill_matrix, dtype=float)

    T_trace = [T]
    t_trace = [t]

    while True:
        if T <= 0:
            break

        alive = alive & (x < max_event - 1) & (y < max_event - 1)
        if not np.any(alive):
            break

        idx = np.where(alive)[0]
        base_rates = r_matrix[x[idx], y[idx]]
        base_rates = np.where(np.isfinite(base_rates) & (base_rates > 0), base_rates, 0.0)

        target_factor = max(T / T0, float(target_rate_floor))
        rates = base_rates * target_factor

        total_rate = float(np.sum(rates))
        if total_rate <= 0:
            break

        dt = float(rng.exponential(1.0 / total_rate))
        if t + dt > max_time:
            break
        t += dt

        probs = rates / total_rate
        i = int(idx[rng.choice(len(idx), p=probs)])

        pi = float(p_kill_matrix[x[i], y[i]])
        if K_rem[i] <= 0:
            pi = 0.0

        if (rng.random() < pi) and (T > 0):
            y[i] += 1
            K_rem[i] -= 1
            T -= 1
            decisions_list[i].append(1)
        else:
            x[i] += 1
            decisions_list[i].append(0)

        times_list[i].append(t)
        T_trace.append(T)
        t_trace.append(t)

    return {
        "decisions_list": [np.array(d, dtype=int) for d in decisions_list],
        "times_list": [np.array(ts, dtype=float) for ts in times_list],
        "x": x,
        "y": y,
        "capacities": capacities,
        "K_remaining": K_rem,
        "t_end": t,
        "n_targets_init": n_targets,
        "targets_remaining": T,
        "targets_trace": np.array(T_trace, dtype=int),
        "targets_times": np.array(t_trace, dtype=float),
        "target_factor_end": max(T / T0, float(target_rate_floor)),
    }


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


def scenario_label(sc, n_killers, n_targets, rate0):
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
    if sc["p_mode"] == "stochastic":
        p_bits.append(f"{sc['p_stochastic_mode']}")
        p_bits.append(f"p0={_fmt_float(sc['p0'],2)}")
        p_bits.append(f"a={_fmt_float(sc['alpha'],2)}")
        p_bits.append(f"b={_fmt_float(sc['beta'],2)}")

    top = " | ".join(p_bits)
    bottom = f"{' , '.join(cap_bits)} | Nkill={int(n_killers)} Ntarg={int(n_targets)} r0={_fmt_float(rate0,2)}"
    return top + "\n" + bottom


def _pick_colors(n, cmap_name="YlGnBu"):
    cmap = plt.get_cmap(cmap_name)
    if n <= 1:
        return [cmap(0.75)]
    xs = np.linspace(0.35, 0.9, n)
    return [cmap(x) for x in xs]


def _apply_legend(ax, loc="upper left"):
    ax.legend(
        loc=loc,
        frameon=True,
        edgecolor="black",
        fontsize=10,
        handlelength=2.0,
        borderpad=0.6,
        labelspacing=0.6,
    )


def decision_map(
    decision_data_group,
    figsize=(9.2, 8.6),
    dpi=300,
    offset_scale=4.0,
    alpha=0.5,
    lw=2.0,
    s=18.0,
    edgecolor="black",
    save_png=False,
    png_path="trajectories.png",
    save_pdf=False,
    pdf_path="trajectories.pdf",
):
    plt.figure(figsize=figsize, dpi=dpi)

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

    plt.xlabel("Non-lethal contacts")
    plt.ylabel("Lethal contacts")
    plt.grid(True, alpha=0.25, linestyle="--")
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    _apply_legend(ax, loc="upper left")
    plt.tight_layout()

    if save_png:
        plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight")


def plot_targets_remaining(
    outs,
    labels,
    colors,
    figsize=(9.2, 4.6),
    dpi=300,
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

    plt.xlabel("Time")
    plt.ylabel("Targets remaining")
    plt.grid(True, alpha=0.25, linestyle="--")
    ax = plt.gca()
    _apply_legend(ax, loc="upper left")
    plt.tight_layout()

    if save_png:
        plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if save_pdf:
        plt.savefig(pdf_path, dpi=dpi, bbox_inches="tight")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Multi-killer Gillespie simulator with target depletion + trajectory plots."
    )

    p.add_argument("--outdir", type=str, default=".", help="Output directory for plots.")
    p.add_argument("--seed", type=int, default=123, help="Base RNG seed.")
    p.add_argument("--n-killers", type=int, default=50, help="Number of killers.")
    p.add_argument("--target-multiplier", type=float, default=4.0, help="Targets = multiplier * killers if --n-targets not set.")
    p.add_argument("--n-targets", type=int, default=None, help="Explicit number of targets (overrides multiplier).")
    p.add_argument("--target-rate-floor", type=float, default=0.0, help="Minimum encounter scaling factor as targets deplete.")
    p.add_argument("--max-event", type=int, default=15, help="Max event count per dimension (x,y grid size).")
    p.add_argument("--max-time", type=float, default=8.0, help="Simulation time horizon.")
    p.add_argument("--rate0", type=float, default=0.5, help="Base constant encounter rate.")

    p.add_argument("--save-png", type=_parse_bool, default=True, help="Save PNG plots.")
    p.add_argument("--save-pdf", type=_parse_bool, default=False, help="Save PDF plots.")

    p.add_argument("--cmap", type=str, default="YlGnBu", help="Matplotlib colormap for scenario colours (default: YlGnBu).")

    p.add_argument(
        "--scenario",
        type=str,
        choices=["default", "stoch_homo", "det_hetero", "stoch_hetero", "all"],
        default="all",
        help="Which scenario(s) to run.",
    )

    p.add_argument("--p-mode", type=str, choices=["deterministic", "stochastic"], default=None, help="Override p_mode (only if --scenario=default).")
    p.add_argument("--p-stochastic-mode", type=str, choices=["constant", "history_exp"], default=None, help="Override p_stochastic_mode (only if --scenario=default).")
    p.add_argument("--p0", type=float, default=None, help="Override p0 (only if --scenario=default).")
    p.add_argument("--alpha", type=float, default=None, help="Override alpha (only if --scenario=default).")
    p.add_argument("--beta", type=float, default=None, help="Override beta (only if --scenario=default).")

    p.add_argument("--capacity-mode", type=str, choices=["homogeneous", "normal", "lognormal", "gamma", "discrete"], default=None, help="Override capacity mode (only if --scenario=default).")
    p.add_argument("--K0", type=float, default=None, help="Override K0 (only if --scenario=default).")
    p.add_argument("--sigma", type=float, default=None, help="Override sigma (only if --scenario=default and mode uses sigma).")
    p.add_argument("--community-proportions", type=_parse_float_list, default=None, help="Comma-separated proportions for discrete capacity mode, e.g. '0.7,0.3'.")
    p.add_argument("--community-K", type=_parse_int_list, default=None, help="Comma-separated capacities per community for discrete capacity mode, e.g. '2,5'.")
    p.add_argument("--min-K", type=int, default=None, help="Override min_K (only if --scenario=default).")
    p.add_argument("--max-K", type=int, default=None, help="Override max_K (only if --scenario=default).")

    p.add_argument("--traj-alpha", type=float, default=0.5, help="Trajectory line alpha.")
    p.add_argument("--traj-lw", type=float, default=2.0, help="Trajectory line width.")
    p.add_argument("--traj-s", type=float, default=18.0, help="Trajectory endpoint marker size.")
    p.add_argument("--offset-scale", type=float, default=4.0, help="Trajectory jitter scale.")
    p.add_argument("--dpi", type=int, default=300, help="Plot DPI.")
    p.add_argument("--traj-figsize", type=float, nargs=2, default=(9.2, 8.6), help="Trajectory figure size (w h).")
    p.add_argument("--targets-figsize", type=float, nargs=2, default=(9.2, 4.6), help="Targets figure size (w h).")

    return p


def main_from_args(args: argparse.Namespace):
    os.makedirs(args.outdir, exist_ok=True)
    rmat = rate_matrix(mode="constant", rate0=args.rate0, max_event=args.max_event)

    default_scenarios = [
        dict(
            key="stoch_homo",
            p_mode="stochastic",
            p_stochastic_mode="history_exp",
            p0=0.30, alpha=0.25, beta=0.15,
            capacity_mode="homogeneous",
            capacity_kwargs={"K0": 3.0, "min_K": 0},
        ),
        dict(
            key="det_hetero",
            p_mode="deterministic",
            p_stochastic_mode="constant",
            p0=0.50, alpha=0.0, beta=0.0,
            capacity_mode="gamma",
            capacity_kwargs={"K0": 3.0, "sigma": 1.2, "min_K": 0},
        ),
        dict(
            key="stoch_hetero",
            p_mode="stochastic",
            p_stochastic_mode="history_exp",
            p0=0.25, alpha=0.30, beta=0.20,
            capacity_mode="discrete",
            capacity_kwargs={"proportions": [0.5, 0.3, 0.2], "community_K": [10, 5, 1], "min_K": 0},
        ),
    ]

    if args.scenario == "default":
        p_mode = args.p_mode or "stochastic"
        p_stoch_mode = args.p_stochastic_mode or "history_exp"
        p0 = 0.30 if args.p0 is None else args.p0
        alpha = 0.25 if args.alpha is None else args.alpha
        beta = 0.15 if args.beta is None else args.beta

        cap_mode = args.capacity_mode or "homogeneous"
        cap_kwargs = {}
        if args.K0 is not None:
            cap_kwargs["K0"] = args.K0
        if args.sigma is not None:
            cap_kwargs["sigma"] = args.sigma
        if args.community_proportions is not None:
            cap_kwargs["proportions"] = args.community_proportions
        if args.community_K is not None:
            cap_kwargs["community_K"] = args.community_K
        if args.min_K is not None:
            cap_kwargs["min_K"] = args.min_K
        if args.max_K is not None:
            cap_kwargs["max_K"] = args.max_K
        if "K0" not in cap_kwargs:
            cap_kwargs["K0"] = 3.0
        if "min_K" not in cap_kwargs:
            cap_kwargs["min_K"] = 0

        if cap_mode == "discrete":
            if ("proportions" not in cap_kwargs) or ("community_K" not in cap_kwargs):
                raise ValueError(
                    "capacity_mode=discrete requires --community-proportions and --community-K (or set them in capacity_kwargs)"
                )

        scenarios = [dict(
            key="default",
            p_mode=p_mode,
            p_stochastic_mode=p_stoch_mode,
            p0=p0, alpha=alpha, beta=beta,
            capacity_mode=cap_mode,
            capacity_kwargs=cap_kwargs,
        )]
    elif args.scenario == "all":
        scenarios = default_scenarios
    else:
        scenarios = [s for s in default_scenarios if s["key"] == args.scenario]

    colors = _pick_colors(len(scenarios), cmap_name=args.cmap)

    outs = []
    groups = []
    labels = []

    for k, (sc, color) in enumerate(zip(scenarios, colors)):
        out = sim_traj_global(
            n_killers=args.n_killers,
            r_matrix=rmat,
            p_mode=sc["p_mode"],
            p_stochastic_mode=sc["p_stochastic_mode"],
            p0=sc["p0"],
            alpha=sc["alpha"],
            beta=sc["beta"],
            capacity_mode=sc["capacity_mode"],
            capacity_kwargs=sc["capacity_kwargs"],
            max_time=args.max_time,
            max_event=args.max_event,
            seed=args.seed + 17 * k,
            n_targets=args.n_targets,
            target_multiplier=args.target_multiplier,
            target_rate_floor=args.target_rate_floor,
        )

        lab = scenario_label(sc, args.n_killers, out["n_targets_init"], args.rate0)
        outs.append(out)
        labels.append(lab)
        groups.append((out["decisions_list"], lab, color))

    decision_map(
        decision_data_group=groups,
        figsize=tuple(args.traj_figsize),
        dpi=args.dpi,
        offset_scale=args.offset_scale,
        alpha=args.traj_alpha,
        lw=args.traj_lw,
        s=args.traj_s,
        save_png=args.save_png,
        png_path=os.path.join(args.outdir, "trajectories_groups.png"),
        save_pdf=args.save_pdf,
        pdf_path=os.path.join(args.outdir, "trajectories_groups.pdf"),
    )

    plot_targets_remaining(
        outs=outs,
        labels=labels,
        colors=colors,
        figsize=tuple(args.targets_figsize),
        dpi=args.dpi,
        save_png=args.save_png,
        png_path=os.path.join(args.outdir, "targets_remaining.png"),
        save_pdf=args.save_pdf,
        pdf_path=os.path.join(args.outdir, "targets_remaining.pdf"),
    )

    return outs


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    main_from_args(args)
