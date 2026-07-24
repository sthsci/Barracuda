DECISION_CATEGORY_ORDER = [
    "No interaction",
    "No kill",
    "Exhausted",
    "Stochastic",
    "Serial-killing",
]

FT_PAPER = "#FFF1E5"
FT_TEXT = "#262A33"
FT_GRID = "#D8CDBF"
FT_SPINE = "#8A7D70"

FT_DECISION_COLORS = {
    "No interaction": "#9B9A91",
    "No kill": "#0F6B78",
    "Exhausted": "#C98B2B",
    "Stochastic": "#990F3D",
    "Serial-killing": "#E95D3C",
}


def classify_decision_history(history):
    h = tuple(int(z) for z in history)

    if len(h) == 0:
        return "No interaction"
    if sum(h) == 0:
        return "No kill"
    if sum(h) == len(h):
        return "Serial-killing"

    first_nonlethal = h.index(0)

    if all(z == 0 for z in h[first_nonlethal:]):
        return "Exhausted"

    return "Stochastic"


def decision_history_to_path(history):
    history = np.asarray(history, dtype=int)
    path = np.zeros((history.size + 1, 2), dtype=float)

    for i, z in enumerate(history, start=1):
        path[i] = path[i - 1]

        if z == 1:
            path[i, 1] += 1
        else:
            path[i, 0] += 1

    return path


def no_treatment_categorised_frame(cells, condition="No treatment"):
    df = cells.loc[cells["condition_label"].eq(condition)].copy()

    df["decision_category"] = pd.Categorical(
        [classify_decision_history(h) for h in df["history"]],
        categories=DECISION_CATEGORY_ORDER,
        ordered=True,
    )
    df["n_lethal"] = [int(np.sum(h)) for h in df["history"]]
    df["n_nonlethal"] = [int(len(h) - np.sum(h)) for h in df["history"]]

    return df


def seeded_stratified_subsample(df, *, seed=20260710, max_per_category=25, max_cells=None):
    rng = np.random.default_rng(seed)

    if max_per_category is not None:
        parts = []

        for category in DECISION_CATEGORY_ORDER:
            sub = df.loc[df["decision_category"].eq(category)]

            if sub.empty:
                continue

            take = min(max_per_category, len(sub))
            chosen = rng.choice(sub.index.to_numpy(), size=take, replace=False)
            parts.append(df.loc[chosen])

        if not parts:
            return df.iloc[0:0].copy()

        return pd.concat(parts, axis=0).sample(frac=1, random_state=seed).reset_index(drop=True)

    if max_cells is None or len(df) <= max_cells:
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    chosen = rng.choice(df.index.to_numpy(), size=max_cells, replace=False)
    return df.loc[chosen].sample(frac=1, random_state=seed).reset_index(drop=True)


def plot_no_treatment_categorised_decision_map(
    cells,
    *,
    seed=20260710,
    max_per_category=15,
    condition="No treatment",
    figure_root=FIGURE_ROOT,
    figsize=(6.6, 6.4),
    dpi=300,
    offset_scale=4.5,
    alpha=0.58,
    lw=1.8,
    s=22,
    axis_max=None,
    show=False,
):
    no_treatment = no_treatment_categorised_frame(cells, condition=condition)

    sampled = seeded_stratified_subsample(
        no_treatment,
        seed=seed,
        max_per_category=max_per_category,
    )

    stats = pd.DataFrame({"decision_category": DECISION_CATEGORY_ORDER})
    total_counts = (
        no_treatment["decision_category"]
        .astype(str)
        .value_counts()
        .rename_axis("decision_category")
        .rename("n_total")
        .reset_index()
    )
    sampled_counts = (
        sampled["decision_category"]
        .astype(str)
        .value_counts()
        .rename_axis("decision_category")
        .rename("n_sampled")
        .reset_index()
    )

    stats = (
        stats.merge(total_counts, on="decision_category", how="left")
        .merge(sampled_counts, on="decision_category", how="left")
        .fillna({"n_total": 0, "n_sampled": 0})
    )
    stats["n_total"] = stats["n_total"].astype(int)
    stats["n_sampled"] = stats["n_sampled"].astype(int)
    stats["total_fraction"] = stats["n_total"] / max(len(no_treatment), 1)
    stats["sampled_fraction"] = stats["n_sampled"] / max(len(sampled), 1)
    stats.insert(0, "condition", condition)
    stats.insert(1, "seed", seed)

    if no_treatment.empty:
        return None, stats

    max_count = int(no_treatment["contact_count"].max()) if axis_max is None else int(axis_max)

    with plt.rc_context(
        {
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 10,
        }
    ):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor(FT_PAPER)
        ax.set_facecolor(FT_PAPER)
        ax.set_axisbelow(True)

        for category_idx, category in enumerate(DECISION_CATEGORY_ORDER):
            sub = sampled.loc[sampled["decision_category"].eq(category)]
            sub = sub.sample(frac=1, random_state=seed + category_idx).reset_index(drop=True)
            n_cells = len(sub)

            if n_cells == 0:
                continue

            color = FT_DECISION_COLORS[category]

            for i, row in sub.iterrows():
                path = decision_history_to_path(row["history"])
                offset = (i - (n_cells - 1) / 2) / (offset_scale * max(n_cells, 1))
                path = path + offset

                ax.plot(
                    path[:, 0],
                    path[:, 1],
                    color=color,
                    alpha=alpha,
                    lw=lw,
                    solid_capstyle="round",
                    zorder=2,
                )
                ax.scatter(
                    path[-1, 0],
                    path[-1, 1],
                    color=color,
                    s=s,
                    edgecolor=FT_TEXT,
                    linewidth=0.45,
                    zorder=4,
                )

        handles = []

        for category in DECISION_CATEGORY_ORDER:
            row = stats.loc[stats["decision_category"].eq(category)].iloc[0]

            if row["n_total"] == 0:
                continue

            handles.append(
                plt.Line2D(
                    [],
                    [],
                    color=FT_DECISION_COLORS[category],
                    lw=5,
                    label=f"{category}",
                )
            )

        ax.legend(
            handles=handles,
            # title="Category",
            loc="upper right",
            frameon=True,
            facecolor=FT_PAPER,
            edgecolor=FT_GRID,
            title_fontsize=11,
        )


        # ax.set_title(f"{condition} decision trajectories", loc="left", color=FT_TEXT, pad=10)
        ax.set_xlabel("Non-lethal contacts", color=FT_TEXT)
        ax.set_ylabel("Lethal contacts", color=FT_TEXT)
        ax.set_xlim(-0.45, 6)
        ax.set_ylim(-0.45, 6)
        ax.set_aspect("equal")
        ax.grid(True, color=FT_GRID, linewidth=0.8, alpha=0.8)

        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.tick_params(axis="both", colors=FT_TEXT, width=0.8, length=4)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(which="minor", bottom=False, left=False, top=False, right=False)
        
        for spine in ax.spines.values():
            spine.set_color(FT_SPINE)
            spine.set_linewidth(0.9)

        figure_root.mkdir(parents=True, exist_ok=True)
        out_path = figure_root / "decision_trajectories_no_treatment_categorised_ft.svg"
        png_path = figure_root / "decision_trajectories_no_treatment_categorised_ft.png"

        fig.savefig(out_path, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
        fig.savefig(png_path, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())

        if show:
            plt.show()
        else:
            plt.close(fig)

    return out_path, stats


no_treatment_decision_path, no_treatment_decision_stats = plot_no_treatment_categorised_decision_map(
    cells,
    seed=20260710,
    max_per_category=15,
    show=True,
)

display(no_treatment_decision_stats)
no_treatment_decision_path