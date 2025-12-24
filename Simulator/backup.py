def inference_one(
    decisions_list,
    draws=100000,
    tune=3000,
    chains=8,
    target_accept=0.98,
    cores=None,
    random_seed=None,
):
    x_hist, y_hist, decisions_flat = build_history_arrays(decisions_list)

    with pm.Model() as model:
        p0 = pm.Beta("p0", alpha=1.5, beta=1.5)
        alpha = pm.HalfNormal("alpha", sigma=0.5)
        beta = pm.HalfNormal("beta", sigma=0.5)

        x_hist_shared = pm.Data("x_hist", x_hist.astype("float64"))
        y_hist_shared = pm.Data("y_hist", y_hist.astype("float64"))

        logit_p0 = pm.math.logit(p0)
        lin = logit_p0 + alpha * x_hist_shared + beta * y_hist_shared
        p = pm.math.sigmoid(lin)

        pm.Bernoulli(
            "obs",
            p=p,
            observed=decisions_flat,
        )

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            cores=(cores or min(chains, os.cpu_count() or 1)),
        )

    return model, idata


def decision_map_general(
    decision_data_group,
    figsize=(6, 6),
    dpi=300,
    offset_scale=7,
    alpha=0.6,
    lw=2,
    s=18,
    zorder=5,
    edgecolor="black",
    PDF_path=None,
    PNG_path=None,
):
    plt.figure(figsize=figsize, dpi=dpi)
    max_x, max_y = 0, 0

    for group_idx, (group, label, color) in enumerate(decision_data_group):
        n_cells = len(group)
        for i, traj in enumerate(group):
            x, y = _coerce_to_xy(traj)
            offset = (i - n_cells / 2) / (offset_scale * n_cells)
            x = x + offset
            y = y + offset
            plt.plot(x, y, alpha=alpha, color=color, lw=lw)
            plt.scatter(
                x[-1],
                y[-1],
                color=color,
                s=s,
                zorder=zorder,
                edgecolor=edgecolor,
            )
            max_x = max(max_x, x[-1])
            max_y = max(max_y, y[-1])
        plt.plot([], [], color=color, label=label, lw=10)

    plt.xlabel("Non-lethal contacts")
    plt.ylabel("Lethal contacts")
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))

    if PDF_path:
        plt.savefig(PDF_path)
    if PNG_path:
        plt.savefig(PNG_path)


def plot_decision_posteriors(
    idatas,
    ground_truth=None,
    parameters=("p0", "alpha", "beta"),
    parameter_display=None,
    hdi_prob=0.95,
    sample_size=200000,
    save_pdf=False,
    pdf_path="decision_posteriors.pdf",
    save_png=False,
    png_path="decision_posteriors.png",
    cmap_name="YlGnBu",
    point_size=3,
    font_scale=0.7,
    diagonal_style="hist",
    marginal_style="circle",
    seed=None,
):
    sns.set_context("talk", font_scale=font_scale)
    cmap = plt.colormaps.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.3, 0.9, len(idatas)))
    rng = np.random.default_rng(seed)

    if parameter_display is None:
        parameter_display = {
            "p0": r"$P_0$",
            "alpha": r"$\alpha$",
            "beta": r"$\beta$",
        }

    label_to_df = {}
    parameters = list(parameters)
    for label, idata in idatas:
        posterior = idata.posterior
        df = pd.DataFrame()
        for p in parameters:
            if p not in posterior:
                continue
            vals = posterior[p].stack(sample=("chain", "draw")).values.ravel()
            vals = vals[np.isfinite(vals)]
            if (sample_size is not None) and (len(vals) > sample_size):
                idx = rng.choice(len(vals), sample_size, replace=False)
                vals = vals[idx]
            df[p] = vals
        if not df.empty:
            df["label"] = label
            label_to_df[label] = df

    npar = len(parameters)
    fig = plt.figure(figsize=(5 * npar, 5 * npar), dpi=250)
    gs = gridspec.GridSpec(npar, npar, wspace=0.2, hspace=0.2)
    gaxes = np.empty((npar, npar), dtype=object)

    for irow, rowpar in enumerate(parameters):
        for icol, colpar in enumerate(parameters):
            ax = plt.subplot(gs[irow, icol])
            gaxes[irow, icol] = ax

            if icol > irow:
                ax.axis("off")
                continue

            for color, (lbl, df) in zip(colors, label_to_df.items()):
                if icol == irow:
                    vals = df[rowpar].dropna().values
                    if diagonal_style == "kde":
                        sns.kdeplot(
                            vals,
                            ax=ax,
                            fill=True,
                            color=color,
                            alpha=0.2,
                            linewidth=1.5,
                            label=lbl if irow == 0 else None,
                        )
                    elif diagonal_style == "hist":
                        sns.histplot(
                            vals,
                            bins=30,
                            stat="density",
                            kde=False,
                            ax=ax,
                            color=color,
                            alpha=0.18,
                            element="step",
                            fill=True,
                        )
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
                            label=lbl if irow == 0 else None,
                        )

                    lo, hi = az.hdi(vals, hdi_prob=hdi_prob)
                    ax.axvspan(lo, hi, color=color, alpha=0.06, linewidth=0)

                    if (
                        ground_truth is not None
                        and lbl in ground_truth
                        and rowpar in ground_truth[lbl]
                    ):
                        ax.axvline(
                            ground_truth[lbl][rowpar],
                            color=color,
                            linestyle="-",
                            linewidth=1.8,
                        )

                    ax.grid(alpha=0.2)

                else:
                    if marginal_style == "circle":
                        sns.kdeplot(
                            x=df[colpar],
                            y=df[rowpar],
                            ax=ax,
                            fill=False,
                            color=color,
                            alpha=0.6,
                            levels=7,
                            linewidths=1.0,
                        )
                    elif marginal_style == "pixel":
                        sns.histplot(
                            x=df[colpar],
                            y=df[rowpar],
                            bins=60,
                            pthresh=0.01,
                            cmap=cmap_name,
                            cbar=False,
                            ax=ax,
                        )

            if icol != irow:
                ax.grid(alpha=0.3)
                for color, (lbl, df) in zip(colors, label_to_df.items()):
                    if ground_truth is not None and lbl in ground_truth:
                        if (
                            colpar in ground_truth[lbl]
                            and rowpar in ground_truth[lbl]
                        ):
                            ax.scatter(
                                ground_truth[lbl][colpar],
                                ground_truth[lbl][rowpar],
                                marker="*",
                                color=color,
                                s=80,
                                linewidths=2.0,
                                zorder=1000,
                            )

            if icol == irow:
                ax.set_xlabel(parameter_display.get(rowpar, rowpar))
                ax.set_ylabel("Density")
            else:
                ax.set_xlabel(parameter_display.get(colpar, colpar))
                ax.set_ylabel(parameter_display.get(rowpar, rowpar))

            ax.margins(0.05)

    handles, labels = gaxes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center",
            bbox_to_anchor=(0.85, 0.85),
            frameon=True,
            edgecolor="black",
            fontsize=10,
        )

    plt.tight_layout()
    if save_pdf:
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
        print(f"Saved joint posterior plot: {pdf_path}")
    if save_png:
        plt.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"Saved joint posterior plot: {png_path}")

