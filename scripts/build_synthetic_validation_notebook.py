"""Build the standalone Orca synthetic-validation demonstration notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = PROJECT_ROOT / "section_1" / "notebook" / "demo_validation_1.ipynb"
WEB_NOTEBOOK = PROJECT_ROOT / "webapp" / "assets" / "downloads" / "orca_synthetic_validation_demo.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def build_notebook():
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    }
    notebook["cells"] = [
        markdown(
            r"""
            # Orca synthetic validation: one-file demonstration

            This notebook contains the simulator, the four event-count models, PyMC Sequential Monte Carlo inference, the joint posterior figure and the Bayes-factor figure. It has no repository-relative imports and runs from top to bottom in one Jupyter session.

            The public example uses the paper model family:

            - $\mathcal{M}_{\mathrm{homo}}$: Homogeneous Poisson
            - $\mathcal{M}_{\mathrm{ZI}}$: Zero inflated Poisson
            - $\mathcal{M}_{\Gamma}$: Heterogeneous Gamma Poisson
            - $\mathcal{M}_{\mathrm{ZI}\Gamma}$: Zero inflated heterogeneous Gamma Poisson

            For cell $i$, $N_i\mid\lambda_i,T\sim\operatorname{Poisson}(\lambda_iT)$. The heterogeneous models use $\lambda_i\sim\operatorname{Gamma}(\mu_\lambda,\sigma_\lambda)$ in the mean/SD parameterisation. A fraction $\phi_0$ can have $\lambda_i=0$.
            """
        ),
        markdown(
            """
            ## 1. Setup and analysis profile

            Install the packages once if they are not already available:

            ```python
            %pip install "pymc==5.25.1" "arviz==0.22.0" "numpy==1.26.4" "pandas==2.2.3" "scipy==1.14.1" "matplotlib==3.9.3"
            ```

            The `quick` profile is intended for an interface demonstration. Change `PROFILE` to `paper` for a much larger calculation. Bayes factors are prior-sensitive, so the exact priors are kept with every output.
            """
        ),
        code(
            r"""
            from __future__ import annotations

            from pathlib import Path
            from typing import Optional

            import arviz as az
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import pymc as pm
            import pytensor.tensor as pt
            from scipy.stats import gaussian_kde

            OUTPUT_DIR = Path("orca_synthetic_validation_outputs")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            PROFILE = "quick"  # choose "quick" or "paper"
            PROFILES = {
                "quick": {"draws": 128, "chains": 1, "cores": 1},
                "paper": {"draws": 3000, "chains": 4, "cores": 4},
            }
            SAMPLING = PROFILES[PROFILE]

            N_CELLS = 100
            T = 1.0
            MU_LAMBDA = 4.0
            SIGMA_LAMBDA = 3.0
            PHI_ZERO = 0.20
            RANDOM_SEED = 2026

            # Priors used in the manuscript Methods.
            RATE_PRIOR_BOUNDS = (-1.5, 1.5)
            SIGMA_PRIOR_SCALE = 3.0
            PHI_PRIOR = (1.0, 1.0)
            SMC_THRESHOLD = 0.5
            SMC_CORRELATION_THRESHOLD = 0.01

            print("Profile:", PROFILE, SAMPLING)
            print("Outputs:", OUTPUT_DIR.resolve())
            """
        ),
        markdown(
            r"""
            ## 2. Embedded simulator

            The generating model is $\mathcal{M}_{\mathrm{ZI}\Gamma}$. The full population rate distribution is $\phi_0\delta_0+(1-\phi_0)\operatorname{Gamma}(\mu_\lambda,\sigma_\lambda)$.
            """
        ),
        code(
            r"""
            def gamma_shape_rate_from_mean_sd(mean: float, sd: float) -> tuple[float, float]:
                mean, sd = float(mean), float(sd)
                if mean <= 0 or sd <= 0:
                    raise ValueError("Gamma mean and SD must both be positive")
                return (mean / sd) ** 2, mean / sd**2


            def simulate_zero_inflated_gamma_poisson(
                n_cells: int,
                observation_time: float,
                mu_lambda: float,
                sigma_lambda: float,
                phi_zero: float,
                seed: Optional[int] = None,
            ) -> pd.DataFrame:
                rng = np.random.default_rng(seed)
                shape, rate = gamma_shape_rate_from_mean_sd(mu_lambda, sigma_lambda)
                rates = rng.gamma(shape=shape, scale=1.0 / rate, size=int(n_cells))
                structural_zero = rng.random(int(n_cells)) < float(phi_zero)
                rates[structural_zero] = 0.0
                counts = rng.poisson(rates * float(observation_time))
                return pd.DataFrame(
                    {
                        "cell_id": [f"cell_{index:04d}" for index in range(1, int(n_cells) + 1)],
                        "lambda_i": rates,
                        "structural_zero": structural_zero,
                        "count": counts.astype(int),
                    }
                )


            synthetic = simulate_zero_inflated_gamma_poisson(
                N_CELLS,
                T,
                MU_LAMBDA,
                SIGMA_LAMBDA,
                PHI_ZERO,
                RANDOM_SEED,
            )
            synthetic.to_csv(OUTPUT_DIR / "synthetic_counts_and_rates.csv", index=False)
            synthetic.head()
            """
        ),
        code(
            r"""
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
            axes[0].hist(synthetic["count"], bins=np.arange(synthetic["count"].max() + 2) - 0.5, color="#304B3D", edgecolor="white")
            axes[0].set(xlabel="Event count $N_i$", ylabel="Cells", title="Generated counts")

            positive_rates = synthetic.loc[synthetic["lambda_i"] > 0, "lambda_i"]
            axes[1].hist(positive_rates, bins=25, density=True, color="#2E7471", alpha=0.35, edgecolor="#2E7471")
            axes[1].bar([0], [PHI_ZERO], width=max(positive_rates.max() * 0.025, 0.05), color="#9A4938", label=rf"point mass $\phi_0={PHI_ZERO:g}$")
            axes[1].set(xlabel=r"Cell-specific rate $\lambda_i$", ylabel="Density / mass", title="Full population rate structure")
            axes[1].legend(frameon=False)
            plt.show()
            """
        ),
        markdown(
            r"""
            ## 3. Embedded PyMC models and SMC inference

            PyMC's native SMC progress display reports the chain, tempering stage and current $\beta$. The fitted models all use the same input counts and priors.
            """
        ),
        code(
            r"""
            MODEL_ORDER = ["homo", "z2p", "dis2p", "hetero3"]
            MODEL_LABELS = {
                "homo": r"$\mathcal{M}_{\mathrm{homo}}$",
                "z2p": r"$\mathcal{M}_{\mathrm{ZI}}$",
                "dis2p": r"$\mathcal{M}_{\Gamma}$",
                "hetero3": r"$\mathcal{M}_{\mathrm{ZI}\Gamma}$",
            }
            MODEL_COLOURS = {
                "homo": "#9BD7FF",
                "z2p": "#56B4E9",
                "dis2p": "#2F80ED",
                "hetero3": "#0B1F5B",
            }


            def gamma_shape_rate_symbolic(mu, sd):
                sd = pt.maximum(sd, 1e-12)
                variance = sd**2
                return mu**2 / variance, mu / variance


            def gamma_marginal_log_integral(n_tensor, observation_time, mu, sd, max_count):
                shape, rate = gamma_shape_rate_symbolic(mu, sd)
                log_rising = pt.zeros_like(n_tensor)
                for k in range(int(max_count)):
                    log_rising = pt.switch(
                        n_tensor > float(k),
                        log_rising + pt.log(shape + float(k)),
                        log_rising,
                    )
                return log_rising - n_tensor * pt.log(rate) - (shape + n_tensor) * pt.log1p(float(observation_time) / rate)


            def poisson_count_constant(n_tensor, observation_time):
                return n_tensor * pt.log(float(observation_time)) - pt.gammaln(n_tensor + 1.0)


            def final_log_evidence(trace, chains: int) -> np.ndarray:
                raw = trace.report.log_marginal_likelihood
                if all(not isinstance(item, (list, tuple, np.ndarray)) for item in raw):
                    values = np.asarray(raw, dtype=float)
                    values = values[np.isfinite(values)]
                    return np.full(chains, values[-1], dtype=float)
                values = []
                for item in raw:
                    array = np.asarray(item, dtype=float).reshape(-1)
                    array = array[np.isfinite(array)]
                    if len(array):
                        values.append(float(array[-1]))
                return np.asarray(values[:chains], dtype=float)


            def sample_model(model_key: str, counts: np.ndarray, observation_time: float, seed: int) -> az.InferenceData:
                counts = np.asarray(counts, dtype=int)
                n_tensor = pt.as_tensor_variable(counts.astype(float))
                max_count = int(counts.max(initial=0))

                with pm.Model() as model:
                    eta = pm.Uniform("eta", lower=RATE_PRIOR_BOUNDS[0], upper=RATE_PRIOR_BOUNDS[1])
                    if model_key in {"homo", "z2p"}:
                        rate = pm.Deterministic("lambda", 10.0**eta)
                    else:
                        rate = pm.Deterministic("mu_lambda", 10.0**eta)
                        rate_sd = pm.HalfNormal("sigma_lambda", sigma=SIGMA_PRIOR_SCALE)
                    if model_key in {"z2p", "hetero3"}:
                        phi_zero = pm.Beta("p_zero", alpha=PHI_PRIOR[0], beta=PHI_PRIOR[1])

                    if model_key == "homo":
                        pm.Poisson("counts", mu=rate * observation_time, observed=counts)
                    elif model_key == "z2p":
                        pm.ZeroInflatedPoisson("counts", psi=1.0 - phi_zero, mu=rate * observation_time, observed=counts)
                    elif model_key == "dis2p":
                        log_likelihood = poisson_count_constant(n_tensor, observation_time) + gamma_marginal_log_integral(
                            n_tensor, observation_time, rate, rate_sd, max_count
                        )
                        pm.Potential("gamma_marginal_counts", pt.sum(log_likelihood))
                    else:
                        log_active = pt.log1p(-phi_zero) + poisson_count_constant(n_tensor, observation_time) + gamma_marginal_log_integral(
                            n_tensor, observation_time, rate, rate_sd, max_count
                        )
                        log_zero = pt.logaddexp(
                            pt.log(phi_zero),
                            pt.log1p(-phi_zero) + gamma_marginal_log_integral(
                                pt.zeros_like(n_tensor), observation_time, rate, rate_sd, 0
                            ),
                        )
                        pm.Potential("zi_gamma_marginal_counts", pt.sum(pt.switch(counts == 0, log_zero, log_active)))

                    trace = pm.sample_smc(
                        draws=SAMPLING["draws"],
                        chains=SAMPLING["chains"],
                        cores=SAMPLING["cores"],
                        random_seed=seed,
                        progressbar=True,
                        return_inferencedata=False,
                        threshold=SMC_THRESHOLD,
                        correlation_threshold=SMC_CORRELATION_THRESHOLD,
                    )
                    idata = pm.to_inference_data(trace, log_likelihood=True)
                    log_evidence = final_log_evidence(trace, SAMPLING["chains"])
                    idata.attrs["log_marginal_likelihood"] = log_evidence.tolist()
                return idata
            """
        ),
        code(
            r"""
            counts = synthetic["count"].to_numpy(dtype=int)
            idatas = {}
            evidence_rows = []

            for model_index, model_key in enumerate(MODEL_ORDER):
                print(f"\nFitting {MODEL_LABELS[model_key]} ({model_index + 1}/{len(MODEL_ORDER)})")
                idata = sample_model(model_key, counts, T, RANDOM_SEED + model_index)
                idatas[model_key] = idata
                path = OUTPUT_DIR / f"posterior_{model_key}_smc.nc"
                idata.to_netcdf(path)
                log_evidence = float(np.nanmean(np.asarray(idata.attrs["log_marginal_likelihood"], dtype=float)))
                evidence_rows.append({"model_key": model_key, "log_evidence": log_evidence})

            evidence = pd.DataFrame(evidence_rows)
            best = float(evidence["log_evidence"].max())
            evidence["log10_BF_best_vs_model"] = (best - evidence["log_evidence"]) / np.log(10.0)
            evidence["is_best"] = np.isclose(evidence["log_evidence"], best)
            evidence = evidence.sort_values("log_evidence", ascending=False).reset_index(drop=True)
            evidence.to_csv(OUTPUT_DIR / "model_evidence.csv", index=False)
            evidence
            """
        ),
        markdown(
            """
            ## 4. Joint posterior distributions

            The lower-triangle plot uses paired chain/draw rows. The homogeneous and zero-inflated models map their shared `lambda` samples to the common mean-rate axis. Dashed lines and stars mark the known generating values; shaded diagonal regions are 95% HDIs.
            """
        ),
        code(
            r"""
            COMMON_PARAMETERS = ["mu_lambda", "sigma_lambda", "p_zero"]
            COMMON_LABELS = {
                "mu_lambda": r"$\mu_\lambda$ / $\lambda$",
                "sigma_lambda": r"$\sigma_\lambda$",
                "p_zero": r"$\phi_0$",
            }
            SOURCE_PARAMETERS = {
                "homo": {"mu_lambda": "lambda"},
                "z2p": {"mu_lambda": "lambda", "p_zero": "p_zero"},
                "dis2p": {"mu_lambda": "mu_lambda", "sigma_lambda": "sigma_lambda"},
                "hetero3": {"mu_lambda": "mu_lambda", "sigma_lambda": "sigma_lambda", "p_zero": "p_zero"},
            }
            TRUTH = {"mu_lambda": MU_LAMBDA, "sigma_lambda": SIGMA_LAMBDA, "p_zero": PHI_ZERO}


            def paired_draw_frame(model_key: str, idata: az.InferenceData) -> pd.DataFrame:
                frame = pd.DataFrame()
                for public_name, source_name in SOURCE_PARAMETERS[model_key].items():
                    frame[public_name] = np.asarray(idata.posterior[source_name], dtype=float).reshape(-1)
                frame["model_key"] = model_key
                return frame


            posterior_frames = {key: paired_draw_frame(key, idata) for key, idata in idatas.items()}
            posterior_samples = pd.concat(posterior_frames.values(), ignore_index=True)
            posterior_samples.to_csv(OUTPUT_DIR / "posterior_samples.csv", index=False)

            fig, axes = plt.subplots(3, 3, figsize=(11, 11))
            for row_index, row_parameter in enumerate(COMMON_PARAMETERS):
                for column_index, column_parameter in enumerate(COMMON_PARAMETERS):
                    axis = axes[row_index, column_index]
                    if column_index > row_index:
                        axis.axis("off")
                        continue
                    for model_key in MODEL_ORDER:
                        frame = posterior_frames[model_key]
                        colour = MODEL_COLOURS[model_key]
                        if row_index == column_index:
                            values = frame[row_parameter].dropna().to_numpy() if row_parameter in frame else np.array([])
                            if not len(values):
                                continue
                            axis.hist(values, bins=30, density=True, histtype="step", linewidth=2, color=colour, label=MODEL_LABELS[model_key])
                            lower, upper = az.hdi(values, hdi_prob=0.95)
                            axis.axvspan(lower, upper, color=colour, alpha=0.08)
                        else:
                            if column_parameter not in frame or row_parameter not in frame:
                                continue
                            paired = frame[[column_parameter, row_parameter]].dropna()
                            if len(paired) < 8:
                                continue
                            x_values = paired[column_parameter].to_numpy()
                            y_values = paired[row_parameter].to_numpy()
                            density = gaussian_kde(np.vstack([x_values, y_values]))
                            x_grid = np.linspace(np.quantile(x_values, 0.005), np.quantile(x_values, 0.995), 55)
                            y_grid = np.linspace(np.quantile(y_values, 0.005), np.quantile(y_values, 0.995), 55)
                            xx, yy = np.meshgrid(x_grid, y_grid)
                            zz = density(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                            axis.contour(xx, yy, zz, levels=6, colors=[colour], linewidths=1.2)

                    if row_index == column_index:
                        axis.axvline(TRUTH[row_parameter], color="#9A4938", linestyle="--", linewidth=2)
                    else:
                        axis.axvline(TRUTH[column_parameter], color="#9A4938", linestyle="--", linewidth=1.5)
                        axis.axhline(TRUTH[row_parameter], color="#9A4938", linestyle="--", linewidth=1.5)
                        axis.scatter(TRUTH[column_parameter], TRUTH[row_parameter], marker="*", s=100, color="#9A4938", edgecolor="black", zorder=10)
                    if row_index == 2:
                        axis.set_xlabel(COMMON_LABELS[column_parameter])
                    if column_index == 0:
                        axis.set_ylabel("Posterior density" if row_index == 0 else COMMON_LABELS[row_parameter])
                    axis.grid(alpha=0.2)

            fig.suptitle(
                rf"Ground truth: $\mu_\lambda={MU_LAMBDA:g}$, $\sigma_\lambda={SIGMA_LAMBDA:g}$, $\phi_0={PHI_ZERO:g}$",
                y=0.985,
            )
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.952),
                ncol=4,
                frameon=False,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.90))
            fig.savefig(OUTPUT_DIR / "joint_posterior.png", dpi=300, bbox_inches="tight")
            fig.savefig(OUTPUT_DIR / "joint_posterior.pdf", bbox_inches="tight")
            plt.show()
            """
        ),
        markdown(
            r"""
            ## 5. Bayes factors from SMC marginal likelihoods

            This uses the same convention as the Orca demo figure: $\log_{10}\mathrm{BF}(\mathcal{M}_{\mathrm{best}}/\mathcal{M})$. The best fitted model is at zero; larger bars indicate greater evidence in favour of the best fitted model over that candidate.
            """
        ),
        code(
            r"""
            plot_evidence = evidence.sort_values("log10_BF_best_vs_model", ascending=True)
            colours = ["black" if is_best else MODEL_COLOURS[key] for key, is_best in zip(plot_evidence["model_key"], plot_evidence["is_best"])]
            labels = [MODEL_LABELS[key] for key in plot_evidence["model_key"]]

            fig, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
            bars = axis.barh(labels, plot_evidence["log10_BF_best_vs_model"], color=colours, edgecolor="black")
            maximum = max(1.0, float(plot_evidence["log10_BF_best_vs_model"].max()))
            for bar, value in zip(bars, plot_evidence["log10_BF_best_vs_model"]):
                axis.text(value + 0.02 * maximum, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center")
            axis.set_xlim(0, 1.18 * maximum)
            axis.set_xlabel(r"$\log_{10}\mathrm{BF}(\mathcal{M}_{\mathrm{best}}/\mathcal{M})$")
            axis.set_ylabel("Fitted model")
            axis.set_title(rf"Ground truth: $\mu_\lambda={MU_LAMBDA:g}$, $\sigma_\lambda={SIGMA_LAMBDA:g}$, $\phi_0={PHI_ZERO:g}$")
            axis.grid(axis="x", alpha=0.25)
            fig.savefig(OUTPUT_DIR / "bayes_factors.png", dpi=300, bbox_inches="tight")
            fig.savefig(OUTPUT_DIR / "bayes_factors.pdf", bbox_inches="tight")
            plt.show()
            """
        ),
        markdown(
            """
            ## 6. Posterior summaries and reopening an InferenceData file

            The website hides large tables by default, but the downloadable CSV and NetCDF files preserve them for further work.
            """
        ),
        code(
            r"""
            summaries = []
            for model_key, idata in idatas.items():
                available = [name for name in ("lambda", "mu_lambda", "sigma_lambda", "p_zero") if name in idata.posterior]
                table = az.summary(idata, var_names=available, hdi_prob=0.95).rename_axis("parameter").reset_index()
                table.insert(0, "model_key", model_key)
                summaries.append(table)
            posterior_summary = pd.concat(summaries, ignore_index=True)
            posterior_summary.to_csv(OUTPUT_DIR / "posterior_summary.csv", index=False)

            reopened = az.from_netcdf(OUTPUT_DIR / "posterior_hetero3_smc.nc")
            print(az.summary(reopened, var_names=["mu_lambda", "sigma_lambda", "p_zero"]))
            print("\nAll files written to", OUTPUT_DIR.resolve())
            """
        ),
    ]
    return notebook


def main() -> None:
    notebook = build_notebook()
    for path in (SOURCE_NOTEBOOK, WEB_NOTEBOOK):
        path.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(notebook, path)
        print(path)


if __name__ == "__main__":
    main()
