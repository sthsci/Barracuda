"""Build the Colab-ready ORCA teaching and analysis notebooks."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
COLAB_ROOT = "https://colab.research.google.com/github/sthsci/Orca/blob/main"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip() + "\n")


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip() + "\n")


def heading(filename: str, title: str, category: str, summary: str):
    return markdown(
        f"""
        # {title}

        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB_ROOT}/notebooks/{filename})

        **{category}.** {summary}

        Run the cells from top to bottom. Values collected near the start of each notebook are safe places to experiment. Bayesian SMC fitting is deliberately disabled by default in the analysis notebooks because it can take several minutes; set `RUN_INFERENCE = True` when the data checks and descriptive plots look right.

        Use synthetic or approved anonymised data only. Do not upload names, clinical metadata, raw microscopy, or a donor key that could identify participants.
        """
    )


PACKAGE_SETUP = r"""
from pathlib import Path
import importlib.util
import subprocess
import sys


def find_orca_checkout():
    start = Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "src" / "bayesorca").is_dir():
            return candidate
    return None


ORCA_ROOT = find_orca_checkout()
if ORCA_ROOT is not None:
    sys.path[:0] = [str(ORCA_ROOT), str(ORCA_ROOT / "src")]
elif importlib.util.find_spec("bayesorca") is None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("ORCA currently requires a Python 3.12 Colab runtime.")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "git+https://github.com/sthsci/Orca.git@main",
        ]
    )

import bayesorca

print("bayesorca", bayesorca.__version__)
print("Python", sys.version.split()[0])
"""


def build_web_app_notebook():
    filename = "00_run_the_orca_web_app.ipynb"
    return filename, [
        heading(
            filename,
            "Run the complete Bayesian ORCA webpage in Jupyter",
            "Start here",
            "Launch the same Dash learning and analysis interface used by the project website inside a Google Colab output cell.",
        ),
        markdown(
            """
            ## What this notebook provides

            This is the shortest route to the complete interface: Bayesian inference 101, synthetic validation, donor-ignorant event counts, donor-aware event counts, trajectory analysis, result downloads, and the optional workspace screen. The scientific pages work without an account; the separate account service is not started in Colab.

            The setup cell clones the public GitHub repository only in Colab and installs its pinned dependencies. A local checkout is reused when this notebook runs from the repository.
            """
        ),
        code(
            r"""
            from pathlib import Path
            import importlib.util
            import subprocess
            import sys


            def find_orca_checkout():
                start = Path.cwd().resolve()
                for candidate in (start, *start.parents):
                    if (candidate / "webapp" / "dashapp.py").is_file():
                        return candidate
                return None


            IN_COLAB = importlib.util.find_spec("google.colab") is not None
            ORCA_ROOT = find_orca_checkout()

            if ORCA_ROOT is None and IN_COLAB:
                if sys.version_info[:2] != (3, 12):
                    raise RuntimeError("ORCA currently requires a Python 3.12 Colab runtime.")
                ORCA_ROOT = Path("/content/Orca")
                if not (ORCA_ROOT / ".git").is_dir():
                    subprocess.check_call(
                        [
                            "git",
                            "clone",
                            "--depth",
                            "1",
                            "https://github.com/sthsci/Orca.git",
                            str(ORCA_ROOT),
                        ]
                    )
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "-q",
                        "-r",
                        str(ORCA_ROOT / "requirements.txt"),
                    ]
                )

            if ORCA_ROOT is None:
                raise RuntimeError("Run this notebook in Colab or from an ORCA checkout.")

            sys.path[:0] = [str(ORCA_ROOT), str(ORCA_ROOT / "src")]
            print("ORCA checkout:", ORCA_ROOT)
            print("Python:", sys.version.split()[0])
            """
        ),
        markdown(
            """
            ## Check the application before launching it

            Building the application confirms that every page and callback can be imported. The route table below is also a compact map of what the webpage covers.
            """
        ),
        code(
            """
            from webapp.dashapp import PAGES, create_app

            app = create_app()
            [(page.TITLE, page.PATH) for page in PAGES]
            """
        ),
        markdown(
            """
            ## Launch the webpage inline

            Change `RUN_WEB_APP` to `True` and run the next cell. Colab will display the full Dash application below it. Keep the cell running while you use the interface. Computation-heavy SMC fits continue only while the Colab runtime is connected.

            The account/CSV-sharing workspace needs the separate platform API and is intentionally not enabled here. Every scientific workflow remains available without it.
            """
        ),
        code(
            """
            RUN_WEB_APP = False  # change to True in Colab

            if RUN_WEB_APP:
                app.run(jupyter_mode="inline", debug=False, port=8050)
            else:
                print("Ready. Set RUN_WEB_APP = True to display Bayesian ORCA here.")
            """
        ),
        markdown(
            """
            ## Where to go next

            - Use **Bayesian inference 101** for the conceptual primer.
            - Use **Event count analysis** when each row is a cell and the outcome is a total count.
            - Add donor labels for the **donor-aware** hierarchy.
            - Use **Trajectory inference** when each cell has an ordered sequence of unsuccessful (`0`) and successful (`1`) contacts.
            - Use the dedicated notebooks in this collection when you prefer editable code and tables over the webpage controls.

            Preview settings are for learning and workflow checks. Record the priors, SMC settings, random seed, package version, and input data when producing scientific results.
            """
        ),
    ]


def build_bayesian_101_notebook():
    filename = "01_bayesian_inference_101.ipynb"
    return filename, [
        heading(
            filename,
            "Bayesian inference 101",
            "Teaching notebook",
            "Use a coin-toss example to connect priors, likelihoods, posteriors, credible intervals, marginal likelihoods, and Bayes factors.",
        ),
        markdown(
            r"""
            ## 1. The Bayesian update

            Bayes' theorem combines a prior with evidence from the data:

            $$p(\theta\mid y)=\frac{p(y\mid\theta)p(\theta)}{p(y)}.$$

            - $p(\theta)$ is the **prior**.
            - $p(y\mid\theta)$ is the **likelihood**.
            - $p(\theta\mid y)$ is the **posterior**.
            - $p(y)$ is the **marginal likelihood**, the average likelihood under a model's prior.

            Edit the values below. A Beta prior and Binomial likelihood give a Beta posterior exactly, so no sampler is needed.
            """
        ),
        code(
            """
            import math

            import matplotlib.pyplot as plt
            import numpy as np
            from scipy.special import betaln, gammaln
            from scipy.stats import beta

            PRIOR_A = 2.0
            PRIOR_B = 2.0
            TOSSES = 20
            HEADS = 14

            if not (0 <= HEADS <= TOSSES):
                raise ValueError("HEADS must be between zero and TOSSES.")
            if PRIOR_A <= 0 or PRIOR_B <= 0:
                raise ValueError("Beta prior parameters must be positive.")

            POSTERIOR_A = PRIOR_A + HEADS
            POSTERIOR_B = PRIOR_B + TOSSES - HEADS
            """
        ),
        code(
            """
            theta = np.linspace(0.001, 0.999, 600)
            prior_density = beta.pdf(theta, PRIOR_A, PRIOR_B)
            likelihood = theta**HEADS * (1 - theta) ** (TOSSES - HEADS)
            likelihood /= np.trapz(likelihood, theta)
            posterior_density = beta.pdf(theta, POSTERIOR_A, POSTERIOR_B)

            interval = beta.ppf([0.025, 0.975], POSTERIOR_A, POSTERIOR_B)
            posterior_mean = POSTERIOR_A / (POSTERIOR_A + POSTERIOR_B)

            fig, ax = plt.subplots(figsize=(9, 4.8))
            ax.plot(theta, prior_density, label="Prior", linewidth=2)
            ax.plot(theta, likelihood, label="Scaled likelihood", linewidth=2)
            ax.plot(theta, posterior_density, label="Posterior", linewidth=3)
            ax.axvspan(*interval, color="#34C759", alpha=0.15, label="95% credible interval")
            ax.axvline(posterior_mean, color="#304B3D", linestyle="--")
            ax.set(xlabel="Probability of heads, θ", ylabel="Density")
            ax.legend(frameon=False)
            plt.show()

            print(f"Posterior mean: {posterior_mean:.3f}")
            print(f"95% credible interval: [{interval[0]:.3f}, {interval[1]:.3f}]")
            """
        ),
        markdown(
            r"""
            ## 2. Compare models with a Bayes factor

            Let $\mathcal M_0$ fix $\theta=0.5$. Let $\mathcal M_1$ allow $\theta$ to vary under the chosen Beta prior. Their marginal likelihoods average over all parameter values each model permits.

            $$\mathrm{BF}_{10}=\frac{p(y\mid\mathcal M_1)}{p(y\mid\mathcal M_0)}.$$

            A value above one favours $\mathcal M_1$ relative to $\mathcal M_0$; a value below one favours $\mathcal M_0$. This is relative evidence, not the probability that either model is true.
            """
        ),
        code(
            """
            log_choose = gammaln(TOSSES + 1) - gammaln(HEADS + 1) - gammaln(TOSSES - HEADS + 1)
            log_evidence_fixed = log_choose + TOSSES * math.log(0.5)
            log_evidence_flexible = (
                log_choose
                + betaln(HEADS + PRIOR_A, TOSSES - HEADS + PRIOR_B)
                - betaln(PRIOR_A, PRIOR_B)
            )
            bf_10 = math.exp(log_evidence_flexible - log_evidence_fixed)

            print(f"log p(data | fixed fair coin): {log_evidence_fixed:.3f}")
            print(f"log p(data | flexible coin):   {log_evidence_flexible:.3f}")
            print(f"BF_10: {bf_10:.3f}")
            """
        ),
        markdown(
            """
            ## 3. Why ORCA uses SMC

            The coin posterior is available in closed form. ORCA's hierarchical and trajectory models are not, so it represents the posterior with samples.

            - **MCMC** constructs a correlated chain whose long-run distribution is the posterior.
            - **Sequential Monte Carlo (SMC)** moves a population of particles from the prior toward the posterior through intermediate distributions.
            - ORCA uses PyMC SMC because the same run supplies posterior particles and an estimate of the marginal likelihood used for Bayes factors.

            More particles and independent chains usually improve stability but cost more computation. Bayes factors are sensitive to prior choices, so report priors and check whether the scientific conclusion survives reasonable alternatives.

            **Try next:** change `HEADS`, `TOSSES`, `PRIOR_A`, and `PRIOR_B`; rerun the notebook; then continue to the event-count model tutorial.
            """
        ),
    ]


def build_event_count_tutorial():
    filename = "02_event_count_model_tutorial.ipynb"
    return filename, [
        heading(
            filename,
            "Teaching the four ORCA event-count models",
            "Teaching notebook",
            "Simulate the four population structures, inspect their observable signatures, and optionally recover the generating model with SMC Bayes factors.",
        ),
        markdown(
            r"""
            ## 1. Model family

            For cell $i$ observed for a common time $T$,

            $$N_i\mid\lambda_i,T\sim\operatorname{Poisson}(\lambda_iT).$$

            | Key | Population structure | Parameters |
            |---|---|---|
            | `homo` | one shared positive rate | $\lambda$ |
            | `z2p` | structural nonengagers plus one shared positive rate | $\lambda,\phi_0$ |
            | `dis2p` | Gamma-distributed positive rates | $\mu_\lambda,\sigma_\lambda$ |
            | `hetero3` | structural nonengagers plus Gamma-distributed positive rates | $\mu_\lambda,\sigma_\lambda,\phi_0$ |

            Zero inflation raises the fraction of zeros. Continuous rate heterogeneity usually creates overdispersion: the count variance exceeds the count mean. Those clues are useful, but model evidence evaluates the complete distributions.
            """
        ),
        code(PACKAGE_SETUP),
        code(
            """
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd

            from bayesorca.event_counts import (
                MODEL_SPECS,
                InferenceSettings,
                evidence_table,
                run_count_models,
                simulate_event_counts,
                summary_table,
            )

            N_CELLS = 250
            OBSERVATION_TIME = 1.0
            MEAN_RATE = 4.0
            SEED = 2026

            MODEL_PARAMETERS = {
                "homo": {"sigma_lambda": 0.0, "p_zero": 0.0},
                "z2p": {"sigma_lambda": 0.0, "p_zero": 0.25},
                "dis2p": {"sigma_lambda": 2.0, "p_zero": 0.0},
                "hetero3": {"sigma_lambda": 2.0, "p_zero": 0.25},
            }
            """
        ),
        code(
            """
            simulations = {}
            truths = {}
            for offset, (model_key, parameters) in enumerate(MODEL_PARAMETERS.items()):
                frame, truth = simulate_event_counts(
                    model_key,
                    n_cells=N_CELLS,
                    obs_time=OBSERVATION_TIME,
                    mu_lambda=MEAN_RATE,
                    seed=SEED + offset,
                    **parameters,
                )
                simulations[model_key] = frame
                truths[model_key] = truth

            summary = pd.DataFrame(
                [
                    {
                        "model": MODEL_SPECS[key].short_label,
                        "mean_count": frame["count"].mean(),
                        "variance": frame["count"].var(),
                        "variance/mean": frame["count"].var() / frame["count"].mean(),
                        "zero_fraction": frame["count"].eq(0).mean(),
                    }
                    for key, frame in simulations.items()
                ]
            )
            summary.round(3)
            """
        ),
        code(
            """
            max_count = max(int(frame["count"].max()) for frame in simulations.values())
            bins = np.arange(-0.5, max_count + 1.5)
            fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)

            for ax, (model_key, frame) in zip(axes.flat, simulations.items()):
                ax.hist(frame["count"], bins=bins, color="#304B3D", alpha=0.82)
                ax.set_title(MODEL_SPECS[model_key].label)
                ax.set(xlabel="Events per cell", ylabel="Cells")

            fig.suptitle("Observable counts under the four population structures")
            fig.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## 2. Optional Bayesian recovery

            The next cell fits every candidate model to the synthetic `hetero3` dataset. It is off by default so the notebook executes quickly. The 128-particle setting is for learning, not publication. Repeat runs and increase particles/chains before interpreting close Bayes factors.
            """
        ),
        code(
            """
            RUN_INFERENCE = False
            DATASET_TO_FIT = "hetero3"

            if RUN_INFERENCE:
                settings = InferenceSettings(draws=128, chains=1, cores=1, seed=2026)
                results = run_count_models(
                    simulations[DATASET_TO_FIT],
                    observation_time=OBSERVATION_TIME,
                    settings=settings,
                    model_keys=list(MODEL_SPECS),
                )
                display(evidence_table(results))
                display(summary_table(results))
            else:
                print("Set RUN_INFERENCE = True to fit the four models.")
            """
        ),
        markdown(
            """
            ## Interpretation checklist

            1. Confirm that the observation time has the intended units and is common to all rows.
            2. Look at the empirical mean, variance, and zero fraction, but do not select a model from one statistic alone.
            3. Rank models by marginal likelihood/Bayes factor and inspect posterior uncertainty under scientifically plausible models.
            4. Treat a wide rate distribution as population heterogeneity, not proof of a particular molecular mechanism.
            5. Record the generating seed here; for real analyses, record the input checksum and inference configuration instead.
            """
        ),
    ]


def build_trajectory_tutorial():
    filename = "03_trajectory_model_tutorial.ipynb"
    return filename, [
        heading(
            filename,
            "Teaching ORCA trajectory models",
            "Teaching notebook",
            "Simulate ordered successful and unsuccessful contacts, summarise decision states, and optionally compare stable heterogeneity with history dependence.",
        ),
        markdown(
            r"""
            ## 1. Why order matters

            Total kills discard the order of contacts. A history such as `0,0,1,0` records two unsuccessful contacts, one successful contact, then another unsuccessful contact. ORCA trajectory models ask two different questions:

            - **Stable heterogeneity:** do cells have persistently different baseline killing propensities ($\sigma_\eta>0$)?
            - **History dependence:** does the probability of the next success change after previous failures ($\beta_f$) or successes ($\beta_s$)?

            Combining homogeneous/heterogeneous with history-independent/history-dependent assumptions produces four candidate models.
            """
        ),
        code(PACKAGE_SETUP),
        code(
            """
            import matplotlib.pyplot as plt
            import pandas as pd

            from bayesorca.trajectories import (
                TRAJECTORY_MODEL_SPECS,
                TrajectorySettings,
                expanded_trajectory_frame,
                run_trajectory_conditions,
                simulate_trajectory_frame,
                trajectory_evidence_frame,
                trajectory_summary_frame,
            )

            trajectories, truth = simulate_trajectory_frame(
                condition="Synthetic",
                n_cells=120,
                mu_lambda=4.0,
                sigma_lambda=2.0,
                p0=0.20,
                sigma_eta=0.75,
                beta_f=0.8,
                beta_s=-0.8,
                observation_time=1.0,
                seed=2026,
            )
            print("Generating model:", truth["Synthetic"]["true_model_key"])
            trajectories.head()
            """
        ),
        code(
            """
            contacts = expanded_trajectory_frame(trajectories)
            state_summary = (
                contacts.groupby(
                    ["previous_nonlethal_contacts", "previous_lethal_contacts"],
                    as_index=False,
                )
                .agg(contacts=("outcome", "size"), kill_probability=("outcome", "mean"))
            )

            fig, ax = plt.subplots(figsize=(8, 5.5))
            points = ax.scatter(
                state_summary["previous_nonlethal_contacts"],
                state_summary["previous_lethal_contacts"],
                s=25 + 5 * state_summary["contacts"],
                c=state_summary["kill_probability"],
                cmap="viridis",
                vmin=0,
                vmax=1,
                alpha=0.85,
            )
            fig.colorbar(points, ax=ax, label="Observed next-contact success fraction")
            ax.set(
                xlabel="Previous unsuccessful contacts",
                ylabel="Previous successful contacts",
                title="Empirical decision states (point size = contacts observed)",
            )
            plt.show()

            state_summary.sort_values("contacts", ascending=False).head(12)
            """
        ),
        markdown(
            """
            ## 2. Optional model comparison

            Empirical state fractions mix true history effects, cell-to-cell differences, and uneven sampling. The trajectory likelihood models them jointly. The preview below is intentionally small; increase particles, chains, and quadrature points for a scientific analysis.
            """
        ),
        code(
            """
            RUN_INFERENCE = False

            if RUN_INFERENCE:
                settings = TrajectorySettings(
                    draws=128,
                    chains=1,
                    cores=1,
                    seed=2026,
                    n_quad=10,
                )
                results = run_trajectory_conditions(
                    trajectories,
                    observation_time=1.0,
                    settings=settings,
                    model_keys=list(TRAJECTORY_MODEL_SPECS),
                )
                display(trajectory_evidence_frame(results))
                display(trajectory_summary_frame(results))
            else:
                print("Set RUN_INFERENCE = True to fit the four trajectory models.")
            """
        ),
        markdown(
            """
            ## Interpretation checklist

            - A blank history is a valid cell with zero observed contacts; keep it in the input.
            - `beta_f` and `beta_s` describe associations with prior outcomes after accounting for the modelled baseline structure.
            - The order of the binary history must match experimental time.
            - Sparse states are noisy; point size above shows how many decisions support each empirical fraction.
            - Compare all four models before describing an effect as stable heterogeneity or history dependence.
            """
        ),
    ]


def build_event_count_analysis():
    filename = "04_event_count_analysis.ipynb"
    return filename, [
        heading(
            filename,
            "Analyse event counts without donor labels",
            "Analysis notebook",
            "Upload one to four experimental conditions, validate the public ORCA schema, explore the counts, fit population models, and export a reproducible result archive.",
        ),
        markdown(
            """
            ## Input schema

            One row represents one cell and one count outcome (for example total contacts or kills).

            ```csv
            cell_id,condition,count
            cell_001,Control,3
            cell_002,Control,0
            cell_001,Treatment,5
            ```

            `condition` is optional; missing values are assigned to one group. Cell IDs must be unique within a condition. The public workflow accepts 5–1,000 cells per condition, integer counts from 0–100, and at least one positive count per condition.
            """
        ),
        code(PACKAGE_SETUP),
        code(
            """
            from io import BytesIO
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd

            from bayesorca.event_counts import (
                MODEL_SPECS,
                InferenceSettings,
                build_condition_results_zip,
                evidence_table,
                normalize_condition_frame,
                run_condition_models,
                sample_count_frame,
                summary_table,
                validate_condition_frame,
            )

            USE_UPLOAD = False
            OBSERVATION_TIME = 1.0
            RUN_INFERENCE = False
            MODEL_KEYS = list(MODEL_SPECS)
            """
        ),
        code(
            """
            def upload_one_csv():
                try:
                    from google.colab import files
                except ImportError as exc:
                    raise RuntimeError("Set USE_UPLOAD=True in Google Colab, or replace raw_data directly.") from exc
                uploaded = files.upload()
                if len(uploaded) != 1:
                    raise ValueError("Upload exactly one CSV file.")
                return pd.read_csv(BytesIO(next(iter(uploaded.values()))))


            if USE_UPLOAD:
                raw_data = upload_one_csv()
            else:
                control = sample_count_frame().assign(condition="Control")
                treatment = sample_count_frame().assign(
                    condition="Treatment",
                    count=lambda frame: frame["count"] + [1, 1, 0, 2, 1, 0, 2, 1, 1, 2, 0, 1],
                )
                raw_data = pd.concat([control, treatment], ignore_index=True).loc[
                    :, ["cell_id", "condition", "count"]
                ]

            mapped_data, mapping_message = normalize_condition_frame(raw_data, donor_aware=False)
            data = validate_condition_frame(mapped_data, donor_aware=False)
            print(mapping_message)
            print(f"Validated {len(data):,} cells across {data['condition'].nunique()} condition(s).")
            data.head()
            """
        ),
        code(
            """
            descriptive = (
                data.groupby("condition", as_index=False)
                .agg(
                    cells=("cell_id", "size"),
                    mean_count=("count", "mean"),
                    variance=("count", "var"),
                    zero_fraction=("count", lambda values: values.eq(0).mean()),
                )
            )
            display(descriptive.round(3))

            axes = data.hist(
                column="count",
                by="condition",
                bins=range(int(data["count"].max()) + 2),
                figsize=(10, 4),
                sharex=True,
                sharey=True,
                color="#304B3D",
                rwidth=0.9,
            )
            plt.suptitle("Event-count distributions by condition")
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## Fit and export

            Each condition is fitted independently with the same model set and priors. The preview uses one chain and 128 SMC particles. For reported results, increase computation, repeat with multiple seeds/chains, inspect posterior stability, and justify the prior sensitivity of Bayes factors.
            """
        ),
        code(
            """
            if RUN_INFERENCE:
                settings = InferenceSettings(draws=128, chains=1, cores=1, seed=2026)
                results = run_condition_models(
                    data,
                    observation_time=OBSERVATION_TIME,
                    settings=settings,
                    model_keys=MODEL_KEYS,
                    donor_aware=False,
                )

                evidence = pd.concat(
                    [evidence_table(models).assign(condition=condition) for condition, models in results.items()],
                    ignore_index=True,
                )
                posterior_summary = pd.concat(
                    [summary_table(models).assign(condition=condition) for condition, models in results.items()],
                    ignore_index=True,
                )
                display(evidence)
                display(posterior_summary)

                archive_path = Path("orca_event_count_analysis.zip")
                archive_path.write_bytes(
                    build_condition_results_zip(
                        results,
                        data,
                        OBSERVATION_TIME,
                        settings,
                        donor_aware=False,
                    )
                )
                print("Saved", archive_path.resolve())
                try:
                    from google.colab import files
                    files.download(str(archive_path))
                except ImportError:
                    pass
            else:
                print("Data checks complete. Set RUN_INFERENCE = True when you are ready to fit.")
            """
        ),
        markdown(
            """
            ## Report with the result

            Record the event definition, observation-time units, inclusion/exclusion rules, cell count per condition, model keys, prior settings, SMC particles/chains, random seed, ORCA version, and any sensitivity runs. A Bayes factor ranks only the models that were compared; it does not establish that the best candidate is biologically complete.
            """
        ),
    ]


def build_donor_analysis():
    filename = "05_donor_aware_analysis.ipynb"
    return filename, [
        heading(
            filename,
            "Analyse event counts with donor structure",
            "Analysis notebook",
            "Upload donor-resolved counts, verify representation within every condition, separate within- and between-donor variation, and export the fitted hierarchy.",
        ),
        markdown(
            """
            ## Input schema

            ```csv
            cell_id,donor_id,condition,count
            cell_001,donor_A,Control,3
            cell_002,donor_A,Control,0
            cell_001,donor_A,Treatment,5
            ```

            Donor codes can still be pseudonymised personal data if another key reconnects them to individuals. Use approved anonymised labels and never upload the key. Every condition is fitted independently and needs 2–12 represented donors, at least three cells per donor, at least five cells overall, and at least one positive count.
            """
        ),
        code(PACKAGE_SETUP),
        code(
            """
            from io import BytesIO
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd

            from bayesorca.event_counts import (
                MODEL_SPECS,
                InferenceSettings,
                build_condition_results_zip,
                evidence_table,
                normalize_condition_frame,
                run_condition_models,
                sample_donor_frame,
                summary_table,
                validate_condition_frame,
            )

            USE_UPLOAD = False
            OBSERVATION_TIME = 1.0
            RUN_INFERENCE = False
            MODEL_KEYS = list(MODEL_SPECS)
            """
        ),
        code(
            """
            def upload_one_csv():
                try:
                    from google.colab import files
                except ImportError as exc:
                    raise RuntimeError("Set USE_UPLOAD=True in Google Colab, or replace raw_data directly.") from exc
                uploaded = files.upload()
                if len(uploaded) != 1:
                    raise ValueError("Upload exactly one CSV file.")
                return pd.read_csv(BytesIO(next(iter(uploaded.values()))))


            if USE_UPLOAD:
                raw_data = upload_one_csv()
            else:
                control = sample_donor_frame().assign(condition="Control")
                treatment = sample_donor_frame().assign(
                    condition="Treatment",
                    count=lambda frame: frame["count"] + [1, 1, 0, 2, 1, 0, 2, 1, 1, 2, 0, 1],
                )
                raw_data = pd.concat([control, treatment], ignore_index=True).loc[
                    :, ["cell_id", "donor_id", "condition", "count"]
                ]

            mapped_data, mapping_message = normalize_condition_frame(raw_data, donor_aware=True)
            data = validate_condition_frame(mapped_data, donor_aware=True)
            print(mapping_message)
            print(f"Validated {len(data):,} cells, {data['donor_id'].nunique()} donor codes, and {data['condition'].nunique()} condition(s).")
            data.head()
            """
        ),
        code(
            """
            donor_summary = (
                data.groupby(["condition", "donor_id"], as_index=False)
                .agg(
                    cells=("cell_id", "size"),
                    mean_count=("count", "mean"),
                    variance=("count", "var"),
                    zero_fraction=("count", lambda values: values.eq(0).mean()),
                )
            )
            display(donor_summary.round(3))

            pivot = donor_summary.pivot(index="donor_id", columns="condition", values="mean_count")
            pivot.plot(kind="bar", figsize=(9, 4.8), color=["#304B3D", "#C05A3D"])
            plt.ylabel("Mean count per cell")
            plt.title("Raw donor means (descriptive, not partial-pooled estimates)")
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.show()
            """
        ),
        markdown(
            """
            ## Fit and export

            The hierarchy estimates population-level distributions and donor-level deviations while retaining cell-level count variation. The preview below is intentionally small. Donor-aware posteriors can be weakly identified with few donors, so report the donor count and examine sensitivity to the donor-deviation prior.
            """
        ),
        code(
            """
            if RUN_INFERENCE:
                settings = InferenceSettings(draws=128, chains=1, cores=1, seed=2026)
                results = run_condition_models(
                    data,
                    observation_time=OBSERVATION_TIME,
                    settings=settings,
                    model_keys=MODEL_KEYS,
                    donor_aware=True,
                )

                evidence = pd.concat(
                    [evidence_table(models).assign(condition=condition) for condition, models in results.items()],
                    ignore_index=True,
                )
                posterior_summary = pd.concat(
                    [summary_table(models).assign(condition=condition) for condition, models in results.items()],
                    ignore_index=True,
                )
                display(evidence)
                display(posterior_summary)

                archive_path = Path("orca_donor_aware_analysis.zip")
                archive_path.write_bytes(
                    build_condition_results_zip(
                        results,
                        data,
                        OBSERVATION_TIME,
                        settings,
                        donor_aware=True,
                    )
                )
                print("Saved", archive_path.resolve())
                try:
                    from google.colab import files
                    files.download(str(archive_path))
                except ImportError:
                    pass
            else:
                print("Data checks complete. Set RUN_INFERENCE = True when you are ready to fit.")
            """
        ),
        markdown(
            """
            ## Report with the result

            Include the number of donors and cells per donor/condition, whether donor labels repeat across conditions, event and time units, hierarchy and priors, SMC settings, seed, ORCA version, and sensitivity analyses. Population differences in this cohort should not be generalised beyond the sampled donor population without an appropriate study design.
            """
        ),
    ]


def build_trajectory_analysis():
    filename = "06_trajectory_analysis.ipynb"
    return filename, [
        heading(
            filename,
            "Analyse ordered contact trajectories",
            "Analysis notebook",
            "Upload binary cell histories, validate their order, inspect empirical decision states, compare four trajectory models, and export posterior results.",
        ),
        markdown(
            """
            ## Input schema

            ```csv
            cell_id,condition,history
            cell_001,Control,"0,0,1,0"
            cell_002,Control,"1,1"
            cell_003,Control,""
            ```

            `0` means an unsuccessful contact and `1` a successful contact, in experimental time order. A blank history retains a cell with zero observed contacts. `condition` is optional; the public workflow supports one to four independently fitted conditions.
            """
        ),
        code(PACKAGE_SETUP),
        code(
            """
            from io import BytesIO
            from pathlib import Path

            import matplotlib.pyplot as plt
            import pandas as pd

            from bayesorca.trajectories import (
                TRAJECTORY_MODEL_SPECS,
                TrajectorySettings,
                TrajectorySimulationSpec,
                build_trajectory_archive,
                expanded_trajectory_frame,
                run_trajectory_conditions,
                simulate_trajectory_frame,
                trajectory_evidence_frame,
                trajectory_summary_frame,
                validate_trajectory_frame,
            )

            USE_UPLOAD = False
            OBSERVATION_TIME = 1.0
            RUN_INFERENCE = False
            MODEL_KEYS = list(TRAJECTORY_MODEL_SPECS)
            """
        ),
        code(
            """
            def upload_one_csv():
                try:
                    from google.colab import files
                except ImportError as exc:
                    raise RuntimeError("Set USE_UPLOAD=True in Google Colab, or replace raw_data directly.") from exc
                uploaded = files.upload()
                if len(uploaded) != 1:
                    raise ValueError("Upload exactly one CSV file.")
                return pd.read_csv(BytesIO(next(iter(uploaded.values()))), keep_default_na=False)


            if USE_UPLOAD:
                raw_data = upload_one_csv()
                truth = None
            else:
                raw_data, truth = simulate_trajectory_frame(
                    [
                        TrajectorySimulationSpec(condition="Control", n_cells=80, seed=2026),
                        TrajectorySimulationSpec(
                            condition="Treatment",
                            n_cells=80,
                            mu_lambda=5.0,
                            sigma_eta=0.45,
                            beta_f=0.35,
                            beta_s=-0.35,
                            seed=2027,
                        ),
                    ]
                )

            data = validate_trajectory_frame(raw_data)
            print(f"Validated {len(data):,} cells across {data['condition'].nunique()} condition(s).")
            data.head()
            """
        ),
        code(
            """
            per_cell = data.assign(
                contacts=data["history"].map(len),
                kills=data["history"].map(sum),
            )
            descriptive = (
                per_cell.groupby("condition", as_index=False)
                .agg(
                    cells=("cell_id", "size"),
                    mean_contacts=("contacts", "mean"),
                    mean_kills=("kills", "mean"),
                    zero_contact_fraction=("contacts", lambda values: values.eq(0).mean()),
                )
            )
            display(descriptive.round(3))

            fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
            for condition, group in per_cell.groupby("condition", sort=False):
                axes[0].hist(group["contacts"], alpha=0.55, label=condition)
                axes[1].hist(group["kills"], alpha=0.55, label=condition)
            axes[0].set(xlabel="Contacts per cell", ylabel="Cells")
            axes[1].set(xlabel="Successful contacts per cell", ylabel="Cells")
            axes[1].legend(frameon=False)
            fig.suptitle("Trajectory summaries before model fitting")
            fig.tight_layout()
            plt.show()
            """
        ),
        code(
            """
            contacts = expanded_trajectory_frame(data)
            state_summary = (
                contacts.groupby(
                    ["condition", "previous_nonlethal_contacts", "previous_lethal_contacts"],
                    as_index=False,
                )
                .agg(contacts=("outcome", "size"), kill_probability=("outcome", "mean"))
                .sort_values(["condition", "contacts"], ascending=[True, False])
            )
            state_summary.head(15)
            """
        ),
        markdown(
            """
            ## Fit and export

            The four models cross homogeneous/heterogeneous baseline decision propensities with history-independent/history-dependent decisions. The preview uses 128 particles and 10 quadrature points; increase these and repeat independent runs before reporting model evidence.
            """
        ),
        code(
            """
            if RUN_INFERENCE:
                settings = TrajectorySettings(
                    draws=128,
                    chains=1,
                    cores=1,
                    seed=2026,
                    n_quad=10,
                )
                results = run_trajectory_conditions(
                    data,
                    observation_time=OBSERVATION_TIME,
                    settings=settings,
                    model_keys=MODEL_KEYS,
                )
                display(trajectory_evidence_frame(results))
                display(trajectory_summary_frame(results))

                archive_path = Path("orca_trajectory_analysis.zip")
                archive_path.write_bytes(
                    build_trajectory_archive(
                        results,
                        data,
                        OBSERVATION_TIME,
                        settings,
                        truth=truth,
                    )
                )
                print("Saved", archive_path.resolve())
                try:
                    from google.colab import files
                    files.download(str(archive_path))
                except ImportError:
                    pass
            else:
                print("Data checks complete. Set RUN_INFERENCE = True when you are ready to fit.")
            """
        ),
        markdown(
            """
            ## Report with the result

            Define successful/unsuccessful contacts, confirm chronological encoding, report cells and contacts per condition, observation-time units, candidate models and priors, SMC/quadrature settings, seed, ORCA version, and sensitivity runs. History coefficients describe the fitted association after the model's baseline structure; they do not alone demonstrate a causal biological memory mechanism.
            """
        ),
    ]


BUILDERS = (
    build_web_app_notebook,
    build_bayesian_101_notebook,
    build_event_count_tutorial,
    build_trajectory_tutorial,
    build_event_count_analysis,
    build_donor_analysis,
    build_trajectory_analysis,
)


def build_all() -> list[Path]:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for builder in BUILDERS:
        filename, cells = builder()
        stem = Path(filename).stem
        for index, cell in enumerate(cells):
            cell["id"] = f"{stem[:56]}-{index:02d}"
        notebook = nbf.v4.new_notebook(
            cells=cells,
            metadata={
                "colab": {"name": filename, "provenance": []},
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.12"},
            },
        )
        path = NOTEBOOK_DIR / filename
        nbf.write(notebook, path)
        written.append(path)
    return written


if __name__ == "__main__":
    for output in build_all():
        print(output.relative_to(PROJECT_ROOT))
