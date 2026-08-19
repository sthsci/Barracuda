# Orca

**Bayesian inference for heterogeneity in immune cell decision making**

Research code supporting the manuscript *A Bayesian framework reveals heterogeneous and stochastic decision-making in NK cell cytotoxicity*. Orca uses single-cell contact and kill histories to ask whether variation in natural killer (NK) cell behaviour arises from stochastic events, stable cell-to-cell differences, donor effects, interaction history, or a combination of these mechanisms.

## Framework

- **Event-count inference** compares four nested population models: homogeneous Poisson, zero-inflated Poisson, Gamma-Poisson continuous heterogeneity, and zero-inflated Gamma-Poisson. A donor-aware hierarchy separates within-donor cellular variation from between-donor differences.
- **Trajectory inference** retains the order of lethal and non-lethal contacts, separating stable baseline killing propensity from changes associated with previous successful or unsuccessful encounters.
- **Bayesian computation** uses PyMC Sequential Monte Carlo to estimate posterior distributions and marginal likelihoods for Bayes-factor model comparison.

In the analysed time-lapse imaging dataset, continuous heterogeneity was supported across untreated, rituximab-treated, and bispecific-antibody-treated NK-cell populations. Rituximab primarily increased mean killing activity, whereas the bispecific antibody produced a more homogeneous cytotoxic response. Donor-aware and trajectory analyses further revealed between-donor variation and history-dependent changes in killing behaviour. These conclusions are specific to the experimental dataset and donor cohort studied.

## Repository map

| Path | Purpose | Suggested entry point |
|---|---|---|
| [`section_1/`](section_1/) | Synthetic event-count validation, parameter recovery, sample-size analysis, and comparison of four population structures | [`demo_validation_1.ipynb`](section_1/notebook/demo_validation_1.ipynb) |
| [`section_2/`](section_2/) | Experimental contact/kill counts, donor-ignorant and donor-aware inference, variance decomposition, and treatment contrasts | [`analysis_1.ipynb`](section_2/notebooks/analysis_1.ipynb) and [`analysis_1_donor.ipynb`](section_2/notebooks/analysis_1_donor.ipynb) |
| [`section_3/`](section_3/) | Synthetic ordered contact-kill trajectories, parameter recovery, and trajectory-model validation | [`plot_1_trajmodel.ipynb`](section_3/notebooks/plot_1_trajmodel.ipynb) |
| [`section_4/`](section_4/) | Trajectory inference for untreated, rituximab, and bispecific-antibody conditions | [`analysis_2.ipynb`](section_4/notebooks/analysis_2.ipynb) |
| [`data/`](data/) | Derived per-cell contact-history tables used by the analyses | - |
| [`figures/`](figures/) | Graphic abstract and assembled main/supplementary manuscript figures | - |

Each section contains its model implementation under `src/`, analysis notebooks, execution scripts where applicable, and exported figures. Large posterior traces and generated `results/` directories are intentionally excluded from version control and can be regenerated from the corresponding workflows.

## Getting started

The simplest introduction is [`section_1/notebook/demo_validation_1.ipynb`](section_1/notebook/demo_validation_1.ipynb), which simulates event counts, fits the four candidate models, visualises posterior recovery, and compares their evidence.

The scientific stack is built around Python, PyMC, PyTensor, ArviZ, NumPy, pandas, SciPy, Matplotlib, and xarray. The analyses are computationally intensive: manuscript-scale SMC runs use substantially more particles and chains than exploratory checks.

## Jupyter and Google Colab

The curated [notebook collection](notebooks/README.md) mirrors the main
teaching and analysis workflows. Each notebook uses synthetic data by default,
requires Python 3.12, and leaves computationally intensive SMC fitting off
until the user enables it.

| Category | Notebook | Google Colab |
|---|---|---|
| Start here | [`00_run_the_orca_web_app.ipynb`](notebooks/00_run_the_orca_web_app.ipynb) | [Open in Colab](https://colab.research.google.com/github/sthsci/Orca/blob/main/notebooks/00_run_the_orca_web_app.ipynb) |
| Teaching | [`01_bayesian_inference_101.ipynb`](notebooks/01_bayesian_inference_101.ipynb) | [Open in Colab](https://colab.research.google.com/github/sthsci/Orca/blob/main/notebooks/01_bayesian_inference_101.ipynb) |
| Teaching | [`02_event_count_model_tutorial.ipynb`](notebooks/02_event_count_model_tutorial.ipynb) | [Open in Colab](https://colab.research.google.com/github/sthsci/Orca/blob/main/notebooks/02_event_count_model_tutorial.ipynb) |
| Teaching | [`03_trajectory_model_tutorial.ipynb`](notebooks/03_trajectory_model_tutorial.ipynb) | [Open in Colab](https://colab.research.google.com/github/sthsci/Orca/blob/main/notebooks/03_trajectory_model_tutorial.ipynb) |
| Analysis | [`04_event_count_analysis.ipynb`](notebooks/04_event_count_analysis.ipynb) | [Open in Colab](https://colab.research.google.com/github/sthsci/Orca/blob/main/notebooks/04_event_count_analysis.ipynb) |
| Analysis | [`05_donor_aware_analysis.ipynb`](notebooks/05_donor_aware_analysis.ipynb) | [Open in Colab](https://colab.research.google.com/github/sthsci/Orca/blob/main/notebooks/05_donor_aware_analysis.ipynb) |
| Analysis | [`06_trajectory_analysis.ipynb`](notebooks/06_trajectory_analysis.ipynb) | [Open in Colab](https://colab.research.google.com/github/sthsci/Orca/blob/main/notebooks/06_trajectory_analysis.ipynb) |

See the catalog for input schemas, runtime expectations, privacy guidance, and
reproducibility notes.

## Python package

The reusable scientific API is packaged as `bayesorca`. It covers donor
ignorant and donor aware event count inference, condition-wise analysis,
synthetic data generation, ordered trajectory inference, evidence tables,
posterior draw tables, and reproducible result archives.

Install a released version with:

```bash
python -m pip install bayesorca
```

For development from this repository:

```bash
python -m pip install -e ".[test,build]"
```

See [`PACKAGE_README.md`](PACKAGE_README.md) for the public API and runnable
examples. The Python package deliberately excludes the Dash presentation layer.

## Dash web application

The `codex/dash-app` branch contains a focused Dash interface for learning and testing the framework:

1. A project home page and guide to the available sections.
2. Bayesian inference 101, including Bayes' theorem, MCMC, SMC, marginal likelihoods, and Bayes factors.
3. Donor ignorant event count inference, beginning with a choice between synthetic validation and the user's own data. Inference can run independently for one to four experimental conditions.
4. Donor aware hierarchical inference for one to four conditions, including Bayes factors, within-versus-between donor heterogeneity, population and donor posterior views, and particle-level condition contrasts.
5. Donor ignorant trajectory inference from synthetic or uploaded ordered contact histories, with empirical state maps, model evidence, marginal posteriors, and full joint posteriors.

The application intentionally defaults to small SMC settings. Preview results are illustrative and are not publication-grade. The scientific simulation, validation, inference, and result-export functions are kept separate from the Dash interface.

### Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python dash_app.py
```

Alternatively, build the included container:

```bash
docker build -t orca-dash-app .
docker run --rm -p 8501:8501 orca-dash-app
```

### Input schemas

Donor-ignorant CSV:

```csv
cell_id,condition,count
cell_001,Control,3
cell_002,Control,0
cell_003,Control,1
cell_004,Treatment,2
cell_005,Treatment,0
```

Donor-aware CSV:

```csv
cell_id,donor_id,condition,count
cell_001,donor_A,Control,3
cell_002,donor_A,Control,0
cell_003,donor_A,Control,1
cell_004,donor_B,Treatment,2
cell_005,donor_B,Treatment,0
cell_006,donor_B,Treatment,1
```

Trajectory CSV:

```csv
cell_id,condition,history
cell_001,Control,"0,0,1,0"
cell_002,Control,"1,1"
cell_003,Control,""
cell_004,Treatment,"0,1,1"
```

Each count upload contains one count outcome and may contain one to four experimental conditions. Each trajectory upload contains one ordered binary history per cell; a blank history retains a cell with zero observed contacts. If `condition` is omitted, all rows are assigned to `Condition 1`. Inference runs independently for each condition with the same selected models and prior settings. The likelihood uses one observation time entered in the interface for every row. Do not upload names, clinical metadata, raw microscopy, or unapproved donor data.

The public demo accepts 5–1,000 cells per condition, integer counts from 0–100, and at least one positive count in each condition. Donor aware inputs need 2–12 donors with at least three cells per donor in every condition; larger donor groups are strongly preferred for stable donor estimates.

### Deployment status

This branch is suitable for local testing with synthetic or approved anonymous data. It is not yet an Imperial production deployment. Before public hosting, complete the ASK ICT process and agree the domain/runtime, authentication needs, retention and deletion policy, server-side job queue and compute limits, security review, privacy notice, and WCAG accessibility audit with Imperial ICT.

## Data availability

Derived analysis tables are included where appropriate. Raw time-lapse microscopy data and the complete experimental dataset are not distributed in this repository. Data-access enquiries should be directed to Elephes Sung at [eu23@ic.ac.uk](mailto:eu23@ic.ac.uk).

## Status and licence

This is research software accompanying a manuscript in preparation. Interfaces and model implementations may continue to evolve. Released code is provided under the [MIT License](LICENSE).
