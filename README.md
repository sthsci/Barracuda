# Orca

**Bayesian inference for heterogeneity in immune-cell decision-making**

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

An interactive Streamlit preview is under development on the [`codex/streamlit-demo`](https://github.com/sthsci/Orca/tree/codex/streamlit-demo) branch.

## Data availability

Derived analysis tables are included where appropriate. Raw time-lapse microscopy data and the complete experimental dataset are not distributed in this repository. Data-access enquiries should be directed to Elephes Sung at [eu23@ic.ac.uk](mailto:eu23@ic.ac.uk).

## Status and licence

This is research software accompanying a manuscript in preparation. Interfaces and model implementations may continue to evolve. Released code is provided under the [MIT License](LICENSE).
