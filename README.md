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

## Streamlit preview

The `codex/streamlit-demo` branch contains a six-page interface for learning and testing the framework:

1. A project home page and guide to the available sections.
2. Bayesian inference 101, including Bayes' theorem, MCMC, SMC, marginal likelihoods, and Bayes factors.
3. Synthetic data generation followed by parameter recovery and model comparison.
4. Donor-ignorant event-count inference from an example, uploaded CSV, or editable browser table.
5. A small donor-aware hierarchical extension.
6. A roadmap page for the trajectory interface under development.

The preview intentionally uses small SMC settings. Its default results are illustrative and are not publication-grade.

### Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Alternatively, build the included container:

```bash
docker build -t orca-streamlit-demo .
docker run --rm -p 8501:8501 orca-streamlit-demo
```

### Input schemas

Donor-ignorant CSV:

```csv
cell_id,count
cell_001,3
cell_002,0
cell_003,1
cell_004,2
cell_005,0
```

Donor-aware CSV:

```csv
cell_id,donor_id,count
cell_001,donor_A,3
cell_002,donor_A,0
cell_003,donor_A,1
cell_004,donor_B,2
cell_005,donor_B,0
cell_006,donor_B,1
```

Each upload should contain one outcome and one experimental condition. The current likelihood uses a single observation time entered in the interface for every row. Do not upload names, clinical metadata, raw microscopy, or unapproved donor-derived data.

The public demo accepts 5–1,000 cells, integer counts from 0–100, and at least one positive count. Donor-aware inputs need 2–12 donors with at least three cells per donor; larger donor groups are strongly preferred for stable donor-specific estimates.

### Deployment status

This branch is suitable for local testing with synthetic or approved anonymous data. It is not yet an Imperial production deployment. Before public hosting, complete the ASK ICT process and agree the domain/runtime, authentication needs, retention and deletion policy, server-side job queue and compute limits, security review, privacy notice, and WCAG accessibility audit with Imperial ICT.

## Data availability

Derived analysis tables are included where appropriate. Raw time-lapse microscopy data and the complete experimental dataset are not distributed in this repository. Data-access enquiries should be directed to Elephes Sung at [eu23@ic.ac.uk](mailto:eu23@ic.ac.uk).

## Status and licence

This is research software accompanying a manuscript in preparation. Interfaces and model implementations may continue to evolve. Released code is provided under the [MIT License](LICENSE).
