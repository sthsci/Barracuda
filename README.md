# Orca

Bayesian modelling code for exploring heterogeneity in NK-cell interaction event counts.

This repository contains simulation, inference, plotting, and validation notebooks for the first section of the project. The current focus is a simple event-count framework where each cell has an event rate, and population-level heterogeneity is tested with four candidate models: homogeneous, zero-inflated, Gamma-heterogeneous, and zero-inflated Gamma-heterogeneous.

## Repository Structure

```text
Orca/
├── README.md
├── LICENSE
├── section_1/
│   ├── src/
│   │   ├── simulator.py      # Synthetic event-count data generation
│   │   └── inference.py      # Bayesian inference models and SMC evidence
│   ├── script/
│   │   ├── operation_1.py    # Command-line simulation + inference workflow
│   │   └── run_1.sh          # Batch runner for sample-size experiments
│   ├── notebook/
│   │   ├── demo_validation_1.ipynb  # Reader-friendly demo notebook
│   │   ├── plot_1.ipynb             # Section 1 posterior/sample-size plots
│   │   └── plot_2.ipynb             # Four-scenario model comparison plots
│   ├── figures/
│   │   ├── part_1/          # Figures for sample-size validation
│   │   ├── part_2/          # Figures for Bayes-factor model comparison
│   │   └── demo_validation_1_four_models/
│   └── results/
│       ├── part_1/          # Saved outputs for sample-size validation
│       ├── part_2/          # Saved outputs for four-scenario comparison
│       └── demo_validation_1_four_models/
```

## Where to Start

For readers who want to play with the framework, start with:

```text
section_1/notebook/demo_validation_1.ipynb
```

This notebook demonstrates the main Section 1 workflow on synthetic data: generating event counts, running Bayesian inference, plotting posterior distributions, and comparing candidate models with Bayes factors.

## Section 1 Contents

- `section_1/src/simulator.py` defines the synthetic event-count simulator.
- `section_1/src/inference.py` defines the Bayesian models and SMC log-evidence extraction.
- `section_1/script/operation_1.py` runs simulation and inference from the command line.
- `section_1/notebook/plot_1.ipynb` reproduces posterior recovery and sample-size validation figures.
- `section_1/notebook/plot_2.ipynb` compares four parameter scenarios and plots posterior/model-comparison results.
- `section_1/figures/` stores exported figures.
- `section_1/results/` stores generated numerical results such as summaries and Bayes-factor tables.

Large posterior trace files (`*.nc`) are generated outputs and are not recommended for normal GitHub commits. They can be regenerated from the notebooks/scripts.

## Streamlit Bayesian Lab (demo branch)

The `codex/streamlit-demo` branch contains a five-page research preview for learning and testing the event-count framework:

1. Bayesian inference 101, including Bayes' theorem, MCMC, SMC, marginal likelihoods, Bayes factors, diagnostics, and Thomas Bayes.
2. Synthetic data generation followed by parameter recovery and model comparison.
3. Donor-ignorant event-count inference from an example, uploaded CSV, or editable browser table.
4. A small donor-aware hierarchical extension.
5. A roadmap page for the trajectory model under development.

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
