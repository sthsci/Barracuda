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
