# Bayesian ORCA notebooks

This collection provides seven small, output-free notebooks for learning the
ORCA framework and adapting its public analysis workflows. They run with
synthetic examples by default; uploads and Bayesian inference are opt-in.

## Start here

| Notebook | Purpose | Google Colab |
|---|---|---|
| [`00_run_the_orca_web_app.ipynb`](00_run_the_orca_web_app.ipynb) | Launch the complete Dash learning and analysis interface inside a notebook. | [Open in Colab](https://colab.research.google.com/github/sthsci/Orca/blob/main/notebooks/00_run_the_orca_web_app.ipynb) |

Set `RUN_WEB_APP = True` after the setup and application checks. The scientific
pages work without an account; the optional account and CSV-sharing service is
not started in Colab.

## Teaching notebooks

| Notebook | Purpose | Google Colab |
|---|---|---|
| [`01_bayesian_inference_101.ipynb`](01_bayesian_inference_101.ipynb) | Explore priors, likelihoods, posteriors, credible intervals, marginal likelihoods, and Bayes factors with an exact coin example. | [Open in Colab](https://colab.research.google.com/github/sthsci/Orca/blob/main/notebooks/01_bayesian_inference_101.ipynb) |
| [`02_event_count_model_tutorial.ipynb`](02_event_count_model_tutorial.ipynb) | Simulate the four ORCA event-count populations and optionally compare them with SMC. | [Open in Colab](https://colab.research.google.com/github/sthsci/Orca/blob/main/notebooks/02_event_count_model_tutorial.ipynb) |
| [`03_trajectory_model_tutorial.ipynb`](03_trajectory_model_tutorial.ipynb) | Simulate ordered contact outcomes and separate stable heterogeneity from history dependence. | [Open in Colab](https://colab.research.google.com/github/sthsci/Orca/blob/main/notebooks/03_trajectory_model_tutorial.ipynb) |

## Analysis notebooks

| Notebook | Purpose | Google Colab |
|---|---|---|
| [`04_event_count_analysis.ipynb`](04_event_count_analysis.ipynb) | Validate, explore, fit, and export donor-ignorant counts for one to four conditions. | [Open in Colab](https://colab.research.google.com/github/sthsci/Orca/blob/main/notebooks/04_event_count_analysis.ipynb) |
| [`05_donor_aware_analysis.ipynb`](05_donor_aware_analysis.ipynb) | Analyse counts while separating within-donor and between-donor variation. | [Open in Colab](https://colab.research.google.com/github/sthsci/Orca/blob/main/notebooks/05_donor_aware_analysis.ipynb) |
| [`06_trajectory_analysis.ipynb`](06_trajectory_analysis.ipynb) | Validate ordered binary histories, compare trajectory models, and export results. | [Open in Colab](https://colab.research.google.com/github/sthsci/Orca/blob/main/notebooks/06_trajectory_analysis.ipynb) |

## Input schemas

Event-count analysis uses one integer outcome per cell. `condition` is optional;
when present, one to four conditions are fitted independently.

```csv
cell_id,condition,count
cell_001,Control,3
cell_002,Control,0
```

Donor-aware analysis adds an approved anonymised donor code. Each condition
needs 2–12 donors, at least three cells per donor, at least five cells overall,
and at least one positive count.

```csv
cell_id,donor_id,condition,count
cell_001,donor_A,Control,3
cell_002,donor_A,Control,0
```

Trajectory analysis uses chronological binary histories: `0` is an
unsuccessful contact, `1` is a successful contact, and an empty value retains a
cell with no observed contacts.

```csv
cell_id,condition,history
cell_001,Control,"0,0,1,0"
cell_002,Control,""
```

## Runtime and reproducibility

- ORCA currently requires Python 3.12. The setup cells reuse a local checkout
  or install the public repository when running in Colab.
- `RUN_INFERENCE` is `False` by default. The optional 128-particle, one-chain
  runs are workflow previews and can take several minutes; they are not
  publication-grade settings.
- The Colab links and setup cells follow the `main` branch. For reported work,
  use one commit for both the notebook URL and package installation, and record
  the ORCA version, priors, SMC settings, random seed, input checksum, and
  software environment.
- Bayes factors are sensitive to candidate models and priors. Repeat important
  fits with more particles, multiple chains or seeds, and reasonable prior
  sensitivity checks.
- Colab runtimes are temporary. Download result archives before disconnecting
  and store them with the corresponding input and analysis metadata.

## Privacy

Google Colab is a remote service. Upload only synthetic or institutionally
approved anonymised data. Do not upload names, clinical metadata, raw
microscopy, or a donor key that could reconnect codes to people. Treat exported
archives as sensitive when they contain the uploaded analysis table, and never
commit private data or results to this public repository.
