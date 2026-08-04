#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

###############################################################################
# User settings
###############################################################################

# Use "auto" unless you want to force a specific Python executable.
python_bin="auto"

# A fresh run label is required whenever any setting below changes.
run_label="homogeneous_vs_heterogeneous_hd_seed309"

# Each replicate simulates max(sample_sizes) cells once. Smaller datasets are
# nested prefixes of that same simulated population.
sample_sizes="5, 10, 50, 100, 250, 500, 750, 1000"
base_seed=309
replicates=5
T=1.0

# Controlled ground truths:
#   No1: heterogeneous history-dependent
#        sigma_eta=0.75, beta_x=0.8, beta_y=-0.8
#   No3: homogeneous history-dependent
#        sigma_eta=0.00, beta_x=0.8, beta_y=-0.8
scenarios="No1,No3"

# Fit only the two models needed to isolate decision-level heterogeneity.
models="homogeneous_history_dependent,heterogeneous_history_dependent"

# SMC controls.
chains=6
smc_particles=10000
smc_cores=6
prior_draws=0
n_quad=45

# Priors used by both fitted models.
lambda_prior_lower=-1.0
lambda_prior_upper=1.5
sigma_lambda_prior=2.0
p0_prior_alpha=1.0
p0_prior_beta=1.0
sigma_eta_prior=1.0
beta_prior_sd=1.0

# SMC tempering controls.
threshold=0.5
correlation_threshold=0.01

# Environment overrides:
#   FORCE_RERUN=1  regenerate simulations and refit every model
#   SHOW_PROGRESS=0 hide PyMC progress bars
#   DRY_RUN=1      print the command without running it
force_rerun="${FORCE_RERUN:-0}"
show_progress="${SHOW_PROGRESS:-1}"
dry_run="${DRY_RUN:-0}"

results_root="../results/part_2_bf_homogeneous_vs_heterogeneous/${run_label}"
pytensor_base_compiledir="${results_root}/pytensor_cache"
pytensor_compile_timeout=600
pytensor_compile_wait=10

###############################################################################
# End user settings
###############################################################################

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/.cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$pytensor_base_compiledir"
pytensor_base_compiledir="$(cd "$pytensor_base_compiledir" && pwd)"

if [[ -z "${PYTENSOR_FLAGS:-}" ]]; then
    PYTENSOR_FLAGS="optimizer_excluding=fusion"
fi

if [[ "$PYTENSOR_FLAGS" != *"base_compiledir="* ]]; then
    PYTENSOR_FLAGS="${PYTENSOR_FLAGS},base_compiledir=${pytensor_base_compiledir}"
fi

if [[ "$PYTENSOR_FLAGS" != *"compile__timeout="* ]]; then
    PYTENSOR_FLAGS="${PYTENSOR_FLAGS},compile__timeout=${pytensor_compile_timeout}"
fi

if [[ "$PYTENSOR_FLAGS" != *"compile__wait="* ]]; then
    PYTENSOR_FLAGS="${PYTENSOR_FLAGS},compile__wait=${pytensor_compile_wait}"
fi

export PYTENSOR_FLAGS

resolve_python() {
    local candidates=()
    local candidate

    if [[ "$python_bin" == "auto" ]]; then
        candidates=(
            "/opt/homebrew/Caskroom/miniforge/base/bin/python3"
            "python3"
        )
    else
        candidates=("$python_bin")
    fi

    for candidate in "${candidates[@]}"; do
        if ! command -v "$candidate" >/dev/null 2>&1; then
            continue
        fi

        if "$candidate" - <<'PY' >/dev/null 2>&1
import arviz
import numpy
import pandas
import pymc
import pytensor
PY
        then
            "$candidate" - <<'PY'
import sys
print(sys.executable)
PY
            return 0
        fi
    done

    return 1
}

if ! resolved_python="$(resolve_python)"; then
    echo "No usable Python found."
    echo "This experiment needs arviz, numpy, pandas, pymc, and pytensor."
    echo "Set python_bin to a compatible environment or install the dependencies."
    exit 1
fi

python_bin="$resolved_python"
mkdir -p "$results_root"

# operation_2_bf_trajectory.py can resume existing posterior files, but it does
# not itself reject every possible configuration mismatch. Refuse to mix an
# existing run with changed settings unless the user explicitly forces a rerun.
"$python_bin" - "$results_root" "$force_rerun" <<PY
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
path = root / "run_config.json"
force_rerun = int(sys.argv[2])

scenario_order = ["No1", "No2", "No3", "No4"]
model_order = [
    "homogeneous_history_independent",
    "homogeneous_history_dependent",
    "heterogeneous_history_independent",
    "heterogeneous_history_dependent",
]
wanted_scenarios = {
    value for value in "${scenarios}".replace(",", " ").split() if value
}
wanted_models = {
    value for value in "${models}".replace(",", " ").split() if value
}

current = {
    "sample_sizes": sorted(
        {int(value) for value in "${sample_sizes}".replace(",", " ").split()}
    ),
    "T": float("${T}"),
    "base_seed": int("${base_seed}"),
    "replicates": int("${replicates}"),
    "scenarios": [
        scenario for scenario in scenario_order if scenario in wanted_scenarios
    ],
    "models": [model for model in model_order if model in wanted_models],
    "chains": int("${chains}"),
    "smc_particles": int("${smc_particles}"),
    "smc_cores": int("${smc_cores}"),
    "prior_draws": int("${prior_draws}"),
    "n_quad": int("${n_quad}"),
    "lambda_prior_bounds": [
        float("${lambda_prior_lower}"),
        float("${lambda_prior_upper}"),
    ],
    "sigma_lambda_prior": float("${sigma_lambda_prior}"),
    "p0_prior": [
        float("${p0_prior_alpha}"),
        float("${p0_prior_beta}"),
    ],
    "sigma_eta_prior": float("${sigma_eta_prior}"),
    "beta_prior_sd": float("${beta_prior_sd}"),
    "threshold": float("${threshold}"),
    "correlation_threshold": float("${correlation_threshold}"),
}

if path.exists():
    previous = json.loads(path.read_text())
    if previous != current and force_rerun != 1:
        raise SystemExit(
            f"{path} contains different settings. Change run_label for a new "
            "experiment. FORCE_RERUN=1 will refit the requested scope, but "
            "directories from settings no longer requested may remain."
        )
else:
    substantive_outputs = (
        list(root.rglob("posterior.nc"))
        + list(root.rglob("simulation_full.csv"))
        + list(root.glob("logml_summary.csv"))
    )
    if substantive_outputs and force_rerun != 1:
        raise SystemExit(
            f"{root} contains results but has no run_config.json. Use a fresh "
            "run_label, or set FORCE_RERUN=1 to regenerate the requested scope."
        )
PY

echo "Using Python: $python_bin"
echo "Writing results to: $results_root"
echo "Ground truths: No1 (heterogeneous HD), No3 (homogeneous HD)"
echo "Fitted models: $models"

cmd=("$python_bin" operation_2_bf_trajectory.py
    --sample_sizes "$sample_sizes"
    --base_seed "$base_seed"
    --replicates "$replicates"
    --scenarios "$scenarios"
    --models "$models"
    --T "$T"
    --chains "$chains"
    --smc_particles "$smc_particles"
    --smc_cores "$smc_cores"
    --prior_draws "$prior_draws"
    --n_quad "$n_quad"
    --lambda_prior_lower "$lambda_prior_lower"
    --lambda_prior_upper "$lambda_prior_upper"
    --sigma_lambda_prior "$sigma_lambda_prior"
    --p0_prior_alpha "$p0_prior_alpha"
    --p0_prior_beta "$p0_prior_beta"
    --sigma_eta_prior "$sigma_eta_prior"
    --beta_prior_sd "$beta_prior_sd"
    --threshold "$threshold"
    --correlation_threshold "$correlation_threshold"
    --out_dir "$results_root")

if [[ "$force_rerun" == "1" ]]; then
    cmd+=(--force)
fi

if [[ "$show_progress" == "0" ]]; then
    cmd+=(--no_progressbar)
fi

printf "Command:"
printf " %q" "${cmd[@]}"
printf "\n"

if [[ "$dry_run" == "1" ]]; then
    echo "DRY_RUN=1: command not executed."
    exit 0
fi

"${cmd[@]}"

# Convert the long model-evidence table into one directly usable pairwise
# Bayes-factor row per scenario, replicate, and cell count.
"$python_bin" - \
    "$results_root/logml_summary.csv" \
    "$results_root/pairwise_bayes_factors.csv" <<'PY'
import sys
from pathlib import Path

import numpy as np
import pandas as pd

summary_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
summary = pd.read_csv(summary_path)

homogeneous = "homogeneous_history_dependent"
heterogeneous = "heterogeneous_history_dependent"
required_models = {homogeneous, heterogeneous}
expected_scenarios = {"No1", "No3"}

required_columns = {
    "scenario",
    "scenario_label",
    "replicate",
    "simulation_seed",
    "n_cells",
    "gt_mean_lambda",
    "gt_sigma_lambda",
    "gt_p0",
    "gt_sigma_eta",
    "gt_beta_x",
    "gt_beta_y",
    "true_model",
    "model",
    "logml",
}
missing_columns = required_columns.difference(summary.columns)
if missing_columns:
    raise SystemExit(
        "Cannot calculate pairwise Bayes factors; missing columns: "
        + ", ".join(sorted(missing_columns))
    )

observed_scenarios = set(summary["scenario"].astype(str))
if observed_scenarios != expected_scenarios:
    raise SystemExit(
        "Expected exactly scenarios No1 and No3, found: "
        + ", ".join(sorted(observed_scenarios))
    )

observed_truths = set(summary["true_model"].astype(str))
if not observed_truths.issubset(required_models):
    raise SystemExit(
        "Unexpected true_model values: "
        + ", ".join(sorted(observed_truths.difference(required_models)))
    )

pair = summary[summary["model"].isin(required_models)].copy()
if not np.isfinite(pair["logml"].to_numpy(dtype=float)).all():
    raise SystemExit("Cannot calculate pairwise Bayes factors: logml is not finite.")

dataset_columns = ["scenario", "replicate", "n_cells"]
duplicate_counts = pair.groupby(
    [*dataset_columns, "model"],
    dropna=False,
).size()
duplicate_counts = duplicate_counts[duplicate_counts != 1]
if not duplicate_counts.empty:
    raise SystemExit(
        "Expected exactly one logml row per scenario/replicate/n_cells/model; "
        f"invalid groups: {duplicate_counts.to_dict()}"
    )

model_coverage = pair.groupby(dataset_columns, dropna=False)["model"].agg(set)
incomplete = model_coverage[
    model_coverage.map(lambda value: value != required_models)
]
if not incomplete.empty:
    raise SystemExit(
        "Missing a required model for dataset groups: "
        f"{incomplete.to_dict()}"
    )

metadata_columns = [
    "scenario",
    "scenario_label",
    "replicate",
    "simulation_seed",
    "n_cells",
    "gt_mean_lambda",
    "gt_sigma_lambda",
    "gt_p0",
    "gt_sigma_eta",
    "gt_beta_x",
    "gt_beta_y",
    "true_model",
]

metadata_consistency = pair.groupby(
    dataset_columns,
    dropna=False,
)[metadata_columns].nunique(dropna=False)
inconsistent_metadata = metadata_consistency[
    (metadata_consistency > 1).any(axis=1)
]
if not inconsistent_metadata.empty:
    raise SystemExit(
        "Metadata disagree between the two model rows for dataset groups: "
        f"{inconsistent_metadata.index.tolist()}"
    )

metadata = (
    pair.groupby(dataset_columns, as_index=False, dropna=False)[
        metadata_columns
    ]
    .first()
)
logml = (
    pair.pivot(
        index=dataset_columns,
        columns="model",
        values="logml",
    )
    .reset_index()
    .rename_axis(columns=None)
    .rename(
        columns={
            homogeneous: "logml_homogeneous_history_dependent",
            heterogeneous: "logml_heterogeneous_history_dependent",
        }
    )
)
wide = metadata.merge(
    logml,
    on=dataset_columns,
    how="inner",
    validate="one_to_one",
)

log_bf_heterogeneous_vs_homogeneous = (
    wide["logml_heterogeneous_history_dependent"]
    - wide["logml_homogeneous_history_dependent"]
)
wide["log_bf_heterogeneous_vs_homogeneous"] = (
    log_bf_heterogeneous_vs_homogeneous
)
wide["log10_bf_heterogeneous_vs_homogeneous"] = (
    log_bf_heterogeneous_vs_homogeneous / np.log(10.0)
)

wide["best_selected_model"] = np.select(
    [
        log_bf_heterogeneous_vs_homogeneous > 0,
        log_bf_heterogeneous_vs_homogeneous < 0,
    ],
    [heterogeneous, homogeneous],
    default="tie",
)
wide["ground_truth"] = wide["true_model"]
wide["log_bf_true_vs_alternative"] = np.where(
    wide["true_model"].eq(heterogeneous),
    log_bf_heterogeneous_vs_homogeneous,
    -log_bf_heterogeneous_vs_homogeneous,
)
wide["log10_bf_true_vs_alternative"] = (
    wide["log_bf_true_vs_alternative"] / np.log(10.0)
)

wide = wide.sort_values(["scenario", "replicate", "n_cells"]).reset_index(
    drop=True
)
temporary_path = output_path.with_name(f".{output_path.name}.tmp")
wide.to_csv(temporary_path, index=False)
temporary_path.replace(output_path)
print("Saved:", output_path)
PY

echo "Done."
echo "Long evidence summary: $results_root/logml_summary.csv"
echo "Pairwise BF summary:  $results_root/pairwise_bayes_factors.csv"
