#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

###############################################################################
# User settings
# Edit this block only for normal runs.
###############################################################################

# Use "auto" unless you want to force a specific Python executable.
# The selected Python must have arviz, pymc, and pytensor installed.
python_bin="auto"
run_label="complex_part1_main_seed309"
force_rerun=1
show_progress=1

# Cumulative sample sizes. The same simulated population is reused and sliced.
sample_sizes=(10 50 100 300 500 700 1000)

# Synthetic ground truth.
seed=309
T=1.0
gt_mean_lambda=4.0
gt_sigma_lambda=2.0
gt_p0=0.25
gt_sigma_eta=0.75
gt_beta_x=0.8
gt_beta_y=-0.8

# SMC controls.
chains=6
smc_particles=5000
smc_cores=6
prior_draws=2000
n_quad=60

# Priors used by the fitted model.
lambda_prior_lower=-1.0
lambda_prior_upper=1.5
sigma_lambda_prior=2.0
p0_prior_alpha=1.0
p0_prior_beta=1.0
sigma_eta_prior=1.0
beta_prior_sd=1.0

# SMC tempering controls. Usually leave these alone.
threshold=0.5
correlation_threshold=0.01

###############################################################################
# End user settings
###############################################################################

export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TMPDIR:-/tmp}/.cache}"
mkdir -p "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

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
    echo "This script needs arviz, pymc, and pytensor."
    echo "Either set python_bin in the User settings block to a working Python,"
    echo "or install dependencies in your active environment, for example:"
    echo "  conda install -c conda-forge arviz pymc pytensor"
    exit 1
fi

python_bin="$resolved_python"
cells_string="${sample_sizes[*]}"
results_root="../results/part_1_complex_runs/${run_label}"
latest_run_file="../results/part_1_complex_latest.txt"
cumulative_sim_path="${results_root}/cumulative/simulation_data.csv"

echo "Using Python: $python_bin"
echo "Writing results to: $results_root"
mkdir -p "$results_root"

"$python_bin" - "$results_root/run_config.json" <<PY
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
current = {
    "cells": [int(x) for x in "${cells_string}".split()],
    "seed": int("${seed}"),
    "T": float("${T}"),
    "gt_mean_lambda": float("${gt_mean_lambda}"),
    "gt_sigma_lambda": float("${gt_sigma_lambda}"),
    "gt_p0": float("${gt_p0}"),
    "gt_sigma_eta": float("${gt_sigma_eta}"),
    "gt_beta_x": float("${gt_beta_x}"),
    "gt_beta_y": float("${gt_beta_y}"),
    "chains": int("${chains}"),
    "smc_particles": int("${smc_particles}"),
    "smc_cores": int("${smc_cores}"),
    "prior_draws": int("${prior_draws}"),
    "n_quad": int("${n_quad}"),
    "sigma_lambda_prior": float("${sigma_lambda_prior}"),
    "p0_prior": [float("${p0_prior_alpha}"), float("${p0_prior_beta}")],
    "sigma_eta_prior": float("${sigma_eta_prior}"),
    "beta_prior_sd": float("${beta_prior_sd}"),
    "lambda_prior_bounds": [float("${lambda_prior_lower}"), float("${lambda_prior_upper}")],
    "threshold": float("${threshold}"),
    "correlation_threshold": float("${correlation_threshold}"),
}

if path.exists():
    previous = json.loads(path.read_text())
    if previous != current and int("${force_rerun}") != 1:
        raise SystemExit(
            f"{path} already exists with different settings. "
            "Change run_label or set force_rerun=1 in run_1.sh."
        )

path.write_text(json.dumps(current, indent=2, sort_keys=True))
PY

printf "part_1_complex_runs/%s\n" "$run_label" > "$latest_run_file"

total=${#sample_sizes[@]}
count=1

for n in "${sample_sizes[@]}"; do
    echo "[$count/$total] n_cells=$n scenario=complex_history_dependent"

    cmd=("$python_bin" operation_1.py
        --n_cell "$n"
        --T "$T"
        --gt_mean_lambda "$gt_mean_lambda"
        --gt_sigma_lambda "$gt_sigma_lambda"
        --gt_p0 "$gt_p0"
        --gt_sigma_eta "$gt_sigma_eta"
        --gt_beta_x "$gt_beta_x"
        --gt_beta_y "$gt_beta_y"
        --seed "$seed"
        --chains "$chains"
        --smc_particles "$smc_particles"
        --smc_cores "$smc_cores"
        --prior_draws "$prior_draws"
        --n_quad "$n_quad"
        --sigma_lambda_prior "$sigma_lambda_prior"
        --p0_prior_alpha "$p0_prior_alpha"
        --p0_prior_beta "$p0_prior_beta"
        --sigma_eta_prior "$sigma_eta_prior"
        --beta_prior_sd "$beta_prior_sd"
        --lambda_prior_lower "$lambda_prior_lower"
        --lambda_prior_upper "$lambda_prior_upper"
        --threshold "$threshold"
        --correlation_threshold "$correlation_threshold"
        --out_dir "${results_root}/${n}"
        --cumulative_sim_path "$cumulative_sim_path")

    if [[ "$force_rerun" == "1" ]]; then
        cmd+=(--force)
    fi

    if [[ "$show_progress" == "0" ]]; then
        cmd+=(--no_progressbar)
    fi

    "${cmd[@]}"
    ((count++))
done
