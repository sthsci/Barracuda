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

sample_sizes="10,50,100,300,500,700,1000"
base_seed=309
replicates=5
scenarios="all"
T=1.0

# SMC controls.
chains=6
smc_particles=8000
smc_cores=6
prior_draws=0
n_quad=45

# Priors used by the fitted models.
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

results_root="../results/part_2_bf_trajectory"
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

echo "Using Python: $python_bin"
echo "Writing results to: $results_root"
mkdir -p "$results_root"

cmd=("$python_bin" operation_2_bf_trajectory.py
    --sample_sizes "$sample_sizes"
    --base_seed "$base_seed"
    --replicates "$replicates"
    --scenarios "$scenarios"
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

if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
    cmd+=(--force)
fi

if [[ "${NO_PROGRESSBAR:-0}" == "1" ]]; then
    cmd+=(--no_progressbar)
fi

"${cmd[@]}"
