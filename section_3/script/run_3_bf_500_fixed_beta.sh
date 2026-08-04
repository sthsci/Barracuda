#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

###############################################################################
# User settings
###############################################################################

python_bin="${PYTHON_BIN:-/opt/homebrew/Caskroom/miniforge/base/bin/python3}"

n_cells=500
replicates=5
scenarios="No1,No3"
base_seed=309
T=1.0

chains=6
smc_particles=10000
smc_cores=6
prior_draws=0
n_quad=60

lambda_prior_lower=-1.0
lambda_prior_upper=1.5
sigma_lambda_prior=2.0
p0_prior_alpha=1.0
p0_prior_beta=1.0
sigma_eta_prior=1.0
threshold=0.5
correlation_threshold=0.01

# Resume saved simulations and fits by default. Set FORCE_RERUN=1 only when
# every requested simulation and model fit should be regenerated.
force_rerun="${FORCE_RERUN:-0}"
show_progress="${SHOW_PROGRESS:-1}"
dry_run="${DRY_RUN:-0}"

results_root="../results/part_3_bf_500_fixed_beta"

###############################################################################
# End user settings
###############################################################################

if [[ ! -x "$python_bin" ]]; then
    echo "Python executable not found: $python_bin"
    exit 1
fi

cmd=("$python_bin" operation_3_bf_500_fixed_beta.py
    --n_cells "$n_cells"
    --replicates "$replicates"
    --scenarios "$scenarios"
    --base_seed "$base_seed"
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

echo "Done."
echo "Results: $results_root/logml_summary_n${n_cells}.csv"
