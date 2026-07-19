#!/bin/bash

set -euo pipefail

# Run from this script's directory so relative paths always work.
cd "$(dirname "$0")"

# Fixed-N No1 ground-truth sensitivity analysis.
# Extra arguments are passed to Python, for example:
#   ./run_2_bf_groundtruth.sh --dry_run
#   ./run_2_bf_groundtruth.sh --force
python operation_2_bf_groundtruth.py \
    --n_cells 500 \
    --T 1.0 \
    --mu_lambda 4.0 \
    --baseline_sigma_lambda 3.0 \
    --baseline_p_zero 0.2 \
    --sigma_lambda_values "0.5,1,2,3,4,5,6" \
    --p_zero_values "0.05,0.1,0.2,0.3,0.4,0.5" \
    --base_seed 309 \
    --replicates 5 \
    --chains 6 \
    --smc_particles 10000 \
    --smc_cores 0 \
    --std_prior_factor 5.0 \
    --lambda_prior_lower -1.0 \
    --lambda_prior_upper 1.5 \
    --p_prior_alpha 1.0 \
    --p_prior_beta 1.0 \
    --threshold 0.6 \
    --correlation_threshold 0.01 \
    --reference_model hetero3 \
    --out_dir ../results/part_2_bf_groundtruth \
    "$@"
