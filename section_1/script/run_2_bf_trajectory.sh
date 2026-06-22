#!/bin/bash

set -euo pipefail

sample_sizes="10,20,30,50,100,200,300,400,500,1000"
base_seed=2026
replicates=5
chains=6
smc_particles=10000
std_prior_factor=5.0
lambda_prior_lower=-1.0
lambda_prior_upper=1.5
threshold=0.6
correlation_threshold=0.01

results_root="../results/part_2_bf_trajectory"

cmd=(python operation_2_bf_trajectory.py \
    --sample_sizes "$sample_sizes" \
    --base_seed "$base_seed" \
    --replicates "$replicates" \
    --chains "$chains" \
    --smc_particles "$smc_particles" \
    --std_prior_factor "$std_prior_factor" \
    --lambda_prior_lower "$lambda_prior_lower" \
    --lambda_prior_upper "$lambda_prior_upper" \
    --threshold "$threshold" \
    --correlation_threshold "$correlation_threshold" \
    --out_dir "$results_root")

if [[ "${FORCE_RERUN:-0}" == "1" ]]; then
    cmd+=(--force)
fi

"${cmd[@]}"
