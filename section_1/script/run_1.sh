#!/bin/bash

set -euo pipefail

cells=(1000 900 800 700 600 500 400 300 200 100 50 30 20 10)
seed=26
gt_mu_lambda=4.0
gt_sigma_lambda=3.0
gt_p0_lambda=0.2
std_prior_factor=2.0
lambda_prior_lower=-1.0
lambda_prior_upper=1.5

run_label="mu${gt_mu_lambda}_sigma${gt_sigma_lambda}_p0${gt_p0_lambda}_seed${seed}"
results_root="../results/part_1_runs/${run_label}"
latest_run_file="../results/part_1_latest.txt"
cumulative_sim_path="${results_root}/cumulative/simulation_data_ZI-gamma.npz"

echo "Writing results to: $results_root"
mkdir -p "$results_root"

python - "$results_root/run_config.json" <<PY
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
current = {
    "seed": int("${seed}"),
    "gt_mu_lambda": float("${gt_mu_lambda}"),
    "gt_sigma_lambda": float("${gt_sigma_lambda}"),
    "gt_p0_lambda": float("${gt_p0_lambda}"),
    "std_prior_factor": float("${std_prior_factor}"),
    "lambda_prior_bounds": [float("${lambda_prior_lower}"), float("${lambda_prior_upper}")],
}

if path.exists():
    previous = json.loads(path.read_text())
    if previous != current:
        raise SystemExit(
            f"{path} already exists with different settings. "
            "Change run_label in run_1.sh, or move/remove that result folder."
        )
else:
    path.write_text(json.dumps(current, indent=2, sort_keys=True))
PY

printf "part_1_runs/%s\n" "$run_label" > "$latest_run_file"

total=${#cells[@]}
count=1

for n in "${cells[@]}"; do
        echo "[$count/$total] n_cells=$n scenario=ZI-gamma"
        python operation_1.py \
            --n_cell "$n" \
            --gt_mu_lambda "$gt_mu_lambda" \
            --gt_sigma_lambda "$gt_sigma_lambda" \
            --gt_p0_lambda "$gt_p0_lambda" \
            --chains 6 \
            --smc_particles 10000 \
            --std_prior_factor "$std_prior_factor" \
            --lambda_prior_lower "$lambda_prior_lower" \
            --lambda_prior_upper "$lambda_prior_upper" \
            --out_dir "${results_root}/${n}/" \
            --cumulative_sim_path "$cumulative_sim_path" \
            --seed "$seed"
        ((count++))
done
