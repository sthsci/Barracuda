#!/bin/bash

set -euo pipefail

cells=(100 200 400 800 1000)
total_t=(40)

# Distributions
simul_dists=(gamma)
# auto = match simulation distribution
infer_dists=(gamma)

# infer_mode choices are: counts | both
modes=(counts both)

total=$(( ${#cells[@]} * ${#total_t[@]} * ${#modes[@]} * ${#simul_dists[@]} * ${#infer_dists[@]} ))
count=1

for n in "${cells[@]}"; do
    for ti in "${total_t[@]}"; do
        for dist_simul in "${simul_dists[@]}"; do
            for dist_infer in "${infer_dists[@]}"; do
                for mode in "${modes[@]}"; do
                    echo "[$count/$total] n_cells=$n T=$ti mode=$mode dist_simul=$dist_simul dist_infer=$dist_infer"
                    python simul_infer.py \
                        --n_cells "$n" \
                        --T "$ti" \
                        --infer_mode "$mode" \
                        --dist_simul "$dist_simul" \
                        --dist_infer "$dist_infer"
                    ((count++))
                done
            done
        done
    done
done