#!/bin/bash

set -euo pipefail

cells=(50 100 200 400 500 1000)
total_t=(80 120)
total=$(( ${#cells[@]} * ${#total_t[@]} * 3 ))
count=1

for n in "${cells[@]}"; do
    for ti in "${total_t[@]}"; do
        for mode in dt counts both; do
            echo "[$count/$total] n_cells=$n T=$ti mode=$mode"
            python simul_infer_full.py --n_cells "$n" --infer_mode "$mode" --T "$ti"
            ((count++))
        done
    done
done