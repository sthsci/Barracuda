#!/bin/bash

set -euo pipefail

cells=(50 100 150 200 300 400 500 600 700 800)

total=${#cells[@]}
count=1

for n in "${cells[@]}"; do
        echo "[$count/$total] n_cells=$n scenario=ZI-gamma"
        python operation_1.py \
            --n_cell "$n" \
            --chains 10 \
            --smc_particles 10000 \
            --out_dir "../results/part_1/${n}/" 
        ((count++))
done