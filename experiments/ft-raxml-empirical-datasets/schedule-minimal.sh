#!/bin/bash

NUM_PE=(1 2 4 8 16 32 64 128 256 512)

for p in "${NUM_PE[@]}"; do
    PARTITION="micro"
    if [[ $p -gt 16 ]]; then
        PARTITION="general"
    fi

    echo sbatch \
        --partition="$PARTITION" \
        --nodes="$p" \
        simulated_minimal.sbatch
done
