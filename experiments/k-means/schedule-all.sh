#!/bin/bash

NUM_PE=(1 2 4 8 16 32 64 128 256 512)

if [ $# == 1 ]; then
    EXEC=$1
else
    EXEC=k-means-gnu-intelmpi.sbatch
fi

for p in "${NUM_PE[@]}"; do
    PARTITION="micro"
    if [[ $p -gt 16 ]]; then
        PARTITION="general"
    fi

    echo sbatch                     \
        --partition="$PARTITION"    \
        --nodes="$p"                \
        --time="30:00"              \
        "$EXEC"
done    
