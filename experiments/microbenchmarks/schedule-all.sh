#!/bin/bash

MODES=("random-ids" "no-random-ids")
NUM_PE=(1 2 4 8 16 32 64 128 256 512)

for MODE in "${MODES[@]}"; do
    for p in "${NUM_PE[@]}"; do
        PARTITION="micro"
        TIME="00:30:00"
        BENCHMARK_FILTER='".*"'
        if [[ $p -gt 16 ]]; then
            PARTITION="general"
        fi

        # The time limit depends on p an whether we are running
        # the disk bencharks (only active when in random-id mode).
        if [[ $MODE == "random-ids" ]]; then
            if [[ $p -ge 256 ]]; then
                TIME="03:00:00"
            else
                TIME="02:00:00"
            fi
        else
            BENCHMARK_FILTER="\"submit|pull\""
            if [[ $p -ge 256 ]]; then
                TIME="00:45:00"
            else
                TIME="00:30:00"
            fi
        fi

        echo sbatch                         \
            --partition="$PARTITION"        \
            --nodes="$p"                    \
            --export="ALL,MODE=$MODE,BENCHMARK_FILTER=$BENCHMARK_FILTER"       \
            --time="$TIME"                  \
            --job-name="microbenchmarks-$MODE"  \
            microbenchmarks.sbatch
    done    
done
