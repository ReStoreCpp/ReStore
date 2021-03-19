#!/bin/bash

PROFILING_RANK=0
EXE_NAME="microbenchmarks"

if [ "$OMPI_COMM_WORLD_RANK" == "$PROFILING_RANK" ]; then
    #LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libprofiler.so.0' CPUPROFILE="$EXE_NAME.prof" "./$EXE_NAME"
    valgrind --tool=cachegrind "./$EXE_NAME"
else
    "./$EXE_NAME"
fi
