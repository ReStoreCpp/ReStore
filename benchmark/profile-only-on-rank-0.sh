#!/bin/bash

PROFILING_RANK="${PROFILING_RANK-0}"
GPERF_LIB="/usr/lib/x86_64-linux-gnu/libprofiler.so.0"
PPROF="${PPROF-$(which pprof)}"
if [ "$PPROF" == "" ]; then
    echo "pprof not found, please specify \$PPROF"
    exit 2
fi

print-usage-and-exit() {
    echo "Usage: $0 <tool> <binary> [<arg1>, ...]"
    echo "where <tool> is one of: callgrind, gperf"
    exit 1
}

# Parse arguments
if [ $# -lt 1 ]; then
    print-usage-and-exit
fi

PROFILING_TOOL="$1"
if [ "$PROFILING_TOOL" != "callgrind" ] && [ "$PROFILING_TOOL" != "gperf" ]; then
    print-usage-and-exit
fi

shift 1
BINARY=$1
CMD=("$@")

if [ "$OMPI_COMM_WORLD_RANK" == "$PROFILING_RANK" ]; then
    if [ "$PROFILING_TOOL" == "callgrind" ]; then
        valgrind --tool=callgrind "${CMD[@]}"
    elif [ "$PROFILING_TOOL" == "gperf" ]; then
        LD_PRELOAD="$GPERF_LIB" CPUPROFILE="$BINARY.prof" "${CMD[@]}"
        $PPROF --callgrind "$BINARY" "$BINARY.prof" > "$BINARY.callgrind"
    fi
else
    "${CMD[@]}"
fi
