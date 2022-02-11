#!/bin/bash

DATA_DIR="simulated_minimal"

add-field() {
    FIELD_NAME=$1
    FIELD_VALUE=$2
    sed -E "2,\$s/^/$FIELD_VALUE,/" | sed -E "1s/^/$FIELD_NAME,/"
}

remove-superfluous-csv-headers() {
    sed -E "2,$ { /^numberOfNodes,/d }"
}

concat-files() {
    for file in "$DATA_DIR"/rep*.simulated.seed*.p*.profiler.csv; do
        if [[ $(basename "$file") =~ rep([0-9]).simulated.seed([0-9]+).p([0-9]+).profiler.csv ]]; then
            REPETITION="${BASH_REMATCH[1]}"
            SEED="${BASH_REMATCH[2]}"
            NUM_NODES="${BASH_REMATCH[3]}"

            cat "$file"                                     \
                | add-field "repetition" "$REPETITION"      \
                | add-field "seed" "$SEED"                  \
                | add-field "numberOfNodes" "$NUM_NODES"
        else
            2> echo "$file did not match pattern!"
        fi
    done
}

concat-files | remove-superfluous-csv-headers
