#!/bin/bash

BASENAME="microbenchmarks"

remove-summary-rows() {
    sed -e "/_mean/d" -e "/_median/d" -e "/_stddev/d"
}

keep-first-five-fields() {
    cut -d, -f1-5
}

expand-name-field() {
    # sed -E "2,\$s/\"BM_(pushBlocksSmallRange)\/([0-9]+)\/([0-9]+)\/([0-9]+)\/([0-9]+)\/([0-9]+)\/manual_time\"/\1,\2,\3,\4,\5,\6/" | 
    # sed -E "2,\$s/\"BM_(pullBlocksSmallRange)\/([0-9]+)\/([0-9]+)\/([0-9]+)\/([0-9]+)\/([0-9]+)\/manual_time\"/\1,\2,\3,\4,\5,\6/" | 
    # sed -E "2,\$s/\"BM_(submitBlocks)\/([0-9]+)\/([0-9]+)\/([0-9]+)\/([0-9]+)\/manual_time\"/\1,\2,\3,\4,,\5/" | 
    sed -E "2,\$s/\"BM_([a-zA-Z]+)\/([0-9]+)\/([0-9]+)\/([0-9]+)\/([0-9]+)\/([0-9]+)\/iterations:1\/manual_time\"/\1,\2,\3,\4,\5,\6/" | 

    sed -E "1s/name,/benchmark,bytesPerBlock,replicationLevel,bytesPerRank,blocksPerPermutationRange,promilleOfRanksThatFail,/"
}

add-field() {
    FIELD_NAME=$1
    FIELD_VALUE=$2
    sed -E "2,\$s/^/$FIELD_VALUE,/" | sed -E "1s/^/$FIELD_NAME,/"
}

remove-googlebenchmark-preample() {
    sed -E "1,10d"
}

concat-files() {
    for mode in "random-ids" "no-random-ids"; do
        for file in "$BASENAME-$mode-"*.csv; do
            p="${file%.csv}"
            p="${p#$BASENAME-$mode-}"

            cat "$file" \
                | remove-googlebenchmark-preample \
                | remove-summary-rows \
                | keep-first-five-fields \
                | expand-name-field \
                | add-field "numberOfNodes" "$p" \
                | add-field "idRandomization" "$mode"
        done 
    done
}

concat-files | sed \
    -e "2,$ { /^idRandomization/d }" \
    -e "s/^no-random-ids/FALSE/" \
    -e "s/random-ids/TRUE/"

