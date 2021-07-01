#!/bin/bash

NO_COMPRESSION_FILENAME=microbenchmarks-no-compression.csv
YES_COMPRESSION_FILENAME=microbenchmarks-yes-compression.csv
TUNED_COMPRESSION_FILENAME=microbenchmarks-tuned-compression.csv

remove-summary-rows() {
    sed -e "/_mean/d" -e "/_median/d" -e "/_stddev/d"
}

keep-first-five-fields() {
    cut -d, -f1-5
}

expand-name-field() {
    sed -E "2,\$s/\"BM_([a-zA-Z]+)\/([0-9]+)\/([0-9]+)\/([0-9]+)\/manual_time\"/\1,\2,\3,\4/" | 
    sed -E "1s/^name,/benchmark,bytesPerBlock,replicationLevel,bytesPerRank,/"
}

add-code-field() {
    sed -E "2,\$s/^/$1,/" | sed -E "1s/^/code,/"
}

remove-csv-header() {
    sed -E "1d"
}

cat "$YES_COMPRESSION_FILENAME" | remove-summary-rows | keep-first-five-fields | expand-name-field | add-code-field "with-id-compression"
cat "$NO_COMPRESSION_FILENAME" | remove-summary-rows | keep-first-five-fields | expand-name-field | add-code-field "without-id-compression" | remove-csv-header
cat "$TUNED_COMPRESSION_FILENAME" | remove-summary-rows | keep-first-five-fields | expand-name-field | add-code-field "tuned-id-compression" | remove-csv-header
