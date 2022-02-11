#!/bin/bash

check_llh() {
    DATASET_DIR=$(grep chdir "$1"*.sbatch | sed -e "s/^.*=//")
    pushd "$DATASET_DIR" > /dev/null

    # for file in *.raxml.log; do
    #     filename_regex="^rep([[:digit:]]+)\.([^.]+)\.seed([[:digit:]]+)\.([[:alpha:]]+)\.raxml\.log$"
    #     if [[ $file =~ $filename_regex ]]; then
    #         rep="${BASH_REMATCH[1]}"
    #         dataset="${BASH_REMATCH[2]}"
    #         seed="${BASH_REMATCH[3]}"
    #         code="${BASH_REMATCH[4]}"
    #     else
    #         >&2 echo "Error: File $file does not match the pattern."
    #         exit 1
    #     fi

    # elapsed_time_regex="Elapsed time: ([0-9.]+) seconds"
    # if [[ $(grep -F "Elapsed time: " "$file") =~ $elapsed_time_regex ]]; then
    #     elapsed_time="${BASH_REMATCH[1]}"
    # else 
    #     >&2 echo "Error: File $file does not match the elapsed time regex"
    #     exit 1
    # fi

    file=out.txt
    for file in rep*.out; do
        llh_regex="Final LogLikelihood: (-[0-9.]+)"
        if [[ $(grep -F "Final LogLikelihood: " "$file") =~ $llh_regex ]]; then
            llh="${BASH_REMATCH[1]}"
        else
            >&2 echo "Error: File $file does not match the LLH regex"
            exit 1
        fi
            
        dataset=$1
        seed=0
        echo "$dataset,$seed,$rep,$llh"
    done
    
    popd > /dev/null
}

for ds in aa_rokasA1 aa_rokasA4 aa_rokasA8 dna_rokasD1 dna_PeteD8 dna_rokasD4 dna_rokasD7; do
    check_llh "$ds"
done

