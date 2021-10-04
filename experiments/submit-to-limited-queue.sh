#!/bin/bash

# Parse command line arguments
if [[ "$#" -ne 1 ]]; then
    echo "Usage $0 <sbatch-calls-file>"
    exit 1
fi

SBATCH_CALL_FILE=$1
TMP_FILE=".$SBATCH_CALL_FILE.tmp"
PROGRESS_FILE=".$SBATCH_CALL_FILE.progress"

# Check if our temporary file already exists.
if [[ -f "$TMP_FILE" ]]; then
    echo "The temporary file $TMP_FILE already exists."
fi

cmds_remaining=0
# Process a single command from the input file
process-command() {
    cmd="$1"
    outfile="$2"

    if ! eval "$cmd"; then
        echo "$cmd" >> "$outfile"
        cmds_remaining=$(( cmds_remaining + 1 ))
    fi
}

# Main loop. Read the input file line by line, assuming one command per line.
while read -r line; do
    process-command "$line" "$TMP_FILE"
done < "$SBATCH_CALL_FILE"

# Move the temporary output file to the input files location. This means only
# those command who where not successfully run remain in the input file.
mv "$TMP_FILE" "$SBATCH_CALL_FILE"

# Output progress report
echo "$cmds_remaining jobs remain to be scheduled."
