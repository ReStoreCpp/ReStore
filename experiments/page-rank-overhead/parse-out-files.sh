#!/bin/bash

# Ignore the jobs that TIMEOUTed
if [ ! -d timeout ]; then
    mkdir timeout
fi

grep --files-with-matches TIMEOUT *.out | xargs -I {} -n1 mv {} timeout/

# Extract the measurements from each slurm-*.out file.
for file in slurm-*.out; do
    basename="${file%.out}"

    # Only keep the CSV output.
    grep '^numRanks,' --after 1 "$file" > "$basename.csv";

    # Split the csv output in fault-tolerance on/off
    head -2 "$basename.csv" > "$basename.csv.ft-on"
    tail -2 "$basename.csv" > "$basename.csv.ft-off"
done

# Collect the measurments in one file for ft ON an OFF respectively.
FT_ON_OUT_FILE="measurements-ft-on.csv"
FT_OFF_OUT_FILE="measurements-ft-off.csv"
if [ -f "$FT_ON_OUT_FILE" -o -f "$FT_OFF_OUT_FILE" ]; then
    echo "At least on of the output files $FT_ON_OUT_FILE or $FT_OFF_OUT_FILE exists."
    echo "To not override them, we'll stop here. Please delete both of those files" \
         "and rerun this script"
    exit 1
fi
cat *.csv.ft-on > "$FT_ON_OUT_FILE"
cat *.csv.ft-off > "$FT_OFF_OUT_FILE"

sed -i -e "2,$ {/numRanks,/d}" measurements-ft-on.csv
sed -i -e "2,$ {/numRanks,/d}" measurements-ft-off.csv

# Clean up
rm slurm-*.csv
rm slurm-*.csv.ft-on
rm slurm-*.csv.ft-off
