#!/bin/bash

MPI_EXECUTABLE=$1
NUMBER_OF_RANKS=$2
TEST_EXECUTABLE=${*:3}

# Create a temporary directory to save the output files with the return codes to.
tmpdir=$(mktemp --directory)

# Create the output_return_code.sh wrapper script. This script runs the given executable and 
# outputs the return code to a temporary file in the given directory.
cat > "$tmpdir/output_exit_code.sh" << EOF
#!/bin/bash
ec_file=\$(mktemp --tmpdir=$tmpdir --suffix=.ec)
"\$@"
echo \$? > "\$ec_file"
EOF
chmod +x "$tmpdir/output_exit_code.sh"

# Run the test executable under MPI with the given number of ranks.
"$MPI_EXECUTABLE" -n "$NUMBER_OF_RANKS" "$tmpdir/output_exit_code.sh" "$TEST_EXECUTABLE"

# Check the return codes, returning 1 if at least one of them failed
my_exit_code=0
for file in "$tmpdir"/*.ec; do
    ec=$(cat "$file")
    if [[ "$ec" != "0" && "$ec" != "" ]]; then
        if [[ "$ec" == "42" ]]; then
            echo "A rank simulated a failue."
        else
            echo "A rank FAILED with exit code $ec"
            my_exit_code=1
        fi
    else
        echo "Rank returned $ec"
    fi
done

# Clean up and return
rm --recursive "$tmpdir"
exit "$my_exit_code"
