#!/bin/bash
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem=80000mb
#SBATCH --partition=test
#SBATCH --export=ALL
#SBATCH --mail-type=END
#SBATCH --mail-user=<CHANGE>
#SBATCH --account=<CHANGE>
#SBATCH --output=logs/%x.%j.out
#SBATCH --ear=off
#SBATCH --switches=1

# See https://github.com/open-mpi/ompi/issues/6981
export PMIX_MCA_gds=hash

module restore gnu
module list

NUM_REPETITIONS=10
MODE="${MODE:-random-ids}"
BENCHMARK_FILTER="${BENCHMARK_FILTER:-.*}"

echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of repititions: $NUM_REPETITIONS"
echo "Mode: $MODE"

mpirun -n $SLURM_NTASKS \
    "./microbenchmarks-$MODE" \
    --benchmark_out="microbenchmarks-$MODE-$SLURM_JOB_NUM_NODES.csv" \
    --benchmark_out_format=csv \
    --benchmark_repetitions="$NUM_REPETITIONS" \
    --benchmark_iterations=1 \
    --benchmark_filter="$BENCHMARK_FILTER"


