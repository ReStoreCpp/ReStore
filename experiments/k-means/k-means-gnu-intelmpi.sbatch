#!/bin/bash
#SBATCH --nodes=4
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

RUN_FT_ON=${RUN_FT_ON:-true}
RUN_FT_OFF=${RUN_FT_OFF:-true}
REPEATS=(0 1 2 3 4 5 6 7 8 9)

REPLICATION_LEVEL=4
NUM_ITERATIONS=500

# 8 byte * 32 dimensions / data point * 65536 data points = 16 MiB of data
NUM_DIMENSIONS=32
NUM_DATA_POINTS=65536

NUM_CENTERS=20
BLOCKS_PER_PERMUTATION_RANGE=4096
SIMULATION_ID="${SIMULATION_ID:-unspecified}"

SEED=0
FAILURE_RATE=0.01

DATA_DIR="$WORK_pn72pu/restore/data/k-means/$SLURM_NTASKS" # Without trailing '/'
OUT_DIR="measurements" # Without trailing '/'
DATA_PREFIX="p-$SLURM_NTASKS.k-$REPLICATION_LEVEL.dp-$NUM_DATA_POINTS" 

# See https://github.com/open-mpi/ompi/issues/6981
#export PMIX_MCA_gds=hash

# Echo configuration
echo "number of ranks: $SLURM_NTASKS"
echo "replication level: $REPLICATION_LEVEL"
echo "number of iterations: $NUM_ITERATIONS"
echo "number of dimensions: $NUM_DIMENSIONS"
echo "number of clusters (k): $NUM_CENTERS"
echo "number of data points: $NUM_DATA_POINTS"
echo "simultion id: $SIMULATION_ID"
echo "seed: $SEED"
echo "failure rate: $FAILURE_RATE"
echo "blocks per permutation range: $BLOCKS_PER_PERMUTATION_RANGE"
echo "prefix: $DATA_PREFIX"
echo "run ft-on: $RUN_FT_ON"
echo "run ft-off: $RUN_FT_OFF"

module restore gnu-intelmpi > /dev/null
module list

# Generate the data
mkdir -p "$DATA_DIR"
echo -n "Generating data ... "
if [[ ! -f "$DATA_DIR/$DATA_PREFIX.0.data" ]]; then
    mpiexec -n $SLURM_NTASKS \
        ./k-means-gnu-intelmpi generate-data          \
        --num-data-points-per-rank "$NUM_DATA_POINTS" \
        --num-dimensions "$NUM_DIMENSIONS"            \
        --seed "$SEED"                                \
        --output="$DATA_DIR/$DATA_PREFIX"
    echo "done."
else
    echo "using existing data."
fi

for REPEAT_ID in "${REPEATS[@]}"; do
    OUT_PREFIX="$DATA_PREFIX.intelmpi.r-$REPEAT_ID" 
    FAILURE_SIMULATOR_SEED="$((SEED + REPEAT_ID))"
    
    #  Run k-means with fault-tolerance enabled and simulated failures
    if $RUN_FT_ON; then
    echo -n "Clustering data (fault-tolerance: ON)... "
        mpiexec -n $SLURM_NTASKS \
            ./k-means-gnu-intelmpi cluster-data           \
            --input="$DATA_DIR/$DATA_PREFIX"              \
            --output="$OUT_DIR/ft-on.$OUT_PREFIX"         \
            --num-centers "$NUM_CENTERS"                  \
            --num-iterations "$NUM_ITERATIONS"            \
            --replication-level "$REPLICATION_LEVEL"      \
            --fault-tolerance=true                        \
            --expected-failure-rate "$FAILURE_RATE"       \
            --simulation-id "$SIMULATION_ID"              \
            --repeat-id "$REPEAT_ID"                      \
            --seed "$SEED"                                \
            --blocks-per-permutation-range="$BLOCKS_PER_PERMUTATION_RANGE" \
            --failure-simulator-seed="$FAILURE_SIMULATOR_SEED" \
            --write-assignment=false
        echo "done"
    fi

    #  Run k-means with fault-tolerance disabled
    if $RUN_FT_OFF; then
        echo -n "Clustering data (fault-tolerance: OFF)... "
        mpiexec -n $SLURM_NTASKS \
            ./k-means-gnu-intelmpi cluster-data           \
            --input="$DATA_DIR/$DATA_PREFIX"              \
            --output="$OUT_DIR/ft-off.$OUT_PREFIX"        \
            --num-centers "$NUM_CENTERS"                  \
            --num-iterations "$NUM_ITERATIONS"            \
            --fault-tolerance=false                       \
            --simulation-id "$SIMULATION_ID"              \
            --repeat-id "$REPEAT_ID"                      \
            --seed "$SEED"                                \
            --blocks-per-permutation-range="$BLOCKS_PER_PERMUTATION_RANGE" \
            --write-assignment=false
        echo "done"
    fi
done
