#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=40
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem=90000mb
#SBATCH --partition=multiple
#SBATCH --export=ALL
#SBATCH --mail-type=END
#SBATCH --mail-user=huebner@kit.edu

REPEAT_ID="${REPEAT_ID:-0}"
REPLICATION_LEVEL=4
NUM_ITERATIONS=100
NUM_DIMENSIONS=10
NUM_CENTERS=5
NUM_DATA_POINTS=2500000
SIMULATION_ID="${SIMULATION_ID:-unspecified}"
SEED=19121954
FAILURE_SIMULATOR_SEED="$((SEED + REPEAT_ID))"
FAILURE_RATE=0.05
MODE="${MODE:-cluster-data}"

PREFIX="p-$SLURM_NTASKS.k-$REPLICATION_LEVEL.dp-$NUM_DATA_POINTS.r-$REPEAT_ID" 

source ./load-modules.sh
module list

# Echo configuration
echo "number of ranks: $SLURM_NTASKS"
echo "replication level: $REPLICATION_LEVEL"
echo "number of iterations: $NUM_ITERATIONS"
echo "number of dimensions: $NUM_DIMENSIONS"
echo "number of clusters (k): $NUM_CENTERS"
echo "number of data points: $NUM_DATA_POINTS"
echo "simultion id: $SIMULATION_ID"
echo "seed: $SEED"
echo "failure simulator seed: $FAILURE_SIMULATOR_SEED"
echo "failure rate: $FAILURE_RATE"
echo "prefix: $PREFIX"

# Generate the data
echo -n "Generating data ... "
mpirun --bind-to core --map-by core -n $SLURM_NTASKS \
    ./k-means generate-data                       \
    --num-data-points-per-rank "$NUM_DATA_POINTS" \
    --num-dimensions "$NUM_DIMENSIONS"            \
    --seed "$SEED"                                \
    --output="data/$PREFIX"
echo "done."

#  Run k-means with fault-tolerance enabled and simulated failures
echo -n "Clustering data (fault-tolerance: ON)... "
mpirun --bind-to core --map-by core -n $SLURM_NTASKS \
    ./k-means cluster-data                        \
    --input="data/$PREFIX"                        \
    --output="ft-on.$PREFIX"                      \
    --num-centers "$NUM_CENTERS"                  \
    --num-iterations "$NUM_ITERATIONS"            \
    --replication-level "$REPLICATION_LEVEL"      \
    --fault-tolerance=true                        \
    --expected-failure-rate "$FAILURE_RATE"       \
    --simulation-id "$SIMULATION_ID"              \
    --repeat-id "$REPEAT_ID"                      \
    --seed "$SEED"                                \
    --failure-simulator-seed="$FAILURE_SIMULATOR_SEED"
echo "done"

#  Run k-means with fault-tolerance disabled
echo -n "Clustering data (fault-tolerance: OFF)... "
mpirun --bind-to core --map-by core -n $SLURM_NTASKS \
    ./k-means cluster-data                        \
    --input="data/$PREFIX"                        \
    --output="ft-off.$PREFIX"                     \
    --num-centers "$NUM_CENTERS"                  \
    --num-iterations "$NUM_ITERATIONS"            \
    --fault-tolerance=false                       \
    --simulation-id "$SIMULATION_ID"              \
    --repeat-id "$REPEAT_ID"                      \
    --seed "$SEED"                                
echo "done"
