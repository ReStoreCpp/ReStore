#!/bin/bash
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=48
##SBATCH -t 01:30:00
#SBATCH -t 00:15:00
#SBATCH -p test
#SBATCH --mem=600000mb
##SBATCH --mem=80000mb
#SBATCH --chdir=simulated_minimal
#SBATCH -o out.txt
#SBATCH -e err.txt
#SBATCH --mail-type=END
#SBATCH --mail-user=<CHANGE>
#SBATCH --account=<CHANGE>
#SBATCH --ear=off
#SBATCH --switches=1

# See https://github.com/open-mpi/ompi/issues/6981
export PMIX_MCA_gds=hash

module restore gnu > /dev/null

DATASET="simulated"
EXEC="ft-restore-raxml-minimal"
NAME="p$SLURM_JOB_NUM_NODES"
SEED=0
PREFIX_DIR="$(pwd)"
REPEATS=(0 1 2 3 4 5 6 7 8 9)
#REPEATS=(0)

# Disable failures
FAIL_EVERY=100000
MAX_FAILURES=0

source ../run_with_restore.sh
