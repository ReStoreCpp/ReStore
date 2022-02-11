#!/bin/bash
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=48
#SBATCH -t 00:30:00
#SBATCH -p test
#SBATCH --mem=80000mb
#SBATCH --chdir=dna_rokasD1_48@10
#SBATCH -o out.txt
#SBATCH -e err.txt
#SBATCH --mail-type=END
#SBATCH --mail-user=<CHANGE>
#SBATCH --account=<CHANGE>
#SBATCH --ear=off
#SBATCH --switches=1

module restore gnu > /dev/null

DATASET="dna_rokasD1"
NAME="restore"
SEED=0
PREFIX_DIR="$(pwd)"
REPEATS=(0 1 2 3 4 5 6 7 8 9)
#REPEATS=(0)

FAIL_EVERY=5000
MAX_FAILURES=2

source ../run_with_restore.sh
