#!/bin/bash
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=48
#SBATCH -t 20:00:00
#SBATCH -p micro
#SBATCH --mem=80000mb
#SBATCH --chdir=dna_PeteD8_48@5
#SBATCH -o out.txt
#SBATCH -e err.txt
#SBATCH --mail-type=END
#SBATCH --mail-user=<CHANGE>
#SBATCH --account=<CHANGE>
#SBATCH --ear=off
#SBATCH --switches=1

module restore gnu > /dev/null

DATASET="dna_PeteD8"
NAME="restore"
SEED=0
PREFIX_DIR="$(pwd)"
#REPEATS=(0 1 2)
REPEATS=(1 2)

FAIL_EVERY=2000
MAX_FAILURES=2

source ../run_with_restore.sh
