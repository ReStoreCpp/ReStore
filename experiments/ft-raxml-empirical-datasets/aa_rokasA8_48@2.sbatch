#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH -t 12:00:00
##SBATCH -t 1:10:00
#SBATCH -p micro
#SBATCH --mem=80000mb
#SBATCH --chdir=aa_rokasA8_48@2
#SBATCH -o out.txt
#SBATCH -e err.txt
#SBATCH --mail-type=END
#SBATCH --mail-user=<CHANGE>
#SBATCH --account=<CHANGE>
#SBATCH --ear=off
#SBATCH --switches=1

module restore gnu > /dev/null

DATASET="aa_rokasA8"
NAME="restore"
SEED=0
PREFIX_DIR="$(pwd)"
REPEATS=(0 1 2 3 4 5 6 7 8 9)
#REPEATS=(0)

FAIL_EVERY=2000
MAX_FAILURES=2

source ../run_with_restore.sh
