#!/bin/bash
#SBATCH --nodes=100
#SBATCH --ntasks-per-node=40
#SBATCH -t 02:40:00
##SBATCH -t 00:50:00
#SBATCH -p general
#SBATCH --mem=80000mb
#SBATCH --chdir=aa_rokasA4_40@100
#SBATCH -o out.txt
#SBATCH -e err.txt
#SBATCH --mail-type=END
#SBATCH --mail-user=<CHANGE>
#SBATCH --account=<CHANGE>
#SBATCH --ear=off
#SBATCH --switches=1

module restore gnu > /dev/null

DATASET="aa_rokasA4"
NAME="restore"
SEED=0
PREFIX_DIR="$(pwd)"
REPEATS=(0 1 2)
#REPEATS=(0)

FAIL_EVERY=2000
MAX_FAILURES=2

source ../run_with_restore.sh
