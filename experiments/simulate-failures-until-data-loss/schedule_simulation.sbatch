#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH -t 24:00:00
#SBATCH -p normal
#SBATCH --mem=60000
#SBATCH -o simulations.out
#SBATCH -e simulations.err

source "$HOME/load-ulfm.sh"
cat simulations/indep_num_blocks.sh | parallel -P 20 -k > simulate_failures_untill_data_loss.csv  
cat simulations/indep_num_ranks_rankwise.sh | parallel -P 20 -k >> simulate_failures_untill_data_loss.csv  
cat simulations/indep_num_ranks_nodewise.sh | parallel -P 20 -k >> simulate_failures_untill_data_loss.csv  
cat simulations/indep_num_ranks_rackwise.sh | parallel -P 20 -k >> simulate_failures_untill_data_loss.csv  
