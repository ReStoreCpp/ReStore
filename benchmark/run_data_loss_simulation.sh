#!/bin/bash

# Alter the number of blocks, keeping the number of ranks and the replication level constant
## ranks = 100, replication level = 2
./simulate-failures-until-data-loss --config "100 1000 2" --repetitions 10
./simulate-failures-until-data-loss --config "100 10000 2" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "100 100000 2" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "100 1000000 2" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "100 10000000 2" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "100 100000000 2" --repetitions 10 --no-header
## ranks = 100, replication level = 3
./simulate-failures-until-data-loss --config "100 1000 3" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "100 10000 3" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "100 100000 3" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "100 1000000 3" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "100 10000000 3" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "100 100000000 3" --repetitions 10 --no-header
## ranks = 100, replication level = 4
./simulate-failures-until-data-loss --config "100 1000 4" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "100 10000 4" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "100 100000 4" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "100 1000000 4" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "100 10000000 4" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "100 100000000 4" --repetitions 10 --no-header
## ranks = 1000, replication level = 2
./simulate-failures-until-data-loss --config "1000 1000 2" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000 10000 2" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000 100000 2" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000 1000000 2" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000 10000000 2" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000 100000000 2" --repetitions 10 --no-header
## ranks = 1000, replication level = 3
./simulate-failures-until-data-loss --config "1000 1000 3" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000 10000 3" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000 100000 3" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000 1000000 3" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000 10000000 3" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000 100000000 3" --repetitions 10 --no-header
## ranks = 1000, replication level = 4
./simulate-failures-until-data-loss --config "1000 1000 4" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000 10000 4" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000 100000 4" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000 1000000 4" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000 10000000 4" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000 100000000 4" --repetitions 10 --no-header

# Alter the number of ranks, keeping the blocks and replication level constant
# 100000000 blocks, k = 2
./simulate-failures-until-data-loss --config "10 100000000 2" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "100 100000000 2" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000 100000000 2" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "10000 100000000 2" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "100000 100000000 2" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 2" --seed 0 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 2" --seed 1 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 2" --seed 2 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 2" --seed 3 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 2" --seed 4 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 2" --seed 5 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 2" --seed 6 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 2" --seed 7 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 2" --seed 8 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 2" --seed 9 --repetitions 1 --no-header
# 100000000 blocks, k = 3
./simulate-failures-until-data-loss --config "10 100000000 3" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "100 100000000 3" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000 100000000 3" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "10000 100000000 3" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "100000 100000000 3" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 3" --seed 0 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 3" --seed 1 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 3" --seed 2 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 3" --seed 3 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 3" --seed 4 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 3" --seed 5 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 3" --seed 6 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 3" --seed 7 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 3" --seed 8 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 3" --seed 9 --repetitions 1 --no-header
# 100000000 blocks, k = 4
./simulate-failures-until-data-loss --config "10 100000000 4" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "100 100000000 4" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000 100000000 4" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "10000 100000000 4" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "100000 100000000 4" --repetitions 10 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 4" --seed 0 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 4" --seed 1 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 4" --seed 2 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 4" --seed 3 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 4" --seed 4 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 4" --seed 5 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 4" --seed 6 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 4" --seed 7 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 4" --seed 8 --repetitions 1 --no-header
./simulate-failures-until-data-loss --config "1000000 100000000 4" --seed 9 --repetitions 1 --no-header
