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

