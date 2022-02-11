#! /usr/bin/python3

import argparse
import numpy as np
import math


def simulate(p, k, repetitions):
    shift = math.floor(p / k)
    blockRanges = np.array([np.roll(np.arange(p), replica * shift) for replica in range(k)])
    # print(blockRanges)
    PEBlockRanges = blockRanges.T
    # print(PEBlockRanges)

    roundsUntilFailure = []

    for repitition in range(repetitions):
        PEFailureOrder = np.random.permutation(np.arange(p))
        failCounts = np.repeat(0, p)

        failed = False
        for i in range(p):
            failedPE = PEFailureOrder[i]
            failedBlockRanges = PEBlockRanges[failedPE]
            for failedBlockRange in failedBlockRanges:
                failCounts[failedBlockRange] += 1
                if failCounts[failedBlockRange] >= k:
                    assert(not failed)
                    failed = True
                    roundsUntilFailure.append(i)
                    break
            if failed:
                break

    return roundsUntilFailure

parser = argparse.ArgumentParser(description="Simulates failures with the block distribution used in our actual implementation")
parser.add_argument("-p", "--processors", help="logarithm of max. number of processors (PEs) to use", type=int)
parser.add_argument("-k", "--replication_level", help="max. number of replicas to use for each block range", type=int, default=3)
parser.add_argument("-r", "--repetitions", help="Number of repetitions of the experiment", type=int, default=10)

args = parser.parse_args()

p = int(args.processors)
k = int(args.replication_level)
repetitions = int(args.repetitions)


logMaxPEs = p
# print(logMaxPEs)


replicationLevelsToSimulate = range(2, k+1)

results = []
for replicationLevel in replicationLevelsToSimulate:
    logMinPEs = math.ceil(math.log2(replicationLevel))
    # print(logMinPEs)
    numPEsToSimulate = [2**i for i in range(logMinPEs, logMaxPEs + 1)]
    # print(numPEsToSimulate)
    for numPEs in numPEsToSimulate:
        roundsUntilFailure = simulate(numPEs, replicationLevel, repetitions)
        resultsToAppend = [[numPEs, replicationLevel, rounds] for rounds in roundsUntilFailure]
        results.extend(resultsToAppend)

print("numPEs,k,roundsUntilDataLoss")
for numPEs, k, rounds in results:
    print(str(numPEs) + "," + str(k) + "," + str(rounds))
