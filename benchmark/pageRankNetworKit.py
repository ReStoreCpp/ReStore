#! /usr/bin/python3

import networkit as nk
import sys
import argparse
import time
import subprocess
import os

import math

def test(inputDir, nodesWithScores, doFT):
    execCommand = [mpirun, "-np", "4", "--oversubscribe", executable]
    args = ["-p", "-s", "-f", "3", "-n", "100", "--seed", "4"]
    if doFT:
        args.extend(["--percentFailures", "0.25", "--enable-ft=true"])
    else:
        args.extend(["--percentFailures", "0.00", "--enable-ft=false"])

    fullCommand = execCommand + [inputDir] + args
    print(" ".join(fullCommand))
    mpiOutput = subprocess.check_output(fullCommand)
    mpiOutput = mpiOutput.decode('UTF-8')

    resultsStarted = False
    checkedCSV = False
    mpiResult = []
    numRanksPosition = -1
    numRanksFailedPosition = -1
    for line in mpiOutput.splitlines():
        print(line)
        if "RESULTS" not in line and not resultsStarted and "numRanks" in line:
            words = line.split(",")
            for i in range(len(words)):
                if words[i] == "numRanks":
                    numRanksPosition = i
                if words[i] == "numRanksFailed":
                    numRanksFailedPosition = i
        if "RESULTS" not in line and not resultsStarted and not "numRanks" in line:
            assert(numRanksPosition >= 0)
            assert(numRanksFailedPosition >= 0)
            words = line.split(",")
            assert(int(words[numRanksPosition]) == 4)
            if doFT:
                assert(int(words[numRanksFailedPosition]) >= 1)
            else:
                assert(int(words[numRanksFailedPosition]) == 0)
            checkedCSV = True
        if "RESULTS" in line:
            resultsStarted = True
        elif resultsStarted:
            words = line.split(" ")
            nodeId = int(words[0])
            nodeScore = float(words[1])
            mpiResult.append((nodeId, nodeScore))

    assert(checkedCSV)

    networkitScores = nodesWithScores
    networkitScores.sort(key=lambda nodeScore: nodeScore[1], reverse=True)
    networkitScores = networkitScores[:20]

    for networkitResult, mpiResult in zip(networkitScores, mpiResult):
        networkitNode = networkitResult[0]
        mpiNode = mpiResult[0]
        assert(networkitNode == mpiNode)
        networkitScore = networkitResult[1]
        mpiScore = mpiResult[1]
        assert(math.isclose(networkitScore, mpiScore, rel_tol=1e-5))
    print("SUCCESS")

parser = argparse.ArgumentParser(description="Calculate PageRank scores using NetworKit")
parser.add_argument('inputDir')
parser.add_argument("-s", "--sort", help="sort output by PageRank value",
                    action="store_true")
parser.add_argument("-o", "--outputPath", help="path to write the output")
parser.add_argument("-r", "--repetitions", help="number of repetitions to run", type=int, default=1)
parser.add_argument("-p", "--print", help="print the first 20 scores of the output",
                    action="store_true")
parser.add_argument("-t", "--test", help="test the mpi implementation against networkit results",
                    action="store_true")
parser.add_argument("-e", "--executable", help="path to the mpi executable")
parser.add_argument("-m", "--mpirun", help="path to the mpirun")

args = parser.parse_args()

inputDir = args.inputDir
outputPath = args.outputPath
sortOutput = args.sort
printOutput=args.print
repetitions = args.repetitions
doTest = args.test
executable = args.executable
mpirun = args.mpirun

if mpirun == "":
    mpirun = "mpirun"


G = nk.graph.Graph(weighted=False, directed=True)
for filename in os.listdir(inputDir):
    inputPath = os.path.join(inputDir, filename)
    if os.path.isfile(inputPath):
        with open(inputPath, "r") as graphFile:
            for line in graphFile:
                words = line.split(" ")
                if line.startswith("p") and G.numberOfNodes() == 0:
                    G.addNodes(int(words[1]))
                if line.startswith("e"):
                    u = int(words[1]) - 1
                    v = int(words[2]) - 1
                    G.addEdge(u,v)

start = time.time()
for i in range(repetitions):
    pr = nk.centrality.PageRank(G, damp=0.85, tol=0.000000001)
    pr.run()
end=time.time()
if not doTest:
    print("time: " +str((end - start)/repetitions))
scores = pr.scores()

assert(len(scores) == G.numberOfNodes())

nodesWithScores = list(zip(range(G.numberOfNodes()), scores))

if sortOutput:
    nodesWithScores.sort(key=lambda nodeScore: nodeScore[1], reverse=True)

if printOutput:
    for i in range(20):
        print(str(nodesWithScores[i]))

if doTest:
    test(inputDir, nodesWithScores, True)
    test(inputDir, nodesWithScores, False)
