#! /usr/bin/python3

import networkit as nk
import sys
import argparse
import time

parser = argparse.ArgumentParser(description="Calculate PageRank scores using NetworKit")
parser.add_argument("inputPath")
parser.add_argument("-s", "--sort", help="sort output by PageRank value",
                    action="store_true")
parser.add_argument("-o", "--outputPath", help="path to write the output")
parser.add_argument("-r", "--repetitions", help="number of repetitions to run", type=int, default=1)
parser.add_argument("-p", "--print", help="print the first 20 scores of the output",
                    action="store_true")

args = parser.parse_args()

inputPath = args.inputPath
outputPath = args.outputPath
sortOutput = args.sort
printOutput=args.print
repetitions = args.repetitions


G = nk.graph.Graph(weighted=False, directed=True)
with open(inputPath, "r") as graphFile:
    for line in graphFile:
        words = line.split(" ")
        if line.startswith("p"):
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
print("time: " +str((end - start)/repetitions))
scores = pr.scores()

assert(len(scores) == G.numberOfNodes())

nodesWithScores = list(zip(range(G.numberOfNodes()), scores))

if sortOutput:
    nodesWithScores.sort(key=lambda nodeScore: nodeScore[1], reverse=True)

if printOutput:
    for i in range(20):
        print(str(nodesWithScores[i]))
