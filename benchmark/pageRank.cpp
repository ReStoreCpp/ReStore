#include "restore/core.hpp"

#include <algorithm>
#include <bits/stdint-uintn.h>
#include <cmath>
#include <cstddef>
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <numeric>
#include <optional>
#include <restore/helpers.hpp>
#include <streambuf>
#include <string.h>
#include <string>
#include <utility>

using node_t = int;

using edge_id_t = int;

struct edge_t {
    node_t from;
    node_t to;
    edge_t(node_t u, node_t v) : from(u), to(v) {}
};

auto comm = MPI_COMM_WORLD;

std::tuple<node_t, edge_id_t, std::vector<edge_t>, std::vector<node_t>> readGraph(std::string graph) {
    int myRank, numRanks;
    MPI_Comm_rank(comm, &myRank);
    MPI_Comm_size(comm, &numRanks);

    std::ifstream infile(graph);

    std::string line;

    node_t              numVertices           = 0;
    edge_id_t           numEdges              = 0;
    edge_id_t           numEdgesRead          = 0;
    edge_id_t           numEdgesPerRank       = 0;
    edge_id_t           numRanksWithMoreEdges = 0;
    std::vector<edge_t> edges;
    std::vector<node_t> outDegrees;
    while (std::getline(infile, line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        char               letter;
        node_t             firstNum, secondNum;
        iss >> letter >> firstNum >> secondNum;
        if (letter == 'p') {
            if (numVertices != 0 || numEdges != 0) {
                if (myRank == 0)
                    std::cout << "Multiple lines starting with p. This is not supported!" << std::endl;
                exit(1);
            }
            assert(numEdgesRead == 0);
            numVertices = firstNum;
            numEdges    = secondNum;
            outDegrees.resize(static_cast<size_t>(numVertices));
            numEdgesPerRank       = numEdges / numRanks;
            numRanksWithMoreEdges = numEdges & numRanks;
        } else if (letter == 'e') {
            --firstNum;
            --secondNum;
            if (numVertices == 0 || numEdges == 0) {
                if (myRank == 0)
                    std::cout << "First edge before specifying number of vertices and edges. This is not supported!"
                              << std::endl;
                exit(1);
            }
            if (firstNum > numVertices) {
                if (myRank == 0)
                    std::cout << "Invalid vertex id " << firstNum << std::endl;
                exit(1);
            }
            if (secondNum > numVertices) {
                if (myRank == 0)
                    std::cout << "Invalid vertex id " << secondNum << std::endl;
                exit(1);
            }
            ++outDegrees[static_cast<size_t>(firstNum)];
            const int lowerBound = numEdgesPerRank * myRank + std::min(myRank, numRanksWithMoreEdges);
            const int upperBound = numEdgesPerRank * (myRank + 1) + std::min(myRank + 1, numRanksWithMoreEdges);
            if (numEdgesRead >= lowerBound && numEdgesRead < upperBound) {
                edges.emplace_back(firstNum, secondNum);
            }
            ++numEdgesRead;
        } else {
            if (myRank == 0)
                std::cout << "Unsupported type: " << letter << std::endl;
            exit(1);
        }
    }
    if (numEdges != numEdgesRead) {
        if (myRank == 0)
            std::cout << "Expected " << numEdges << " edges but found " << numEdgesRead << std::endl;
        exit(1);
    }
    assert(static_cast<edge_id_t>(edges.size()) == numEdgesPerRank + (myRank < numRanksWithMoreEdges));
    return std::make_tuple(numVertices, numEdges, edges, outDegrees);
}

// double calcL2Norm(const std::vector<double>& vec) {
//     double qsum = std::accumulate(
//         vec.begin(), vec.end(), 0.0, [](const double lhs, const double rhs) { return lhs + rhs * rhs; });
//     return std::sqrt(qsum);
// }

constexpr double getActualPageRank(const double storedPagerank, const double teleport, const double dampening) {
    return teleport + dampening * storedPagerank;
}

double calcDiffL2Norm(const std::vector<double>& lhs, const std::vector<double>& rhs) {
    double qsum = 0;
    assert(lhs.size() == rhs.size());
    for (node_t i = 0; i < static_cast<node_t>(lhs.size()); ++i) {
        double diff = lhs[static_cast<size_t>(i)] - rhs[static_cast<size_t>(i)];
        qsum += diff * diff;
    }
    return sqrt(qsum);
}

std::vector<double> pageRank(
    const node_t numVertices, const edge_id_t numEdges, const std::vector<edge_t>& edges,
    const std::vector<node_t>& nodeDegrees, const double dampening, const double tol) {
    UNUSED(numEdges);
    int myRank;
    int numRanks;
    MPI_Comm_rank(comm, &myRank);
    MPI_Comm_size(comm, &numRanks);
    std::vector<double> prevPageRanks(static_cast<size_t>(numVertices), 0);
    std::vector<double> currPageRanks(static_cast<size_t>(numVertices), 1 / (double)numVertices);
    std::vector<double> tempPageRanks(static_cast<size_t>(numVertices), 0);
    const int           n        = static_cast<int>(nodeDegrees.size());
    const double        teleport = (1.0 - dampening) / n;
    while (calcDiffL2Norm(prevPageRanks, currPageRanks) > tol) {
        std::swap(prevPageRanks, currPageRanks);
        std::fill(currPageRanks.begin(), currPageRanks.end(), 0.0);
        for (size_t i = 0; i < edges.size(); ++i) {
            const size_t from             = static_cast<size_t>(edges[i].from);
            const size_t to               = static_cast<size_t>(edges[i].to);
            const size_t prefetchDistance = 20;
            if (i < edges.size() - prefetchDistance) {
                // __builtin_prefetch(&prevPageRanks[static_cast<size_t>(edges[i + prefetchDistance].from)], 0);
                // __builtin_prefetch(&nodeDegrees[static_cast<size_t>(edges[i + prefetchDistance].from)], 0);
                __builtin_prefetch(&currPageRanks[static_cast<size_t>(edges[i + prefetchDistance].to)], 1);
            }
            currPageRanks[to] += prevPageRanks[from] / nodeDegrees[from];
        }
        // TODO Make fault tolerant
        MPI_Allreduce(currPageRanks.data(), tempPageRanks.data(), n, MPI_DOUBLE, MPI_SUM, comm);
        std::swap(currPageRanks, tempPageRanks);
        std::for_each(currPageRanks.begin(), currPageRanks.end(), [teleport, dampening](double& value) {
            value = getActualPageRank(value, teleport, dampening);
        });
    }
    return currPageRanks;
}

void outputPageRanks(
    const std::vector<double>& pageRanks, const bool sort, size_t numVerticesToOutput, std::ostream& stream) {
    std::vector<size_t> indices(pageRanks.size());
    std::iota(indices.begin(), indices.end(), 0);

    if (sort) {
        std::sort(indices.begin(), indices.end(), [&pageRanks](const size_t i, const size_t j) {
            return pageRanks[i] > pageRanks[j];
        });
    }

    for (size_t i = 0; i < numVerticesToOutput; ++i) {
        auto node = indices[i];
        stream << node << " " << pageRanks[node] << std::endl;
    }
}

void writePageRanks(const std::vector<double>& pageRanks, const std::string path, const bool sort) {
    std::ofstream outfile(path);

    outputPageRanks(pageRanks, sort, pageRanks.size(), outfile);

    outfile.close();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int myRank;
    MPI_Comm_rank(comm, &myRank);

    cxxopts::Options cliParser("pageRank", "Benchmarks a fault tolerant page rank algorithm.");

    cliParser.add_options()                                                                                    ///
        ("graph", "path to an input graph file", cxxopts::value<std::string>())                                ///
        ("o,output", "path to the output file", cxxopts::value<std::string>())                                 ///
        ("s,sort", "sort the output", cxxopts::value<bool>()->default_value("false"))                          ///
        ("p,print", "print the first 20 scores of the output", cxxopts::value<bool>()->default_value("false")) ///
        ("d,dampening", "dampening factor.", cxxopts::value<double>()->default_value("0.85"))                  ///
        ("t,tolerance",
         "Tolerance for stopping PageRank iterations. Stops when the l2 norm of the difference between two iterations "
         "drops below the tolerance.",
         cxxopts::value<double>()->default_value("0.000000001"))                                       ///
        ("r,repetitions", "Number of repitions to run", cxxopts::value<size_t>()->default_value("10")) ///
        ("h,help", "Print help message.");

    cliParser.parse_positional({"graph"});
    cliParser.positional_help("<graph>");

    cxxopts::ParseResult options;
    try {
        options = cliParser.parse(argc, argv);
    } catch (cxxopts::OptionException& e) {
        if (myRank == 0)
            std::cout << e.what() << std::endl << std::endl;
        if (myRank == 0)
            std::cout << cliParser.help() << std::endl;
        exit(1);
    }

    if (options.count("help")) {
        if (myRank == 0)
            std::cout << cliParser.help() << std::endl;
        exit(0);
    }

    if (!options.count("graph")) {
        if (myRank == 0)
            std::cout << "Please provide a graph." << std::endl;
        if (myRank == 0)
            std::cout << cliParser.help() << std::endl;
        exit(1);
    }

    std::string outputPath;
    bool        doOutput = false;
    if (options.count("output")) {
        doOutput   = true;
        outputPath = options["output"].as<std::string>();
    }
    const bool sortOutput  = options["sort"].as<bool>();
    const bool printOutput = options["print"].as<bool>();

    const auto   numRepititions = options["repetitions"].as<size_t>();
    const double dampening      = options["dampening"].as<double>();
    const double tolerance      = options["tolerance"].as<double>();

    auto [numVertices, numEdges, edges, nodeDegrees] = readGraph(options["graph"].as<std::string>());
    std::sort(edges.begin(), edges.end(), [](const edge_t lhs, const edge_t rhs) { return lhs.from < rhs.from; });


    std::vector<double> result;
    auto                start = MPI_Wtime();
    for (size_t i = 0; i < numRepititions; ++i) {
        result = pageRank(numVertices, numEdges, edges, nodeDegrees, dampening, tolerance);
    }
    auto end        = MPI_Wtime();
    auto time       = end - start;
    auto timePerRun = time / static_cast<double>(numRepititions);

    if (myRank == 0) {
        std::cout << "Time per run: " << timePerRun << " s" << std::endl;
    }

    if (doOutput && myRank == 0) {
        writePageRanks(result, outputPath, sortOutput);
    }

    if (printOutput && myRank == 0) {
        outputPageRanks(result, sortOutput, 20, std::cout);
    }

    MPI_Finalize();
}
