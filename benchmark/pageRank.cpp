#include "restore/core.hpp"

#include <algorithm>
#include <bits/stdint-uintn.h>
#include <cmath>
#include <cstddef>
#include <cxxopts.hpp>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <numeric>
#include <optional>
#include <restore/helpers.hpp>
#include <string.h>
#include <string>

using node_t = int;

using edge_id_t = int;

struct edge_t {
    node_t from;
    node_t to;
};

auto comm = MPI_COMM_WORLD;

std::tuple<node_t, edge_id_t, std::vector<edge_t>, std::vector<node_t>> readGraph(std::string graph) {
    UNUSED(graph);
    return {};
}

// double calcL2Norm(const std::vector<double>& vec) {
//     double qsum = std::accumulate(
//         vec.begin(), vec.end(), 0.0, [](const double lhs, const double rhs) { return lhs + rhs * rhs; });
//     return std::sqrt(qsum);
// }

constexpr double getActualPageRank(const double storedPagerank, const node_t n, const double dampening) {
    return (1.0 - dampening) / n + dampening * storedPagerank;
}

double calcDiffL2Norm(const std::vector<double>& lhs, const std::vector<double>& rhs, const double dampening) {
    double qsum = 0;
    assert(lhs.size() == rhs.size());
    for (node_t i = 0; i < static_cast<node_t>(lhs.size()); ++i) {
        double diff = getActualPageRank(lhs[static_cast<size_t>(i)], static_cast<int>(lhs.size()), dampening)
                      - getActualPageRank(rhs[static_cast<size_t>(i)], static_cast<int>(lhs.size()), dampening);
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
    const int           n = static_cast<int>(nodeDegrees.size());
    while (calcDiffL2Norm(prevPageRanks, currPageRanks, dampening) > tol) {
        std::swap(prevPageRanks, currPageRanks);
        std::fill(currPageRanks.begin(), currPageRanks.end(), 0.0);
        for (const auto edge: edges) {
            size_t from = static_cast<size_t>(edge.from);
            size_t to   = static_cast<size_t>(edge.to);
            currPageRanks[to] += getActualPageRank(prevPageRanks[from], n, dampening)
                                 / getActualPageRank(nodeDegrees[from], n, dampening);
        }
        // TODO Make fault tolerant
        MPI_Allreduce(currPageRanks.data(), tempPageRanks.data(), n, MPI_DOUBLE, MPI_SUM, comm);
        std::swap(currPageRanks, tempPageRanks);
    }
    std::for_each(currPageRanks.begin(), currPageRanks.end(), [n, dampening](double& value) {
        value = getActualPageRank(value, n, dampening);
    });
    return currPageRanks;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    cxxopts::Options cliParser("pageRank", "Benchmarks a fault tolerant page rank algorithm.");

    cliParser.add_options()                                                                          ///
        ("graph", "path to an input graph file", cxxopts::value<std::string>())                      ///
        ("d,dampening", "dampening factor.", cxxopts::value<unsigned long>()->default_value("0.85")) ///
        ("t,tolerance",
         "Tolerance for stopping PageRank iterations. Stops when the l2 norm of the difference between two iterations "
         "drops below the tolerance.",
         cxxopts::value<unsigned long>()->default_value("0.000000001"))                                ///
        ("r,repetitions", "Number of repitions to run", cxxopts::value<size_t>()->default_value("10")) ///
        ("h,help", "Print help message.");

    cliParser.parse_positional({"graph"});
    cliParser.positional_help("<graph>");

    cxxopts::ParseResult options;
    try {
        options = cliParser.parse(argc, argv);
    } catch (cxxopts::OptionException& e) {
        std::cout << e.what() << std::endl << std::endl;
        std::cout << cliParser.help() << std::endl;
        exit(1);
    }

    if (options.count("help")) {
        std::cout << cliParser.help() << std::endl;
        exit(0);
    }

    if (!options.count("graph")) {
        std::cout << "Please provide a graph." << std::endl;
        std::cout << cliParser.help() << std::endl;
        exit(1);
    }

    const auto   numRepititions = options["repetitions"].as<size_t>();
    const double dampening      = options["dampening"].as<double>();
    const double tolerance      = options["tolerance"].as<double>();

    auto [numVertices, numEdges, edges, nodeDegrees] = readGraph(options["graph"].as<std::string>());

    for (size_t i = 0; i < numRepititions; ++i) {
        pageRank(numVertices, numEdges, edges, nodeDegrees, dampening, tolerance);
    }
}
