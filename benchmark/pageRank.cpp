#include "restore/block_serialization.hpp"
#include "restore/common.hpp"
#include "restore/core.hpp"
#include "restore/equal_load_balancer.hpp"
#include "restore/helpers.hpp"
#include "restore/mpi_context.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <math.h>
#include <mpi.h>
#include <numeric>
#include <optional>
#include <stdint.h>
#include <streambuf>
#include <string.h>
#include <string>
#include <unordered_set>
#include <utility>

#if USE_FTMPI
    #include <mpi-ext.h>
#endif

static_assert(
    USE_FTMPI || SIMULATE_FAILURES,
    "When not simulating failures, you need to use a fault-tolerant MPI implementation.");

using node_t = int;

using edge_id_t = uint64_t;

struct edge_t {
    node_t from;
    node_t to;
    edge_t(node_t u, node_t v) noexcept : from(u), to(v) {}
    edge_t() noexcept : from(0), to(0) {}
};

auto comm    = MPI_COMM_WORLD;
bool amIDead = false;

using block_distribution_t = std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, ReStoreMPI::current_rank_t>>;

std::tuple<node_t, edge_id_t, edge_id_t, std::vector<edge_t>, std::vector<node_t>, block_distribution_t>
readGraph(std::string graph) {
    int myRank, numRanks;
    MPI_Comm_rank(comm, &myRank);
    MPI_Comm_size(comm, &numRanks);

    std::ifstream infile(graph);

    std::string line;

    node_t               numVertices           = 0;
    edge_id_t            numEdges              = 0;
    edge_id_t            numEdgesRead          = 0;
    edge_id_t            numEdgesPerRank       = 0;
    int                  numRanksWithMoreEdges = 0;
    std::vector<edge_t>  edges;
    std::vector<node_t>  outDegrees;
    edge_id_t            lowerBound = 0;
    edge_id_t            upperBound = 0;
    block_distribution_t blockDistribution;
    while (std::getline(infile, line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        char               letter;
        iss >> letter;
        if (letter == 'p') {
            if (numVertices != 0 || numEdges != 0) {
                if (myRank == 0)
                    std::cout << "Multiple lines starting with p. This is not supported!" << std::endl;
                exit(1);
            }
            assert(numEdgesRead == 0);
            node_t    firstNum;
            edge_id_t secondNum;
            iss >> firstNum >> secondNum;
            numVertices = firstNum;
            numEdges    = secondNum;
            outDegrees.resize(static_cast<size_t>(numVertices));
            numEdgesPerRank       = numEdges / asserting_cast<edge_id_t>(numRanks);
            numRanksWithMoreEdges = asserting_cast<int>(numEdges % asserting_cast<edge_id_t>(numRanks));
            lowerBound            = numEdgesPerRank * asserting_cast<edge_id_t>(myRank)
                         + asserting_cast<edge_id_t>(std::min(myRank, numRanksWithMoreEdges));
            upperBound =
                numEdgesPerRank * asserting_cast<edge_id_t>((myRank + 1) + std::min(myRank + 1, numRanksWithMoreEdges));

            for (int rank = 0; rank < numRanks; ++rank) {
                ReStore::block_id_t rankLowerBound = numEdgesPerRank * asserting_cast<edge_id_t>(rank)
                                                     + asserting_cast<edge_id_t>(std::min(rank, numRanksWithMoreEdges));
                size_t rankUpperBound =
                    numEdgesPerRank * asserting_cast<edge_id_t>((rank + 1) + std::min(rank + 1, numRanksWithMoreEdges));
                blockDistribution.emplace_back(std::make_pair(rankLowerBound, rankUpperBound - rankLowerBound), rank);
            }
        } else if (letter == 'e') {
            node_t firstNum, secondNum;
            iss >> firstNum >> secondNum;
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
    return std::make_tuple(numVertices, numEdges, lowerBound, edges, outDegrees, blockDistribution);
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

void recoverFromFailure(
    const edge_id_t numEdges, std::vector<edge_t>& edges, ReStore::ReStore<edge_t>& reStore,
    ReStore::EqualLoadBalancer& loadBalancer, int& myRank, int& numRanks) {
    MPI_Comm_rank(comm, &myRank);
    MPI_Comm_size(comm, &numRanks);
    reStore.updateComm(comm);
    UNUSED(numEdges);

    auto diedRanks = reStore.getRanksDiedSinceLastCall();
    auto requests  = loadBalancer.getNewBlocksAfterFailure(diedRanks);
    reStore.pushBlocksOriginalRankIds(
        requests, [&edges](const std::byte* dataPtr, size_t size, ReStore::block_id_t blockId) {
            assert(sizeof(edge_t) == size);
            UNUSED(size);
            UNUSED(blockId);
            edges.push_back(*reinterpret_cast<const edge_t*>(dataPtr));
        });
    loadBalancer.commitToPreviousCall();
    assert(std::all_of(edges.begin(), edges.end(), [](edge_t edge) { return !(edge.from == 0 && edge.to == 0); }));
}

std::unordered_set<int> ranksToKill;

template <class F>
bool fault_tolerant_mpi_call(const F& mpi_call) {
    assert(comm != MPI_COMM_NULL);

    if (SIMULATE_FAILURES) {
        if (ranksToKill.size() > 0) {
            MPI_Comm newComm;

            int rank;
            MPI_Comm_rank(comm, &rank);
            int color = 0;
            if (ranksToKill.find(rank) != ranksToKill.end()) {
                color   = 1;
                amIDead = true;
            }
            ranksToKill.clear();

            MPI_Comm_split(comm, color, rank, &newComm);
            MPI_Comm_free(&comm);
            comm = newComm;
            return false;
        } else {
            mpi_call();
            return true;
        }

    } else {
#if USE_FTMPI
        if (ranksToKill.size() > 0) {
            int rank;
            MPI_Comm_rank(comm, &rank);
            if (ranksToKill.find(rank) != ranksToKill.end()) {
                exit(42);
            }
            ranksToKill.clear();
        }
        int rc, ec;
        rc = mpi_call();
        MPI_Error_class(rc, &ec);

        if (ec == MPI_ERR_PROC_FAILED || ec == MPI_ERR_REVOKED) {
            if (ec == MPI_ERR_PROC_FAILED) {
                MPIX_Comm_revoke(comm);
            }

            // Build a new communicator without the failed ranks
            MPI_Comm newComm;
            if ((rc = MPIX_Comm_shrink(comm, &newComm)) != MPI_SUCCESS) {
                std::cerr << "A rank failure was detected, but building the new communicator failed" << std::endl;
                exit(1);
            }
            assert(comm != MPI_COMM_NULL);
            // As for the ULFM documentation, freeing the communicator is recommended but will probably
            // not succeed. This is why we do not check for an error here.
            // I checked that --mca mpi_show_handle_leaks 1 does not show a leaked handle
            MPI_Comm_free(&comm);
            comm = newComm;
            return false;
        } else if (rc != MPI_SUCCESS) {
            std::cerr << "MPI call did non fail because of a faulty rank but still did not return MPI_SUCCESS"
                      << std::endl;
            exit(1);
        }
        return true;
#endif
    }
}

std::vector<double> pageRank(
    const node_t numVertices, const edge_id_t numEdges, std::vector<edge_t>& edges,
    const std::vector<node_t>& nodeDegrees, const double dampening, const double tol, ReStore::ReStore<edge_t>& reStore,
    ReStore::EqualLoadBalancer& loadBalancer) {
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
        size_t i            = 0;
        bool   updatedEdges = false;
        bool   anotherPass  = false;
        do {
            anotherPass = false;
            for (; i < edges.size(); ++i) {
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
            static int crashCounter = 0;
            if ((crashCounter++ % 100) == 5) {
                ranksToKill = {0};
            }
            if (!fault_tolerant_mpi_call([&]() {
                    return MPI_Allreduce(currPageRanks.data(), tempPageRanks.data(), n, MPI_DOUBLE, MPI_SUM, comm);
                })) {
                if (amIDead) {
                    return currPageRanks;
                }
                recoverFromFailure(numEdges, edges, reStore, loadBalancer, myRank, numRanks);
                updatedEdges = true;
                anotherPass  = true;
            }
        } while (anotherPass);
        std::swap(currPageRanks, tempPageRanks);
        std::for_each(currPageRanks.begin(), currPageRanks.end(), [teleport, dampening](double& value) {
            value = getActualPageRank(value, teleport, dampening);
        });
        if (updatedEdges) {
            std::sort(
                edges.begin(), edges.end(), [](const edge_t lhs, const edge_t rhs) { return lhs.from < rhs.from; });
        }
    }
    const double sum = std::accumulate(currPageRanks.begin(), currPageRanks.end(), 0.0);
    std::for_each(currPageRanks.begin(), currPageRanks.end(), [sum](double& value) { value /= sum; });
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

void writePageRanks(const std::vector<double>& pageRanks, const std::string& path, const bool sort) {
    std::ofstream outfile(path);

    outputPageRanks(pageRanks, sort, pageRanks.size(), outfile);

    outfile.close();
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN);

    int myRank;
    MPI_Comm_rank(comm, &myRank);
    int numRanks;
    MPI_Comm_size(comm, &numRanks);

    cxxopts::Options cliParser("pageRank", "Benchmarks a fault tolerant page rank algorithm.");

    cliParser.add_options()                                                                                    ///
        ("graph", "path to an input graph file", cxxopts::value<std::string>())                                ///
        ("o,output", "path to the output file", cxxopts::value<std::string>())                                 ///
        ("s,sort", "sort the output", cxxopts::value<bool>()->default_value("false"))                          ///
        ("p,print", "print the first 20 scores of the output", cxxopts::value<bool>()->default_value("false")) ///
        ("d,dampening", "dampening factor.", cxxopts::value<double>()->default_value("0.85"))                  ///
        ("t,tolerance",
         "Tolerance for stopping PageRank iterations. Stops when the l2 norm of the difference between two "
         "iterations "
         "drops below the tolerance.",
         cxxopts::value<double>()->default_value("0.000000001"))                                         ///
        ("r,repetitions", "Number of repetitions to run", cxxopts::value<size_t>()->default_value("10")) ///
        ("f,replications", "Replications for fault tolerance with ReStore",
         cxxopts::value<size_t>()->default_value("3")) ///
        ("h,help", "Print help message.");

    cliParser.parse_positional({"graph"});
    cliParser.positional_help("<graph>");

    cxxopts::ParseResult options;
    try {
        options = cliParser.parse(argc, argv);
    } catch (cxxopts::OptionException& e) {
        if (myRank == 0) {
            std::cout << e.what() << std::endl << std::endl;
            std::cout << cliParser.help() << std::endl;
        }
        exit(1);
    }

    if (options.count("help")) {
        if (myRank == 0)
            std::cout << cliParser.help() << std::endl;
        exit(0);
    }

    if (!options.count("graph")) {
        if (myRank == 0) {
            std::cout << "Please provide a graph." << std::endl;
            std::cout << cliParser.help() << std::endl;
        }
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

    const auto   numRepetitions = options["repetitions"].as<size_t>();
    const double dampening      = options["dampening"].as<double>();
    const double tolerance      = options["tolerance"].as<double>();

    const auto numReplications = std::min(options["replications"].as<size_t>(), static_cast<size_t>(numRanks));

    auto start = MPI_Wtime();
    auto [numVertices, numEdges, firstEdgeId, edges, nodeDegrees, blockDistribution] =
        readGraph(options["graph"].as<std::string>());
    std::sort(edges.begin(), edges.end(), [](const edge_t lhs, const edge_t rhs) { return lhs.from < rhs.from; });
    auto end  = MPI_Wtime();
    auto time = end - start;
    if (myRank == 0) {
        std::cout << "Reading graph took " << time << " s" << std::endl;
    }

    ReStore::EqualLoadBalancer loadBalancer(blockDistribution, numRanks);

    start = MPI_Wtime();
    ReStore::ReStore<edge_t> reStore(
        comm, asserting_cast<uint16_t>(numReplications), ReStore::OffsetMode::constant, sizeof(edge_t));

    edge_id_t currentEdgeId = firstEdgeId;
    auto      getNextBlock  = [firstEdgeId = firstEdgeId, &currentEdgeId,
                         &edges      = edges]() -> std::optional<ReStore::NextBlock<edge_t>> {
        if (currentEdgeId - firstEdgeId >= asserting_cast<edge_id_t>(edges.size())) {
            return std::nullopt;
        } else {
            ++currentEdgeId;
            return ReStore::NextBlock<edge_t>{asserting_cast<ReStore::block_id_t>(currentEdgeId - 1),
                                              edges[asserting_cast<size_t>(currentEdgeId - 1 - firstEdgeId)]};
        }
    };

    auto serializeBlock = [](const edge_t edge, ReStore::SerializedBlockStoreStream& stream) {
        stream << edge.from << edge.to;
    };

    reStore.submitBlocks(serializeBlock, getNextBlock, asserting_cast<ReStore::block_id_t>(numEdges));
    end  = MPI_Wtime();
    time = end - start;
    if (myRank == 0) {
        std::cout << "Initializing ReStore and submitting took " << time << " s" << std::endl;
    }

    if (myRank == 0)
        std::cout << "Starting with " << numRanks << " ranks" << std::endl;

    MPI_Barrier(comm);
    std::vector<double> result;
    start = MPI_Wtime();
    for (size_t i = 0; i < numRepetitions; ++i) {
        result = pageRank(numVertices, numEdges, edges, nodeDegrees, dampening, tolerance, reStore, loadBalancer);
    }
    end             = MPI_Wtime();
    time            = end - start;
    auto timePerRun = time / static_cast<double>(numRepetitions);

    MPI_Comm_rank(comm, &myRank);
    MPI_Comm_size(comm, &numRanks);

    if (myRank == 0 && !amIDead) {
        std::cout << "Finished with " << numRanks << " ranks" << std::endl;
        std::cout << "Time per run: " << timePerRun << " s" << std::endl;
    }

    if (doOutput && myRank == 0 && !amIDead) {
        writePageRanks(result, outputPath, sortOutput);
    }

    if (printOutput && myRank == 0 && !amIDead) {
        outputPageRanks(result, sortOutput, 20, std::cout);
    }

    MPI_Finalize();
}
