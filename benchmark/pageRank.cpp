#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cxxopts.hpp>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <streambuf>
#include <string>
#include <unordered_set>
#include <utility>

#include <mpi.h>

#include "probabilistic_failure_simulator.hpp"
#include "restore/block_serialization.hpp"
#include "restore/common.hpp"
#include "restore/core.hpp"
#include "restore/equal_load_balancer.hpp"
#include "restore/helpers.hpp"
#include "restore/mpi_context.hpp"
#include "restore/restore_vector.hpp"
#include "restore/timer.hpp"

#include "memoryMappedFileReader.h"

#if defined(USE_FTMPI)
    #include <mpi-ext.h>
#elif !defined(SIMULATE_FAILURES)
    #error "If not using a fault-tolerant MPI implementation, you can use only simulated failures."
#endif

using node_t              = int;
const node_t invalid_node = std::numeric_limits<node_t>::max();

using edge_id_t = uint64_t;

struct edge_t {
    node_t from;
    node_t to;
    edge_t(node_t u, node_t v) noexcept : from(u), to(v) {}
    edge_t() noexcept : from(0), to(0) {}

    bool operator==(const edge_t other) const {
        return from == other.from && to == other.to;
    }

    bool operator!=(const edge_t other) const {
        return !(*this == other);
    }
};
const edge_t invalid_edge = edge_t(invalid_node, invalid_node);

auto comm_    = MPI_COMM_WORLD;
bool amIDead  = false;
bool enableFT = true;

using block_distribution_t = std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, ReStoreMPI::current_rank_t>>;

std::tuple<node_t, edge_id_t, edge_id_t, std::vector<edge_t>, std::vector<node_t>, block_distribution_t>
readGraph(const std::vector<std::string>& graphs) {
    int myRank   = -1;
    int numRanks = -1;
    MPI_Comm_rank(comm_, &myRank);
    MPI_Comm_size(comm_, &numRanks);
    assert(myRank >= 0);
    assert(numRanks > 0);

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

    for (const auto& graph: graphs) {
        MemoryMappedFileReader fileReader(graph);
        while (!fileReader.finishedFile()) {
            if (!fileReader.isLetter()) {
                fileReader.skipLine();
            }
            edge_id_t numEdgesInThisFile   = 0;
            char      letter               = fileReader.getLetter();
            bool      hasReadEdgesFromFile = false;
            if (letter == 'p') {
                node_t    firstNum  = fileReader.getInt();
                edge_id_t secondNum = fileReader.getuint64_t();
                if (numVertices != 0 || numEdges != 0) {
                    if (numVertices != firstNum || numEdges != secondNum) {
                        if (myRank == 0)
                            std::cout << "Found another line starting with 'p' that claims different vertex or edge "
                                         "counts than the first one!"
                                      << std::endl;
                        exit(1);
                    }
                } else {
                    assert(numEdgesRead == 0);
                    numVertices = firstNum;
                    numEdges    = secondNum;
                    assert(firstNum > 0);
                    outDegrees.resize(static_cast<size_t>(numVertices));
                    numEdgesPerRank       = numEdges / asserting_cast<edge_id_t>(numRanks);
                    numRanksWithMoreEdges = asserting_cast<int>(numEdges % asserting_cast<edge_id_t>(numRanks));
                    lowerBound            = numEdgesPerRank * asserting_cast<edge_id_t>(myRank)
                                 + asserting_cast<edge_id_t>(std::min(myRank, numRanksWithMoreEdges));
                    upperBound =
                        numEdgesPerRank
                        * asserting_cast<edge_id_t>((myRank + 1) + std::min(myRank + 1, numRanksWithMoreEdges));

                    for (int rank = 0; rank < numRanks; ++rank) {
                        ReStore::block_id_t rankLowerBound =
                            numEdgesPerRank * asserting_cast<edge_id_t>(rank)
                            + asserting_cast<edge_id_t>(std::min(rank, numRanksWithMoreEdges));
                        size_t rankUpperBound =
                            numEdgesPerRank
                            * asserting_cast<edge_id_t>((rank + 1) + std::min(rank + 1, numRanksWithMoreEdges));
                        blockDistribution.emplace_back(
                            std::make_pair(rankLowerBound, rankUpperBound - rankLowerBound), rank);
                    }
                }

            } else if (letter == 'e') {
                hasReadEdgesFromFile = true;
                if (numEdgesRead >= lowerBound && numEdgesRead < upperBound) {
                    node_t firstNum  = fileReader.getInt();
                    node_t secondNum = fileReader.getInt();
                    assert(firstNum > 0);
                    assert(secondNum > 0);
                    --firstNum;
                    --secondNum;
                    if (numVertices == 0 || numEdges == 0) {
                        if (myRank == 0)
                            std::cout
                                << "First edge before specifying number of vertices and edges. This is not supported!"
                                << std::endl;
                        exit(1);
                    }
                    if (firstNum > numVertices) {
                        if (myRank == 0) {
                            std::cout << "Invalid vertex id " << firstNum << std::endl;
                        }
                        exit(1);
                    }
                    if (secondNum > numVertices) {
                        if (myRank == 0) {
                            std::cout << "Invalid vertex id " << secondNum << std::endl;
                        }
                        exit(1);
                    }
                    ++outDegrees[static_cast<size_t>(firstNum)];
                    edges.emplace_back(firstNum, secondNum);
                }
                ++numEdgesRead;
                numEdgesInThisFile = numEdgesInThisFile == 0 ? 0 : numEdgesInThisFile - 1;
                if (numEdgesRead >= upperBound && numEdgesInThisFile > 0) {
                    // We've read all the edges we need, so we can skip over this file
                    // the 'numEdgesInThisFile > 0' makes sure this file had an 'f' line so we can keep track of the
                    // number of edges in the input files. This isn't really necessary. It just gives me some piece of
                    // mind that there is at least some check that the files include as many edges as the 'p' line
                    // claims
                    numEdgesRead += numEdgesInThisFile;
                    break;
                }
            } else if (letter == 'f') {
                if (hasReadEdgesFromFile) {
                    std::cerr << "line with 'f' must occur before any edge in each file" << std::endl;
                    exit(1);
                }

                numEdgesInThisFile = fileReader.getuint64_t();
                if (numEdgesRead < upperBound && numEdgesRead + numEdgesInThisFile > lowerBound) {
                    // This file contains edges that we have to read on this PE
                } else {
                    // This file doesn't contain any edges we need to read. Act like we have seen them and don't read
                    // any further.
                    numEdgesRead += numEdgesInThisFile;
                    break;
                }

            } else {
                if (myRank == 0) {
                    std::cerr << "Unsupported type: " << letter << std::endl;
                }
                exit(1);
            }
            fileReader.skipLine();
        }
    }
    if (numEdges != numEdgesRead) {
        if (myRank == 0) {
            std::cerr << "Expected " << numEdges << " edges but found " << numEdgesRead << std::endl;
        }
        exit(1);
    }

    assert(static_cast<edge_id_t>(edges.size()) == numEdgesPerRank + (myRank < numRanksWithMoreEdges));
    // Get all degrees on all ranks (which we don't have because we skipped over all edges that don't belong to us)
    MPI_Allreduce(MPI_IN_PLACE, outDegrees.data(), numVertices, MPI_INT, MPI_SUM, comm_);
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

std::unordered_set<int> ranksToKill;

template <class F>
bool fault_tolerant_mpi_call(const F& mpi_call) {
    assert(comm_ != MPI_COMM_NULL);
    if (!enableFT) {
        mpi_call();
        return true;
    }

#if defined(SIMULATE_FAILURES)
    if (ranksToKill.size() > 0) {
        MPI_Comm newComm;

        int rank;
        MPI_Comm_rank(comm_, &rank);
        int color = 0;
        if (ranksToKill.find(rank) != ranksToKill.end()) {
            color   = 1;
            amIDead = true;
        }
        ranksToKill.clear();

        MPI_Comm_split(comm_, color, rank, &newComm);
        MPI_Comm_free(&comm_);
        comm_ = newComm;
        return false;
    } else {
        mpi_call();
        return true;
    }
#elif defined(USE_FTMPI)
    if (ranksToKill.size() > 0) {
        int rank;
        MPI_Comm_rank(comm_, &rank);
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
            MPIX_Comm_revoke(comm_);
        }

        // Build a new communicator without the failed ranks
        MPI_Comm newComm;
        if ((rc = MPIX_Comm_shrink(comm_, &newComm)) != MPI_SUCCESS) {
            std::cerr << "A rank failure was detected, but building the new communicator failed" << std::endl;
            exit(1);
        }
        assert(comm_ != MPI_COMM_NULL);
        // As for the ULFM documentation, freeing the communicator is recommended but will probably
        // not succeed. This is why we do not check for an error here.
        // I checked that --mca mpi_show_handle_leaks 1 does not show a leaked handle
        MPI_Comm_free(&comm_);
        comm_ = newComm;
        return false;
    } else if (rc != MPI_SUCCESS) {
        std::cerr << "MPI call did non fail because of a faulty rank but still did not return MPI_SUCCESS" << std::endl;
        exit(1);
    }
    return true;
#endif
}

// Performs a fault-tolerant global MPI barrier. If SIMULATE_FAILURES is defined, this will degrade into a
// MPI_Barrier.
bool ft_barrier() {
    if (!enableFT) {
        return true;
    }
    return fault_tolerant_mpi_call([&]() {
#ifndef SIMULATE_FAILURES
        int flag = 42;
        return MPIX_Comm_agree(comm_, &flag);
#endif
        return MPI_Barrier(comm_);
    });
}

void recoverFromFailure(
    const edge_id_t numEdges, std::vector<edge_t>& edges, ReStore::ReStoreVector<edge_t>& reStoreVectorHelper,
    ReStore::EqualLoadBalancer& loadBalancer, int& myRank, int& numRanks) {
    MPI_Comm_rank(comm_, &myRank);
    MPI_Comm_size(comm_, &numRanks);
    reStoreVectorHelper.updateComm(comm_);
    UNUSED(numEdges);

    auto diedRanks = reStoreVectorHelper.getRanksDiedSinceLastCall();
    auto requests  = loadBalancer.getNewBlocksAfterFailure(diedRanks);
    reStoreVectorHelper.restoreDataAppend(edges, requests);
    // Barrier to check that everyone successfully recovered. In a final production implementation we would have to
    // recover from this as well
    if (!ft_barrier()) {
        std::cerr << "Failure during recovery. Aborting" << std::endl;
        exit(1);
    }
    loadBalancer.commitToPreviousCall();
    assert(
        std::all_of(edges.begin(), edges.end(), [](const edge_t edge) { return !(edge.from == 0 && edge.to == 0); }));
    assert(std::all_of(edges.begin(), edges.end(), [](const edge_t edge) { return edge != invalid_edge; }));
}

std::vector<double> pageRank(
    const node_t numVertices, const edge_id_t numEdges, std::vector<edge_t>& edges,
    const std::vector<node_t>& nodeDegrees, const double dampening, const int numIterations,
    std::optional<ReStore::ReStoreVector<edge_t>>& reStoreVectorHelper, ReStore::EqualLoadBalancer& loadBalancer,
    ProbabilisticFailureSimulator& failureSimulator) {
    UNUSED(numEdges);
    int myRank;
    int numRanks;
    MPI_Comm_rank(comm_, &myRank);
    MPI_Comm_size(comm_, &numRanks);
    std::vector<double> prevPageRanks(static_cast<size_t>(numVertices), 0);
    std::vector<double> currPageRanks(static_cast<size_t>(numVertices), 1 / (double)numVertices);
    std::vector<double> tempPageRanks(static_cast<size_t>(numVertices), 0);
    const int           n        = static_cast<int>(nodeDegrees.size());
    const double        teleport = (1.0 - dampening) / n;
    for (int currentIt = 0; currentIt < numIterations; ++currentIt) {
        std::swap(prevPageRanks, currPageRanks);
        std::fill(currPageRanks.begin(), currPageRanks.end(), 0.0);
        size_t i            = 0;
        bool   updatedEdges = false;
        bool   anotherPass  = false;

        ranksToKill.clear();
        failureSimulator.maybeFailRanks(numRanks, ranksToKill);

        do {
            bool isRecomputation = anotherPass;
            anotherPass          = false;
            if (isRecomputation) {
                assert(enableFT);
                TIME_PUSH_AND_START("Recomputation");
            }
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

            // Failure simulation
            MPI_Comm_rank(comm_, &myRank);
            MPI_Comm_size(comm_, &numRanks);
            // if (myRank == 0) {
            //     std::cout << "Killing ranks ";
            //     for (const auto rankToKill: ranksToKill) {
            //         std::cout << rankToKill << " ";
            //     }
            //     std::cout << std::endl;
            // }

            if (!fault_tolerant_mpi_call([&]() {
                    return MPI_Allreduce(currPageRanks.data(), tempPageRanks.data(), n, MPI_DOUBLE, MPI_SUM, comm_);
                })
                || !ft_barrier()) {
                assert(enableFT);
                if (amIDead) {
                    return currPageRanks;
                }
                TIME_PUSH_AND_START("Recovery");
                assert(reStoreVectorHelper.has_value());
                recoverFromFailure(numEdges, edges, reStoreVectorHelper.value(), loadBalancer, myRank, numRanks);
                TIME_POP("Recovery");
                updatedEdges = true;
                anotherPass  = true;
            }
            if (isRecomputation) {
                TIME_POP("Recomputation");
            }
        } while (anotherPass);
        std::swap(currPageRanks, tempPageRanks);
        std::for_each(currPageRanks.begin(), currPageRanks.end(), [teleport, dampening](double& value) {
            value = getActualPageRank(value, teleport, dampening);
        });
        if (updatedEdges) {
            assert(enableFT);
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

    stream << "----------------------- RESULTS" << std::endl;
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

double getFailureProbabilityForExpectedNumberOfFailures(
    const int numIterations, const int numPEs, const double percentFailures) {
    // Discrete exponential decay
    return 1 - pow(((1.0 * (numPEs - (percentFailures * numPEs))) / numPEs), (1.0 / numIterations));
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(comm_, MPI_ERRORS_RETURN);

    int myRank;
    MPI_Comm_rank(comm_, &myRank);
    int numRanks;
    MPI_Comm_size(comm_, &numRanks);

    cxxopts::Options cliParser("pageRank", "Benchmarks a fault tolerant page rank algorithm.");

    cliParser.add_options()                                                                                    ///
        ("graphs", "paths to the input graph files", cxxopts::value<std::vector<std::string>>())               ///
        ("o,output", "path to the output file", cxxopts::value<std::string>())                                 ///
        ("s,sort", "sort the output", cxxopts::value<bool>()->default_value("false"))                          ///
        ("p,print", "print the first 20 scores of the output", cxxopts::value<bool>()->default_value("false")) ///
        ("d,dampening", "dampening factor.", cxxopts::value<double>()->default_value("0.85"))                  ///
        ("n,numIterations", "Number of PageRank iterations to run",
         cxxopts::value<int>()->default_value("100")) ///
        // ("r,repetitions", "Number of repetitions to run", cxxopts::value<size_t>()->default_value("1")) ///
        ("e,enable-ft", "Enable fault tolerance", cxxopts::value<bool>()->default_value("true")) ///
        ("f,replications", "Replications for fault tolerance with ReStore",
         cxxopts::value<size_t>()->default_value("3")) ///
        ("b,blockSize", "Number of edges to combine into one block for ReStore",
         cxxopts::value<size_t>()->default_value("8")) ///
        ("blocksPerPermutationRange",
         "Number of blocks that get consecutive ids when using a pseudo-random permutation on the block ids.",
         cxxopts::value<size_t>()->default_value("100"))                                                   ///
        ("seed", "The seed for failure simulation.", cxxopts::value<unsigned long>()->default_value("42")) ///
        ("percentFailures", "Expected ratio of PEs to fail during the calculation.",
         cxxopts::value<double>()->default_value("0.1")) ///
        ("h,help", "Print help message.");

    cliParser.parse_positional({"graphs"});
    cliParser.positional_help("<graphs>");

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

    if (!options.count("graphs")) {
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

    // const auto   numRepetitions = options["repetitions"].as<size_t>();
    const auto   numRepetitions = 1;
    const double dampening      = options["dampening"].as<double>();
    const int    numIterations  = options["numIterations"].as<int>();

    enableFT                   = options["enable-ft"].as<bool>();
    const auto numReplications = std::min(options["replications"].as<size_t>(), static_cast<size_t>(numRanks));

    const auto blockSize                 = options["blockSize"].as<size_t>();
    const auto blocksPerPermutationRange = options["blocksPerPermutationRange"].as<size_t>();

    const auto seed            = options["seed"].as<unsigned long>();
    const auto percentFailures = options["percentFailures"].as<double>();

    double failureProbability =
        getFailureProbabilityForExpectedNumberOfFailures(numIterations, numRanks, percentFailures);

    auto failureSimulator = ProbabilisticFailureSimulator(seed, failureProbability);

    auto start = MPI_Wtime();
    // TIME_NEXT_SECTION("Read graph");
    auto [numVertices, numEdges, firstEdgeId, edges, nodeDegrees, blockDistribution] =
        readGraph(options["graphs"].as<std::vector<std::string>>());
    std::sort(edges.begin(), edges.end(), [](const edge_t lhs, const edge_t rhs) { return lhs.from < rhs.from; });
    // TIME_STOP();
    auto end              = MPI_Wtime();
    auto graphReadingTime = end - start;

    TIME_NEXT_SECTION("Init");
    auto restoreVectorHelper =
        enableFT ? std::make_optional<ReStore::ReStoreVector<edge_t>>(
            blockSize, comm_, asserting_cast<uint16_t>(numReplications), blocksPerPermutationRange, invalid_edge)
                 : std::nullopt;
    if (enableFT) {
        assert(restoreVectorHelper.has_value());
        auto numBlocksLocal = restoreVectorHelper.value().submitData(edges);

        auto mpiContext        = ReStoreMPI::MPIContext(comm_);
        auto firstBlockIdLocal = mpiContext.exclusive_scan(numBlocksLocal, MPI_SUM);
        auto blockRangeLocal   = std::make_pair(std::make_pair(firstBlockIdLocal, numBlocksLocal), myRank);
        blockDistribution      = mpiContext.allgather(blockRangeLocal);
    }
    ReStore::EqualLoadBalancer loadBalancer(blockDistribution, numRanks);
    TIME_STOP();

    int numRanksStarting = numRanks;

    MPI_Barrier(comm_);
    std::vector<double> result;
    TIME_NEXT_SECTION("Pagerank");
    for (size_t i = 0; i < numRepetitions; ++i) {
        result = pageRank(
            numVertices, numEdges, edges, nodeDegrees, dampening, numIterations, restoreVectorHelper, loadBalancer,
            failureSimulator);
    }
    TIME_STOP();

    MPI_Comm_rank(comm_, &myRank);
    MPI_Comm_size(comm_, &numRanks);

    int numRanksFailed = numRanksStarting - numRanks;

    if (myRank == 0 && !amIDead) {
        ResultsCSVPrinter resultPrinter(std::cout, true);
        auto              timers = TimerRegister::instance().getAllTimers();
        resultPrinter.allResults("numRanks", numRanksStarting);
        resultPrinter.thisResult("numRanksFailed", numRanksFailed);
        resultPrinter.thisResult("failureProbability", failureProbability);
        resultPrinter.thisResult("graphReadingTime", graphReadingTime);
        resultPrinter.allResults("enableFT", enableFT);
        resultPrinter.allResults("numReplications", numReplications);
        resultPrinter.allResults("blockSize", blockSize);
        resultPrinter.allResults("numIterations", numIterations);
        resultPrinter.thisResult("seed", seed);
        resultPrinter.allResults("numVertices", numVertices);
        resultPrinter.allResults("numEdges", numEdges);
        resultPrinter.allResults("blocksPerPermutationRange", blocksPerPermutationRange);
        resultPrinter.thisResult(timers);
        resultPrinter.finalizeAndPrintResult();
    }

    if (doOutput && myRank == 0 && !amIDead) {
        writePageRanks(result, outputPath, sortOutput);
    }

    if (printOutput && myRank == 0 && !amIDead) {
        outputPageRanks(result, sortOutput, 20, std::cout);
    }

    MPI_Finalize();
}
