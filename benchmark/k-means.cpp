#include <array>
#include <cstdint>
#include <mpi.h>
#include <random>
#include <string>
#include <vector>

#include <cxxopts.hpp>

#include "k-means.hpp"
#include "probabilistic_failure_simulator.hpp"
#include "restore/helpers.hpp"
#include "restore/mpi_context.hpp"
#include "restore/timer.hpp"

class CommandLineOptions {
    public:
    CommandLineOptions(int argc, char* argv[]) {
        cxxopts::Options cliParser(
            "k-means", "Performs a distributed memory k-means clustering. Is capable of handling failed rank.");

        cliParser.add_options()                                                                              ///
            ("num-data-points-per-rank", "Number of data points to use per rank.", cxxopts::value<size_t>()) ///
            ("num-dimensions", "The number of dimensions per data point.", cxxopts::value<size_t>())         ///
            ("num-centers", "The number of cluster centers (k).", cxxopts::value<size_t>())                  ///
            ("num-iterations", "The number of iterations to perform.", cxxopts::value<size_t>())             ///
            ("replication-level", "The number of replicates of each data point to keep (for fault-tolerance).",
             cxxopts::value<uint16_t>()) ///
            ("fault-tolerance", "Enable or disable fault-tolerance?",
             cxxopts::value<bool>()->default_value("false")) ///
            //("failure-probability", "Set the probability 0 < p < 1 of each rank failing during one iteration.",
            // cxxopts::value<double>())                                                               ///
            ("num-failures", "Sets the number of failures to simulate.", cxxopts::value<uint64_t>()) ///)
            ("simulation-id",
             "Simulation id. Will be echoed as is; can be used to identify the results of this experiment.",
             cxxopts::value<std::string>())                                                                 ///
            ("s,seed", "Random seed.", cxxopts::value<unsigned long>()->default_value("0"))                 ///
            ("n,no-header", "Do not print the csv header.", cxxopts::value<bool>()->default_value("false")) ///
            ("r,repetitions", "Number of replicas to run", cxxopts::value<size_t>()->default_value("10"))   ///
            ("h,help", "Print help message.");                                                              ///

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

        if (options.count("num-data-points-per-rank") != 1) {
            _fail("Required option missing or provided more than once: --num-data-points-per-rank");
        } else {
            _numDataPointsPerRank = options["num-data-points-per-rank"].as<size_t>();
        }

        if (options.count("num-dimensions") != 1) {
            _fail("Required option missing or provided more than once: --num-dimensions");
        } else {
            _numDimensions = options["num-dimensions"].as<size_t>();
        }

        if (options.count("num-centers") != 1) {
            _fail("Required option missing or provided more than once: --num-centers");
        } else {
            _numCenters = options["num-centers"].as<size_t>();
        }

        if (options.count("num-iterations") != 1) {
            _fail("Required option missing or provided more than once: --num-iterations");
        } else {
            _numIterations = options["num-iterations"].as<size_t>();
        }

        if (options.count("replication-level") != 1) {
            _fail("Required option missing or provided more than once: --replication-level");
        } else {
            _replicationLevel = options["replication-level"].as<uint16_t>();
        }

        if (options.count("fault-tolerance") != 1) {
            _fail("Required option missing or provided more than once: --fault-tolerance");
        } else {
            _useFaultTolerance = options["fault-tolerance"].as<bool>();
        }

        if (options.count("simulation-id") != 1) {
            _fail("Required option missing or provided more than once: --simulation-id");
        } else {
            _simulationId = options["simulation-id"].as<std::string>();
        }

        if (options.count("seed")) {
            _seed = options["seed"].as<unsigned long>();
        }

        if (options.count("no-header")) {
            _printCSVHeader = !(options["no-header"].as<bool>());
        } else {
            _printCSVHeader = true;
        }

        if (options.count("repetitions")) {
            _numRepetitions = options["no-header"].as<bool>();
        }

        // if (options.count("failure-probability")) {
        //     _failureProbability = options["failure-probability"].as<double>();
        //     if (_failureProbability < 0.0 || _failureProbability > 1.0) {
        //         _fail("Failure probability must be between 0 and 1.");
        //     } else if (!_useFaultTolerance) {
        //         _fail("Failure probability can only be specified when fault-tolerance is enabled.");
        //     } else if (options.count("num-failures")) {
        //         _fail("Failure probability and number of failures cannot be specified at the same time.");
        //     }
        // }

        if (options.count("num-failures")) {
            _numFailures = options["num-failures"].as<uint64_t>();
        }
    }

    size_t numDataPointsPerRank() const {
        assert(_numDataPointsPerRank > 0);
        return _numDataPointsPerRank;
    }

    size_t numCenters() const {
        assert(_numCenters > 0);
        return _numCenters;
    }

    size_t numIterations() const {
        assert(_numIterations > 0);
        return _numIterations;
    }

    uint16_t replicationLevel() const {
        assert(!_useFaultTolerance || _replicationLevel > 0);
        return _replicationLevel;
    }

    bool useFaultTolerance() const {
        return _useFaultTolerance;
    }

    bool printCSVHeader() const {
        return _printCSVHeader;
    }

    std::string simulationId() const {
        return _simulationId;
    }

    bool validConfiguration() const {
        return _validConfiguration;
    }

    unsigned long seed() const {
        return _seed;
    }

    size_t numRepetitions() const {
        assert(_numRepetitions > 0);
        return _numRepetitions;
    }

    size_t numDimensions() const {
        assert(_numDimensions > 0);
        return _numDimensions;
    }

    // double failureProbability() const {
    //     assert(_failureProbability >= 0 && _failureProbability <= 1);
    //     return _failureProbability;
    // }

    bool simulateFailures() const {
        // assert(_failureProbability >= 0 && _failureProbability <= 1);
        // return _failureProbability > 0 || _numFailures > 0;
        return numFailures() > 0;
    }

    uint64_t numFailures() const {
        return _numFailures;
    }

    private:
    size_t        _numDataPointsPerRank;
    size_t        _numCenters;
    size_t        _numIterations;
    size_t        _numDimensions;
    uint16_t      _replicationLevel;
    // double        _failureProbability = 0;
    uint64_t      _numFailures        = 0;
    unsigned long _seed               = 0;
    bool          _useFaultTolerance  = false;
    bool          _printCSVHeader     = true;
    size_t        _numRepetitions     = 1;
    std::string   _simulationId;
    bool          _validConfiguration = true;

    void _fail(const char* msg) {
        std::cerr << msg << std::endl;
        _validConfiguration = false;
    }
};

MPI_Comm
simulateFailure(int myRankId, MPI_Comm currentComm, const std::unordered_set<ReStoreMPI::current_rank_t>& failedRanks) {
    bool iFailed = failedRanks.count(myRankId) > 0;

    MPI_Comm newComm;
    MPI_Comm_split(
        currentComm,
        iFailed,  // color 1
        myRankId, // key
        &newComm);
    // MPI_Comm_free(&currentComm);

    if (iFailed) {
        MPI_Finalize();
        exit(0);
    }
    return newComm;
}

int main(int argc, char** argv) {
    using namespace kmeans;

    // Parse command line arguments and initialize MPI
    CommandLineOptions options(argc, argv);
    if (!options.validConfiguration()) {
        exit(1);
    }
    MPI_Init(&argc, &argv);
    ReStoreMPI::MPIContext mpiContext(MPI_COMM_WORLD);

    // Run the experiment
    TIME_NEXT_SECTION("generate-data");
    auto kmeansInstance = kmeans::kMeansAlgorithm<float, ReStoreMPI::MPIContext>(
        kmeans::generateRandomData<float>(options.numDataPointsPerRank(), options.numDimensions(), options.seed()),
        mpiContext, options.useFaultTolerance(), options.replicationLevel());

    TIME_NEXT_SECTION("pick-centers");
    kmeansInstance.pickCentersRandomly(options.numCenters(), options.seed());

    // Initialize Failure Simulator
    std::optional<ProbabilisticFailureSimulator> failureSimulator = std::nullopt;
    if (options.simulateFailures()) {
        // failureSimulator.emplace(options.seed(), options.failureProbability());
        failureSimulator.emplace(options.seed(), 0.5);
    }

    long unsigned int numSimulatedRankFailures = 0;
    unsigned long     failuresToSimulate       = options.numFailures();
    unsigned long     failEvery                = options.numIterations() / failuresToSimulate;

    // Perform the iterations
    for (uint64_t iteration = 0; iteration < options.numIterations(); ++iteration) {
        // Perform one iteration
        TIME_NEXT_SECTION("perform-iterations");
        kmeansInstance.performIterations(1);
        TIME_STOP();
        
        // Shall we simulate a failure?
        if (options.simulateFailures() && (iteration % failEvery == 0)) {
            assert(failureSimulator.has_value());
            auto                    numAliveRanks = mpiContext.getCurrentSize();
            std::unordered_set<int> ranksToFail;
            failureSimulator->failRanksNow(numAliveRanks, 1, ranksToFail, true);
            numSimulatedRankFailures += ranksToFail.size();
            if (ranksToFail.size() > 0) {
                MPI_Comm newComm = simulateFailure(mpiContext.getMyCurrentRank(), mpiContext.getComm(), ranksToFail);
                mpiContext.simulateFailure(newComm);
            }
        }
    }

    // Print the results of the runtime measurements.
    if (mpiContext.getMyCurrentRank() == 0) {
        ResultsCSVPrinter resultPrinter(std::cerr, options.printCSVHeader());
        resultPrinter.allResults("simulationId", options.simulationId());
        resultPrinter.allResults("numDataPointsPerRank", options.numDataPointsPerRank());
        resultPrinter.allResults("numCenters", options.numCenters());
        resultPrinter.allResults("numIterations", options.numIterations());
        resultPrinter.allResults("numDimensions", options.numDimensions());
        resultPrinter.allResults("useFaultTolerance", options.useFaultTolerance());
        resultPrinter.allResults("replicationLevel", options.replicationLevel());
        resultPrinter.allResults("seed", options.seed());
        resultPrinter.thisResult(TimerRegister::instance().getAllTimers());
        resultPrinter.thisResult("num-simulated-rank-failures", numSimulatedRankFailures);
        resultPrinter.finalizeAndPrintResult();
    }

    // Print the results of the k-means for comparison with the reference implementation.
    auto clusterAssignments = kmeansInstance.collectClusterAssignments();
    for (auto assignment: clusterAssignments) {
        std::cout << assignment << std::endl;
    }

    // Finalize MPI and exit
    MPI_Finalize();
    return 0;
}