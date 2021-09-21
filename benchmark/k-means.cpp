#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
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
    enum class Mode : uint8_t { GenerateData, ClusterData };

    CommandLineOptions(int argc, char* argv[], int numRanks) {
        if (numRanks < 1) {
            throw std::invalid_argument("Number of ranks must be at least 1");
        }

        cxxopts::Options cliParser(
            "k-means", "Performs a distributed memory k-means clustering. Is capable of handling failed rank.");

        cliParser.add_options()                                                                              ///
            ("mode", "<generate-data,cluster-data>", cxxopts::value<std::string>())                          ///
            ("num-data-points-per-rank", "Number of data points to use per rank.", cxxopts::value<size_t>()) ///
            ("num-dimensions", "The number of dimensions per data point.", cxxopts::value<size_t>())         ///
            ("num-centers", "The number of cluster centers (k).", cxxopts::value<size_t>())                  ///
            ("num-iterations", "The number of iterations to perform.", cxxopts::value<size_t>())             ///
            ("replication-level", "The number of replicates of each data point to keep (for fault-tolerance).",
             cxxopts::value<uint16_t>()) ///
            ("fault-tolerance", "Enable or disable fault-tolerance?",
             cxxopts::value<bool>()->default_value("false")) ///
            ("failure-probability", "Set the probability 0 < p < 1 of each rank failing during one iteration.",
             cxxopts::value<double>()) ///
            ("expected-failure-rate",
             "Set the fraction of ranks expected to fail; we'll comput the failure probability for you.",
             cxxopts::value<double>()) ///
            //("num-failures", "Sets the number of failures to simulate.", cxxopts::value<uint64_t>()) ///
            ("simulation-id",
             "Simulation id. Will be echoed as is; can be used to identify the results of this experiment.",
             cxxopts::value<std::string>()) ///
            ("repeat-id", "Repeat id. Will be echoed as is; can be used to identify the results of this experiment.",
             cxxopts::value<std::string>()->default_value("0")) ///
            ("s,seed", "Random seed for the data generation and cluster center selection.",
             cxxopts::value<unsigned long>()->default_value("0"))                                                 ///
            ("failure-simulator-seed", "Random seed for the failure simulator.", cxxopts::value<unsigned long>()) ///
            ("n,no-header", "Do not print the csv header.", cxxopts::value<bool>()->default_value("false"))       ///
            ("i,input", "Name of the input file.", cxxopts::value<std::string>())                                 ///
            ("o,output", "Name of the output file.", cxxopts::value<std::string>())                               ///
            ("h,help", "Print help message.");                                                                    ///

        cliParser.parse_positional({"mode"});
        cliParser.positional_help("<mode>");

        cxxopts::ParseResult options;
        try {
            options = cliParser.parse(argc, argv);
        } catch (cxxopts::OptionException& e) {
            std::cout << e.what() << std::endl << std::endl;
            std::cout << cliParser.help() << std::endl;
            exit(1);
        }

        // Print the help message?
        if (options.count("help")) {
            std::cout << cliParser.help() << std::endl;
            exit(0);
        }

        // Which mode are we in?
        if (options.count("mode")) {
            if (options["mode"].as<std::string>() == "generate-data") {
                _mode = Mode::GenerateData;
            } else if (options["mode"].as<std::string>() == "cluster-data") {
                _mode = Mode::ClusterData;
            } else {
                _fail(
                    std::string("Unknown mode: ") + options["mode"].as<std::string>()
                    + ". Valid modes are 'generate-data' and 'cluster-data'.");
            }
        } else {
            _fail(std::string("Please specify a mode. Please use either 'generate-data' or 'cluster-data'."));
        }

        // First, parse the options common to all modes.
        if (options.count("seed")) {
            _seed = options["seed"].as<unsigned long>();
        }

        if (options.count("no-header")) {
            _printCSVHeader = !(options["no-header"].as<bool>());
        } else {
            _printCSVHeader = true;
        }

        if (options.count("output") != 1) {
            _fail("Required option missing or provided more than once: --output");
        } else {
            _outputFile = options["output"].as<std::string>();
        }

        // Now, parse the options only present in some modes.
        if (mode() == Mode::GenerateData) {
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
        } else if (mode() == Mode::ClusterData) {
            if (options.count("failure-simulator-seed")) {
                _failureSimulatorSeed.emplace(options["failure-simulator-seed"].as<unsigned long>());
            }

            if (options.count("input") != 1) {
                _fail("Required option missing or provided more than once: --input");
            } else {
                _inputFile = options["input"].as<std::string>();
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

            if (options.count("fault-tolerance") != 1) {
                _fail("Required option missing or provided more than once: --fault-tolerance");
            } else {
                _useFaultTolerance = options["fault-tolerance"].as<bool>();
            }

            if (_useFaultTolerance) {
                if (options.count("replication-level") != 1) {
                    _fail("Required option missing or provided more than once: --replication-level");
                } else {
                    _replicationLevel = options["replication-level"].as<uint16_t>();
                }
            } else {
                _replicationLevel = 0;
            }

            if (options.count("simulation-id") != 1) {
                _fail("Required option missing or provided more than once: --simulation-id");
            } else {
                _simulationId = options["simulation-id"].as<std::string>();
            }

            _repeatId = options["repeat-id"].as<std::string>();

            if (_useFaultTolerance) {
                if (options.count("failure-probability") > 0 && options.count("expected-failure-rate") > 0) {
                    _fail("Cannot specify both --failure-probability and --expected-failures");
                } else if (options.count("failure-probability")) {
                    _failureProbability = options["failure-probability"].as<double>();

                    if (_failureProbability <= 0.0 || _failureProbability >= 1.0) {
                        _fail("Failure probability must be between and not equal to 0 and 1.");
                    }
                } else if (options.count("expected-failure-rate")) {
                    auto expectedFailureRate = options["expected-failure-rate"].as<double>();

                    if (expectedFailureRate > 0.0 && expectedFailureRate < 1.0) {
                        assert(numRanks > 0);
                        assert(_numIterations > 0);
                        double numRanks_d            = static_cast<double>(numRanks);
                        double numExpectedFailures_d = static_cast<double>(expectedFailureRate) * numRanks_d;
                        double numIterations_d       = static_cast<double>(_numIterations);
                        _failureProbability =
                            1.0 - std::pow((numRanks_d - numExpectedFailures_d) / numRanks_d, 1.0 / numIterations_d);
                    } else {
                        _fail("The expected failure rate must be greater than 0 and smaller than 1.");
                    }
                }
            }
        }
    }

    size_t numDataPointsPerRank() const {
        assert(_numDataPointsPerRank > 0);
        return _numDataPointsPerRank;
    }

    void numDataPointsPerRank(size_t numDataPoints) {
        if (numDataPoints == 0) {
            throw std::invalid_argument("There must be at least one data point.");
        }
        _numDataPointsPerRank = numDataPoints;
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
        assert(_useFaultTolerance || _replicationLevel == 0);
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

    unsigned long dataGenerationSeed() const {
        return _seed;
    }

    unsigned long clusterCenterSeed() const {
        return _seed;
    }

    unsigned long failureSimulatorSeed() const {
        if (_failureSimulatorSeed) {
            return _failureSimulatorSeed.value();
        } else {
            return _seed;
        }
    }

    size_t numRepetitions() const {
        assert(_numRepetitions > 0);
        return _numRepetitions;
    }

    size_t numDimensions() const {
        assert(_numDimensions > 0);
        return _numDimensions;
    }

    void numDimensions(size_t numDimensions) {
        if (numDimensions == 0) {
            throw std::invalid_argument("There must be at least one dimension.");
        }
        _numDimensions = numDimensions;
    }

    Mode mode() const {
        return _mode;
    }

    double failureProbability() const {
        assert(_failureProbability >= 0 && _failureProbability <= 1);
        return _failureProbability;
    }

    bool simulateFailures() const {
        assert(_failureProbability >= 0 && _failureProbability <= 1);
        return _failureProbability > 0 || _numFailures > 0;
        // return numFailures() > 0;
    }

    // uint64_t numFailures() const {
    //     return _numFailures;
    // }

    std::string inputFile() const {
        return _inputFile;
    }

    std::string dataInputFile(int myRank) const {
        return _inputFile + "." + std::to_string(myRank) + ".data";
    }

    std::string outputFile() const {
        return _outputFile;
    }

    std::string generatedDataOutputFile(int myRank) const {
        return _outputFile + "." + std::to_string(myRank) + ".data";
    }

    std::string measurementOutputFile() const {
        return _outputFile + ".measurements.csv";
    }

    std::string clusterCentersOutputFile() const {
        return _outputFile + ".cluster-centers";
    }

    std::string assignmentOutputFile(int myRank) const {
        return _outputFile + ".rank" + std::to_string(myRank) + ".assignments";
    }

    std::string repeatId() const {
        return _repeatId;
    }

    private:
    size_t                       _numDataPointsPerRank;
    size_t                       _numCenters;
    size_t                       _numIterations;
    size_t                       _numDimensions;
    uint16_t                     _replicationLevel;
    Mode                         _mode;
    std::string                  _inputFile;
    std::string                  _outputFile;
    std::string                  _repeatId;
    double                       _failureProbability   = 0;
    uint64_t                     _numFailures          = 0;
    unsigned long                _seed                 = 0;
    std::optional<unsigned long> _failureSimulatorSeed = std::nullopt;
    bool                         _useFaultTolerance    = false;
    bool                         _printCSVHeader       = true;
    size_t                       _numRepetitions       = 1;
    std::string                  _simulationId;
    bool                         _validConfiguration = true;

    void _fail(const char* msg) {
        std::cerr << msg << std::endl;
        _validConfiguration = false;
    }

    void _fail(const std::string& msg) {
        _fail(msg.c_str());
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

// Print the results of the runtime measurements.
void writeMeasurementsToFile(
    const std::string& outputFile, const CommandLineOptions& options,
    std::vector<std::pair<const char*, double>>& measurements, uint64_t numFailures, int numberOfRanks) {
    std::ofstream file;
    file.open(outputFile.c_str(), std::ios::out | std::ios::trunc);

    ResultsCSVPrinter resultPrinter(file, options.printCSVHeader());
    resultPrinter.allResults("simulationId", options.simulationId());
    resultPrinter.allResults("repeatId", options.repeatId());
    resultPrinter.allResults("numRanks", numberOfRanks);
    resultPrinter.allResults("numDataPointsPerRank", options.numDataPointsPerRank());
    resultPrinter.allResults("numCenters", options.numCenters());
    resultPrinter.allResults("numIterations", options.numIterations());
    resultPrinter.allResults("numDimensions", options.numDimensions());
    resultPrinter.allResults("useFaultTolerance", options.useFaultTolerance());
    resultPrinter.allResults("replicationLevel", options.replicationLevel());
    resultPrinter.allResults("failureSimulatorSeed", options.failureSimulatorSeed());
    resultPrinter.allResults("clusterCenterSeed", options.clusterCenterSeed());

    resultPrinter.thisResult("failureProbability", options.failureProbability());
    resultPrinter.thisResult("numSimulatedRankFailures", numFailures);
    resultPrinter.thisResult(measurements);
    resultPrinter.finalizeAndPrintResult();
}

// Print the results of the k-means for comparison with the reference implementation.
template <class data_t>
void writeClusterCentersToFile(const std::string& outputFile, const kmeans::kMeansData<data_t>& clusterCenters) {
    std::ofstream file;
    file.open(outputFile.c_str(), std::ios::out | std::ios::trunc);

    // Write the number of clusters and dimensions to the file.
    file << clusterCenters.numDataPoints() << " " << clusterCenters.numDimensions() << std::endl;

    // Write the cluster centers to the file.
    for (uint64_t centerIdx = 0; centerIdx < clusterCenters.numDataPoints(); ++centerIdx) {
        for (uint32_t dimension = 0; dimension < clusterCenters.numDimensions(); ++dimension) {
            file << clusterCenters.getElementDimension(centerIdx, dimension);
            if (dimension == clusterCenters.numDimensions() - 1) {
                file << std::endl;
            } else {
                file << " ";
            }
        }
    }
}

// Print the results of the k-means for comparison with the reference implementation.
template <class data_t>
void writeDataAssignmentToFile(
    const std::string& outputFile, const kmeans::kMeansData<data_t>& data, const std::vector<size_t>& assignment) {
    std::ofstream file;
    file.open(outputFile.c_str(), std::ios::out | std::ios::trunc);

    file << data.numDataPoints() << " " << data.numDimensions() << std::endl;
    for (uint64_t dataPointIdx = 0; dataPointIdx < data.numDataPoints(); ++dataPointIdx) {
        for (uint16_t dimension = 0; dimension < data.numDimensions(); ++dimension) {
            file << data.getElementDimension(dataPointIdx, dimension) << " ";
        }
        file << assignment[dataPointIdx] << std::endl;
    }
}

void generateData(ReStoreMPI::MPIContext& mpiContext, const CommandLineOptions& options) {
    TIME_NEXT_SECTION("generate-data");
    auto thisRanksSeed = options.dataGenerationSeed() + asserting_cast<unsigned long>(mpiContext.getMyCurrentRank());
    auto data =
        kmeans::generateRandomData<float>(options.numDataPointsPerRank(), options.numDimensions(), thisRanksSeed);
    kmeans::writeDataToFile(data, options.generatedDataOutputFile(mpiContext.getMyCurrentRank()));
    TIME_STOP();
}

void runKMeansAndReport(ReStoreMPI::MPIContext& mpiContext, CommandLineOptions& options) {
    TIME_NEXT_SECTION("load-data");
    auto inputFile      = options.dataInputFile(mpiContext.getMyCurrentRank());
    auto kmeansInstance = kmeans::kMeansAlgorithm<float, ReStoreMPI::MPIContext>(
        kmeans::loadDataFromFile<float>(inputFile), mpiContext, options.useFaultTolerance(),
        options.replicationLevel());

    options.numDimensions(kmeansInstance.numDimensions());
    options.numDataPointsPerRank(kmeansInstance.numDataPoints());

    TIME_NEXT_SECTION("pick-centers");
    kmeansInstance.pickCentersRandomly(options.numCenters(), options.clusterCenterSeed());
    TIME_STOP();

    // Initialize Failure Simulator
    std::optional<ProbabilisticFailureSimulator> failureSimulator = std::nullopt;
    if (options.simulateFailures()) {
        failureSimulator.emplace(options.failureSimulatorSeed(), options.failureProbability());
    }

    long unsigned int numSimulatedRankFailures = 0;

    // Perform the iterations
    std::unordered_set<int> ranksToFail;
    for (uint64_t iteration = 0; iteration < options.numIterations(); ++iteration) {
        // Perform one iteration
        TIME_NEXT_SECTION("perform-iterations");
        kmeansInstance.performIterations(1);
        TIME_STOP();

        // Shall we simulate a failure?
        if (options.simulateFailures()) {
            assert(failureSimulator.has_value());
            auto numAliveRanks = mpiContext.getCurrentSize();
            ranksToFail.clear();
            failureSimulator->maybeFailRanks(numAliveRanks, ranksToFail);
            numSimulatedRankFailures += ranksToFail.size();
            if (ranksToFail.size() > 0) {
                MPI_Comm newComm = simulateFailure(mpiContext.getMyCurrentRank(), mpiContext.getComm(), ranksToFail);
                mpiContext.simulateFailure(newComm);
            }
        }
    }

    // The measurements and cluster centers are written once by the main rank.
    if (mpiContext.getMyCurrentRank() == 0) {
        auto timers = TimerRegister::instance().getAllTimers();
        writeMeasurementsToFile(
            options.measurementOutputFile(), options, timers, numSimulatedRankFailures, mpiContext.getCurrentSize());

        writeClusterCentersToFile(options.clusterCentersOutputFile(), kmeansInstance.centers());
    }
    // The cluster assignments to data is written per rank, to avoid having to collect the data from all ranks and
    // therefore needing a lot of memory on the main rank.
    writeDataAssignmentToFile(
        options.assignmentOutputFile(mpiContext.getMyCurrentRank()), kmeansInstance.data(),
        kmeansInstance.pointToCenterAssignment().assignedCenter);
}

int main(int argc, char** argv) {
    using namespace kmeans;

    // Parse command line arguments and initialize MPI
    MPI_Init(&argc, &argv);
    ReStoreMPI::MPIContext mpiContext(MPI_COMM_WORLD);

    CommandLineOptions options(argc, argv, mpiContext.getCurrentSize());
    if (!options.validConfiguration()) {
        exit(1);
    }
    // Perform the requested operation
    if (options.mode() == CommandLineOptions::Mode::GenerateData) {
        generateData(mpiContext, options);
    } else if (options.mode() == CommandLineOptions::Mode::ClusterData) {
        runKMeansAndReport(mpiContext, options);
    }

    // Finalize MPI and exit
    MPI_Finalize();
    return 0;
}