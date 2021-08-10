#include <array>
#include <cstdint>
#include <mpi.h>
#include <random>
#include <string>
#include <vector>

#include <cxxopts.hpp>

#include "k-means.hpp"
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

    private:
    size_t        _numDataPointsPerRank;
    size_t        _numCenters;
    size_t        _numIterations;
    size_t        _numDimensions;
    uint16_t      _replicationLevel;
    unsigned long _seed              = 0;
    bool          _useFaultTolerance = false;
    bool          _printCSVHeader    = true;
    size_t        _numRepetitions    = 1;
    std::string   _simulationId;
    bool          _validConfiguration = true;

    void _fail(const char* msg) {
        std::cerr << msg << std::endl;
        _validConfiguration = false;
    }
};

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

    TIME_NEXT_SECTION("perform-iterations");
    kmeansInstance.performIterations(options.numIterations());
    TIME_STOP();

    // Print the results
    if (mpiContext.getMyCurrentRank() == 0) {
        ResultsCSVPrinter resultPrinter(std::cout, options.printCSVHeader());
        resultPrinter.allResults("simulationId", options.simulationId());
        resultPrinter.allResults("numDataPointsPerRank", options.numDataPointsPerRank());
        resultPrinter.allResults("numCenters", options.numCenters());
        resultPrinter.allResults("numIterations", options.numIterations());
        resultPrinter.allResults("numDimensions", options.numDimensions());
        resultPrinter.allResults("useFaultTolerance", options.useFaultTolerance());
        resultPrinter.allResults("replicationLevel", options.replicationLevel());
        resultPrinter.allResults("seed", options.seed());
        resultPrinter.thisResult(TimerRegister::instance().getAllTimers());
        resultPrinter.finalizeAndPrintResult();
    }

    MPI_Finalize();
    return 0;
}