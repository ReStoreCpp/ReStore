#include <algorithm>
#include <assert.h>
#include <cppitertools/range.hpp>
#include <iostream>
#include <memory>
#include <random>
#include <restore/core.hpp>
#include <restore/helpers.hpp>
#include <stdexcept>
#include <stdint.h>
#include <stdlib.h>
#include <string>
#include <vector>

#include <cxxopts.hpp>

#include "../tests/mocks.hpp"

using namespace ::testing;

using iter::range;
using ReStoreMPI::original_rank_t;

bool checkForDataLoss(std::shared_ptr<ReStore::BlockDistribution<MPIContextFake>> blockDistribution) {
    for (auto rangeId: range(blockDistribution->numRanges())) {
        auto blockRange = blockDistribution->blockRangeById(rangeId);
        if (blockDistribution->ranksBlockRangeIsStoredOn(blockRange).empty()) {
            return true;
        }
    }
    return false;
}

struct RankDistributionConfig {
    uint32_t numRanks;
    size_t   numBlocks;
    uint16_t replicationLevel;

    friend std::istream& operator>>(std::istream& stream, RankDistributionConfig& config) {
        stream >> config.numRanks >> config.numBlocks >> config.replicationLevel;
        return stream;
    }

    friend std::ostream& operator<<(std::ostream& stream, const RankDistributionConfig& config) {
        stream << "{numRanks: " << config.numRanks << ", numBlocks: " << config.numBlocks
               << ", replicationLevel: " << config.replicationLevel << "}";
        return stream;
    }
};

enum class SimulationMode { RANK, NODE, RACK, INVALID };

SimulationMode parse_mode_string(const std::string& modeStr) {
    if (modeStr == "rank") {
        return SimulationMode::RANK;
    } else if (modeStr == "node") {
        return SimulationMode::NODE;
    } else if (modeStr == "rack") {
        return SimulationMode::RACK;
    } else {
        return SimulationMode::INVALID;
    }
}

std::string to_string(const SimulationMode& simulationMode) {
    switch (simulationMode) {
        case SimulationMode::RANK:
            return "rank";
        case SimulationMode::NODE:
            return "node";
        case SimulationMode::RACK:
            return "rack";
        case SimulationMode::INVALID:
        default:
            throw std::runtime_error("Invalid SimulationMode");
    }
}

class ResultsPrinter {
    public:
    ResultsPrinter(bool printHeader, const SimulationMode& simulationMode, std::string simulationId) noexcept
        : _simulationId(simulationId),
          _simulationMode(to_string(simulationMode)) {
        if (printHeader) {
            std::cout << "simulation_id,mode,seed,num_ranks,num_blocks,replication_level,failures_until_data_loss"
                      << std::endl;
        }
    }

    void print_result(unsigned long seed, RankDistributionConfig config, size_t failuresUntilDataLoss) {
        _printCSV(
            _simulationId, _simulationMode, seed, config.numRanks, config.numBlocks, config.replicationLevel,
            failuresUntilDataLoss);
    }

    private:
    template <class T, class... Ts>
    void _printCSV(T firstArg, Ts... furtherArgs) {
        std::cout << firstArg;
        if constexpr (sizeof...(furtherArgs) > 0) {
            std::cout << ",";
            _printCSV(furtherArgs...);
        } else {
            std::cout << std::endl;
        }
    }

    const std::string _simulationId;
    const std::string _simulationMode;
};

int main(int argc, char** argv) {
    // Parse command line options
    cxxopts::Options cliParser(
        "simulate-failures-until-data-loss",
        "Simulates rank failures and uses the data distribution to check when irrecoverable data loss occurred.");

    cliParser.add_options()                                                                 ///
        ("mode", "Simulation mode. One of <rank|node|rack>", cxxopts::value<std::string>()) ///
        ("simulation-id",
         "Simulation id. Will be echoed as is; can be used to identify the results of this experiment.",
         cxxopts::value<std::string>()->default_value("rank"))                                        ///
        ("s,seed", "Random seed.", cxxopts::value<unsigned long>()->default_value("0"))               ///
        ("n,no-header", "Do not print the csv header.")                                               ///
        ("r,repetitions", "Number of replicas to run", cxxopts::value<size_t>()->default_value("10")) ///
        ("h,help", "Print help message.")                                                             ///
        ("d,ranks-per-node", "The number of ranks per node or ranks per rack to fail at once.",
         cxxopts::value<original_rank_t>()) ///
        ("c,config",
         "A configuration to simulate. Multiple configurations can be given.\n Format: <numRanks numBlocks "
         "replicationLevel>, e.g.: \"10 1000 3\"",
         cxxopts::value<std::vector<RankDistributionConfig>>());

    cliParser.parse_positional({"mode", "simulation-id"});
    cliParser.positional_help("<mode> <simulation-id>");

    cxxopts::ParseResult options;
    try {
        options = cliParser.parse(argc, argv);
    } catch (cxxopts::exceptions::exception& e) {
        std::cout << e.what() << std::endl << std::endl;
        std::cout << cliParser.help() << std::endl;
        exit(1);
    }

    if (options.count("help")) {
        std::cout << cliParser.help() << std::endl;
        exit(0);
    }

    if (!options.count("config")) {
        std::cout << "Please provide at least one configuration to simulate with --config" << std::endl << std::endl;
        std::cout << cliParser.help() << std::endl;
        exit(1);
    }

    if (!options.count("mode")) {
        std::cout << "Please provide a simulation mode." << std::endl;
        std::cout << cliParser.help() << std::endl;
        exit(1);
    }
    const auto SIMULATION_MODE = parse_mode_string(options["mode"].as<std::string>());
    if (SIMULATION_MODE == SimulationMode::INVALID) {
        std::cout << "The simulation mode must be to simulate failures either <rank|node|rack> wise." << std::endl;
        std::cout << cliParser.help() << std::endl;
        exit(1);
    }

    original_rank_t FAIL_AT_ONCE = 1;
    if ((SIMULATION_MODE == SimulationMode::NODE || SIMULATION_MODE == SimulationMode::RACK)) {
        if (!options.count("ranks-per-node")) {
            std::cout << "When in node or rack mode, you must specify the number of ranks per node/rack." << std::endl;
            std::cout << cliParser.help() << std::endl;
            exit(1);
        } else {
            FAIL_AT_ONCE = options["ranks-per-node"].as<original_rank_t>();
        }
    }

    if (!options.count("simulation-id")) {
        std::cout << "Please provide an id for this simulation." << std::endl;
        std::cout << cliParser.help() << std::endl;
        exit(1);
    }

    const auto RANDOM_SEED      = options["seed"].as<unsigned long>();
    const auto NUM_REPETITIONS  = options["repetitions"].as<size_t>();
    const auto PRINT_CSV_HEADER = !options.count("no-header");
    const auto configurations   = options["config"].as<std::vector<RankDistributionConfig>>();
    const auto SIMULATION_ID    = options["simulation-id"].as<std::string>();

    // Set up the fake MPI Context
    auto mpiContext = MPIContextFake();

    // Set up the CSV output
    ResultsPrinter resultsPrinter(PRINT_CSV_HEADER, SIMULATION_MODE, SIMULATION_ID);

    // Loop over all configurations, simulating multiple replicas each
    for (auto config: configurations) {
        if (!in_range<original_rank_t>(config.numRanks)) {
            std::cout << config.numRanks << " is not a valid rank count (too large)" << std::endl;
            exit(1);
        }
        for (auto repetition: range(NUM_REPETITIONS)) {
            // Create a new block configuration with the given configuration
            std::shared_ptr<ReStore::BlockDistribution<MPIContextFake>> blockDistribution = nullptr;
            try {
                blockDistribution = std::make_shared<ReStore::BlockDistribution<MPIContextFake>>(
                    config.numRanks, config.numBlocks, config.replicationLevel, mpiContext);
            } catch (std::runtime_error& e) {
                std::cout << "Configuration " << config << " is invalid:\n\t" << e.what() << std::endl;
                return 1;
            }

            // Precompute the random order in which the ranks shall fail.
            assert(SIMULATION_MODE != SimulationMode::RANK || FAIL_AT_ONCE == 1);
            assert(SIMULATION_MODE != SimulationMode::NODE || FAIL_AT_ONCE != 1);
            assert(SIMULATION_MODE != SimulationMode::RACK || FAIL_AT_ONCE != 1);
            auto orderOfFailures = std::vector<original_rank_t>();
            for (auto i: range(static_cast<original_rank_t>(
                     config.numRanks / throwing_cast<decltype(config.numRanks)>(FAIL_AT_ONCE)))) {
                orderOfFailures.push_back(i);
            }

            shuffle(
                orderOfFailures.begin(), orderOfFailures.end(), std::default_random_engine(RANDOM_SEED + repetition));

            // Simulate the failures in the precomputed order until a irrecoverable data loss occurs.
            for (auto failing: orderOfFailures) {
                if (SIMULATION_MODE == SimulationMode::RANK) {
                    mpiContext.killRank(failing);
                } else {
                    assert(SIMULATION_MODE == SimulationMode::NODE || SIMULATION_MODE == SimulationMode::RACK);
                    auto firstRankToFail = failing * FAIL_AT_ONCE;
                    auto lastRankToFail  = firstRankToFail + FAIL_AT_ONCE - 1;
                    for (auto failingRank: range(firstRankToFail, lastRankToFail + 1)) {
                        mpiContext.killRank(failingRank);
                    }
                }
                if (checkForDataLoss(blockDistribution)) {
                    resultsPrinter.print_result(RANDOM_SEED + repetition, config, mpiContext.numFailed());
                    break;
                }
            }

            // Reset deadRanks so we can reuse it in the next iteration (replica X config)
            mpiContext.resurrectRanks();
        }
    }

    return 0;
}
