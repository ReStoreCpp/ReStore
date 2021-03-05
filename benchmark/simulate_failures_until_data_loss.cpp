#include <cppitertools/range.hpp>
#include <iostream>
#include <random>
#include <restore/core.hpp>
#include <restore/helpers.hpp>
#include <vector>

#include <cxxopts.hpp>

#include "../tests/mocks.hpp"
#include "restore/helpers.hpp"

using namespace std;
using namespace ::testing;

using iter::range;
using ReStoreMPI::original_rank_t;

bool checkForDataLoss(shared_ptr<ReStore::BlockDistribution<MPIContextFake>> blockDistribution) {
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

    friend istream& operator>>(istream& stream, RankDistributionConfig& config) {
        stream >> config.numRanks >> config.numBlocks >> config.replicationLevel;
        return stream;
    }

    friend ostream& operator<<(ostream& stream, const RankDistributionConfig& config) {
        stream << "{numRanks: " << config.numRanks << ", numBlocks: " << config.numBlocks
               << ", replicationLevel: " << config.replicationLevel << "}";
        return stream;
    }
};

enum class SimulationMode { RANK, NODE, RACK, INVALID };

SimulationMode parse_mode_string(const string& modeStr) {
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

string to_string(const SimulationMode& simulationMode) {
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
    ResultsPrinter(bool printHeader, const SimulationMode& simulationMode, string simulationId) noexcept
        : _simulationId(simulationId),
          _simulationMode(to_string(simulationMode)) {
        if (printHeader) {
            cout << "simulation_id,mode,seed,num_ranks,num_blocks,replication_level,failures_until_data_loss" << endl;
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
        cout << firstArg;
        if constexpr (sizeof...(furtherArgs) > 0) {
            cout << ",";
            _printCSV(furtherArgs...);
        } else {
            cout << endl;
        }
    }

    const string _simulationId;
    const string _simulationMode;
};

int main(int argc, char** argv) {
    // Parse command line options
    cxxopts::Options cliParser(
        "simulate-failures-until-data-loss",
        "Simulates rank failures and uses the data distribution to check when irrecoverable data loss occurred.");

    cliParser.add_options()                                                            ///
        ("mode", "Simulation mode. One of <rank|node|rack>", cxxopts::value<string>()) ///
        ("simulation-id",
         "Simulation id. Will be echoed as is; can be used to identify the results of this experiment.",
         cxxopts::value<string>()->default_value("rank"))                                             ///
        ("s,seed", "Random seed.", cxxopts::value<unsigned long>()->default_value("0"))               ///
        ("n,no-header", "Do not print the csv header.")                                               ///
        ("r,repetitions", "Number of replicas to run", cxxopts::value<size_t>()->default_value("10")) ///
        ("h,help", "Print help message.")                                                             ///
        ("d,ranks-per-node", "The number of ranks per node or ranks per rack to fail at once.",
         cxxopts::value<original_rank_t>()) ///
        ("c,config",
         "A configuration to simulate. Multiple configurations can be given.\n Format: <numRanks numBlocks "
         "replicationLevel>, e.g.: \"10 1000 3\"",
         cxxopts::value<vector<RankDistributionConfig>>());

    cliParser.parse_positional({"mode", "simulation-id"});
    cliParser.positional_help("<mode> <simulation-id>");

    cxxopts::ParseResult options;
    try {
        options = cliParser.parse(argc, argv);
    } catch (cxxopts::OptionException& e) {
        cout << e.what() << endl << endl;
        cout << cliParser.help() << endl;
        exit(1);
    }

    if (options.count("help")) {
        cout << cliParser.help() << endl;
        exit(0);
    }

    if (!options.count("config")) {
        cout << "Please provide at least one configuration to simulate with --config" << endl << endl;
        cout << cliParser.help() << endl;
        exit(1);
    }

    if (!options.count("mode")) {
        cout << "Please provide a simulation mode." << endl;
        cout << cliParser.help() << endl;
        exit(1);
    }
    const auto SIMULATION_MODE = parse_mode_string(options["mode"].as<string>());
    if (SIMULATION_MODE == SimulationMode::INVALID) {
        cout << "The simulation mode must be to simulate failures either <rank|node|rack> wise." << endl;
        cout << cliParser.help() << endl;
        exit(1);
    }

    original_rank_t FAIL_AT_ONCE = 1;
    if ((SIMULATION_MODE == SimulationMode::NODE || SIMULATION_MODE == SimulationMode::RACK)) {
        if (!options.count("ranks-per-node")) {
            cout << "When in node or rack mode, you must specify the number of ranks per node/rack." << endl;
            cout << cliParser.help() << endl;
            exit(1);
        } else {
            FAIL_AT_ONCE = options["ranks-per-node"].as<original_rank_t>();
        }
    }

    if (!options.count("simulation-id")) {
        cout << "Please provie an id for this simulation." << endl;
        cout << cliParser.help() << endl;
        exit(1);
    }

    const auto RANDOM_SEED      = options["seed"].as<unsigned long>();
    const auto NUM_REPETITIONS  = options["repetitions"].as<size_t>();
    const auto PRINT_CSV_HEADER = !options.count("no-header");
    const auto configurations   = options["config"].as<vector<RankDistributionConfig>>();
    const auto SIMULATION_ID    = options["simulation-id"].as<string>();

    // Set up the fake MPI Context
    auto mpiContext = MPIContextFake();

    // Set up the CSV output
    ResultsPrinter resultsPrinter(PRINT_CSV_HEADER, SIMULATION_MODE, SIMULATION_ID);

    // Loop over all configurations, simulating multiple replicas each
    for (auto config: configurations) {
        if (!in_range<original_rank_t>(config.numRanks)) {
            cout << config.numRanks << " is not a valid rank count (too large)" << endl;
            exit(1);
        }
        for (auto repetition: range(NUM_REPETITIONS)) {
            // Create a new block configuration with the given configuration
            shared_ptr<ReStore::BlockDistribution<MPIContextFake>> blockDistribution = nullptr;
            try {
                blockDistribution = make_shared<ReStore::BlockDistribution<MPIContextFake>>(
                    config.numRanks, config.numBlocks, config.replicationLevel, mpiContext);
            } catch (runtime_error& e) {
                cout << "Configuration " << config << " is invalid:\n\t" << e.what() << endl;
                return 1;
            }

            // Precompute the random order in which the ranks shall fail.
            assert(SIMULATION_MODE != SimulationMode::RANK || FAIL_AT_ONCE == 1);
            assert(SIMULATION_MODE != SimulationMode::NODE || FAIL_AT_ONCE != 1);
            assert(SIMULATION_MODE != SimulationMode::RACK || FAIL_AT_ONCE != 1);
            auto orderOfFailures = vector<original_rank_t>();
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
