#include <cppitertools/range.hpp>
#include <iostream>
#include <random>
#include <restore/core.hpp>
#include <restore/helpers.hpp>
#include <vector>

#include <cxxopts.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../tests/mocks.hpp"
#include "restore/helpers.hpp"

using namespace std;
using namespace ::testing;

using iter::range;
using ReStoreMPI::original_rank_t;

bool checkForDataLoss(shared_ptr<ReStore::BlockDistribution<MPIContextMock>> blockDistribution) {
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

class ResultsPrinter {
    public:
    ResultsPrinter(bool printHeader) noexcept {
        if (printHeader) {
            cout << "seed,num_ranks,num_blocks,replication_level,failures_until_data_loss" << endl;
        }
    }

    void print_result(unsigned long seed, RankDistributionConfig config, size_t failuresUntilDataLoss) {
        _printCSV(seed, config.numRanks, config.numBlocks, config.replicationLevel, failuresUntilDataLoss);
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
};

int main(int argc, char** argv) {
    // Parse command line options
    cxxopts::Options cliParser(
        "simulate-failures-until-data-loss",
        "Simulates rank failures and uses the data distribution to check when irrecoverable data loss occurred.");

    cliParser.add_options()                                                                        ///
        ("s,seed", "Random seed.", cxxopts::value<unsigned long>()->default_value("0"))            ///
        ("n,no-header", "Do not print the csv header.")                                            ///
        ("r,repetitions", "Number of replicas to run", cxxopts::value<size_t>()->default_value("10")) ///
        ("h,help", "Print help message.")                                                          ///
        ("c,config",
         "A configuration to simulate. Multiple configurations can be given.\n Format: <numRanks numBlocks "
         "replicationLevel>, e.g.: \"10 1000 3\"",
         cxxopts::value<vector<RankDistributionConfig>>());

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
    }

    const auto RANDOM_SEED      = options["seed"].as<unsigned long>();
    const auto NUM_REPETITIONS     = options["repetitions"].as<size_t>();
    const auto PRINT_CSV_HEADER = !options.count("no-header");
    const auto configurations   = options["config"].as<vector<RankDistributionConfig>>();

    // Set up the fake MPI Context
    auto                    mpiContext = MPIContextMock();
    vector<original_rank_t> deadRanks;

    // deadRanks is passed by reference, we can therefore push_back new failed ranks onto dead ranks when simulating
    // failures.
    EXPECT_CALL(mpiContext, getOnlyAlive(_)).WillRepeatedly([&deadRanks](std::vector<original_rank_t> ranks) {
        return getAliveOnlyFake(deadRanks, ranks);
    });

    // Set up the CSV output
    ResultsPrinter resultsPrinter(PRINT_CSV_HEADER);

    // Loop over all configurations, simulating multiple replicas each
    for (auto config: configurations) {
        if (!in_range<original_rank_t>(config.numRanks)) {
            cout << config.numRanks << " is not a valid rank count (too large)" << endl;
            exit(1);
        }
        for (auto repetition: range(NUM_REPETITIONS)) {
            // Create a new block configuration with the given configuration
            shared_ptr<ReStore::BlockDistribution<MPIContextMock>> blockDistribution = nullptr;
            try {
                blockDistribution = make_shared<ReStore::BlockDistribution<MPIContextMock>>(
                    config.numRanks, config.numBlocks, config.replicationLevel, mpiContext);
            } catch (runtime_error& e) {
                cout << "Configuration " << config << " is invalid:\n\t" << e.what() << endl;
                return 1;
            }

            // Precompute the random order in which the ranks shall fail.
            auto orderOfRankFailures = vector<original_rank_t>();
            for (auto i: range(static_cast<original_rank_t>(config.numRanks))) {
                orderOfRankFailures.push_back(i);
            }
            shuffle(
                orderOfRankFailures.begin(), orderOfRankFailures.end(),
                std::default_random_engine(RANDOM_SEED + repetition));

            // Simulate the failures in the precomputed order until a irrecoverable data loss occurs.
            for (auto failingRank: orderOfRankFailures) {
                deadRanks.push_back(failingRank);
                if (checkForDataLoss(blockDistribution)) {
                    resultsPrinter.print_result(RANDOM_SEED + repetition, config, deadRanks.size());
                    break;
                }
            }

            // Reset deadRanks so we can reuse it in the next iteration (replica X config)
            deadRanks.clear();
        }
    }

    return 0;
}