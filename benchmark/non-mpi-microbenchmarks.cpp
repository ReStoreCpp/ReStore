#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <restore/common.hpp>
#include <string>
#include <utility>

#if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wsuggest-override"
    #include <benchmark/benchmark.h>
    #pragma GCC diagnostic pop
#else
    #include <benchmark/benchmark.h>
#endif

#include "probabilistic_failure_simulator.hpp"
#include <restore/core.hpp>
#include <restore/equal_load_balancer.hpp>
#include <restore/helpers.hpp>
#include <restore/mpi_context.hpp>

static void BM_EqualLoadBalancer(benchmark::State& state) {
    const auto numRanks = throwing_cast<ReStoreMPI::original_rank_t>(state.range(0));

    const uint64_t numBlocksPerRank    = 65536;
    const uint64_t myRank              = 1337;
    const double   numExpectedFailures = ceil(0.01 * static_cast<double>(numRanks));

    for (auto _: state) {
        std::vector<ReStoreMPI::original_rank_t> ranksToDie(asserting_cast<size_t>(numRanks));
        std::iota(ranksToDie.begin(), ranksToDie.end(), 0);
        std::random_shuffle(ranksToDie.begin(), ranksToDie.end());

        std::vector<bool>                        ranksAlive(asserting_cast<size_t>(numRanks), true);
        std::vector<ReStoreMPI::original_rank_t> ranksToDieInThisIteration;

        std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, ReStoreMPI::original_rank_t>> blockRanges;
        for (uint64_t rank = 0; rank < asserting_cast<size_t>(numRanks); ++rank) {
            blockRanges.emplace_back(std::make_pair(rank * numBlocksPerRank, numBlocksPerRank), rank);
        }

        ReStore::EqualLoadBalancer loadBalancer(blockRanges, numRanks);

        auto start = std::chrono::high_resolution_clock::now();
        for (size_t failure = 0; failure < static_cast<size_t>(numExpectedFailures); ++failure) {
            auto rankToFail = ranksToDie[failure];
            if (rankToFail == myRank) {
                continue;
            }
            ranksToDieInThisIteration = {rankToFail};

            auto newBlockRanges = loadBalancer.getNewBlocksAfterFailureForPullBlocks(ranksToDieInThisIteration, myRank);
            loadBalancer.commitToPreviousCall();
            benchmark::DoNotOptimize(newBlockRanges);
            benchmark::ClobberMemory();
        }
        auto end            = std::chrono::high_resolution_clock::now();
        auto elapsedSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        state.SetIterationTime(elapsedSeconds);
        //std::cout << dummy << std::endl;
    }
}

BENCHMARK(BM_EqualLoadBalancer)     ///
    ->UseManualTime()               ///
    ->Unit(benchmark::kMillisecond) ///
    ->RangeMultiplier(2)            ///
    ->Range(48, 6144);              ///

BENCHMARK_MAIN();