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

template <bool pullBlocks>
static void BM_EqualLoadBalancer(benchmark::State& state) {
    const auto numRanks = throwing_cast<ReStoreMPI::original_rank_t>(state.range(0));

    const uint64_t numBlocksPerRank    = 65536;
    const uint64_t myRank              = 1337;
    const double   numExpectedFailures = ceil(0.01 * static_cast<double>(numRanks));

    for (auto _: state) {
        state.PauseTiming();
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
        state.ResumeTiming();
        for (size_t failure = 0; failure < static_cast<size_t>(numExpectedFailures); ++failure) {
            auto rankToFail = ranksToDie[failure];
            if (rankToFail == myRank) {
                continue;
            }
            ranksToDieInThisIteration = {rankToFail};

            auto newBlockRanges = [&]() {
                if constexpr (pullBlocks) {
                    return loadBalancer.getNewBlocksAfterFailureForPullBlocks(ranksToDieInThisIteration, myRank);
                } else {
                    return loadBalancer.getNewBlocksAfterFailureForPushBlocks(ranksToDieInThisIteration);
                }
            }();
            loadBalancer.commitToPreviousCall();
            benchmark::DoNotOptimize(newBlockRanges);
            benchmark::ClobberMemory();
        }
        // std::cout << dummy << std::endl;
    }
}

static void BM_EqualLoadBalancerPullBlocks(benchmark::State& state) {
    BM_EqualLoadBalancer<true>(state);
}

static void BM_EqualLoadBalancerPushBlocks(benchmark::State& state) {
    BM_EqualLoadBalancer<false>(state);
}

static void benchmarkArguments(benchmark::internal::Benchmark* benchmark) {
    const ReStoreMPI::original_rank_t startRankCount = 48;
    const ReStoreMPI::original_rank_t endRankCount   = 512 * 48;

    for (ReStoreMPI::original_rank_t rankCount = startRankCount; rankCount <= endRankCount; rankCount *= 2) {
        benchmark->Args({rankCount});
    }
}


BENCHMARK(BM_EqualLoadBalancerPullBlocks) ///
    ->Unit(benchmark::kMillisecond)       ///
    ->Apply(benchmarkArguments);

BENCHMARK(BM_EqualLoadBalancerPushBlocks) ///
    ->Unit(benchmark::kMillisecond)       ///
    ->Apply(benchmarkArguments);

BENCHMARK_MAIN();
