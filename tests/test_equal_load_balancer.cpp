#include <algorithm>
#include <gtest/gtest.h>
#include <math.h>
#include <numeric>
#include <unordered_map>

#include "restore/common.hpp"
#include "restore/equal_load_balancer.hpp"
#include "restore/mpi_context.hpp"

using namespace ::testing;

TEST(EqualLoadBalancerTest, noFailuresTest) {
    std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, ReStoreMPI::original_rank_t>> initialBlockRanges = {
        {{0, 100}, 0},   {{100, 100}, 1}, {{200, 100}, 2}, {{300, 100}, 3}, {{400, 100}, 4},
        {{500, 100}, 5}, {{600, 100}, 6}, {{700, 100}, 7}, {{800, 100}, 8}, {{900, 100}, 9}};

    auto loadBalancer   = ReStore::EqualLoadBalancer(initialBlockRanges, 10);
    auto newBlockRanges = loadBalancer.getNewBlocksAfterFailure({});

    EXPECT_EQ(newBlockRanges.size(), 0);
}

TEST(EqualLoadBalancerTest, simpleLoadBalancerTest) {
    std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, ReStoreMPI::original_rank_t>> initialBlockRanges = {
        {{0, 100}, 0},   {{100, 100}, 1}, {{200, 100}, 2}, {{300, 100}, 3}, {{400, 100}, 4},
        {{500, 100}, 5}, {{600, 100}, 6}, {{700, 100}, 7}, {{800, 100}, 8}, {{900, 100}, 9}};

    auto loadBalancer   = ReStore::EqualLoadBalancer(initialBlockRanges, 10);
    auto newBlockRanges = loadBalancer.getNewBlocksAfterFailure({0, 1});
    // don't commit previous change and discard it by requesting distribution for different set of dead ranks
    newBlockRanges = loadBalancer.getNewBlocksAfterFailure({5, 7});

    // This should be perfectly splittable into 25 block chunks
    EXPECT_EQ(newBlockRanges.size(), 8);

    // All 200 blocks from ranks 5 and 7 should have a new rank
    size_t count = std::accumulate(newBlockRanges.begin(), newBlockRanges.end(), 0u, [](size_t size, auto blockRange) {
        return size + blockRange.first.second;
    });
    EXPECT_EQ(count, 200);

    // Nothing for 5
    EXPECT_EQ(
        std::find_if(
            newBlockRanges.begin(), newBlockRanges.end(), [](auto blockRange) { return blockRange.second == 5; }),
        newBlockRanges.end());
    // Nothing for 7
    EXPECT_EQ(
        std::find_if(
            newBlockRanges.begin(), newBlockRanges.end(), [](auto blockRange) { return blockRange.second == 7; }),
        newBlockRanges.end());

    // Sort by ascending block id (works because blockRanges are pairs and pairs are sorted lexicographically)
    std::sort(newBlockRanges.begin(), newBlockRanges.end());

    ReStore::block_id_t expectedBlockId = 500;
    for (const auto& blockRange: newBlockRanges) {
        EXPECT_FALSE(expectedBlockId >= 600 && expectedBlockId < 700);
        EXPECT_TRUE(expectedBlockId < 800);
        // Check that all lost blocks are there
        EXPECT_EQ(expectedBlockId, blockRange.first.first);
        expectedBlockId += blockRange.first.second;
        if (expectedBlockId == 600) {
            expectedBlockId = 700;
        }
    }

    // Commit to ranks 5 and 7 being dead
    loadBalancer.commitToPreviousCall();
    // Commit again to make sure this doesn't crash or produce otherwise wrong results
    loadBalancer.commitToPreviousCall();
    // Add 2 more dead ranks
    newBlockRanges = loadBalancer.getNewBlocksAfterFailure({0, 1});

    // All 250 blocks from ranks 0 and 1 should have a new rank
    count = std::accumulate(newBlockRanges.begin(), newBlockRanges.end(), 0u, [](size_t size, auto blockRange) {
        return size + blockRange.first.second;
    });
    EXPECT_EQ(count, 250);

    std::unordered_map<ReStoreMPI::original_rank_t, size_t> receivers;
    for (const auto& blockRange: newBlockRanges) {
        receivers[blockRange.second] += blockRange.first.second;
    }

    // Nothing for dead ranks
    EXPECT_TRUE(receivers.find(0) == receivers.end());
    EXPECT_TRUE(receivers.find(1) == receivers.end());
    EXPECT_TRUE(receivers.find(5) == receivers.end());
    EXPECT_TRUE(receivers.find(7) == receivers.end());

    // Everyone should get something
    EXPECT_EQ(6, receivers.size());

    // Everyone should get the average number of items +- 1
    for (const auto& mapIt: receivers) {
        EXPECT_GE(mapIt.second, floor(250.0 / 6.0));
        EXPECT_LE(mapIt.second, ceil(250.0 / 6.0));
    }
}
