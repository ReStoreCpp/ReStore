#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <gtest/gtest_pred_impl.h>
#include <memory>
#include <utility>

#include "mocks.hpp"
#include "restore/block_retrieval.hpp"
#include "restore/common.hpp"
#include "restore/core.hpp"
#include "restore/mpi_context.hpp"

using namespace ::testing;

TEST(BlockRetrievalTest, getServingRanks) {
    using BlockDistribution = ReStore::BlockDistribution<MPIContextMock>;
    using ReStore::OffsetMode;
    using ReStoreMPI::current_rank_t;
    using ReStoreMPI::RecvMessage;

    MPIContextMock mpiContext;
    EXPECT_CALL(mpiContext, getOnlyAlive(_)).WillRepeatedly([](std::vector<ReStoreMPI::original_rank_t> ranks) {
        return getAliveOnlyFake({}, ranks);
    });
    auto blockDistribution = std::make_shared<BlockDistribution>(10, 100, 3, mpiContext);
    auto blockRange        = blockDistribution->blockRangeById(0);
    ASSERT_EQ(0, blockRange.start());
    ASSERT_EQ(10, blockRange.length());
    ReStore::block_id_t rangeStart         = 1;
    size_t              rangeSize          = 8;
    auto                blockRangeExternal = std::pair<ReStore::block_id_t, size_t>(rangeStart, rangeSize);
    auto                servingRanks = ReStore::getServingRank(blockRange, blockRangeExternal, blockDistribution.get());
    EXPECT_EQ(3, servingRanks.size());

    size_t avgSize = blockRangeExternal.second / servingRanks.size();
    std::sort(servingRanks.begin(), servingRanks.end());
    ReStore::block_id_t nextBlockId = rangeStart;
    for (const auto& servingRank: servingRanks) {
        EXPECT_LT(servingRank.second, 10);
        auto size = servingRank.first.second;
        EXPECT_PRED2(
            [](size_t _avgSize, size_t _size) { return _size == _avgSize || _size == _avgSize + 1; }, avgSize, size);
        EXPECT_EQ(nextBlockId, servingRank.first.first);
        nextBlockId += servingRank.first.second;
    }
    EXPECT_EQ(rangeStart + rangeSize, nextBlockId);
}

TEST(BlockRetrievalTest, getServingRanksWithDeadRanks) {
    using BlockDistribution = ReStore::BlockDistribution<MPIContextMock>;
    using ReStore::OffsetMode;
    using ReStoreMPI::current_rank_t;
    using ReStoreMPI::RecvMessage;

    MPIContextMock mpiContext;
    EXPECT_CALL(mpiContext, getOnlyAlive(_)).WillRepeatedly([](std::vector<ReStoreMPI::original_rank_t> ranks) {
        return getAliveOnlyFake({6, 7, 8, 9}, ranks);
    });
    auto blockDistribution = std::make_shared<BlockDistribution>(10, 100, 3, mpiContext);
    auto blockRange        = blockDistribution->blockRangeById(0);
    ASSERT_EQ(0, blockRange.start());
    ASSERT_EQ(10, blockRange.length());
    ReStore::block_id_t rangeStart         = 1;
    size_t              rangeSize          = 8;
    auto                blockRangeExternal = std::pair<ReStore::block_id_t, size_t>(rangeStart, rangeSize);
    auto                servingRanks = ReStore::getServingRank(blockRange, blockRangeExternal, blockDistribution.get());
    EXPECT_EQ(2, servingRanks.size());

    size_t avgSize = blockRangeExternal.second / servingRanks.size();
    std::sort(servingRanks.begin(), servingRanks.end());
    ReStore::block_id_t nextBlockId = rangeStart;
    for (const auto& servingRank: servingRanks) {
        EXPECT_LT(servingRank.second, 6);
        auto size = servingRank.first.second;
        EXPECT_PRED2(
            [](size_t _avgSize, size_t _size) { return _size == _avgSize || _size == _avgSize + 1; }, avgSize, size);
        EXPECT_EQ(nextBlockId, servingRank.first.first);
        nextBlockId += servingRank.first.second;
    }
    EXPECT_EQ(rangeStart + rangeSize, nextBlockId);
}
