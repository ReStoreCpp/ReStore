
#include <algorithm>
#include <initializer_list>
#include <optional>
#include <stddef.h>

#include <chain.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <range.hpp>

#include "restore/core.hpp"
#include "restore/helpers.hpp"

#include "mocks.hpp"

using namespace ::testing;

TEST(BlockDistributionTest, BlockRange) {
    // BlockRange(size_t range_id, size_t numBlocks, size_t numRanges) {
    // bool includes(block_id_t block) {
    using BlockRange = ReStore::BlockDistribution<MPIContextMock>::BlockRange;
    using block_id_t = ReStore::block_id_t;

    // Mock MPI context to pass to the block distribution
    auto mpiContext = MPIContextMock();
    EXPECT_CALL(mpiContext, getOriginalRank(_)).WillRepeatedly(ReturnArg<0>());
    EXPECT_CALL(mpiContext, getCurrentRank(_)).WillRepeatedly(ReturnArg<0>());
    EXPECT_CALL(mpiContext, isAlive(_)).WillRepeatedly(Return(true));
    EXPECT_CALL(mpiContext, getOnlyAlive(_)).WillRepeatedly(ReturnArg<0>());

    {
        auto blockDistribution = ReStore::BlockDistribution<MPIContextMock>(
            100,  // number of ranks
            1000, // number of blocks
            3,    // replication level,
            mpiContext);

        ASSERT_ANY_THROW(BlockRange(101, &blockDistribution)); // Range id greater than the number of ranges
    }

    {
        auto blockDistribution = ReStore::BlockDistribution<MPIContextMock>(
            10,  // number of ranks
            100, // number of blocks
            3,   // replication level,
            mpiContext);

        auto range = BlockRange(0, &blockDistribution);
        ASSERT_EQ(range.id(), 0);
        ASSERT_EQ(range.start(), 0);
        ASSERT_EQ(range.length(), 10);

        for (block_id_t blockId = 0; blockId < 10; blockId++) {
            ASSERT_TRUE(range.contains(blockId));
        }
        for (block_id_t blockId = 10; blockId < 100; blockId++) {
            ASSERT_FALSE(range.contains(blockId));
        }

        ASSERT_FALSE(range.contains(101));
    }

    {
        auto blockDistribution = ReStore::BlockDistribution<MPIContextMock>(
            10,  // number of ranks
            100, // number of blocks
            3,   // replication level,
            mpiContext);

        auto range = BlockRange(2, &blockDistribution);
        ASSERT_EQ(range.id(), 2);
        ASSERT_EQ(range.start(), 20);
        ASSERT_EQ(range.length(), 10);

        for (block_id_t blockId = 0; blockId < 20; blockId++) {
            ASSERT_FALSE(range.contains(blockId));
        }
        for (block_id_t blockId = 20; blockId < 30; blockId++) {
            ASSERT_TRUE(range.contains(blockId));
        }
        for (block_id_t blockId = 30; blockId < 100; blockId++) {
            ASSERT_FALSE(range.contains(blockId));
        }

        ASSERT_FALSE(range.contains(101));
    }

    {
        auto blockDistribution = ReStore::BlockDistribution<MPIContextMock>(
            3,  // number of ranks
            10, // number of blocks
            2,  // replication level,
            mpiContext);

        // range 0: 0, 1, 2, 3
        // range 1: 4, 5, 6
        // range 2: 7, 8, 9
        auto range0 = BlockRange(0, &blockDistribution);
        ASSERT_TRUE(range0.contains(0));
        ASSERT_TRUE(range0.contains(1));
        ASSERT_TRUE(range0.contains(2));
        ASSERT_TRUE(range0.contains(3));
        ASSERT_FALSE(range0.contains(4));
        ASSERT_FALSE(range0.contains(5));
        ASSERT_FALSE(range0.contains(6));
        ASSERT_FALSE(range0.contains(7));
        ASSERT_FALSE(range0.contains(8));
        ASSERT_FALSE(range0.contains(9));

        auto range1 = BlockRange(1, &blockDistribution);
        ASSERT_FALSE(range1.contains(0));
        ASSERT_FALSE(range1.contains(1));
        ASSERT_FALSE(range1.contains(2));
        ASSERT_FALSE(range1.contains(3));
        ASSERT_TRUE(range1.contains(4));
        ASSERT_TRUE(range1.contains(5));
        ASSERT_TRUE(range1.contains(6));
        ASSERT_FALSE(range1.contains(7));
        ASSERT_FALSE(range1.contains(8));
        ASSERT_FALSE(range1.contains(9));

        auto range2 = BlockRange(2, &blockDistribution);
        ASSERT_FALSE(range2.contains(0));
        ASSERT_FALSE(range2.contains(1));
        ASSERT_FALSE(range2.contains(2));
        ASSERT_FALSE(range2.contains(3));
        ASSERT_FALSE(range2.contains(4));
        ASSERT_FALSE(range2.contains(5));
        ASSERT_FALSE(range2.contains(6));
        ASSERT_TRUE(range2.contains(7));
        ASSERT_TRUE(range2.contains(8));
        ASSERT_TRUE(range2.contains(9));
    }

    {
        auto blockDistribution = ReStore::BlockDistribution<MPIContextMock>(
            4, // number of ranks
            6, // number of blocks
            2, // replication level,
            mpiContext);

        // range0: 0, 1
        // range1: 2, 3
        // range2: 4
        // range3: 5
        auto range0 = BlockRange(0, &blockDistribution);
        ASSERT_TRUE(range0.contains(0));
        ASSERT_TRUE(range0.contains(1));
        ASSERT_FALSE(range0.contains(2));
        ASSERT_FALSE(range0.contains(3));
        ASSERT_FALSE(range0.contains(4));
        ASSERT_FALSE(range0.contains(5));

        auto range1 = BlockRange(1, &blockDistribution);
        ASSERT_FALSE(range1.contains(0));
        ASSERT_FALSE(range1.contains(1));
        ASSERT_TRUE(range1.contains(2));
        ASSERT_TRUE(range1.contains(3));
        ASSERT_FALSE(range1.contains(4));
        ASSERT_FALSE(range1.contains(5));

        auto range2 = BlockRange(2, &blockDistribution);
        ASSERT_FALSE(range2.contains(0));
        ASSERT_FALSE(range2.contains(1));
        ASSERT_FALSE(range2.contains(2));
        ASSERT_FALSE(range2.contains(3));
        ASSERT_TRUE(range2.contains(4));
        ASSERT_FALSE(range2.contains(5));

        auto range3 = BlockRange(3, &blockDistribution);
        ASSERT_FALSE(range3.contains(0));
        ASSERT_FALSE(range3.contains(1));
        ASSERT_FALSE(range3.contains(2));
        ASSERT_FALSE(range3.contains(3));
        ASSERT_FALSE(range3.contains(4));
        ASSERT_TRUE(range3.contains(5));
    }

    {
        auto blockDistribution1 = ReStore::BlockDistribution<MPIContextMock>(
            50,   // number of ranks
            6000, // number of blocks
            2,    // replication level,
            mpiContext);
        auto blockDistribution2 = ReStore::BlockDistribution<MPIContextMock>(
            60,   // number of ranks
            6000, // number of blocks
            2,    // replication level,
            mpiContext);

        // Different block distributions
        ASSERT_NE(BlockRange(0, &blockDistribution1), BlockRange(1, &blockDistribution2));
        ASSERT_NE(BlockRange(1, &blockDistribution1), BlockRange(0, &blockDistribution2));

        // Different id
        ASSERT_NE(BlockRange(0, &blockDistribution1), BlockRange(1, &blockDistribution1));
        ASSERT_NE(BlockRange(0, &blockDistribution2), BlockRange(1, &blockDistribution2));

        // Different id and different block distribution
        ASSERT_NE(BlockRange(42, &blockDistribution1), BlockRange(1, &blockDistribution2));
        ASSERT_NE(BlockRange(42, &blockDistribution2), BlockRange(1, &blockDistribution1));

        // Same id and block distribution
        ASSERT_EQ(BlockRange(13, &blockDistribution1), BlockRange(13, &blockDistribution1));
        ASSERT_EQ(BlockRange(0, &blockDistribution2), BlockRange(0, &blockDistribution2));
        ASSERT_EQ(BlockRange(12, &blockDistribution2), BlockRange(12, &blockDistribution2));
    }
}

TEST(BlockDistributionTest, Basic) {
    using BlockDistribution = ReStore::BlockDistribution<MPIContextMock>;
    using block_id_t        = ReStore::block_id_t;
    using original_rank_t   = ReStoreMPI::original_rank_t;

    // Mock MPI context to pass to the block distribution
    auto mpiContext = MPIContextMock();
    EXPECT_CALL(mpiContext, getOriginalRank(_)).WillRepeatedly(ReturnArg<0>());
    EXPECT_CALL(mpiContext, getCurrentRank(_)).WillRepeatedly(ReturnArg<0>());
    EXPECT_CALL(mpiContext, isAlive(_)).WillRepeatedly(Return(true));
    EXPECT_CALL(mpiContext, getOnlyAlive(_)).WillRepeatedly(ReturnArg<0>());

    // Constructor - invalid arguments
    ASSERT_ANY_THROW(BlockDistribution(0, 1, 1, mpiContext));     // No ranks
    ASSERT_ANY_THROW(BlockDistribution(1, 1, 2, mpiContext));     // replication level > #ranks
    ASSERT_ANY_THROW(BlockDistribution(1, 0, 1, mpiContext));     // No blocks
    ASSERT_ANY_THROW(BlockDistribution(2, 1, 1, mpiContext));     // Less blocks than ranks
    ASSERT_ANY_THROW(BlockDistribution(1, 1, 0, mpiContext));     // Replication level of zero
    ASSERT_ANY_THROW(BlockDistribution(10, 100, 11, mpiContext)); // replication level > #ranks

    // Comparison of BlockDistribution objects
    {
        // The mpiContext is not checked for equality
        ASSERT_EQ(BlockDistribution(1, 1, 1, mpiContext), BlockDistribution(1, 1, 1, mpiContext));
        ASSERT_EQ(BlockDistribution(12, 1000, 2, mpiContext), BlockDistribution(12, 1000, 2, mpiContext));
        ASSERT_EQ(BlockDistribution(10, 1000, 3, mpiContext), BlockDistribution(10, 1000, 3, mpiContext));
        ASSERT_EQ(BlockDistribution(10, 1000, 3, mpiContext), BlockDistribution(10, 1000, 3, mpiContext));

        ASSERT_NE(BlockDistribution(11, 1000, 3, mpiContext), BlockDistribution(10, 1000, 3, mpiContext));
        ASSERT_NE(BlockDistribution(10, 1000, 3, mpiContext), BlockDistribution(11, 1000, 3, mpiContext));

        ASSERT_NE(BlockDistribution(10, 1000, 3, mpiContext), BlockDistribution(10, 1001, 3, mpiContext));
        ASSERT_NE(BlockDistribution(10, 1001, 3, mpiContext), BlockDistribution(10, 1000, 3, mpiContext));

        ASSERT_NE(BlockDistribution(10, 1000, 3, mpiContext), BlockDistribution(10, 1000, 4, mpiContext));
        ASSERT_NE(BlockDistribution(10, 1000, 4, mpiContext), BlockDistribution(10, 1000, 3, mpiContext));

        ASSERT_NE(BlockDistribution(10, 1000, 3, mpiContext), BlockDistribution(10, 1001, 4, mpiContext));
        ASSERT_NE(BlockDistribution(10, 1001, 4, mpiContext), BlockDistribution(10, 1000, 3, mpiContext));

        ASSERT_NE(BlockDistribution(11, 1001, 1, mpiContext), BlockDistribution(10, 1000, 3, mpiContext));
        ASSERT_NE(BlockDistribution(10, 1000, 3, mpiContext), BlockDistribution(11, 1001, 1, mpiContext));
    }

    {
        // 10 ranks, 100 blocks, (replication level) k = 3
        auto blockDistribution = std::make_shared<BlockDistribution>(10, 100, 3, mpiContext);
        ASSERT_EQ(blockDistribution->shiftWidth(), 3);
        ASSERT_EQ(blockDistribution->numBlocks(), 100);
        ASSERT_EQ(blockDistribution->numRanks(), 10);
        ASSERT_EQ(blockDistribution->replicationLevel(), 3);
        ASSERT_EQ(blockDistribution->numRanges(), 10);
        ASSERT_EQ(blockDistribution->blocksPerRange(), 10);
        ASSERT_EQ(blockDistribution->numRangesWithAdditionalBlock(), 0);

        // Exactly first ten blocks should be in range 0
        for (auto blockId: iter::range<block_id_t>(0, 10)) {
            ASSERT_EQ(blockDistribution->rangeOfBlock(blockId).id(), 0);
        }
        for (auto blockId: iter::range<block_id_t>(10, 100)) {
            ASSERT_NE(blockDistribution->rangeOfBlock(blockId).id(), 0);
        }

        // Blocks 30..39 should be in range 3
        for (auto blockId: iter::range<block_id_t>(0, 30)) {
            ASSERT_NE(blockDistribution->rangeOfBlock(blockId).id(), 3);
        }
        for (auto blockId: iter::range<block_id_t>(30, 40)) {
            ASSERT_EQ(blockDistribution->rangeOfBlock(blockId).id(), 3);
        }
        for (auto blockId: iter::range<block_id_t>(40, 100)) {
            ASSERT_NE(blockDistribution->rangeOfBlock(blockId).id(), 3);
        }

        // Rank 7 stores ranges 7, 4, and 1
        ASSERT_THAT(
            blockDistribution->rangesStoredOnRank(7),
            UnorderedElementsAre(
                blockDistribution->blockRangeById(7), blockDistribution->blockRangeById(4),
                blockDistribution->blockRangeById(1)));

        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(0), 7));
        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(1), 7));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(2), 7));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(3), 7));
        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(4), 7));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(5), 7));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(6), 7));
        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(7), 7));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(8), 7));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(9), 7));

        // ... and so should the blocks 10..19, 40..49 and 70..79
        for (block_id_t blockId: iter::chain(
                 iter::range<block_id_t>(10, 20), iter::range<block_id_t>(40, 50), iter::range<block_id_t>(70, 80))) {
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 7));
        }
        for (block_id_t blockId: iter::chain(
                 iter::range<block_id_t>(0, 10), iter::range<block_id_t>(20, 40), iter::range<block_id_t>(50, 70),
                 iter::range<block_id_t>(80, 100))) {
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 7));
        }

        // Rank 1 stores ranges 1, 8, and 5
        ASSERT_THAT(
            blockDistribution->rangesStoredOnRank(1),
            UnorderedElementsAre(
                blockDistribution->blockRangeById(1), blockDistribution->blockRangeById(8),
                blockDistribution->blockRangeById(5)));

        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(0), 1));
        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(1), 1));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(2), 1));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(3), 1));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(4), 1));
        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(5), 1));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(6), 1));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(7), 1));
        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(8), 1));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(9), 1));

        // ... and so should the blocks 10..19, 50..59 and 80..89
        for (block_id_t blockId: iter::chain(
                 iter::range<block_id_t>(10, 20), iter::range<block_id_t>(50, 60), iter::range<block_id_t>(80, 90))) {
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 1));
        }
        for (block_id_t blockId: iter::chain(
                 iter::range<block_id_t>(0, 10), iter::range<block_id_t>(20, 50), iter::range<block_id_t>(60, 80),
                 iter::range<block_id_t>(90, 100))) {
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 1));
        }

        // The first ten blocks should be on rank 0, 3 and 6 but not on any other ranks
        for (auto blockId: iter::range<block_id_t>(0, 10)) {
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 0));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 1));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 2));
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 3));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 4));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 5));
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 6));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 7));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 8));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 9));

            ASSERT_THAT(blockDistribution->ranksBlockIsStoredOn(blockId), UnorderedElementsAre(0, 3, 6));

            const auto blockRange = blockDistribution->rangeOfBlock(blockId);
            for (uint64_t seed = 0; seed < 10; seed++) {
                ASSERT_THAT(blockDistribution->randomAliveRankBlockRangeIsStoredOn(blockRange, seed), AnyOf(0, 3, 6));
            }
        }

        // The blocks 70..79 should be on rank 7, 0 and 3 but not on any other ranks
        for (block_id_t blockId: iter::range<block_id_t>(70, 80)) {
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 0));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 1));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 2));
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 3));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 4));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 5));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 6));
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 7));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 8));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 9));

            ASSERT_THAT(blockDistribution->ranksBlockIsStoredOn(blockId), UnorderedElementsAre(7, 0, 3));

            const auto blockRange = blockDistribution->rangeOfBlock(blockId);
            for (uint64_t seed = 0; seed < 10; seed++) {
                ASSERT_THAT(blockDistribution->randomAliveRankBlockRangeIsStoredOn(blockRange, seed), AnyOf(7, 0, 3));
            }
        }

        // ranksBlockIsStored() and isStoredOn() yield consistent results
        for (block_id_t blockId: iter::range<block_id_t>(0, 100)) {
            auto ranksOfThisBlock = blockDistribution->ranksBlockIsStoredOn(blockId);
            for (auto rankId: iter::range<original_rank_t>(0, 10)) {
                if (std::find(ranksOfThisBlock.begin(), ranksOfThisBlock.end(), rankId) != ranksOfThisBlock.end()) {
                    ASSERT_TRUE(blockDistribution->isStoredOn(blockId, rankId));
                } else {
                    ASSERT_FALSE(blockDistribution->isStoredOn(blockId, rankId));
                }
            }
        }

        // rangesStoredOnRank() and isStoredOn() yield consistent results
        for (auto rankId: iter::range<original_rank_t>(0, 10)) {
            auto rangesOnThisRank = blockDistribution->rangesStoredOnRank(rankId);
            for (block_id_t blockId: iter::range<block_id_t>(0, 10)) {
                auto blockRange = blockDistribution->blockRangeById(blockId);
                if (std::find(rangesOnThisRank.begin(), rangesOnThisRank.end(), blockRange) != rangesOnThisRank.end()) {
                    ASSERT_TRUE(blockDistribution->isStoredOn(blockRange, rankId));
                } else {
                    ASSERT_FALSE(blockDistribution->isStoredOn(blockRange, rankId));
                }
            }
        }
    }
}

TEST(BlockDistributionTest, Advanced) {
    using BlockDistribution = ReStore::BlockDistribution<MPIContextMock>;
    using block_id_t        = ReStore::block_id_t;
    using original_rank_t   = ReStoreMPI::original_rank_t;

    // Mock MPI context to pass to the block distribution
    auto mpiContext = MPIContextMock();
    EXPECT_CALL(mpiContext, getOriginalRank(_)).WillRepeatedly(ReturnArg<0>());
    EXPECT_CALL(mpiContext, getCurrentRank(_)).WillRepeatedly(ReturnArg<0>());
    EXPECT_CALL(mpiContext, isAlive(_)).WillRepeatedly(Return(true));
    EXPECT_CALL(mpiContext, getOnlyAlive(_)).WillRepeatedly(ReturnArg<0>());

    {
        // 20 ranks, 81 blocks, (replication level) k = 3
        auto blockDistribution = std::make_shared<BlockDistribution>(20, 81, 3, mpiContext);
        ASSERT_EQ(blockDistribution->shiftWidth(), 6);
        ASSERT_EQ(blockDistribution->numBlocks(), 81);
        ASSERT_EQ(blockDistribution->numRanks(), 20);
        ASSERT_EQ(blockDistribution->replicationLevel(), 3);
        ASSERT_EQ(blockDistribution->numRanges(), 20);
        ASSERT_EQ(blockDistribution->blocksPerRange(), 4);
        ASSERT_EQ(blockDistribution->numRangesWithAdditionalBlock(), 1);

        // Exactly first five blocks should be in range 0
        for (auto blockId: iter::range<block_id_t>(0, 5)) {
            ASSERT_EQ(blockDistribution->rangeOfBlock(blockId).id(), 0);
        }
        for (auto blockId: iter::range<block_id_t>(5, 81)) {
            ASSERT_NE(blockDistribution->rangeOfBlock(blockId).id(), 0);
        }

        // Blocks 9..12 should be in range 2
        for (auto blockId: iter::range<block_id_t>(0, 9)) {
            ASSERT_NE(blockDistribution->rangeOfBlock(blockId).id(), 2);
        }
        for (auto blockId: iter::range<block_id_t>(9, 13)) {
            ASSERT_EQ(blockDistribution->rangeOfBlock(blockId).id(), 2);
        }
        for (auto blockId: iter::range<block_id_t>(13, 81)) {
            ASSERT_NE(blockDistribution->rangeOfBlock(blockId).id(), 2);
        }

        // Rank 19 stores ranges 7, 13, and 19
        ASSERT_THAT(
            blockDistribution->rangesStoredOnRank(19),
            UnorderedElementsAre(
                blockDistribution->blockRangeById(19), blockDistribution->blockRangeById(13),
                blockDistribution->blockRangeById(7)));

        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(0), 19));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(1), 19));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(2), 19));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(3), 19));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(4), 19));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(5), 19));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(6), 19));
        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(7), 19));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(8), 19));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(9), 19));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(11), 19));
        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(13), 19));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(18), 19));
        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(19), 19));

        // ... and so should the blocks 28..31, 52..55, 77..81
        for (block_id_t blockId: iter::chain(
                 iter::range<block_id_t>(29, 33), iter::range<block_id_t>(53, 57), iter::range<block_id_t>(77, 81))) {
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 19));
        }
        for (block_id_t blockId: iter::chain(
                 iter::range<block_id_t>(0, 29), iter::range<block_id_t>(33, 53), iter::range<block_id_t>(57, 77))) {
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 19));
        }

        // Rank 0 stores ranges 0, 14 and 8
        ASSERT_THAT(
            blockDistribution->rangesStoredOnRank(0),
            UnorderedElementsAre(
                blockDistribution->blockRangeById(0), blockDistribution->blockRangeById(14),
                blockDistribution->blockRangeById(8)));

        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(0), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(1), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(2), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(3), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(4), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(5), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(6), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(7), 0));
        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(8), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(9), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(10), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(11), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(12), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(13), 0));
        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(14), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(15), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(16), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(17), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(18), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(19), 0));

        // ... and so should the blocks 0..4, 57..60 and 33..36
        for (block_id_t blockId: iter::chain(
                 iter::range<block_id_t>(0, 5), iter::range<block_id_t>(57, 61), iter::range<block_id_t>(33, 37))) {
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 0));
        }
        for (block_id_t blockId: iter::chain(
                 iter::range<block_id_t>(5, 33), iter::range<block_id_t>(37, 57), iter::range<block_id_t>(61, 81))) {
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 0));
        }

        // The first five blocks should be on rank 0, 6 and 12 but not on any other ranks
        for (auto blockId: iter::range<block_id_t>(0, 5)) {
            for (ReStoreMPI::original_rank_t rankId: iter::range(0, 20)) {
                if (rankId == 0 || rankId == 6 || rankId == 12) {
                    ASSERT_TRUE(blockDistribution->isStoredOn(blockId, rankId));
                } else {
                    ASSERT_FALSE(blockDistribution->isStoredOn(blockId, rankId));
                }
            }

            ASSERT_THAT(blockDistribution->ranksBlockIsStoredOn(blockId), UnorderedElementsAre(0, 6, 12));

            const auto blockRange = blockDistribution->rangeOfBlock(blockId);
            for (uint64_t seed = 0; seed < 10; seed++) {
                ASSERT_THAT(blockDistribution->randomAliveRankBlockRangeIsStoredOn(blockRange, seed), AnyOf(0, 6, 12));
            }
        }

        // The blocks 77..80 should be on rank 19, 5 and 11 but not on any other ranks
        for (auto blockId: iter::range<block_id_t>(77, 81)) {
            for (auto rankId: iter::range(0, 20)) {
                if (rankId == 19 || rankId == 5 || rankId == 11) {
                    ASSERT_TRUE(blockDistribution->isStoredOn(blockId, rankId));
                } else {
                    ASSERT_FALSE(blockDistribution->isStoredOn(blockId, rankId));
                }
            }

            ASSERT_THAT(blockDistribution->ranksBlockIsStoredOn(blockId), UnorderedElementsAre(19, 5, 11));

            const auto blockRange = blockDistribution->rangeOfBlock(blockId);
            for (uint64_t seed = 0; seed < 10; seed++) {
                ASSERT_THAT(blockDistribution->randomAliveRankBlockRangeIsStoredOn(blockRange, seed), AnyOf(19, 5, 11));
            }
        }

        // ranksBlockIsStored() and isStoredOn() yield consistent results
        for (block_id_t blockId: iter::range<block_id_t>(0, 81)) {
            auto ranksOfThisBlock = blockDistribution->ranksBlockIsStoredOn(blockId);
            for (auto rankId: iter::range<original_rank_t>(0, 20)) {
                if (std::find(ranksOfThisBlock.begin(), ranksOfThisBlock.end(), rankId) != ranksOfThisBlock.end()) {
                    ASSERT_TRUE(blockDistribution->isStoredOn(blockId, rankId));
                } else {
                    ASSERT_FALSE(blockDistribution->isStoredOn(blockId, rankId));
                }
            }
        }

        // rangesStoredOnRank() and isStoredOn() yield consistent results
        for (auto rankId: iter::range<original_rank_t>(0, 20)) {
            auto rangesOnThisRank = blockDistribution->rangesStoredOnRank(rankId);
            for (block_id_t blockId: iter::range<block_id_t>(0, 20)) {
                auto blockRange = blockDistribution->blockRangeById(blockId);
                if (std::find(rangesOnThisRank.begin(), rangesOnThisRank.end(), blockRange) != rangesOnThisRank.end()) {
                    ASSERT_TRUE(blockDistribution->isStoredOn(blockRange, rankId));
                } else {
                    ASSERT_FALSE(blockDistribution->isStoredOn(blockRange, rankId));
                }
            }
        }
    }
}

TEST(BlockDistributionTest, Basic_with_failures) {
    using BlockDistribution = ReStore::BlockDistribution<MPIContextMock>;
    using block_id_t        = ReStore::block_id_t;
    using original_rank_t   = ReStoreMPI::original_rank_t;

    {
        // Mock MPI context to pass to the block distribution
        auto mpiContext = MPIContextMock();
        EXPECT_CALL(mpiContext, isAlive(_)).WillRepeatedly(Return(true));
        EXPECT_CALL(mpiContext, isAlive(1)).WillRepeatedly(Return(false));
        EXPECT_CALL(mpiContext, getOnlyAlive(_)).WillRepeatedly([](std::vector<original_rank_t> ranks) {
            return getAliveOnlyFake({1}, ranks);
        });

        // 10 ranks, 100 blocks, (replication level) k = 3
        auto blockDistribution = std::make_shared<BlockDistribution>(10, 100, 3, mpiContext);

        // Rank 7 stores ranges 7, 4, and 1, this is not influenced by the simulated failure of 1
        ASSERT_THAT(
            blockDistribution->rangesStoredOnRank(7),
            UnorderedElementsAre(
                blockDistribution->blockRangeById(7), blockDistribution->blockRangeById(4),
                blockDistribution->blockRangeById(1)));

        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(0), 7));
        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(1), 7));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(2), 7));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(3), 7));
        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(4), 7));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(5), 7));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(6), 7));
        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(7), 7));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(8), 7));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(9), 7));

        // ... and so should the blocks 10..19, 40..49 and 70..79
        for (block_id_t blockId: iter::chain(
                 iter::range<block_id_t>(10, 20), iter::range<block_id_t>(40, 50), iter::range<block_id_t>(70, 80))) {
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 7));
        }
        for (block_id_t blockId: iter::chain(
                 iter::range<block_id_t>(0, 10), iter::range<block_id_t>(20, 40), iter::range<block_id_t>(50, 70),
                 iter::range<block_id_t>(80, 100))) {
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 7));
        }

        // Rank 1 stores ranges 1, 8, and 5, but rank 1 failed, so it should store no more blocks/blockRanges
        // all other ranks should still store 3 elements.
        for (original_rank_t rankId: iter::range<original_rank_t>(0, 10)) {
            if (rankId == 1) {
                ASSERT_THAT(blockDistribution->rangesStoredOnRank(rankId), IsEmpty());
            } else {
                ASSERT_THAT(blockDistribution->rangesStoredOnRank(rankId), BeginEndDistanceIs(3));
            }
        }

        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(0), 1));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(1), 1));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(2), 1));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(3), 1));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(4), 1));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(5), 1));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(6), 1));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(7), 1));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(8), 1));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(9), 1));

        for (block_id_t blockId: iter::chain(iter::range<block_id_t>(0, 100))) {
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 1));
        }

        // The first ten blocks should be on rank 0, 3 and 6 but not on any other ranks.
        // As only rank one failed, this should still be true.
        for (auto blockId: iter::range<block_id_t>(0, 10)) {
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 0));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 1));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 2));
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 3));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 4));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 5));
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 6));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 7));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 8));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 9));

            ASSERT_THAT(blockDistribution->ranksBlockIsStoredOn(blockId), UnorderedElementsAre(0, 3, 6));

            const auto blockRange = blockDistribution->rangeOfBlock(blockId);
            for (uint64_t seed = 0; seed < 10; seed++) {
                ASSERT_THAT(blockDistribution->randomAliveRankBlockRangeIsStoredOn(blockRange, seed), AnyOf(0, 3, 6));
            }
        }

        // The blocks 70..79 should be on rank 7, 0 and 3 but not on any other ranks
        // As only rank 1 failed, this should still be true;
        for (block_id_t blockId: iter::range<block_id_t>(70, 80)) {
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 0));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 1));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 2));
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 3));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 4));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 5));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 6));
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 7));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 8));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 9));

            ASSERT_THAT(blockDistribution->ranksBlockIsStoredOn(blockId), UnorderedElementsAre(7, 0, 3));

            const auto blockRange = blockDistribution->rangeOfBlock(blockId);
            for (uint64_t seed = 0; seed < 10; seed++) {
                ASSERT_THAT(blockDistribution->randomAliveRankBlockRangeIsStoredOn(blockRange, seed), AnyOf(7, 0, 3));
            }
        }

        // The blocks 10..19 should be on ranks 1, 4 and 7. As rank 1 failed, they should only
        // be on ranks 4 and 7.
        for (block_id_t blockId: iter::range<block_id_t>(10, 20)) {
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 0));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 1));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 2));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 3));
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 4));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 5));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 6));
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 7));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 8));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 9));

            ASSERT_THAT(blockDistribution->ranksBlockIsStoredOn(blockId), UnorderedElementsAre(4, 7));

            const auto blockRange = blockDistribution->rangeOfBlock(blockId);
            for (uint64_t seed = 0; seed < 10; seed++) {
                ASSERT_THAT(blockDistribution->randomAliveRankBlockRangeIsStoredOn(blockRange, seed), AnyOf(4, 7));
            }
        }

        // ranksBlockIsStored() and isStoredOn() yield consistent results
        for (block_id_t blockId: iter::range<block_id_t>(0, 100)) {
            auto ranksOfThisBlock = blockDistribution->ranksBlockIsStoredOn(blockId);
            for (ReStoreMPI::original_rank_t rankId: iter::range<original_rank_t>(0, 10)) {
                if (std::find(ranksOfThisBlock.begin(), ranksOfThisBlock.end(), rankId) != ranksOfThisBlock.end()) {
                    ASSERT_TRUE(blockDistribution->isStoredOn(blockId, rankId));
                } else {
                    ASSERT_FALSE(blockDistribution->isStoredOn(blockId, rankId));
                }
            }
        }

        // rangesStoredOnRank() and isStoredOn() yield consistent results
        for (ReStoreMPI::original_rank_t rankId: iter::range<original_rank_t>(0, 10)) {
            auto rangesOnThisRank = blockDistribution->rangesStoredOnRank(rankId);
            for (block_id_t blockId: iter::range<block_id_t>(0, 10)) {
                auto blockRange = blockDistribution->blockRangeById(blockId);
                if (std::find(rangesOnThisRank.begin(), rangesOnThisRank.end(), blockRange) != rangesOnThisRank.end()) {
                    ASSERT_TRUE(blockDistribution->isStoredOn(blockRange, rankId));
                } else {
                    ASSERT_FALSE(blockDistribution->isStoredOn(blockRange, rankId));
                }
            }
        }
    }
}

TEST(BlockDistributionTest, Advanced_with_failure) {
    using BlockDistribution = ReStore::BlockDistribution<MPIContextMock>;
    using block_id_t        = ReStore::block_id_t;
    using original_rank_t   = ReStoreMPI::original_rank_t;

    {
        // Mock MPI context to pass to the block distribution
        auto mpiContext = MPIContextMock();
        EXPECT_CALL(mpiContext, isAlive(_)).WillRepeatedly(Return(true));
        EXPECT_CALL(mpiContext, isAlive(19)).WillRepeatedly(Return(false));
        EXPECT_CALL(mpiContext, getOnlyAlive(_)).WillRepeatedly([](std::vector<original_rank_t> ranks) {
            return getAliveOnlyFake({19}, ranks);
        });

        // 20 ranks, 81 blocks, (replication level) k = 3
        auto blockDistribution = std::make_shared<BlockDistribution>(20, 81, 3, mpiContext);
        ASSERT_EQ(blockDistribution->shiftWidth(), 6);
        ASSERT_EQ(blockDistribution->numBlocks(), 81);
        ASSERT_EQ(blockDistribution->numRanks(), 20);
        ASSERT_EQ(blockDistribution->replicationLevel(), 3);
        ASSERT_EQ(blockDistribution->numRanges(), 20);
        ASSERT_EQ(blockDistribution->blocksPerRange(), 4);
        ASSERT_EQ(blockDistribution->numRangesWithAdditionalBlock(), 1);

        // Rank 19 stores ranges 7, 13, and 19, but rank 19 is dead, it should not store anything.
        ASSERT_THAT(blockDistribution->rangesStoredOnRank(19), IsEmpty());

        for (size_t rangeId: iter::range<size_t>(0, 20)) {
            ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(rangeId), 19));
        }

        for (block_id_t blockId: iter::range<size_t>(0, 81)) {
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 19));
        }

        // Rank 0 stores ranges 0, 14 and 8; rank 0 is still alive, so nothing should change.
        ASSERT_THAT(
            blockDistribution->rangesStoredOnRank(0),
            UnorderedElementsAre(
                blockDistribution->blockRangeById(0), blockDistribution->blockRangeById(14),
                blockDistribution->blockRangeById(8)));

        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(0), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(1), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(2), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(3), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(4), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(5), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(6), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(7), 0));
        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(8), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(9), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(10), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(11), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(12), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(13), 0));
        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(14), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(15), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(16), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(17), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(18), 0));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(19), 0));

        // ... and so should the blocks 0..4, 57..60 and 33..36
        for (block_id_t blockId: iter::chain(
                 iter::range<block_id_t>(0, 5), iter::range<block_id_t>(57, 61), iter::range<block_id_t>(33, 37))) {
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 0));
        }
        for (block_id_t blockId: iter::chain(
                 iter::range<block_id_t>(5, 33), iter::range<block_id_t>(37, 57), iter::range<block_id_t>(61, 81))) {
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 0));
        }

        // The first five blocks should be on rank 0, 6 and 12 but not on any other ranks.
        // None of these ranks is affected by a failure, so this should not change.
        for (auto blockId: iter::range<block_id_t>(0, 5)) {
            for (original_rank_t rankId: iter::range<original_rank_t>(0, 20)) {
                if (rankId == 0 || rankId == 6 || rankId == 12) {
                    ASSERT_TRUE(blockDistribution->isStoredOn(blockId, rankId));
                } else {
                    ASSERT_FALSE(blockDistribution->isStoredOn(blockId, rankId));
                }
            }

            ASSERT_THAT(blockDistribution->ranksBlockIsStoredOn(blockId), UnorderedElementsAre(0, 6, 12));

            const auto blockRange = blockDistribution->rangeOfBlock(blockId);
            for (uint64_t seed = 0; seed < 10; seed++) {
                ASSERT_THAT(blockDistribution->randomAliveRankBlockRangeIsStoredOn(blockRange, seed), AnyOf(0, 6, 12));
            }
        }

        // The blocks 77..80 should be on rank 19, 5 and 11 but not on any other ranks
        // Rank 19 failed, it should not be listed as storing anything.
        for (auto blockId: iter::range<block_id_t>(77, 81)) {
            for (auto rankId: iter::range<original_rank_t>(0, 20)) {
                if (rankId == 5 || rankId == 11) {
                    ASSERT_TRUE(blockDistribution->isStoredOn(blockId, rankId));
                } else {
                    ASSERT_FALSE(blockDistribution->isStoredOn(blockId, rankId));
                }
            }

            ASSERT_THAT(blockDistribution->ranksBlockIsStoredOn(blockId), UnorderedElementsAre(5, 11));

            const auto blockRange = blockDistribution->rangeOfBlock(blockId);
            for (uint64_t seed = 0; seed < 10; seed++) {
                ASSERT_THAT(blockDistribution->randomAliveRankBlockRangeIsStoredOn(blockRange, seed), AnyOf(5, 11));
            }
        }

        // ranksBlockIsStored() and isStoredOn() yield consistent results
        for (block_id_t blockId: iter::range<block_id_t>(0, 81)) {
            auto ranksOfThisBlock = blockDistribution->ranksBlockIsStoredOn(blockId);
            for (auto rankId: iter::range<original_rank_t>(0, 20)) {
                if (std::find(ranksOfThisBlock.begin(), ranksOfThisBlock.end(), rankId) != ranksOfThisBlock.end()) {
                    ASSERT_TRUE(blockDistribution->isStoredOn(blockId, rankId));
                } else {
                    ASSERT_FALSE(blockDistribution->isStoredOn(blockId, rankId));
                }
            }
        }

        // rangesStoredOnRank() and isStoredOn() yield consistent results
        for (auto rankId: iter::range<original_rank_t>(0, 20)) {
            auto rangesOnThisRank = blockDistribution->rangesStoredOnRank(rankId);
            for (block_id_t blockId: iter::range<block_id_t>(0, 20)) {
                auto blockRange = blockDistribution->blockRangeById(blockId);
                if (std::find(rangesOnThisRank.begin(), rangesOnThisRank.end(), blockRange) != rangesOnThisRank.end()) {
                    ASSERT_TRUE(blockDistribution->isStoredOn(blockRange, rankId));
                } else {
                    ASSERT_FALSE(blockDistribution->isStoredOn(blockRange, rankId));
                }
            }
        }
    }
}

TEST(BlockDistributionTest, Multiple_failures) {
    using BlockDistribution = ReStore::BlockDistribution<MPIContextMock>;
    using block_id_t        = ReStore::block_id_t;
    using original_rank_t   = ReStoreMPI::original_rank_t;

    {
        // Mock MPI context to pass to the block distribution
        auto mpiContext = MPIContextMock();
        EXPECT_CALL(mpiContext, isAlive(_)).WillRepeatedly(Return(true));
        EXPECT_CALL(mpiContext, isAlive(0)).WillRepeatedly(Return(false));
        EXPECT_CALL(mpiContext, isAlive(3)).WillRepeatedly(Return(false));
        EXPECT_CALL(mpiContext, isAlive(5)).WillRepeatedly(Return(false));
        EXPECT_CALL(mpiContext, isAlive(6)).WillRepeatedly(Return(false));
        EXPECT_CALL(mpiContext, isAlive(8)).WillRepeatedly(Return(false));
        EXPECT_CALL(mpiContext, getOnlyAlive(_)).WillRepeatedly([](std::vector<original_rank_t> ranks) {
            return getAliveOnlyFake({0, 3, 5, 6, 8}, ranks);
        });

        // 10 ranks, 100 blocks, (replication level) k = 3
        auto blockDistribution = std::make_shared<BlockDistribution>(10, 100, 3, mpiContext);

        // Rank 7 stores ranges 7, 4, and 1, this is not influenced by the simulated failure of 0, 3, 5, 6 or 8
        ASSERT_THAT(
            blockDistribution->rangesStoredOnRank(7),
            UnorderedElementsAre(
                blockDistribution->blockRangeById(7), blockDistribution->blockRangeById(4),
                blockDistribution->blockRangeById(1)));

        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(0), 7));
        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(1), 7));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(2), 7));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(3), 7));
        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(4), 7));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(5), 7));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(6), 7));
        ASSERT_TRUE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(7), 7));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(8), 7));
        ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(9), 7));

        // ... and so should the blocks 10..19, 40..49 and 70..79
        for (block_id_t blockId: iter::chain(
                 iter::range<block_id_t>(10, 20), iter::range<block_id_t>(40, 50), iter::range<block_id_t>(70, 80))) {
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 7));
        }
        for (block_id_t blockId: iter::chain(
                 iter::range<block_id_t>(0, 10), iter::range<block_id_t>(20, 40), iter::range<block_id_t>(50, 70),
                 iter::range<block_id_t>(80, 100))) {
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 7));
        }

        // Ranks 0, 3, 5, 6 and 8 failed, they should not store any elements. All other ranks should still store 3
        // elements.
        for (original_rank_t rankId: iter::range<original_rank_t>(0, 10)) {
            if (rankId == 0 || rankId == 3 || rankId == 5 || rankId == 6 || rankId == 8) {
                ASSERT_THAT(blockDistribution->rangesStoredOnRank(rankId), IsEmpty());
            } else {
                ASSERT_THAT(blockDistribution->rangesStoredOnRank(rankId), BeginEndDistanceIs(3));
            }
        }

        for (original_rank_t rankId: {0, 3, 5, 6, 8}) {
            for (block_id_t blockId: iter::range<block_id_t>(0, 81)) {
                ASSERT_FALSE(blockDistribution->isStoredOn(blockId, rankId));
            }
        }

        // The first ten blocks should be on rank 0, 3 and 6 but not on any other ranks.
        // All of these ranks failed, so the first ten blocks should be stored nowere.
        for (original_rank_t rankId: iter::range<original_rank_t>(0, 10)) {
            for (block_id_t blockId: iter::range<block_id_t>(0, 10)) {
                ASSERT_FALSE(blockDistribution->isStoredOn(blockId, rankId));
            }
        }
        // as should range 0
        for (original_rank_t rankId: iter::range<original_rank_t>(0, 10)) {
            ASSERT_FALSE(blockDistribution->isStoredOn(blockDistribution->blockRangeById(0), rankId));
        }

        // The blocks 70..79 should be on rank 7, 0 and 3 but not on any other ranks
        // Rank 0 and 3 failed, so they should still be on rank 7.
        for (block_id_t blockId: iter::range<block_id_t>(70, 80)) {
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 8));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 1));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 2));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 3));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 4));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 5));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 6));
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 7));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 8));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 9));

            ASSERT_THAT(blockDistribution->ranksBlockIsStoredOn(blockId), UnorderedElementsAre(7));

            const auto blockRange = blockDistribution->rangeOfBlock(blockId);
            for (uint64_t seed = 0; seed < 10; seed++) {
                ASSERT_THAT(blockDistribution->randomAliveRankBlockRangeIsStoredOn(blockRange, seed), AnyOf(7));
            }
        }

        // The blocks 10..19 should be on ranks 1, 4 and 7. None of these ranks failed, this should still be true
        for (block_id_t blockId: iter::range<block_id_t>(10, 20)) {
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 0));
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 1));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 2));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 3));
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 4));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 5));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 6));
            ASSERT_TRUE(blockDistribution->isStoredOn(blockId, 7));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 8));
            ASSERT_FALSE(blockDistribution->isStoredOn(blockId, 9));

            ASSERT_THAT(blockDistribution->ranksBlockIsStoredOn(blockId), UnorderedElementsAre(4, 7, 1));

            const auto            blockRange = blockDistribution->rangeOfBlock(blockId);
            std::vector<uint64_t> ranks(10, 0);
            assert(ranks.size() == 10);
            assert(ranks[4] == 0);
            assert(ranks[1] == 0);
            assert(ranks[7] == 0);
            for (uint64_t seed = 0; seed < 1000; seed++) {
                const auto servingRank = blockDistribution->randomAliveRankBlockRangeIsStoredOn(blockRange, seed);
                ASSERT_THAT(servingRank, AnyOf(4, 7, 1));
                ranks[asserting_cast<size_t>(servingRank)]++;
            }
            EXPECT_GT(ranks[4], 200);
            EXPECT_GT(ranks[7], 200);
            EXPECT_GT(ranks[1], 200);
        }

        // ranksBlockIsStored() and isStoredOn() yield consistent results
        for (block_id_t blockId: iter::range<block_id_t>(0, 100)) {
            auto ranksOfThisBlock = blockDistribution->ranksBlockIsStoredOn(blockId);
            for (ReStoreMPI::original_rank_t rankId: iter::range<original_rank_t>(0, 10)) {
                if (std::find(ranksOfThisBlock.begin(), ranksOfThisBlock.end(), rankId) != ranksOfThisBlock.end()) {
                    ASSERT_TRUE(blockDistribution->isStoredOn(blockId, rankId));
                } else {
                    ASSERT_FALSE(blockDistribution->isStoredOn(blockId, rankId));
                }
            }
        }

        // rangesStoredOnRank() and isStoredOn() yield consistent results
        for (ReStoreMPI::original_rank_t rankId: iter::range<original_rank_t>(0, 10)) {
            auto rangesOnThisRank = blockDistribution->rangesStoredOnRank(rankId);
            for (block_id_t blockId: iter::range<block_id_t>(0, 10)) {
                auto blockRange = blockDistribution->blockRangeById(blockId);
                if (std::find(rangesOnThisRank.begin(), rangesOnThisRank.end(), blockRange) != rangesOnThisRank.end()) {
                    ASSERT_TRUE(blockDistribution->isStoredOn(blockRange, rankId));
                } else {
                    ASSERT_FALSE(blockDistribution->isStoredOn(blockRange, rankId));
                }
            }
        }
    }
}
