#include <algorithm>

#include "chain.hpp"
#include "range.hpp"
#include <algorithm>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi/mpi.h>

#include "restore/core.hpp"
#include "restore/helpers.hpp"

using namespace ::testing;

class StoreTest : public ::testing::Environment {
    // You can remove any or all of the following functions if its body
    // is empty.

    StoreTest(){};

    virtual ~StoreTest() {
        // You can do clean-up work that doesn't throw exceptions here.
    }

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    virtual void SetUp() {
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    virtual void TearDown() {
        // Code here will be called immediately after each test (right
        // before the destructor).
    }

    // Objects declared here can be used by all tests in the test case for Foo.
};

TEST(StoreTest, ReStore_Constructor) {
    // Construction of a ReStore object
    ASSERT_NO_THROW(ReStore<int>(MPI_COMM_WORLD, 3, ReStore<int>::OffsetMode::lookUpTable));
    ASSERT_NO_THROW(ReStore<int>(MPI_COMM_WORLD, 3, ReStore<int>::OffsetMode::constant, sizeof(int)));

    ASSERT_ANY_THROW(ReStore<int>(MPI_COMM_WORLD, 3, ReStore<int>::OffsetMode::lookUpTable, sizeof(int)));
    ASSERT_ANY_THROW(ReStore<int>(MPI_COMM_WORLD, 3, ReStore<int>::OffsetMode::constant, 0));
    ASSERT_ANY_THROW(ReStore<int>(MPI_COMM_WORLD, 0, ReStore<int>::OffsetMode::lookUpTable));
    ASSERT_ANY_THROW(ReStore<int>(MPI_COMM_WORLD, 0, ReStore<int>::OffsetMode::constant, sizeof(int)));

    // TODO Test a replication level that is larger than the number of ranks
    // TODO Test a replication level that cannot be archived because of memory
    // constraints

    // Replication level and offset mode getters
    {
        auto store = ReStore<uint8_t>(MPI_COMM_WORLD, 3, ReStore<uint8_t>::OffsetMode::constant, sizeof(uint8_t));
        ASSERT_EQ(store.replicationLevel(), 3);
        auto [offsetMode, constOffset] = store.offsetMode();
        ASSERT_EQ(offsetMode, ReStore<uint8_t>::OffsetMode::constant);
        ASSERT_EQ(constOffset, sizeof(uint8_t));
    }

    {
        auto store = ReStore<uint8_t>(MPI_COMM_WORLD, 10, ReStore<uint8_t>::OffsetMode::lookUpTable);
        ASSERT_EQ(store.replicationLevel(), 10);
        auto [offsetMode, constOffset] = store.offsetMode();
        ASSERT_EQ(offsetMode, ReStore<uint8_t>::OffsetMode::lookUpTable);
        ASSERT_EQ(constOffset, 0);
    }
}

/*
TEST(StoreTest, ReStore_submitBlocks) {
    auto store         = ReStore<uint8_t>(MPI_COMM_WORLD, 3, ReStore<uint8_t>::OffsetMode::constant, sizeof(uint8_t));
    auto serializeFunc = [](const ReStore<uint8_t>::block_id_t& block, void* buffer) {
        UNUSED(block);
        UNUSED(buffer);
        return size_t(1);
    };
    auto nextBlock = []() {
        return std::optional<std::pair<ReStore<uint8_t>::block_id_t, const uint8_t&>>();
    };
    store.submitBlocks(serializeFunc, nextBlock);
}
*/

TEST(StoreTest, ReStore_BlockRange) {
    // BlockRange(size_t range_id, size_t numBlocks, size_t numRanges) {
    // bool includes(block_id_t block) {
    typedef ReStore<u_int8_t>::BlockDistribution::BlockRange BlockRange;

    ASSERT_ANY_THROW(BlockRange(100, 1, 10)); // Range id greater than the number of ranges
    ASSERT_ANY_THROW(BlockRange(0, 1, 2));    // More ranges than blocks

    {
        auto range = BlockRange(0, 100, 10);
        ASSERT_EQ(range.id, 0);
        ASSERT_EQ(range.start, 0);
        ASSERT_EQ(range.length, 10);

        for (ReStore<uint8_t>::block_id_t blockId = 0; blockId < 10; blockId++) {
            ASSERT_TRUE(range.contains(blockId));
        }
        for (ReStore<uint8_t>::block_id_t blockId = 10; blockId < 100; blockId++) {
            ASSERT_FALSE(range.contains(blockId));
        }

        ASSERT_FALSE(range.contains(101));
    }

    {
        auto range = BlockRange(2, 100, 10);
        ASSERT_EQ(range.id, 2);
        ASSERT_EQ(range.start, 20);
        ASSERT_EQ(range.length, 10);

        for (ReStore<uint8_t>::block_id_t blockId = 0; blockId < 20; blockId++) {
            ASSERT_FALSE(range.contains(blockId));
        }
        for (ReStore<uint8_t>::block_id_t blockId = 20; blockId < 30; blockId++) {
            ASSERT_TRUE(range.contains(blockId));
        }
        for (ReStore<uint8_t>::block_id_t blockId = 30; blockId < 100; blockId++) {
            ASSERT_FALSE(range.contains(blockId));
        }

        ASSERT_FALSE(range.contains(101));
    }

    {
        // range 0: 0, 1, 2, 3
        // range 1: 4, 5, 6
        // range 2: 7, 8, 9
        auto range0 = BlockRange(0, 10, 3);
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

        auto range1 = BlockRange(1, 10, 3);
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

        auto range2 = BlockRange(2, 10, 3);
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
        // range0: 0, 1
        // range1: 2, 3
        // range2: 4
        // range3: 5
        auto range0 = BlockRange(0, 6, 4);
        ASSERT_TRUE(range0.contains(0));
        ASSERT_TRUE(range0.contains(1));
        ASSERT_FALSE(range0.contains(2));
        ASSERT_FALSE(range0.contains(3));
        ASSERT_FALSE(range0.contains(4));
        ASSERT_FALSE(range0.contains(5));

        auto range1 = BlockRange(1, 6, 4);
        ASSERT_FALSE(range1.contains(0));
        ASSERT_FALSE(range1.contains(1));
        ASSERT_TRUE(range1.contains(2));
        ASSERT_TRUE(range1.contains(3));
        ASSERT_FALSE(range1.contains(4));
        ASSERT_FALSE(range1.contains(5));

        auto range2 = BlockRange(2, 6, 4);
        ASSERT_FALSE(range2.contains(0));
        ASSERT_FALSE(range2.contains(1));
        ASSERT_FALSE(range2.contains(2));
        ASSERT_FALSE(range2.contains(3));
        ASSERT_TRUE(range2.contains(4));
        ASSERT_FALSE(range2.contains(5));

        auto range3 = BlockRange(3, 6, 4);
        ASSERT_FALSE(range3.contains(0));
        ASSERT_FALSE(range3.contains(1));
        ASSERT_FALSE(range3.contains(2));
        ASSERT_FALSE(range3.contains(3));
        ASSERT_FALSE(range3.contains(4));
        ASSERT_TRUE(range3.contains(5));
    }
}

TEST(StoreTest, ReStore_BlockDistribution) {
    using BlockDistribution = ReStore<uint16_t>::BlockDistribution;
    using block_id_t        = ReStore<uint16_t>::block_id_t;

    // Dummy MPI context to pass to the block distribution
    auto mpiContext = ReStoreMPI::MPIContext(MPI_COMM_WORLD);

    // Constructor - invalid arguments
    ASSERT_ANY_THROW(BlockDistribution(0, 1, 1, mpiContext));     // No ranks
    ASSERT_ANY_THROW(BlockDistribution(1, 1, 2, mpiContext));     // replication level > #ranks
    ASSERT_ANY_THROW(BlockDistribution(1, 0, 1, mpiContext));     // No blocks
    ASSERT_ANY_THROW(BlockDistribution(2, 1, 1, mpiContext));     // Less blocks than ranks
    ASSERT_ANY_THROW(BlockDistribution(1, 1, 0, mpiContext));     // Replication level of zero
    ASSERT_ANY_THROW(BlockDistribution(10, 100, 11, mpiContext)); // replication level > #ranks

    {
        // 10 ranks, 100 blocks, (replication level) k = 3
        auto blockDistribution = BlockDistribution(10, 100, 3, mpiContext);
        ASSERT_EQ(blockDistribution.shiftWidth(), 3);
        ASSERT_EQ(blockDistribution.numBlocks(), 100);
        ASSERT_EQ(blockDistribution.numRanks(), 10);
        ASSERT_EQ(blockDistribution.replicationLevel(), 3);
        ASSERT_EQ(blockDistribution.numRanges(), 10);
        ASSERT_EQ(blockDistribution.blocksPerRange(), 10);
        ASSERT_EQ(blockDistribution.numRangesWithAdditionalBlock(), 0);

        // Exactly first ten blocks should be in range 0
        for (auto blockId: iter::range<block_id_t>(0, 10)) {
            ASSERT_EQ(blockDistribution.rangeOfBlock(blockId).id, 0);
        }
        for (auto blockId: iter::range<block_id_t>(10, 100)) {
            ASSERT_NE(blockDistribution.rangeOfBlock(blockId).id, 0);
        }

        // Blocks 30..39 should be in range 0
        for (auto blockId: iter::range<block_id_t>(0, 30)) {
            ASSERT_NE(blockDistribution.rangeOfBlock(blockId).id, 3);
        }
        for (auto blockId: iter::range<block_id_t>(30, 40)) {
            ASSERT_EQ(blockDistribution.rangeOfBlock(blockId).id, 3);
        }
        for (auto blockId: iter::range<block_id_t>(40, 100)) {
            ASSERT_NE(blockDistribution.rangeOfBlock(blockId).id, 3);
        }

        // Rank 7 stores ranges 7, 4, and 1
        ASSERT_THAT(
            blockDistribution.rangesStoredOnRank(7),
            ElementsAre(
                blockDistribution.blockRangeById(7), blockDistribution.blockRangeById(4),
                blockDistribution.blockRangeById(1)));

        ASSERT_FALSE(blockDistribution.isStoredOn(blockDistribution.blockRangeById(0), 7));
        ASSERT_TRUE(blockDistribution.isStoredOn(blockDistribution.blockRangeById(1), 1));
        ASSERT_FALSE(blockDistribution.isStoredOn(blockDistribution.blockRangeById(2), 7));
        ASSERT_FALSE(blockDistribution.isStoredOn(blockDistribution.blockRangeById(3), 7));
        ASSERT_TRUE(blockDistribution.isStoredOn(blockDistribution.blockRangeById(4), 7));
        ASSERT_FALSE(blockDistribution.isStoredOn(blockDistribution.blockRangeById(5), 7));
        ASSERT_FALSE(blockDistribution.isStoredOn(blockDistribution.blockRangeById(6), 7));
        ASSERT_TRUE(blockDistribution.isStoredOn(blockDistribution.blockRangeById(7), 7));
        ASSERT_FALSE(blockDistribution.isStoredOn(blockDistribution.blockRangeById(8), 7));
        ASSERT_FALSE(blockDistribution.isStoredOn(blockDistribution.blockRangeById(9), 7));

        // ... and so should the blocks 10..19, 40..49 and 70..79
        for (block_id_t blockId: iter::chain(iter::range(10, 20), iter::range(40, 50), iter::range(70, 80))) {
            ASSERT_TRUE(blockDistribution.isStoredOn(blockId, 7));
        }
        for (block_id_t blockId:
             iter::chain(iter::range(0, 10), iter::range(20, 40), iter::range(50, 70), iter::range(80, 100))) {
            ASSERT_FALSE(blockDistribution.isStoredOn(blockId, 7));
        }

        // Rank 1 stores ranges 1, 8, and 5
        ASSERT_THAT(
            blockDistribution.rangesStoredOnRank(1),
            ElementsAre(
                blockDistribution.blockRangeById(1), blockDistribution.blockRangeById(8),
                blockDistribution.blockRangeById(5)));

        ASSERT_FALSE(blockDistribution.isStoredOn(blockDistribution.blockRangeById(0), 1));
        ASSERT_TRUE(blockDistribution.isStoredOn(blockDistribution.blockRangeById(1), 1));
        ASSERT_FALSE(blockDistribution.isStoredOn(blockDistribution.blockRangeById(2), 1));
        ASSERT_FALSE(blockDistribution.isStoredOn(blockDistribution.blockRangeById(3), 1));
        ASSERT_FALSE(blockDistribution.isStoredOn(blockDistribution.blockRangeById(4), 1));
        ASSERT_TRUE(blockDistribution.isStoredOn(blockDistribution.blockRangeById(5), 1));
        ASSERT_FALSE(blockDistribution.isStoredOn(blockDistribution.blockRangeById(6), 1));
        ASSERT_FALSE(blockDistribution.isStoredOn(blockDistribution.blockRangeById(7), 1));
        ASSERT_TRUE(blockDistribution.isStoredOn(blockDistribution.blockRangeById(8), 1));
        ASSERT_FALSE(blockDistribution.isStoredOn(blockDistribution.blockRangeById(9), 1));

        // ... and so should the blocks 10..19, 50..59 and 80..89
        for (block_id_t blockId: iter::chain(iter::range(10, 20), iter::range(50, 60), iter::range(80, 90))) {
            ASSERT_TRUE(blockDistribution.isStoredOn(blockId, 1));
        }
        for (block_id_t blockId:
             iter::chain(iter::range(0, 10), iter::range(20, 50), iter::range(60, 80), iter::range(90, 100))) {
            ASSERT_FALSE(blockDistribution.isStoredOn(blockId, 1));
        }

        // The first ten blocks should be on rank 0, 3 and 6 but not on any other ranks
        // TODO Test in the presence of failures
        for (auto blockId: iter::range<block_id_t>(0, 10)) {
            ASSERT_TRUE(blockDistribution.isStoredOn(blockId, 0));
            ASSERT_FALSE(blockDistribution.isStoredOn(blockId, 1));
            ASSERT_FALSE(blockDistribution.isStoredOn(blockId, 2));
            ASSERT_TRUE(blockDistribution.isStoredOn(blockId, 3));
            ASSERT_FALSE(blockDistribution.isStoredOn(blockId, 4));
            ASSERT_FALSE(blockDistribution.isStoredOn(blockId, 5));
            ASSERT_TRUE(blockDistribution.isStoredOn(blockId, 6));
            ASSERT_FALSE(blockDistribution.isStoredOn(blockId, 7));
            ASSERT_FALSE(blockDistribution.isStoredOn(blockId, 8));
            ASSERT_FALSE(blockDistribution.isStoredOn(blockId, 9));

            ASSERT_THAT(blockDistribution.ranksBlockIsStoredOn(blockId), ElementsAre(0, 3, 6));
        }

        // The blocks 70..79 should be on rank 7, 0 and 3 but not on any other ranks
        for (block_id_t blockId: iter::range<block_id_t>(70, 80)) {
            ASSERT_TRUE(blockDistribution.isStoredOn(blockId, 0));
            ASSERT_FALSE(blockDistribution.isStoredOn(blockId, 1));
            ASSERT_FALSE(blockDistribution.isStoredOn(blockId, 2));
            ASSERT_TRUE(blockDistribution.isStoredOn(blockId, 3));
            ASSERT_FALSE(blockDistribution.isStoredOn(blockId, 4));
            ASSERT_FALSE(blockDistribution.isStoredOn(blockId, 5));
            ASSERT_FALSE(blockDistribution.isStoredOn(blockId, 6));
            ASSERT_TRUE(blockDistribution.isStoredOn(blockId, 7));
            ASSERT_FALSE(blockDistribution.isStoredOn(blockId, 8));
            ASSERT_FALSE(blockDistribution.isStoredOn(blockId, 9));

            ASSERT_THAT(blockDistribution.ranksBlockIsStoredOn(blockId), ElementsAre(7, 0, 3));
        }

        // ranksBlockIsStored() and isStoredOn() yield consistent results
        for (block_id_t blockId: iter::range(0, 100)) {
            auto ranksOfThisBlock = blockDistribution.ranksBlockIsStoredOn(blockId);
            for (ReStoreMPI::original_rank_t rankId: iter::range(0, 10)) {
                if (std::find(ranksOfThisBlock.begin(), ranksOfThisBlock.end(), rankId) != ranksOfThisBlock.end()) {
                    ASSERT_TRUE(blockDistribution.isStoredOn(blockId, rankId));
                } else {
                    ASSERT_FALSE(blockDistribution.isStoredOn(blockId, rankId));
                }
            }
        }

        // rangesStoredOnRank() and isStoredOn() yield consistent results
        for (ReStoreMPI::original_rank_t rankId: iter::range(0, 10)) {
            auto rangesOnThisRank = blockDistribution.rangesStoredOnRank(rankId);
            for (BlockDistribution::BlockRange blockRange: rangesOnThisRank) {
                if (std::find(rangesOnThisRank.begin(), rangesOnThisRank.end(), blockRange) != rangesOnThisRank.end()) {
                    ASSERT_TRUE(blockDistribution.isStoredOn(blockRange.id, rankId));
                } else {
                    ASSERT_FALSE(blockDistribution.isStoredOn(blockRange.id, rankId));
                }
            }
        }
    }
    /*
        BlockRange rangeOfBlock(block_id_t block) const {
        std::vector<ReStoreMPI::original_rank_t> ranksBlockIsStoredOn(block_id_t block) const {
        std::vector<BlockRange> rangesStoredOnRank(ReStoreMPI::original_rank_t rankId) const {
        bool isStoredOn(BlockRange blockRange, ReStoreMPI::original_rank_t rankId) const {
        bool isStoredOn(block_id_t block, ReStoreMPI::original_rank_t rankId) const {}
    */
}

int main(int argc, char** argv) {
    // Filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int result = RUN_ALL_TESTS();

    return result;
}
