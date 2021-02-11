#include <bits/stdint-uintn.h>
#include <cstddef>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <stdexcept>

#include "restore/block_serialization.hpp"
#include "restore/common.hpp"
#include "restore/helpers.hpp"

using namespace ::testing;

class MPIContextMock {
    using original_rank_t = ReStoreMPI::original_rank_t;
    using current_rank_t  = ReStoreMPI::current_rank_t;

    public:
    MOCK_METHOD(original_rank_t, getOriginalRank, (const current_rank_t), (const));
    MOCK_METHOD(current_rank_t, getCurrentRank, (const original_rank_t), (const));
    MOCK_METHOD(bool, isAlive, (const original_rank_t), (const));
    MOCK_METHOD(std::vector<original_rank_t>, getOnlyAlive, (const std::vector<original_rank_t>&), (const));


    /*
    std::vector<current_rank_t> getAliveCurrentRanks(const std::vector<original_rank_t>& originalRanks) const {
        return _rankManager.getAliveCurrentRanks(originalRanks);
    }

    std::vector<Message>
    SparseAllToAll(const std::vector<Message>& messages, const int tag = RESTORE_SPARSE_ALL_TO_ALL_TAG) const {
        return ReStoreMPI::SparseAllToAll(messages, _comm, tag);
    }
    */
};

TEST(SerializedBlockStorageTest, forAllBlocks) {
    using BlockDistribution                           = ReStore::BlockDistribution<MPIContextMock>;
    auto                            mpiContext        = MPIContextMock();
    auto                            blockDistribution = BlockDistribution(10, 100, 3, mpiContext);
    ReStore::SerializedBlockStorage storage(ReStore::OffsetMode::constant, blockDistribution, 1);
    auto                            blockrange0   = blockDistribution.blockRangeById(0);
    auto                            blockrange1   = blockDistribution.blockRangeById(1);
    auto                            blockrange2   = blockDistribution.blockRangeById(2);
    auto                            blockrange4   = blockDistribution.blockRangeById(4);
    auto                            blockRangeVec = {blockrange0, blockrange1, blockrange2, blockrange4};
    storage.registerRanges(blockRangeVec);
    for (uint8_t block = 0; block < 30; ++block) {
        storage.writeBlock(block, &block);
    }

    for (uint8_t block = 40; block < 50; ++block) {
        storage.writeBlock(block, &block);
    }

    std::pair<ReStore::block_id_t, size_t> range1(5, 20);
    std::pair<ReStore::block_id_t, size_t> range2(40, 10);
    std::pair<ReStore::block_id_t, size_t> rangeInvalid(30, 10);

    uint8_t expected = 5;
    storage.forAllBlocks(range1, [&expected](const uint8_t* ptr, size_t size) {
        EXPECT_EQ(1, size);
        EXPECT_EQ(expected, *ptr);
        ++expected;
    });
    EXPECT_EQ(25, expected);

    expected = 40;
    storage.forAllBlocks(range2, [&expected](const uint8_t* ptr, size_t size) {
        EXPECT_EQ(1, size);
        EXPECT_EQ(expected, *ptr);
        ++expected;
    });
    EXPECT_EQ(50, expected);

    ASSERT_THROW(storage.forAllBlocks(
        rangeInvalid,
        [&expected](const uint8_t* ptr, size_t size) {
            ++expected;
            UNUSED(ptr);
            UNUSED(size);
        });
                 , std::invalid_argument);
}

int main(int argc, char** argv) {
    // Filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int result = RUN_ALL_TESTS();

    return result;
}
