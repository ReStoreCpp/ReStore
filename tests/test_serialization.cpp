#include <bits/stdint-uintn.h>
#include <cstddef>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <mpi.h>
#include <stdexcept>

#include "restore/block_serialization.hpp"
#include "restore/common.hpp"
#include "restore/helpers.hpp"

#include "mocks.hpp"

using namespace ::testing;

TEST(SerializedBlockStorageTest, forAllBlocks) {
    using BlockDistribution = ReStore::BlockDistribution<MPIContextMock>;
    auto mpiContext         = MPIContextMock();
    auto blockDistribution  = std::make_shared<BlockDistribution>(BlockDistribution(10, 100, 3, mpiContext));
    ReStore::SerializedBlockStorage<MPIContextMock> storage(blockDistribution, ReStore::OffsetMode::constant, 1);
    auto                                            blockrange0 = blockDistribution->blockRangeById(0);
    auto                                            blockrange1 = blockDistribution->blockRangeById(1);
    auto                                            blockrange2 = blockDistribution->blockRangeById(2);
    auto                                            blockrange4 = blockDistribution->blockRangeById(4);
    auto blockRangeVec                                          = {blockrange0, blockrange1, blockrange2, blockrange4};
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

TEST(SerializedBlockStoreStream, Constructor) {
    using BuffersType = std::map<ReStoreMPI::current_rank_t, std::vector<uint8_t>>;
    using RanksArrayType = std::vector<ReStoreMPI::current_rank_t>;

    auto buffers = std::make_shared<BuffersType>();
    auto ranks = std::make_shared<RanksArrayType>();

    // nullptr arguments
    ASSERT_ANY_THROW(ReStore::SerializedBlockStoreStream(nullptr, ranks));
    ASSERT_ANY_THROW(ReStore::SerializedBlockStoreStream(buffers, nullptr));
    ASSERT_ANY_THROW(ReStore::SerializedBlockStoreStream(nullptr, nullptr));

    // no ranks
    ASSERT_ANY_THROW(ReStore::SerializedBlockStoreStream(buffers, ranks));
    
    // all fine
    ranks->push_back(0);    
    ASSERT_NO_THROW(ReStore::SerializedBlockStoreStream(buffers, ranks));
    ranks->push_back(1);    
    ranks->push_back(2);    
    ASSERT_NO_THROW(ReStore::SerializedBlockStoreStream(buffers, ranks));
    
    // for completeness
    ASSERT_ANY_THROW(ReStore::SerializedBlockStoreStream(nullptr, ranks));
}

TEST(SerializedBlockStoreStream, InStream) {
    using BuffersType = std::map<ReStoreMPI::current_rank_t, std::vector<uint8_t>>;
    using RanksArrayType = std::vector<ReStoreMPI::current_rank_t>;

    auto buffers = std::make_shared<BuffersType>();
    auto ranks = std::make_shared<RanksArrayType>();
    ranks->push_back(0);    
    ranks->push_back(3);    

    ReStore::SerializedBlockStoreStream stream(buffers, ranks);

    stream << 0x42_byte;
    ASSERT_EQ(buffers->at(0)[0], 0x42_byte);
    ASSERT_EQ(buffers->find(1), buffers->end());
    ASSERT_EQ(buffers->find(2), buffers->end());
    ASSERT_EQ(buffers->at(3)[0], 0x42_byte);
    ASSERT_EQ(stream.bytesWritten(), 1);

    stream << 0x00_uint8;
    ASSERT_EQ(buffers->at(0)[0], 0x42_byte);
    ASSERT_EQ(buffers->find(1), buffers->end());
    ASSERT_EQ(buffers->find(2), buffers->end());
    ASSERT_EQ(buffers->at(3)[0], 0x42_byte);

    ASSERT_EQ(buffers->at(0)[1], 0x00_byte);
    ASSERT_EQ(buffers->at(3)[1], 0x00_byte);
    ASSERT_EQ(stream.bytesWritten(), 2);
}

int main(int argc, char** argv) {
    // Filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int result = RUN_ALL_TESTS();

    return result;
}
