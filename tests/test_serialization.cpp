#include <bits/stdint-uintn.h>
#include <cstddef>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "restore/block_serialization.hpp"
#include "restore/common.hpp"
#include "restore/helpers.hpp"

#include "mocks.hpp"

using namespace ::testing;

TEST(SerializedBlockStorageTest, forAllBlocks) {
    using BlockDistribution = ReStore::BlockDistribution<MPIContextMock>;
    auto mpiContext         = MPIContextMock();
    auto blockDistribution  = std::make_shared<BlockDistribution>(BlockDistribution(10, 100, 3, mpiContext));
    // TODO: Test LUT mode when that is implemented in the rest of the class
    ReStore::SerializedBlockStorage<MPIContextMock> storage(blockDistribution, ReStore::OffsetMode::constant, 1);
    // auto                                            blockrange0 = blockDistribution->blockRangeById(0);
    // auto                                            blockrange1 = blockDistribution->blockRangeById(1);
    // auto                                            blockrange2 = blockDistribution->blockRangeById(2);
    // auto                                            blockrange4 = blockDistribution->blockRangeById(4);
    // auto blockRangeVec                                          = {blockrange0, blockrange1, blockrange2,
    // blockrange4}; storage.registerRanges(blockRangeVec);
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

    ASSERT_THROW(
        storage.forAllBlocks(
            rangeInvalid,
            [&expected](const uint8_t* ptr, size_t size) {
                ++expected;
                UNUSED(ptr);
                UNUSED(size);
            }),
        std::invalid_argument);
}

TEST(SerializedBlockStorageTest, Constructor) {
    auto blockDistribution = std::make_shared<ReStore::BlockDistribution<MPIContextMock>>(10, 100, 3, MPIContextMock());

    ASSERT_ANY_THROW(ReStore::SerializedBlockStorage<MPIContextMock>(blockDistribution, ReStore::OffsetMode::constant));
    ASSERT_ANY_THROW(
        ReStore::SerializedBlockStorage<MPIContextMock>(blockDistribution, ReStore::OffsetMode::constant, 0));
    ASSERT_ANY_THROW(
        ReStore::SerializedBlockStorage<MPIContextMock>(blockDistribution, ReStore::OffsetMode::lookUpTable, 1));

    ASSERT_NO_THROW(
        ReStore::SerializedBlockStorage<MPIContextMock>(blockDistribution, ReStore::OffsetMode::constant, 2));
    ASSERT_NO_THROW(
        ReStore::SerializedBlockStorage<MPIContextMock>(blockDistribution, ReStore::OffsetMode::lookUpTable));
}

TEST(SerializedBlockStorageTest, Writing) {
    using block_id_t       = ReStore::block_id_t;
    auto blockDistribution = std::make_shared<ReStore::BlockDistribution<MPIContextMock>>(10, 30, 3, MPIContextMock());
    assert(blockDistribution->numRanges() == 10);
    assert(blockDistribution->numRangesWithAdditionalBlock() == 0);
    assert(blockDistribution->blocksPerRange() == 3);

    {
        auto storage = ReStore::SerializedBlockStorage<MPIContextMock>(
            blockDistribution, ReStore::OffsetMode::constant, sizeof(uint8_t));
        std::vector<uint8_t> values = {13, 37, 42};
        for (size_t blockId = 0; blockId < values.size(); blockId++) {
            storage.writeBlock(blockId, &values[blockId]);
        }

        size_t rankId = 0;
        storage.forAllBlocks(std::make_pair<block_id_t, size_t>(0, 3), [&](const uint8_t* data, size_t lengthInBytes) {
            EXPECT_EQ(lengthInBytes, 1);
            EXPECT_EQ(values[rankId++], *data);
        });
    }

    {
        auto storage = ReStore::SerializedBlockStorage<MPIContextMock>(
            blockDistribution, ReStore::OffsetMode::constant, sizeof(uint16_t));
        std::vector<uint16_t> values = {1000, 1001, 1002};

        for (size_t blockId = 0; blockId < values.size(); blockId++) {
            storage.writeBlock(blockId, reinterpret_cast<const uint8_t*>(&values[blockId]));
        }

        size_t rankId = 0;
        storage.forAllBlocks(std::make_pair<block_id_t, size_t>(0, 3), [&](const uint8_t* data, size_t lengthInBytes) {
            EXPECT_EQ(lengthInBytes, 2);
            EXPECT_EQ(values[rankId++], *reinterpret_cast<const uint16_t*>(data));
        });
    }

    {
        auto storage = ReStore::SerializedBlockStorage<MPIContextMock>(
            blockDistribution, ReStore::OffsetMode::constant, sizeof(uint64_t));
        std::vector<uint64_t> values = {1000, 1001, 1002};

        for (size_t blockId = 0; blockId < values.size(); blockId++) {
            storage.writeBlock(blockId, reinterpret_cast<const uint8_t*>(&values[blockId]));
        }

        size_t rankId = 0;
        storage.forAllBlocks(std::make_pair<block_id_t, size_t>(0, 3), [&](const uint8_t* data, size_t lengthInBytes) {
            EXPECT_EQ(lengthInBytes, 8);
            EXPECT_EQ(values[rankId++], *reinterpret_cast<const uint64_t*>(data));
        });
    }

    {
        auto storage = ReStore::SerializedBlockStorage<MPIContextMock>(
            blockDistribution, ReStore::OffsetMode::constant, sizeof(uint64_t));
        std::vector<uint64_t> values = {1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009};

        for (size_t blockId = 0; blockId < values.size(); blockId++) {
            storage.writeBlock(blockId, reinterpret_cast<const uint8_t*>(&values[blockId]));
        }

        size_t rankId = 0;
        storage.forAllBlocks(std::make_pair<block_id_t, size_t>(0, 9), [&](const uint8_t* data, size_t lengthInBytes) {
            EXPECT_EQ(lengthInBytes, 8);
            EXPECT_EQ(values[rankId++], *reinterpret_cast<const uint64_t*>(data));
        });
    }
}

TEST(SerializedBlockStoreStream, Constructor) {
    using BuffersType    = std::unordered_map<ReStoreMPI::current_rank_t, std::vector<uint8_t>>;
    using RanksArrayType = std::vector<ReStoreMPI::current_rank_t>;

    auto buffers = BuffersType();
    auto ranks   = RanksArrayType();

    // no ranks
    ASSERT_ANY_THROW(ReStore::SerializedBlockStoreStream(buffers, ranks));

    // all fine
    ranks.push_back(0);
    ASSERT_NO_THROW(ReStore::SerializedBlockStoreStream(buffers, ranks));
    ranks.push_back(1);
    ranks.push_back(2);
    ASSERT_NO_THROW(ReStore::SerializedBlockStoreStream(buffers, ranks));
}

TEST(SerializedBlockStoreStream, InStream) {
    using BuffersType    = std::unordered_map<ReStoreMPI::current_rank_t, std::vector<uint8_t>>;
    using RanksArrayType = std::vector<ReStoreMPI::current_rank_t>;

    auto buffers = BuffersType();
    auto ranks   = RanksArrayType();
    ranks.push_back(0);
    ranks.push_back(3);

    ReStore::SerializedBlockStoreStream stream(buffers, ranks);

    stream << 0x42_byte;
    ASSERT_EQ(buffers.at(0)[0], 0x42_byte);
    ASSERT_EQ(buffers.find(1), buffers.end());
    ASSERT_EQ(buffers.find(2), buffers.end());
    ASSERT_EQ(buffers.at(3)[0], 0x42_byte);
    ASSERT_EQ(stream.bytesWritten(), 1);
    ASSERT_EQ(buffers.at(0).size(), 1);
    ASSERT_EQ(buffers.at(3).size(), 1);

    stream << 0x00_uint8;
    ASSERT_EQ(buffers.at(0)[0], 0x42_byte);
    ASSERT_EQ(buffers.find(1), buffers.end());
    ASSERT_EQ(buffers.find(2), buffers.end());
    ASSERT_EQ(buffers.at(3)[0], 0x42_byte);

    ASSERT_EQ(buffers.at(0)[1], 0x00_byte);
    ASSERT_EQ(buffers.at(3)[1], 0x00_byte);
    ASSERT_EQ(stream.bytesWritten(), 2);
    ASSERT_EQ(buffers.at(0).size(), 2);
    ASSERT_EQ(buffers.at(3).size(), 2);

    // I refrain from asserting anything about the order in which bytes are stored in memory
    // TODO test that read ° store = identity

    stream << 0xFFFF_uint16;
    ASSERT_EQ(buffers.at(0)[0], 0x42_byte);
    ASSERT_EQ(buffers.find(1), buffers.end());
    ASSERT_EQ(buffers.find(2), buffers.end());
    ASSERT_EQ(buffers.at(3)[0], 0x42_byte);

    ASSERT_EQ(buffers.at(0)[1], 0x00_byte);
    ASSERT_EQ(buffers.at(3)[1], 0x00_byte);
    ASSERT_EQ(buffers.at(0)[2], 0xFF_byte);
    ASSERT_EQ(buffers.at(3)[2], 0xFF_byte);
    ASSERT_EQ(buffers.at(0)[3], 0xFF_byte);
    ASSERT_EQ(buffers.at(3)[3], 0xFF_byte);
    ASSERT_EQ(stream.bytesWritten(), 4);
    ASSERT_EQ(buffers.at(0).size(), 4);
    ASSERT_EQ(buffers.at(3).size(), 4);

    stream << 0x10133701_uint32;
    ASSERT_THAT(buffers.at(0), UnorderedElementsAre(0x00, 0x42, 0x01, 0x10, 0x13, 0x37, 0xFF, 0xFF));
    ASSERT_EQ(buffers.find(1), buffers.end());
    ASSERT_EQ(buffers.find(2), buffers.end());
    ASSERT_THAT(buffers.at(3), UnorderedElementsAre(0x00, 0x42, 0x01, 0x10, 0x13, 0x37, 0xFF, 0xFF));
    ASSERT_EQ(stream.bytesWritten(), 8);
    ASSERT_EQ(buffers.at(0).size(), 8);
    ASSERT_EQ(buffers.at(3).size(), 8);
}

int main(int argc, char** argv) {
    // Filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int result = RUN_ALL_TESTS();

    return result;
}