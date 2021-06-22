#include <algorithm>
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <mpi.h>
#include <stdexcept>
#include <utility>

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
        storage.writeBlock(block, reinterpret_cast<std::byte*>(&block));
    }

    for (uint8_t block = 40; block < 50; ++block) {
        storage.writeBlock(block, reinterpret_cast<std::byte*>(&block));
    }

    std::pair<ReStore::block_id_t, size_t> range1(5, 20);
    std::pair<ReStore::block_id_t, size_t> range2(40, 10);
    std::pair<ReStore::block_id_t, size_t> rangeInvalid(30, 10);

    std::byte expected = (std::byte)5;
    storage.forAllBlocks(range1, [&expected](const std::byte* ptr, size_t size) {
        EXPECT_EQ(1, size);
        EXPECT_EQ(expected, *ptr);
        expected = static_cast<std::byte>(static_cast<unsigned char>(expected) + 1);
    });
    EXPECT_EQ((std::byte)25, expected);

    expected = (std::byte)40;
    storage.forAllBlocks(range2, [&expected](const std::byte* ptr, size_t size) {
        EXPECT_EQ(1, size);
        EXPECT_EQ(expected, *ptr);
        expected = static_cast<std::byte>(static_cast<unsigned char>(expected) + 1);
    });
    EXPECT_EQ((std::byte)50, expected);

    ASSERT_THROW(
        storage.forAllBlocks(
            rangeInvalid,
            [&expected](const std::byte* ptr, size_t size) {
                expected = static_cast<std::byte>(static_cast<unsigned char>(expected) + 1);
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
            storage.writeBlock(blockId, reinterpret_cast<std::byte*>(&values[blockId]));
        }

        size_t rankId = 0;
        storage.forAllBlocks(
            std::make_pair<block_id_t, size_t>(0, 3), [&](const std::byte* data, size_t lengthInBytes) {
                EXPECT_EQ(lengthInBytes, 1);
                EXPECT_EQ(values[rankId++], *(reinterpret_cast<const uint8_t*>(data)));
            });
    }

    {
        auto storage = ReStore::SerializedBlockStorage<MPIContextMock>(
            blockDistribution, ReStore::OffsetMode::constant, sizeof(uint16_t));
        std::vector<uint16_t> values = {1000, 1001, 1002};

        for (size_t blockId = 0; blockId < values.size(); blockId++) {
            storage.writeBlock(blockId, reinterpret_cast<const std::byte*>(&values[blockId]));
        }

        size_t rankId = 0;
        storage.forAllBlocks(
            std::make_pair<block_id_t, size_t>(0, 3), [&](const std::byte* data, size_t lengthInBytes) {
                EXPECT_EQ(lengthInBytes, 2);
                EXPECT_EQ(values[rankId++], *reinterpret_cast<const uint16_t*>(data));
            });
    }

    {
        auto storage = ReStore::SerializedBlockStorage<MPIContextMock>(
            blockDistribution, ReStore::OffsetMode::constant, sizeof(uint64_t));
        std::vector<uint64_t> values = {1000, 1001, 1002};

        for (size_t blockId = 0; blockId < values.size(); blockId++) {
            storage.writeBlock(blockId, reinterpret_cast<const std::byte*>(&values[blockId]));
        }

        size_t rankId = 0;
        storage.forAllBlocks(
            std::make_pair<block_id_t, size_t>(0, 3), [&](const std::byte* data, size_t lengthInBytes) {
                EXPECT_EQ(lengthInBytes, 8);
                EXPECT_EQ(values[rankId++], *reinterpret_cast<const uint64_t*>(data));
            });
    }

    {
        auto storage = ReStore::SerializedBlockStorage<MPIContextMock>(
            blockDistribution, ReStore::OffsetMode::constant, sizeof(uint64_t));
        std::vector<uint64_t> values = {1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009};

        for (size_t blockId = 0; blockId < values.size(); blockId++) {
            storage.writeBlock(blockId, reinterpret_cast<const std::byte*>(&values[blockId]));
        }

        size_t rankId = 0;
        storage.forAllBlocks(
            std::make_pair<block_id_t, size_t>(0, 9), [&](const std::byte* data, size_t lengthInBytes) {
                EXPECT_EQ(lengthInBytes, 8);
                EXPECT_EQ(values[rankId++], *reinterpret_cast<const uint64_t*>(data));
            });
    }
}

TEST(SerializedBlockStoreStream, Constructor) {
    using BuffersType    = std::vector<std::vector<std::byte>>;
    using RanksArrayType = std::vector<ReStoreMPI::current_rank_t>;

    auto buffers = BuffersType();
    auto ranks   = RanksArrayType();

    // no ranks or buffers not allocated
    ASSERT_ANY_THROW(ReStore::SerializedBlockStoreStream(buffers, 0));
    ASSERT_ANY_THROW(ReStore::SerializedBlockStoreStream(buffers, 4).setDestinationRanks(ranks));
    buffers.resize(4);
    ASSERT_ANY_THROW(ReStore::SerializedBlockStoreStream(buffers, 0).setDestinationRanks(ranks));

    // all fine
    ranks.push_back(0);
    ASSERT_NO_THROW(ReStore::SerializedBlockStoreStream(buffers, 4).setDestinationRanks(ranks));
    ranks.push_back(1);
    ranks.push_back(2);
    ASSERT_NO_THROW(ReStore::SerializedBlockStoreStream(buffers, 4).setDestinationRanks(ranks));
}

TEST(SerializedBlockStoreStream, InStream) {
    using BuffersType    = std::vector<std::vector<std::byte>>;
    using RanksArrayType = std::vector<ReStoreMPI::current_rank_t>;

    auto                        buffers  = BuffersType();
    auto                        ranks    = RanksArrayType();
    ReStoreMPI::original_rank_t numRanks = 4;
    ranks.push_back(0);
    ranks.push_back(3);
    buffers.resize(asserting_cast<size_t>(numRanks));

    ReStore::SerializedBlockStoreStream stream(buffers, numRanks);
    stream.setDestinationRanks(ranks);

    stream << 0x42_byte;
    ASSERT_EQ(buffers.at(0)[0], 0x42_byte);
    ASSERT_EQ(buffers.at(1).size(), 0);
    ASSERT_EQ(buffers.at(2).size(), 0);
    ASSERT_EQ(buffers.at(3)[0], 0x42_byte);
    ASSERT_EQ(stream.bytesWritten(0), 1);
    ASSERT_EQ(stream.bytesWritten(3), 1);
    ASSERT_EQ(buffers.at(0).size(), 1);
    ASSERT_EQ(buffers.at(3).size(), 1);

    stream << 0x00_uint8;
    ASSERT_EQ(buffers.at(0)[0], 0x42_byte);
    ASSERT_EQ(buffers.at(1).size(), 0);
    ASSERT_EQ(buffers.at(2).size(), 0);
    ASSERT_EQ(buffers.at(3)[0], 0x42_byte);

    ASSERT_EQ(buffers.at(0)[1], 0x00_byte);
    ASSERT_EQ(buffers.at(3)[1], 0x00_byte);
    ASSERT_EQ(stream.bytesWritten(0), 2);
    ASSERT_EQ(stream.bytesWritten(3), 2);
    ASSERT_EQ(buffers.at(0).size(), 2);
    ASSERT_EQ(buffers.at(3).size(), 2);

    stream << 0xFFFF_uint16;
    ASSERT_EQ(buffers.at(0)[0], 0x42_byte);
    ASSERT_EQ(buffers.at(1).size(), 0);
    ASSERT_EQ(buffers.at(2).size(), 0);
    ASSERT_EQ(buffers.at(3)[0], 0x42_byte);

    ASSERT_EQ(buffers.at(0)[1], 0x00_byte);
    ASSERT_EQ(buffers.at(3)[1], 0x00_byte);
    ASSERT_EQ(buffers.at(0)[2], 0xFF_byte);
    ASSERT_EQ(buffers.at(3)[2], 0xFF_byte);
    ASSERT_EQ(buffers.at(0)[3], 0xFF_byte);
    ASSERT_EQ(buffers.at(3)[3], 0xFF_byte);
    ASSERT_EQ(stream.bytesWritten(0), 4);
    ASSERT_EQ(stream.bytesWritten(3), 4);
    ASSERT_EQ(buffers.at(0).size(), 4);
    ASSERT_EQ(buffers.at(3).size(), 4);

    stream << 0x10133701_uint32;
    ASSERT_THAT(
        buffers.at(0),
        UnorderedElementsAre(0x00_byte, 0x42_byte, 0x01_byte, 0x10_byte, 0x13_byte, 0x37_byte, 0xFF_byte, 0xFF_byte));
    ASSERT_EQ(buffers.at(1).size(), 0);
    ASSERT_EQ(buffers.at(2).size(), 0);
    ASSERT_THAT(
        buffers.at(3),
        UnorderedElementsAre(0x00_byte, 0x42_byte, 0x01_byte, 0x10_byte, 0x13_byte, 0x37_byte, 0xFF_byte, 0xFF_byte));
    ASSERT_EQ(stream.bytesWritten(0), 8);
    ASSERT_EQ(stream.bytesWritten(3), 8);
    ASSERT_EQ(buffers.at(0).size(), 8);
    ASSERT_EQ(buffers.at(3).size(), 8);

    auto buffers2 = BuffersType();
    buffers2.resize(asserting_cast<size_t>(numRanks));
    ReStore::SerializedBlockStoreStream stream2(buffers2, numRanks);
    stream2.setDestinationRanks(ranks);

    stream2 << 0x1F1F_uint16;
    auto handle1p0 = stream2.reserveBytesForWriting(0, sizeof(uint8_t));
    auto handle1p3 = stream2.reserveBytesForWriting(3, sizeof(uint8_t));
    stream2 << 0x0101_uint16;
    auto handle2 = stream2.reserveBytesForWriting(3, sizeof(uint16_t));
    stream2 << 0x2222_uint16;
    stream2.writeToReservedBytes(handle1p0, 0xFF_uint8);
    stream2.writeToReservedBytes(handle1p3, 0xFF_uint8);
    stream2.writeToReservedBytes(handle2, 0x3333_uint16);

    ASSERT_EQ(stream2.bytesWritten(0), 7);
    ASSERT_EQ(stream2.bytesWritten(3), 9);

    ASSERT_THAT(
        buffers2.at(0), ElementsAre(0x1F_byte, 0x1F_byte, 0xFF_byte, 0x01_byte, 0x01_byte, 0x22_byte, 0x22_byte));
    ASSERT_THAT(
        buffers2.at(3),
        ElementsAre(0x1F_byte, 0x1F_byte, 0xFF_byte, 0x01_byte, 0x01_byte, 0x33_byte, 0x33_byte, 0x22_byte, 0x22_byte));
}

int main(int argc, char** argv) {
    // Filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int result = RUN_ALL_TESTS();

    return result;
}
