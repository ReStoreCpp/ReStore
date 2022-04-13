// #include <algorithm>
// #include <assert.h>
// #include <cstddef>
// #include <optional>
// #include <vector>
//
// #include "range.hpp"
#include <cassert>
#include <gmock/gmock.h>
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <mpi.h>
// #include <utility>

#include "restore/common.hpp"
#include "restore/core.hpp"

#include "mpi_helpers.hpp"

using namespace ::testing;

TEST(ReStoreTest, EndToEnd_AlreadySerializedData) {
    // The logic of this tests assumes that there are four ranks
    assert(numRanks() == 4);

    const uint8_t replicationLevel = 3;
    const size_t  constantOffset   = sizeof(int);

    ReStore::ReStore<int> store(MPI_COMM_WORLD, replicationLevel, ReStore::OffsetMode::constant, constantOffset);
    std::vector<int>      data;
    std::vector<ReStore::SerializedBlocksDescriptor> blockDescriptors;
    switch (myRankId()) {
        case 0:
            data = {0, 2, 3};
            blockDescriptors.emplace_back(0, 1, reinterpret_cast<std::byte*>(data.data()));
            blockDescriptors.emplace_back(2, 4, reinterpret_cast<std::byte*>(data.data()) + constantOffset);
            break;
        case 1:
            data = {1, 4, 5, 6};
            blockDescriptors.emplace_back(1, 2, reinterpret_cast<std::byte*>(data.data()));
            blockDescriptors.emplace_back(4, 7, reinterpret_cast<std::byte*>(data.data()) + constantOffset);
            break;
        case 2:
            data = {10, 11, 12};
            blockDescriptors.emplace_back(10, 13, reinterpret_cast<std::byte*>(data.data()));
            break;
        case 3:
            data = {7, 8, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
            blockDescriptors.emplace_back(7, 10, reinterpret_cast<std::byte*>(data.data()));
            blockDescriptors.emplace_back(13, 24, reinterpret_cast<std::byte*>(data.data()) + 3 * constantOffset);
            break;
    }
    const ReStore::block_id_t globalNumberOfBlocks = 24;

    store.submitSerializedBlocks(blockDescriptors, globalNumberOfBlocks);

    { // Request all data on all processes.
        std::vector<std::pair<ReStore::block_id_t, size_t>> requests;
        requests.emplace_back(std::make_pair(0, globalNumberOfBlocks));

        std::vector<int> dataReceived(globalNumberOfBlocks);
        std::fill(dataReceived.begin(), dataReceived.end(), -1);
        ReStore::block_id_t numBlocksReceived = 0;
        store.pullBlocks(
            requests,
            [&dataReceived, &numBlocksReceived](const std::byte* dataPtr, size_t size, ReStore::block_id_t blockId) {
                ASSERT_EQ(sizeof(int), size);
                assert(blockId < dataReceived.size());
                EXPECT_EQ(dataReceived[blockId], -1);
                dataReceived[blockId] = *reinterpret_cast<const int*>(dataPtr);
                EXPECT_NE(dataReceived[blockId], -1);
                EXPECT_EQ(dataReceived[blockId], blockId);
                numBlocksReceived++;
            });
        EXPECT_EQ(globalNumberOfBlocks, numBlocksReceived);
        EXPECT_EQ(dataReceived.size(), numBlocksReceived);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    { // Request equal parts of the data on all ranks, no element requested by two different processes.
        assert(globalNumberOfBlocks % asserting_cast<unsigned long>(numRanks()) == 0);
        const ReStore::block_id_t blocksPerRank = globalNumberOfBlocks / asserting_cast<unsigned long>(numRanks());
        const ReStore::block_id_t firstBlockOfThisRank = blocksPerRank * asserting_cast<unsigned long>(myRankId());
        const ReStore::block_id_t lastBlockOfThisRank  = firstBlockOfThisRank + blocksPerRank - 1;
        std::vector<std::pair<ReStore::block_id_t, size_t>> requests;
        requests.emplace_back(std::make_pair(firstBlockOfThisRank, blocksPerRank));

        std::vector<int> dataReceived(globalNumberOfBlocks);
        std::fill(dataReceived.begin(), dataReceived.end(), -1);
        ReStore::block_id_t numBlocksReceived = 0;
        MPI_Barrier(MPI_COMM_WORLD);
        store.pullBlocks(
            requests, [&dataReceived, &numBlocksReceived, firstBlockOfThisRank,
                       lastBlockOfThisRank](const std::byte* dataPtr, size_t size, ReStore::block_id_t blockId) {
                ASSERT_EQ(sizeof(int), size);
                EXPECT_GE(blockId, firstBlockOfThisRank);
                EXPECT_LE(blockId, lastBlockOfThisRank);
                assert(blockId < dataReceived.size());
                EXPECT_EQ(dataReceived[blockId], -1);
                dataReceived[blockId] = *reinterpret_cast<const int*>(dataPtr);
                EXPECT_NE(dataReceived[blockId], -1);
                EXPECT_EQ(dataReceived[blockId], blockId);
                numBlocksReceived++;
            });
        MPI_Barrier(MPI_COMM_WORLD);
        EXPECT_EQ(blocksPerRank, numBlocksReceived);
    }
}

int main(int argc, char** argv) {
    // Filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Set errorhandler to return so we have a chance to mitigate failures
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    // Add object that will finalize MPI on exit; Google Test owns this pointer
    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);

    // Get the event listener list.
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

    // Remove default listener: the default printer and the default XML printer
    ::testing::TestEventListener* l = listeners.Release(listeners.default_result_printer());

    // Adds MPI listener; Google Test owns this pointer
    listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l, MPI_COMM_WORLD));

    int result = RUN_ALL_TESTS();

    return result;
}
