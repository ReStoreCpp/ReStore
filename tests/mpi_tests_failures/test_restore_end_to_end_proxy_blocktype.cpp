#include <algorithm>
#include <assert.h>
#include <cstddef>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <mpi.h>
#include <utility>

#include "restore/common.hpp"
#include "restore/core.hpp"

#include "mpi_helpers.hpp"
#include "restore/mpi_context.hpp"

using namespace ::testing;

TEST(ReStoreTest, EndToEnd_ProxyBlockType) {
    // The logic of this tests assumes that there are four ranks
    assert(numRanks() == 4);

    const uint16_t NUM_DIMENSIONS = 3;

    using BlockProxy = int*;

    ReStore::ReStore<BlockProxy> store(
        MPI_COMM_WORLD, 1, ReStore::OffsetMode::constant, sizeof(int) * NUM_DIMENSIONS, 1);
    std::vector<int> data{100, 101, 102, 200, 201, 202, 300, 301, 302, 400, 401, 402};
    assert(data.size() % NUM_DIMENSIONS == 0);
    const size_t numDataPointsLocal = data.size() / NUM_DIMENSIONS;

    // Reserve space for the current proxy (we are pushing only BlockType references around during submitBlocks).
    BlockProxy currentProxy = nullptr;
    unsigned   counter      = 0;
    store.submitBlocks(
        [](BlockProxy blockProxy, ReStore::SerializedBlockStoreStream& stream) {
            assert(blockProxy != nullptr);
            stream.writeBytes(reinterpret_cast<const std::byte*>(blockProxy), sizeof(int) * NUM_DIMENSIONS);
        },
        [&counter, data = data.data(), numDataPointsLocal, &currentProxy]() {
            assert(data != nullptr);
            std::optional<ReStore::NextBlock<BlockProxy>> nextBlock = std::nullopt;
            if (counter < numDataPointsLocal) {
                auto blockId = counter + static_cast<size_t>(myRankId()) * numDataPointsLocal;
                currentProxy = BlockProxy{data + counter * NUM_DIMENSIONS};
                nextBlock.emplace(blockId, currentProxy);
                counter++; // We cannot put this in the above line, as we can't assume if the first argument of the pair
                           // is bound before or after the increment.
            }
            return nextBlock;
        },
        data.size() * asserting_cast<size_t>(numRanks()));

    // // No failure
    // {
    //     std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, ReStoreMPI::current_rank_t>> requests;
    //     for (int rank = 0; rank < numRanks(); ++rank) {
    //         requests.emplace_back(std::make_pair(std::make_pair(0, data.size()), rank));
    //     }

    //     std::vector<int>    dataReceived;
    //     ReStore::block_id_t nextBlockId = 0;
    //     store.pushBlocksCurrentRankIds(
    //         requests,
    //         [&dataReceived, &nextBlockId](const std::byte* dataPtr, size_t size, ReStore::block_id_t blockId) {
    //             EXPECT_EQ(nextBlockId, blockId);
    //             ++nextBlockId;
    //             ASSERT_EQ(sizeof(int), size);
    //             dataReceived.emplace_back(*reinterpret_cast<const int*>(dataPtr));
    //         });
    //     EXPECT_EQ(data.size(), nextBlockId);

    //     ASSERT_EQ(data.size(), dataReceived.size());
    //     for (size_t i = 0; i < data.size(); ++i) {
    //         EXPECT_EQ(data[i], dataReceived[i]);
    //     }
    // }

    // {
    //     std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, ReStoreMPI::current_rank_t>> requests;
    //     for (int rank = 0; rank < numRanks(); ++rank) {
    //         requests.emplace_back(
    //             std::make_pair(std::make_pair(static_cast<size_t>(rank) * data.size(), data.size()), rank));
    //     }

    //     std::vector<int>    dataReceived;
    //     ReStore::block_id_t nextBlockId = static_cast<size_t>(myRankId()) * data.size();
    //     store.pushBlocksCurrentRankIds(
    //         requests,
    //         [&dataReceived, &nextBlockId](const std::byte* dataPtr, size_t size, ReStore::block_id_t blockId) {
    //             EXPECT_EQ(nextBlockId, blockId);
    //             ++nextBlockId;
    //             ASSERT_EQ(sizeof(int), size);
    //             dataReceived.emplace_back(*reinterpret_cast<const int*>(dataPtr));
    //         });
    //     EXPECT_EQ(static_cast<size_t>(myRankId()) * data.size() + data.size(), nextBlockId);

    //     ASSERT_EQ(data.size(), dataReceived.size());
    //     for (size_t i = 0; i < data.size(); ++i) {
    //         EXPECT_EQ(data[i], dataReceived[i]);
    //     }
    // }
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
