#include <algorithm>
#include <assert.h>
#include <cstddef>
#include <optional>
#include <vector>

#include "range.hpp"
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

using iter::range;

TEST(ReStoreTest, EndToEnd_Simple1_PushBlocks) {
    // The most basic test case. Each rank submits exactly the same data. The replication level is set to one. There is
    // no rank failure.

    // The logic of this tests assumes that there are four ranks
    assert(numRanks() == 4);

    const uint8_t         replicationLevel          = 1;
    const size_t          constantOffset            = sizeof(int);
    const uint64_t        blocksPerPermutationRange = 1;
    ReStore::ReStore<int> store(
        MPI_COMM_WORLD, replicationLevel, ReStore::OffsetMode::constant, constantOffset, blocksPerPermutationRange);
    std::vector<int> data{0, 1, 2, 3, 42, 1337};

    unsigned counter = 0;
    store.submitBlocks(
        [](const int& value, ReStore::SerializedBlockStoreStream& stream) { stream << value; },
        [&counter, &data]() {
            auto ret = data.size() == counter
                           ? std::nullopt
                           : std::make_optional(ReStore::NextBlock<int>(
                               {counter + static_cast<size_t>(myRankId()) * data.size(), data[counter]}));
            counter++; // We cannot put this in the above line, as we can't assume if the first argument of the pair is
                       // bound before or after the increment.
            return ret;
        },
        data.size() * static_cast<size_t>(numRanks()));

    // No failure
    {
        std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, ReStoreMPI::current_rank_t>> requests;
        for (int rank = 0; rank < numRanks(); ++rank) {
            requests.emplace_back(std::make_pair(std::make_pair(0, data.size()), rank));
        }

        std::vector<int> dataReceived(data.size());
        std::fill(dataReceived.begin(), dataReceived.end(), -1);
        ReStore::block_id_t numBlocksReceived = 0;
        store.pushBlocksCurrentRankIds(
            requests,
            [&dataReceived, &numBlocksReceived](const std::byte* dataPtr, size_t size, ReStore::block_id_t blockId) {
                ASSERT_EQ(sizeof(int), size);
                ASSERT_LT(blockId, dataReceived.size());
                EXPECT_EQ(dataReceived[blockId], -1);
                dataReceived[blockId] = *reinterpret_cast<const int*>(dataPtr);
                EXPECT_NE(dataReceived[blockId], -1);
                numBlocksReceived++;
            });
        EXPECT_EQ(data.size(), numBlocksReceived);

        ASSERT_EQ(data.size(), dataReceived.size());
        for (size_t i = 0; i < data.size(); ++i) {
            EXPECT_EQ(data[i], dataReceived[i]);
        }
    }

    {
        std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, ReStoreMPI::current_rank_t>> requests;
        for (int rank = 0; rank < numRanks(); ++rank) {
            requests.emplace_back(
                std::make_pair(std::make_pair(static_cast<size_t>(rank) * data.size(), data.size()), rank));
        }

        std::vector<int> dataReceived(data.size());
        std::fill(dataReceived.begin(), dataReceived.end(), -1);
        ReStore::block_id_t numBlocksReceived      = 0;
        const auto          thisRanksBlockIdOffset = data.size() * static_cast<size_t>(myRankId());
        store.pushBlocksCurrentRankIds(
            requests, [&dataReceived, &numBlocksReceived,
                       thisRanksBlockIdOffset](const std::byte* dataPtr, size_t size, ReStore::block_id_t blockId) {
                const auto idxDataReceived = blockId - thisRanksBlockIdOffset;
                ASSERT_EQ(sizeof(int), size);
                ASSERT_LT(idxDataReceived, dataReceived.size());
                EXPECT_EQ(dataReceived[idxDataReceived], -1);
                dataReceived[idxDataReceived] = *reinterpret_cast<const int*>(dataPtr);
                EXPECT_NE(dataReceived[idxDataReceived], -1);
                numBlocksReceived++;
            });
        EXPECT_EQ(data.size(), numBlocksReceived);

        ASSERT_EQ(data.size(), dataReceived.size());
        for (size_t i = 0; i < data.size(); ++i) {
            EXPECT_EQ(data[i], dataReceived[i]);
        }
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
