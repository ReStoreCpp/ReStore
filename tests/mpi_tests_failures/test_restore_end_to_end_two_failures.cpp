#include <algorithm>
#include <functional>
#include <signal.h>
#include <sstream>

#include "itertools.hpp"
#include <gmock/gmock.h>
#include <gtest-mpi-listener/include/gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <mpi.h>
#include <utility>

#include "restore/common.hpp"
#include "restore/core.hpp"
#include "restore/helpers.hpp"

#include "mocks.hpp"
#include "mpi_helpers.hpp"
#include "restore/mpi_context.hpp"

using namespace ::testing;

using iter::range;

TEST(ReStoreTest, EndToEnd_TwoFailures) {
    // Each rank submits different data. The replication level is set to 3. There are two rank failures.
    ReStore::ReStore<int> store(MPI_COMM_WORLD, 3, ReStore::OffsetMode::constant, sizeof(int));

    std::vector<int> data;
    for (int value: range(1000 * myRankId(), 1000 * myRankId() + 1000)) {
        data.push_back(-1 * value);
    }

    int      numRanksInitial = numRanks();
    size_t   numBlocks       = static_cast<size_t>(numRanksInitial) * 1000;
    unsigned counter         = 0;
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
        numBlocks);
    EXPECT_EQ(1001, counter);

    // Two failures
    constexpr int failingRank1 = 1;
    constexpr int failingRank2 = 2;
    failRank(failingRank1);
    failRank(failingRank2);
    ASSERT_NE(myRankId(), failingRank1);
    ASSERT_NE(myRankId(), failingRank2);

    auto newComm = getFixedCommunicator();

    store.updateComm(newComm);

    size_t              numBlocksPerRank       = numBlocks / static_cast<size_t>(numRanks(newComm));
    size_t              numRanksWithMoreBlocks = numBlocks % static_cast<size_t>(numRanks(newComm));
    ReStore::block_id_t currentBlock           = 0;
    std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, ReStoreMPI::current_rank_t>> requests;
    for (int rank = 0; rank < numRanks(newComm); ++rank) {
        const size_t numBlockForThisRank = numBlocksPerRank + (static_cast<size_t>(rank) < numRanksWithMoreBlocks);
        requests.emplace_back(std::make_pair(std::make_pair(currentBlock, numBlockForThisRank), rank));
        currentBlock += numBlockForThisRank;
    }
    EXPECT_EQ(numBlocks, currentBlock);

    ReStore::block_id_t numBlocksToReceive =
        numBlocksPerRank + (static_cast<size_t>(myRankId(newComm)) < numRanksWithMoreBlocks);
    std::vector<int>    dataReceived(numBlocksToReceive, 1);
    ReStore::block_id_t firstBlockId = numBlocksPerRank * static_cast<size_t>(myRankId(newComm))
                                       + std::min(numRanksWithMoreBlocks, static_cast<size_t>(myRankId(newComm)));
    ReStore::block_id_t numBlocksReceived = 0;
    store.pushBlocks(
        requests, [&dataReceived, &numBlocksReceived,
                   firstBlockId](const std::byte* dataPtr, size_t size, ReStore::block_id_t blockId) {
            ASSERT_GE(blockId, firstBlockId);
            size_t index = blockId - firstBlockId;
            ASSERT_LT(index, dataReceived.size());
            EXPECT_EQ(1, dataReceived[index]);
            ++numBlocksReceived;
            ASSERT_EQ(sizeof(int), size);
            EXPECT_EQ(-1 * static_cast<int>(blockId), *reinterpret_cast<const int*>(dataPtr));
            dataReceived[index] = *reinterpret_cast<const int*>(dataPtr);
        });
    EXPECT_EQ(numBlocksToReceive, numBlocksReceived);

    for (const auto& receivedInt: dataReceived) {
        EXPECT_LE(receivedInt, 0);
    }
}


int main(int argc, char** argv) {
    // Filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Set errorhandler to return so we have a chance to mitigate failures
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    // // Add object that will finalize MPI on exit; Google Test owns this pointer
    // ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);

    // // Get the event listener list.
    // ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

    // // Remove default listener: the default printer and the default XML printer
    // ::testing::TestEventListener* l = listeners.Release(listeners.default_result_printer());

    // // Adds MPI listener; Google Test owns this pointer
    // listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l, MPI_COMM_WORLD));

    int result = RUN_ALL_TESTS();

    return result;
}
