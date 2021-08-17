#include <cstddef>
#include <cstdint>

#include <gmock/gmock.h>
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <mpi.h>

#include "restore/restore_vector.hpp"

#include "test_with_failures_fixture.hpp"

using namespace ReStore;
using namespace testing;

// This test does not simulate a failure and thus does not need to be in a separate executable.
TEST_F(ReStoreVectorTest, ArgumentChecking) {
    const size_t   blockSize        = 10;
    const MPI_Comm mpiComm          = MPI_COMM_WORLD;
    const uint16_t replicationLevel = 3;

    // Invalid constructor arguments
    ASSERT_THROW(ReStoreVector<int>(0, mpiComm, replicationLevel), std::invalid_argument);
    ASSERT_THROW(ReStoreVector<int>(blockSize, MPI_COMM_NULL, replicationLevel), std::invalid_argument);
    ASSERT_THROW(ReStoreVector<int>(blockSize, mpiComm, 0), std::invalid_argument);

    // All arguments valid
    ASSERT_NO_THROW(ReStoreVector<int>(blockSize, mpiComm, replicationLevel));

    // Empty data vector
    auto             store = ReStoreVector<int>(blockSize, mpiComm, replicationLevel);
    std::vector<int> vec(0);
    ASSERT_THROW(store.submitData(vec), std::invalid_argument);

    // Updating with invalid communictor
    ASSERT_THROW(store.updateComm(MPI_COMM_NULL), std::invalid_argument);

    // Nobody gets new blocks
    ReStoreVector<int>::BlockRangeToRestoreList newBlocksPerRank;
    ASSERT_NO_THROW(store.restoreDataAppend(vec, newBlocksPerRank));
}

TEST_F(ReStoreVectorTest, EndToEnd) {
    const size_t   blockSize          = 2;
    const MPI_Comm mpiComm            = MPI_COMM_WORLD;
    const uint16_t replicationLevel   = 3;
    const int      numBlocksPerRank   = 5;
    const int      numElementsPerRank = blockSize * numBlocksPerRank;

    // Build input data
    auto             myRank = myRankId();
    std::vector<int> vec;
    for (int value = myRank * numElementsPerRank; value < (myRank + 1) * numElementsPerRank; value++) {
        vec.push_back(value);
    }

    // Submit the data into the ReStore
    auto store = ReStoreVector<int>(blockSize, mpiComm, replicationLevel);
    store.submitData(vec);

    // Simulate failures
    EXIT_IF_FAILED(!_rankFailureManager.everyoneStillRunning());
    auto newComm = _rankFailureManager.failRanks({1, 3});
    EXIT_IF_FAILED(_rankFailureManager.iFailed());

    ASSERT_NE(myRankId(), 1);
    ASSERT_NE(myRankId(), 3);
    ASSERT_EQ(numRanks(newComm), 2);
    myRank = myRankId(newComm);
    ASSERT_NE(myRank, 2);
    ASSERT_NE(myRank, 3);

    // Update the communicator of the ReStore and fetch the missing data.
    store.updateComm(newComm);

    ReStoreVector<int>::BlockRangeToRestoreList newBlocksPerRank;
    newBlocksPerRank.push_back(std::make_pair(std::make_pair(5, 5), 0));
    newBlocksPerRank.push_back(std::make_pair(std::make_pair(15, 5), 2));

    store.restoreDataAppend(vec, newBlocksPerRank);

    // Check if we have the right data
    ASSERT_EQ(vec.size(), 20);
    if (myRank == 0) {
        ASSERT_THAT(vec, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19));
    } else if (myRank == 1) {
        ASSERT_THAT(vec, ElementsAre(20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39));
    } else {
        throw std::runtime_error("This test was designed with 4 ranks in mind.");
    }
}

int main(int argc, char** argv) {
    // Filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Set errorhandler to return so we have a chance to mitigate failures
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    if constexpr (SIMULATE_FAILURES) {
        // Add object that will finalize MPI on exit; Google Test owns this pointer
        ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);

        // Get the event listener list.
        ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

        // Remove default listener: the default printer and the default XML printer
        ::testing::TestEventListener* l = listeners.Release(listeners.default_result_printer());

        // Adds MPI listener; Google Test owns this pointer
        listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l, MPI_COMM_WORLD));
    }

    int result = RUN_ALL_TESTS();

    return result;
}
