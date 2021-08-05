#include <cstddef>
#include <gmock/gmock.h>
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>

#include "k-means.hpp"
#include "restore/mpi_context.hpp"
#include "test_with_failures_fixture.hpp"

using namespace ::testing;
using namespace kmeans;
using namespace ReStoreMPI;

const uint16_t REPLICATION_LEVEL = 3;

TEST_F(kMeansTestWithFailures, updateCentersThrowsOnFailure) {
    using MPIContext = ReStoreMPI::MPIContext;
    MPIContext mpiContext(MPI_COMM_WORLD);

    // All ranks supply the same data.
    auto kmeansInstance = kMeansAlgorithm<float, MPIContext>({0, 0, 1, 1}, 1, mpiContext, REPLICATION_LEVEL);
    // One failure
    EXIT_IF_FAILED(!_rankFailureManager.everyoneStillRunning());
    auto newComm = _rankFailureManager.failRanks({1});
    EXIT_IF_FAILED(_rankFailureManager.iFailed());
    EXPECT_NE(myRankId(), 1);
    EXPECT_EQ(numRanks(newComm), 3);

    // setCenters() is local
    ASSERT_NO_THROW((kmeansInstance.setCenters({0, 1})));
    ASSERT_THAT(kmeansInstance.centers(), ElementsAre(0, 1));

    // assignPointsToCenters is local
    ASSERT_NO_THROW(kmeansInstance.assignPointsToCenters());
    ASSERT_THAT(kmeansInstance.pointToCenterAssignment().assignedCenter, ElementsAre(0, 0, 1, 1));
    ASSERT_THAT(kmeansInstance.pointToCenterAssignment().numPointsAssignedToCenter, ElementsAre(2, 2));

#ifndef SIMULATE_FAILURES
    // Update centers should fail as a rank failure is detected
    ASSERT_THROW(kmeansInstance.updateCenters(), ReStoreMPI::FaultException);
#endif
}

int main(int argc, char** argv) {
    // Filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Set errorhandler to return so we have a chance to mitigate failures
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    // These tests have been designed with 4 ranks in mind
    int numRanks = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    if (numRanks != 4) {
        throw std::runtime_error("Please run these tests with 4 mpi ranks.");
    }

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
