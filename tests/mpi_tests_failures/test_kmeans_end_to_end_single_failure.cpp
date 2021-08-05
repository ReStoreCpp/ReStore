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

TEST_F(kMeansTestWithFailures, SingleFailure) {
    // Each rank submits different data. The replication level is set to 3. There is a single rank failure.
    const uint16_t REPLICATION_LEVEL = 3;

    // Initialize the MPI context
    ReStoreMPI::MPIContext mpiContext(MPI_COMM_WORLD);

    // Initialize data
    std::vector<double> data;
    switch (mpiContext.getMyCurrentRank()) {
        case 0:
            data = {1, 1};
            break;
        case 1:
            data = {2, 10};
            break;
        case 2:
            data = {13, 18};
            break;
        case 3:
            data = {20, 21};
            break;
        default:
            assert(false && "Invalid number of ranks for this test");
    }

    // Initialize the K-Means algorithm
    auto kmeansInstance = kMeansAlgorithm(std::move(data), 1, mpiContext, REPLICATION_LEVEL);
    kmeansInstance.setCenters({5, 17});
    ASSERT_THAT(kmeansInstance.centers(), ElementsAre(5, 17));

    // Perform one iteration before the failure
    kmeansInstance.performIterations(1);
    ASSERT_THAT(kmeansInstance.pointToCenterAssignment().numPointsAssignedToCenter, ElementsAre(4, 4));
    ASSERT_THAT(kmeansInstance.centers(), ElementsAre(3.5, 18));

    // Generate or simulate one failure
    EXIT_IF_FAILED(!_rankFailureManager.everyoneStillRunning());
    auto newComm = _rankFailureManager.failRanks({1});
    EXIT_IF_FAILED(_rankFailureManager.iFailed());
    EXPECT_NE(myRankId(), 1);
    EXPECT_EQ(numRanks(newComm), 3);
#ifdef SIMULATE_FAILURES
    mpiContext.simulateFailure(newComm);
#endif

    // Perform one iteration after the failure
    kmeansInstance.performIterations(1);
    //ASSERT_THAT(kmeansInstance.pointToCenterAssignment().numPointsAssignedToCenter, ElementsAre(4, 4));
    ASSERT_THAT(kmeansInstance.centers(), ElementsAre(3.5, 18));

    // Check the result
    auto clusterAssignments = kmeansInstance.collectClusterAssignments();
    if (mpiContext.getMyCurrentRank() == 0) {
        ASSERT_THAT(clusterAssignments, UnorderedElementsAre(0, 0, 0, 0, 1, 1, 1, 1));
    }
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
