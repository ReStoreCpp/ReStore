
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
#include "test_with_failures_fixture.hpp"

using namespace ::testing;

using iter::range;

TEST(ReStoreTest, Constructor) {
    // Construction of a ReStore object
    ASSERT_NO_THROW(ReStore::ReStore<int>(MPI_COMM_WORLD, 3, ReStore::OffsetMode::lookUpTable));
    ASSERT_NO_THROW(ReStore::ReStore<int>(MPI_COMM_WORLD, 3, ReStore::OffsetMode::constant, sizeof(int)));

    ASSERT_ANY_THROW(ReStore::ReStore<int>(MPI_COMM_WORLD, 3, ReStore::OffsetMode::lookUpTable, sizeof(int)));
    ASSERT_ANY_THROW(ReStore::ReStore<int>(MPI_COMM_WORLD, 3, ReStore::OffsetMode::constant, 0));
    ASSERT_ANY_THROW(ReStore::ReStore<int>(MPI_COMM_WORLD, 0, ReStore::OffsetMode::lookUpTable));
    ASSERT_ANY_THROW(ReStore::ReStore<int>(MPI_COMM_WORLD, 0, ReStore::OffsetMode::constant, sizeof(int)));

    // TODO Test a replication level that is larger than the number of ranks
    // TODO Test a replication level that cannot be archived because of memory
    // constraints

    // Replication level and offset mode getters
    {
        auto store = ReStore::ReStore<uint8_t>(MPI_COMM_WORLD, 3, ReStore::OffsetMode::constant, sizeof(uint8_t));
        ASSERT_EQ(store.replicationLevel(), 3);
        auto [offsetMode, constOffset] = store.offsetMode();
        ASSERT_EQ(offsetMode, ReStore::OffsetMode::constant);
        ASSERT_EQ(constOffset, sizeof(uint8_t));
    }

    {
        auto store = ReStore::ReStore<uint8_t>(MPI_COMM_WORLD, 10, ReStore::OffsetMode::lookUpTable);
        ASSERT_EQ(store.replicationLevel(), 10);
        auto [offsetMode, constOffset] = store.offsetMode();
        ASSERT_EQ(offsetMode, ReStore::OffsetMode::lookUpTable);
        ASSERT_EQ(constOffset, 0);
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
