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

TEST(ReStoreTest, EndToEnd_ComplexDataType) {
    // Each rank submits different data. The replication level is set to 3. There are two rank failures.
    // We use a struct as a data type in this test case.

    struct AwesomeDataType {
        signed int   number;
        unsigned int absNumber;
        bool         divisibleByTwo;
        bool         divisibleByThree;

        AwesomeDataType(
            signed int _number, unsigned int _absNumber, bool _divisibleByTwo, bool _divisibleByThree) noexcept
            : number(_number),
              absNumber(_absNumber),
              divisibleByTwo(_divisibleByTwo),
              divisibleByThree(_divisibleByThree) {}
    };

    ReStore::ReStore<AwesomeDataType> store(MPI_COMM_WORLD, 3, ReStore::OffsetMode::constant, sizeof(int));
    std::vector<AwesomeDataType>      data;

    signed int myStart = (myRankId() - numRanks() / 2) * 1000;
    signed int myEnd   = myStart + 1000;
    for (int number: range(myStart, myEnd)) {
        data.emplace_back(number, abs(number), number % 2 == 0, number % 3 == 0);
    }

    unsigned counter = 0;
    store.submitBlocks(
        [](const auto& value, ReStore::SerializedBlockStoreStream& stream) {
            stream << value.number;
            stream << value.absNumber;
            stream << value.divisibleByTwo;
            stream << value.divisibleByThree;
        },
        [&counter, &data]() {
            auto ret = data.size() == counter
                           ? std::nullopt
                           : std::make_optional(ReStore::NextBlock<AwesomeDataType>({counter, data[counter]}));
            counter++; // We cannot put this in the above line, as we can't assume if the first argument of the pair is
                       // bound before or after the increment.
            return ret;
        },
        asserting_cast<ReStore::block_id_t>(numRanks()) * data.size());

    // Two failures
    constexpr int failingRank1 = 0;
    constexpr int failingRank2 = 1;
    failRank(failingRank1);
    failRank(failingRank2);
    ASSERT_NE(myRankId(), failingRank1);
    ASSERT_NE(myRankId(), failingRank2);

    // TODO @Demian Assert stuff
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
