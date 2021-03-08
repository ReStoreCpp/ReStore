#include <algorithm>
#include <functional>
#include <signal.h>
#include <sstream>

#include "itertools.hpp"
#include <gmock/gmock.h>
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

TEST(ReStoreTest, EndToEnd_Simple2) {
    // Each rank submits different data. The replication level is set to 3. There is no rank failure.
    ReStore::ReStore<int> store(MPI_COMM_WORLD, 3, ReStore::OffsetMode::constant, sizeof(int));

    std::vector<int> data;
    for (int value: range(1000 * myRankId(), 1000 * myRankId() + 1000)) {
        data.push_back(value);
    }

    unsigned counter = 0;
    store.submitBlocks(
        [](const int& value, ReStore::SerializedBlockStoreStream stream) { stream << value; },
        [&counter, &data]() {
            auto ret = data.size() == counter ? std::nullopt
                                              : std::make_optional(ReStore::NextBlock<int>({counter, data[counter]}));
            counter++; // We cannot put this in the above line, as we can't assume if the first argument of the pair is
                       // bound before or after the increment.
            return ret;
        },
        data.size());

    // No failure

    // TODO @Demian Assert stuff
}

int main(int argc, char** argv) {
    // Filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Set errorhandler to return so we have a chance to mitigate failures
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    int result = RUN_ALL_TESTS();

    return result;
}
