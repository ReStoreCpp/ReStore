#include <algorithm>
#include <functional>
#include <sstream>

#include "itertools.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mpi.h>

#include "restore/core.hpp"
#include "restore/helpers.hpp"

#include "mocks.hpp"

using namespace ::testing;

class StoreTest : public ::testing::Environment {
    // You can remove any or all of the following functions if its body
    // is empty.

    StoreTest(){};

    virtual ~StoreTest() {
        // You can do clean-up work that doesn't throw exceptions here.
    }

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    virtual void SetUp() override {
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    virtual void TearDown() override {
        // Code here will be called immediately after each test (right
        // before the destructor).
    }

    // Objects declared here can be used by all tests in the test case for Foo.
};

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

TEST(ReStoreTest, submitBlocks) {
    return;
    ReStore::ReStore<int> store(MPI_COMM_WORLD, 1, ReStore::OffsetMode::constant, sizeof(int));
    unsigned              counter = 0;
    std::vector<int>      data{0, 1, 2, 3, 42, 1337};
    store.submitBlocks(
        [](const int& value, ReStore::SerializedBlockStoreStream stream) { stream << value; },
        [&counter, &data]() {
            auto ret =
                data.size() == counter ? std::nullopt : std::make_optional(std::make_pair(counter, data[counter]));
            counter++; // We cannot put this in the above line, as we can't assume if the first argument of the pair is
                       // bound before or after the increment.
            return ret;
        },
        data.size());
}

int main(int argc, char** argv) {
    // Filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int result = RUN_ALL_TESTS();

    return result;
}
