#include "restore/core.hpp"
#include "restore/helpers.hpp"
#include <gtest/gtest.h>

#include <mpi/mpi.h>

using namespace std;
class StoreTest : public ::testing::Environment {
    // You can remove any or all of the following functions if its body
    // is empty.

    StoreTest(){};

    virtual ~StoreTest() {
        // You can do clean-up work that doesn't throw exceptions here.
    }

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:

    virtual void SetUp() {
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    virtual void TearDown() {
        // Code here will be called immediately after each test (right
        // before the destructor).
    }

    // Objects declared here can be used by all tests in the test case for Foo.
};

TEST(StoreTest, ReStore_Constructor) {
    // Construction of a ReStore object
    ASSERT_NO_THROW(ReStore<int>(MPI_COMM_WORLD, 3, ReStore<int>::OffsetMode::lookUpTable));
    ASSERT_NO_THROW(ReStore<int>(MPI_COMM_WORLD, 3, ReStore<int>::OffsetMode::constant, sizeof(int)));

    ASSERT_ANY_THROW(ReStore<int>(MPI_COMM_WORLD, 3, ReStore<int>::OffsetMode::lookUpTable, sizeof(int)));
    ASSERT_ANY_THROW(ReStore<int>(MPI_COMM_WORLD, 3, ReStore<int>::OffsetMode::constant, 0));
    ASSERT_ANY_THROW(ReStore<int>(MPI_COMM_WORLD, 0, ReStore<int>::OffsetMode::lookUpTable));
    ASSERT_ANY_THROW(ReStore<int>(MPI_COMM_WORLD, 0, ReStore<int>::OffsetMode::constant, sizeof(int)));

    // TODO Test a replication level that is larger than the number of ranks
    // TODO Test a replication level that cannot be archived because of memory
    // constraints

    // Replication level and offset mode getters
    {
        auto store = ReStore<uint8_t>(MPI_COMM_WORLD, 3, ReStore<uint8_t>::OffsetMode::constant, sizeof(uint8_t));
        ASSERT_EQ(store.replicationLevel(), 3);
        auto [offsetMode, constOffset] = store.offsetMode();
        ASSERT_EQ(offsetMode, ReStore<uint8_t>::OffsetMode::constant);
        ASSERT_EQ(constOffset, sizeof(uint8_t));
    }

    {
        auto store = ReStore<uint8_t>(MPI_COMM_WORLD, 10, ReStore<uint8_t>::OffsetMode::lookUpTable);
        ASSERT_EQ(store.replicationLevel(), 10);
        auto [offsetMode, constOffset] = store.offsetMode();
        ASSERT_EQ(offsetMode, ReStore<uint8_t>::OffsetMode::lookUpTable);
        ASSERT_EQ(constOffset, 0);
    }
}

TEST(StoreTest, ReStore_submitBlocks) {
    auto store         = ReStore<uint8_t>(MPI_COMM_WORLD, 3, ReStore<uint8_t>::OffsetMode::constant, sizeof(uint8_t));
    auto serializeFunc = [](const ReStore<uint8_t>::block_id_t& block, void* buffer) {
        UNUSED(block);
        UNUSED(buffer);
        return size_t(1);
    };
    auto nextBlock = []() { return std::optional<std::pair<ReStore<uint8_t>::block_id_t, const uint8_t&>>(); };
    store.submitBlocks(serializeFunc, nextBlock);
}
