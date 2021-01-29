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
    auto nextBlock = []() {
        return std::optional<std::pair<ReStore<uint8_t>::block_id_t, const uint8_t&>>();
    };
    store.submitBlocks(serializeFunc, nextBlock);
}

TEST(StoreTest, ReStore_BlockDistribution) {
    // BlockRange(size_t range_id, size_t numBlocks, size_t numRanges) {
    // bool includes(block_id_t block) {
    typedef ReStore<u_int8_t>::BlockDistribution::BlockRange BlockRange;

    ASSERT_ANY_THROW(BlockRange(100, 1, 10)); // Range id greater than the number of ranges
    ASSERT_ANY_THROW(BlockRange(0, 1, 2));    // More ranges than blocks

    {
        auto range = BlockRange(0, 100, 10);
        ASSERT_EQ(range.id, 0);
        ASSERT_EQ(range.start, 0);
        ASSERT_EQ(range.length, 10);

        for (size_t blockId = 0; blockId < 10; blockId++) {
            ASSERT_TRUE(range.contains(blockId));
        }
        for (size_t blockId = 10; blockId < 100; blockId++) {
            ASSERT_FALSE(range.contains(blockId));
        }

        ASSERT_FALSE(range.contains(101));
    }

    {
        auto range = BlockRange(2, 100, 10);
        ASSERT_EQ(range.id, 2);
        ASSERT_EQ(range.start, 20);
        ASSERT_EQ(range.length, 10);

        for (size_t blockId = 0; blockId < 20; blockId++) {
            ASSERT_FALSE(range.contains(blockId));
        }
        for (size_t blockId = 20; blockId < 30; blockId++) {
            ASSERT_TRUE(range.contains(blockId));
        }
        for (size_t blockId = 30; blockId < 100; blockId++) {
            ASSERT_FALSE(range.contains(blockId));
        }

        ASSERT_FALSE(range.contains(101));
    }

    {
        // range 0: 0, 1, 2, 3
        // range 1: 4, 5, 6
        // range 2: 7, 8, 9
        auto range0 = BlockRange(0, 10, 3);
        ASSERT_TRUE(range0.contains(0));
        ASSERT_TRUE(range0.contains(1));
        ASSERT_TRUE(range0.contains(2));
        ASSERT_TRUE(range0.contains(3));
        ASSERT_FALSE(range0.contains(4));
        ASSERT_FALSE(range0.contains(5));
        ASSERT_FALSE(range0.contains(6));
        ASSERT_FALSE(range0.contains(7));
        ASSERT_FALSE(range0.contains(8));
        ASSERT_FALSE(range0.contains(9));

        auto range1 = BlockRange(1, 10, 3);
        ASSERT_FALSE(range1.contains(0));
        ASSERT_FALSE(range1.contains(1));
        ASSERT_FALSE(range1.contains(2));
        ASSERT_FALSE(range1.contains(3));
        ASSERT_TRUE(range1.contains(4));
        ASSERT_TRUE(range1.contains(5));
        ASSERT_TRUE(range1.contains(6));
        ASSERT_FALSE(range1.contains(7));
        ASSERT_FALSE(range1.contains(8));
        ASSERT_FALSE(range1.contains(9));

        auto range2 = BlockRange(2, 10, 3);
        ASSERT_FALSE(range2.contains(0));
        ASSERT_FALSE(range2.contains(1));
        ASSERT_FALSE(range2.contains(2));
        ASSERT_FALSE(range2.contains(3));
        ASSERT_FALSE(range2.contains(4));
        ASSERT_FALSE(range2.contains(5));
        ASSERT_FALSE(range2.contains(6));
        ASSERT_TRUE(range2.contains(7));
        ASSERT_TRUE(range2.contains(8));
        ASSERT_TRUE(range2.contains(9));
    }

    {
        // range0: 0, 1
        // range1: 2, 3
        // range2: 4
        // range3: 5
        auto range0 = BlockRange(0, 6, 4);
        ASSERT_TRUE(range0.contains(0));
        ASSERT_TRUE(range0.contains(1));
        ASSERT_FALSE(range0.contains(2));
        ASSERT_FALSE(range0.contains(3));
        ASSERT_FALSE(range0.contains(4));
        ASSERT_FALSE(range0.contains(5));

        auto range1 = BlockRange(1, 6, 4);
        ASSERT_FALSE(range1.contains(0));
        ASSERT_FALSE(range1.contains(1));
        ASSERT_TRUE(range1.contains(2));
        ASSERT_TRUE(range1.contains(3));
        ASSERT_FALSE(range1.contains(4));
        ASSERT_FALSE(range1.contains(5));

        auto range2 = BlockRange(2, 6, 4);
        ASSERT_FALSE(range2.contains(0));
        ASSERT_FALSE(range2.contains(1));
        ASSERT_FALSE(range2.contains(2));
        ASSERT_FALSE(range2.contains(3));
        ASSERT_TRUE(range2.contains(4));
        ASSERT_FALSE(range2.contains(5));

        auto range3 = BlockRange(3, 6, 4);
        ASSERT_FALSE(range3.contains(0));
        ASSERT_FALSE(range3.contains(1));
        ASSERT_FALSE(range3.contains(2));
        ASSERT_FALSE(range3.contains(3));
        ASSERT_FALSE(range3.contains(4));
        ASSERT_TRUE(range3.contains(5));
    }
}

int main(int argc, char** argv) {
    // Filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);

    int result = RUN_ALL_TESTS();

    return result;
}
