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
#include "restore/mpi_context.hpp"

using namespace ::testing;

using iter::range;

// Some MPI helpers
int myRankId() {
    int rankId;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    return rankId;
}

int numRanks() {
    int numRanks;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    return numRanks;
}

void failRank(int rankId) {
    if (myRankId() == rankId) {
        raise(SIGKILL);
    }
}

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

TEST(ReStoreTest, EndToEnd_Simple1) {
    // The most basic test base. Each rank submits exactly the same data. The replication level is set to one. There is
    // no rank failure.

    ReStore::ReStore<int> store(MPI_COMM_WORLD, 1, ReStore::OffsetMode::constant, sizeof(int));
    std::vector<int>      data{0, 1, 2, 3, 42, 1337};

    unsigned counter = 0;
    store.submitBlocks(
        [](const int& value, ReStore::SerializedBlockStoreStream stream) { stream << value; },
        [&counter, &data]() {
            auto ret = data.size() == counter
                           ? std::nullopt
                           : std::make_optional(ReStore::NextBlock<int>(
                               {counter + static_cast<size_t>(myRankId()) * data.size(), data[counter]}));
            counter++; // We cannot put this in the above line, as we can't assume if the first argument of the pair is
                       // bound before or after the increment.
            return ret;
        },
        data.size() * static_cast<size_t>(numRanks()));

    // No failure

    {
        std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, ReStoreMPI::current_rank_t>> requests;
        for (int rank = 0; rank < numRanks(); ++rank) {
            requests.emplace_back(std::make_pair(std::make_pair(0, data.size()), rank));
        }

        std::vector<int>    dataReceived;
        ReStore::block_id_t nextBlockId = 0;
        store.pushBlocks(
            requests,
            [&dataReceived, &nextBlockId](const std::byte* dataPtr, size_t size, ReStore::block_id_t blockId) {
                EXPECT_EQ(nextBlockId, blockId);
                ++nextBlockId;
                ASSERT_EQ(sizeof(int), size);
                dataReceived.emplace_back(*reinterpret_cast<const int*>(dataPtr));
            });
        EXPECT_EQ(data.size(), nextBlockId);

        ASSERT_EQ(data.size(), dataReceived.size());
        for (size_t i = 0; i < data.size(); ++i) {
            EXPECT_EQ(data[i], dataReceived[i]);
        }
    }

    {
        std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, ReStoreMPI::current_rank_t>> requests;
        for (int rank = 0; rank < numRanks(); ++rank) {
            requests.emplace_back(
                std::make_pair(std::make_pair(static_cast<size_t>(rank) * data.size(), data.size()), rank));
        }

        std::vector<int>    dataReceived;
        ReStore::block_id_t nextBlockId = static_cast<size_t>(myRankId()) * data.size();
        store.pushBlocks(
            requests,
            [&dataReceived, &nextBlockId](const std::byte* dataPtr, size_t size, ReStore::block_id_t blockId) {
                EXPECT_EQ(nextBlockId, blockId);
                ++nextBlockId;
                ASSERT_EQ(sizeof(int), size);
                dataReceived.emplace_back(*reinterpret_cast<const int*>(dataPtr));
            });
        EXPECT_EQ(static_cast<size_t>(myRankId()) * data.size() + data.size(), nextBlockId);

        ASSERT_EQ(data.size(), dataReceived.size());
        for (size_t i = 0; i < data.size(); ++i) {
            EXPECT_EQ(data[i], dataReceived[i]);
        }
    }
}

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

TEST(ReStoreTest, EndToEnd_SingleFailure) {
    // Each rank submits different data. The replication level is set to 3. There is a single rank failure.
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

    // One failure
    constexpr int failingRank = 1;
    failRank(failingRank);
    ASSERT_NE(myRankId(), failingRank);

    // TODO @Demian Assert stuff
}

TEST(ReStoreTest, EndToEnd_TwoFailures) {
    // Each rank submits different data. The replication level is set to 3. There are two rank failures.
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

    // Two failures
    constexpr int failingRank1 = 0;
    constexpr int failingRank2 = 1;
    failRank(failingRank1);
    failRank(failingRank2);
    ASSERT_NE(myRankId(), failingRank1);
    ASSERT_NE(myRankId(), failingRank2);

    // TODO @Demian Assert stuff
}

TEST(ReStoreTest, EndToEnd_IrrecoverableDataLoss) {
    // Each rank submits different data. The replication level is set to 2. There are two rank failures. Therefore, some
    // data should be irrecoverably lost.
    ReStore::ReStore<int> store(MPI_COMM_WORLD, 2, ReStore::OffsetMode::constant, sizeof(int));

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

    // Two failures
    constexpr int failingRank1 = 0;
    constexpr int failingRank2 = 1;
    failRank(failingRank1);
    failRank(failingRank2);
    ASSERT_NE(myRankId(), failingRank1);
    ASSERT_NE(myRankId(), failingRank2);

    // TODO @Demian Assert stuff
}

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
        [](const auto& value, ReStore::SerializedBlockStoreStream stream) {
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
        data.size());

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

    int result = RUN_ALL_TESTS();

    return result;
}
