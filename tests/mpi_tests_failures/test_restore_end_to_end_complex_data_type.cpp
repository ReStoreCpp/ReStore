#include <algorithm>
#include <assert.h>
#include <cstddef>
#include <memory>
#include <sstream>
#include <stdlib.h>
#include <utility>

#include "range.hpp"
#include <gmock/gmock.h>
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <mpi.h>

#include "restore/common.hpp"
#include "restore/core.hpp"
#include "restore/helpers.hpp"

#include "mpi_helpers.hpp"
#include "restore/mpi_context.hpp"
#include "test_with_failures_fixture.hpp"

using namespace ::testing;

using iter::range;

struct AwesomeDataType {
    signed int   number;
    unsigned int absNumber;
    bool         divisibleByTwo;
    bool         divisibleByThree;

    AwesomeDataType(signed int _number, unsigned int _absNumber, bool _divisibleByTwo, bool _divisibleByThree) noexcept
        : number(_number),
          absNumber(_absNumber),
          divisibleByTwo(_divisibleByTwo),
          divisibleByThree(_divisibleByThree) {}

    AwesomeDataType() noexcept : number(0), absNumber(0), divisibleByTwo(false), divisibleByThree(false) {}

    bool operator==(const AwesomeDataType& other) const {
        return number == other.number && absNumber == other.absNumber && divisibleByTwo == other.divisibleByTwo
               && divisibleByThree == other.divisibleByThree;
    }

    friend void PrintTo(const AwesomeDataType& obj, std::ostream* os) {
        *os << "AwesomeDataType(number=" << obj.number << ", absNumber=" << obj.absNumber
            << ", divisibleByTwo=" << obj.divisibleByTwo << ", divisibleByThree=" << obj.divisibleByThree << ")";
    }
};

TEST_F(ReStoreTestWithFailures, ComplexDataType) {
    // Each rank submits different data. The replication level is set to 3. There are two rank failures.
    // We use a struct as a data type in this test case.

    // The logic of this tests assumes that there are four ranks
    assert(numRanks() == 4);

    const uint8_t replicationLevel = 3;
    const size_t  constantOffset   = 10;
    const uint64_t blocksPerPermutationRange = 2;

    ReStore::ReStore<AwesomeDataType> store(
        MPI_COMM_WORLD, replicationLevel, ReStore::OffsetMode::constant, constantOffset, blocksPerPermutationRange);
    std::vector<AwesomeDataType> data;

    signed int myStart = (myRankId() - (numRanks() / 2)) * 1000;
    signed int myEnd   = myStart + 1000;

    for (int number = myStart; number < myEnd; number++) {
        data.emplace_back(number, abs(number), number % 2 == 0, number % 3 == 0);
    }

    std::vector<AwesomeDataType> allData;

    for (const auto& rank: range(numRanks())) {
        signed int start = (rank - (numRanks() / 2)) * 1000;
        signed int end   = start + 1000;
        for (int number = start; number < end; number++) {
            allData.emplace_back(number, abs(number), number % 2 == 0, number % 3 == 0);
        }
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
                           : std::make_optional(ReStore::NextBlock<AwesomeDataType>(
                               {counter + asserting_cast<ReStore::block_id_t>(myRankId()) * 1000, data[counter]}));
            counter++; // We cannot put this in the above line, as we can't assume if the first argument of the pair is
                       // bound before or after the increment.
            return ret;
        },
        asserting_cast<ReStore::block_id_t>(numRanks()) * data.size());

    // Two failures
    EXIT_IF_FAILED(!_rankFailureManager.everyoneStillRunning());
    auto newComm = _rankFailureManager.failRanks({1, 2});
    EXIT_IF_FAILED(_rankFailureManager.iFailed());
    EXPECT_NE(myRankId(), 1);
    EXPECT_NE(myRankId(), 2);
    EXPECT_EQ(numRanks(newComm), 2);

    store.updateComm(newComm);

    size_t              numBlocks              = asserting_cast<ReStore::block_id_t>(numRanks()) * data.size();
    size_t              numBlocksPerRank       = numBlocks / static_cast<size_t>(numRanks(newComm));
    size_t              numRanksWithMoreBlocks = numBlocks % static_cast<size_t>(numRanks(newComm));
    ReStore::block_id_t currentBlock           = 0;
    std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, ReStoreMPI::current_rank_t>> requests;
    for (int rank = 0; rank < numRanks(newComm); ++rank) {
        const size_t numBlockForThisRank = numBlocksPerRank + (static_cast<size_t>(rank) < numRanksWithMoreBlocks);
        ReStore::block_id_t start        = asserting_cast<ReStore::block_id_t>(rank) * numBlocksPerRank
                                    + std::min(numRanksWithMoreBlocks, asserting_cast<size_t>(myRankId(newComm)));
        requests.emplace_back(std::make_pair(std::make_pair(start, numBlockForThisRank), rank));
        currentBlock += numBlockForThisRank;
    }
    EXPECT_EQ(numBlocks, currentBlock);

    ReStore::block_id_t numBlocksToReceive =
        numBlocksPerRank + (static_cast<size_t>(myRankId(newComm)) < numRanksWithMoreBlocks);
    std::vector<AwesomeDataType> dataReceived(numBlocksToReceive);
    ReStore::block_id_t          firstBlockId = numBlocksPerRank * static_cast<size_t>(myRankId(newComm))
                                       + std::min(numRanksWithMoreBlocks, static_cast<size_t>(myRankId(newComm)));
    ReStore::block_id_t numBlocksReceived = 0;
    EXIT_IF_FAILED(!_rankFailureManager.everyoneStillRunning());
    store.pushBlocksCurrentRankIds(
        requests, [&dataReceived, &numBlocksReceived,
                   firstBlockId](const std::byte* dataPtr, size_t size, ReStore::block_id_t blockId) {
            ASSERT_GE(blockId, firstBlockId);
            size_t index = blockId - firstBlockId;
            ASSERT_LT(index, dataReceived.size());
            ++numBlocksReceived;
            ASSERT_EQ(sizeof(signed int) + sizeof(unsigned int) + sizeof(bool) + sizeof(bool), size);
            signed int   number         = *reinterpret_cast<const signed int*>(dataPtr);
            unsigned int absNumber      = *reinterpret_cast<const unsigned int*>(dataPtr + sizeof(int));
            bool         divisibleByTwo = *reinterpret_cast<const bool*>(dataPtr + sizeof(int) + sizeof(unsigned int));
            bool         divisibleByThree =
                *reinterpret_cast<const bool*>(dataPtr + sizeof(int) + sizeof(unsigned int) + sizeof(bool));
            dataReceived[index] = AwesomeDataType(number, absNumber, divisibleByTwo, divisibleByThree);
        });
    EXPECT_EQ(numBlocksToReceive, numBlocksReceived);

    for (const auto index: range(numBlocksPerRank)) {
        EXPECT_EQ(allData[index + firstBlockId], dataReceived[index]);
    }

#ifdef SIMULATE_FAILURES
    assert(numRanks(newComm) == 2);
#endif
}

int main(int argc, char** argv) {
    // Filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Set errorhandler to return so we have a chance to mitigate failures
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

#ifdef SIMULATE_FAILURES
    // Add object that will finalize MPI on exit; Google Test owns this pointer
    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);

    // Get the event listener list.
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

    // Remove default listener: the default printer and the default XML printer
    ::testing::TestEventListener* l = listeners.Release(listeners.default_result_printer());

    // Adds MPI listener; Google Test owns this pointer
    listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l, MPI_COMM_WORLD));
#endif

    int result = RUN_ALL_TESTS();

    return result;
}
