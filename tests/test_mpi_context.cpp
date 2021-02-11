#include <bits/stdint-uintn.h>
#include <cassert>
#include <cstdint>
#include <gtest-mpi-listener/include/gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <mpi.h>
#include <mpi/mpi.h>
#include <numeric>
#include <optional>
#include <vector>

#include "restore/mpi_context.hpp"

TEST(MPIContext, SparseAllToAll) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int      rank;
    int      size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    ReStoreMPI::MPIContext               context(comm);
    std::vector<ReStoreMPI::SendMessage> sendMessages;
    for (int target = (rank + 2) % size; target != rank && target != (rank + 1) % size; target = (target + 2) % size) {
        sendMessages.emplace_back(
            ReStoreMPI::SendMessage{reinterpret_cast<const uint8_t*>(new const int[2]{rank, target}), 2 * sizeof(int),
                                    static_cast<ReStoreMPI::current_rank_t>(target)});
    }
    auto receiveMessages = context.SparseAllToAll(sendMessages);

    std::vector<ReStoreMPI::RecvMessage> receiveMessagesExpected;
    for (int source = (rank - 2 + 2 * size) % size; source != rank && source != (rank - 1 + 2 * size) % size;
         source     = (source - 2 + 2 * size) % size) {
        int                  intMes[] = {source, rank};
        auto                 uint8Mes = reinterpret_cast<uint8_t*>(intMes);
        std::vector<uint8_t> mes(uint8Mes, uint8Mes + 2 * sizeof(int));
        receiveMessagesExpected.emplace_back(
            ReStoreMPI::RecvMessage{mes, static_cast<ReStoreMPI::current_rank_t>(source)});
    }

    std::sort(
        receiveMessages.begin(), receiveMessages.end(),
        [](const ReStoreMPI::RecvMessage& lhs, const ReStoreMPI::RecvMessage& rhs) {
            return lhs.srcRank < rhs.srcRank;
        });
    std::sort(
        receiveMessagesExpected.begin(), receiveMessagesExpected.end(),
        [](const ReStoreMPI::RecvMessage& lhs, const ReStoreMPI::RecvMessage& rhs) {
            return lhs.srcRank < rhs.srcRank;
        });

    ASSERT_EQ(receiveMessagesExpected.size(), receiveMessages.size());
    for (size_t i = 0; i < receiveMessages.size(); ++i) {
        auto received = receiveMessages[i];
        auto expected = receiveMessagesExpected[i];
        EXPECT_EQ(expected.srcRank, received.srcRank);
        EXPECT_EQ(expected.data.size(), received.data.size());
        EXPECT_EQ(2 * sizeof(int), received.data.size());
        int* dataReceived   = reinterpret_cast<int*>(received.data.data());
        int* dataExpected   = reinterpret_cast<int*>(expected.data.data());
        int  sourceExpected = dataExpected[0];
        int  sourceReceived = dataReceived[0];
        EXPECT_EQ(sourceExpected, sourceReceived);
        int targetExpected = dataExpected[1];
        int targetReceived = dataReceived[1];
        EXPECT_EQ(targetExpected, targetReceived);
    }
    for (auto message: sendMessages) {
        delete[] reinterpret_cast<const int*>(message.data);
    }
}

TEST(MPIContext, SparseAllToAllSmallerComm) {
    MPI_Comm comm = MPI_COMM_WORLD;
    int      rank;
    int      size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    ReStoreMPI::MPIContext context(comm);
    MPI_Comm_split(MPI_COMM_WORLD, rank == 0, rank, &comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    context.updateComm(comm);
    std::vector<ReStoreMPI::SendMessage> sendMessages;
    for (int target = (rank + 2) % size; target != rank && target != (rank + 1) % size; target = (target + 2) % size) {
        sendMessages.emplace_back(
            ReStoreMPI::SendMessage{reinterpret_cast<const uint8_t*>(new const int[2]{rank, target}), 2 * sizeof(int),
                                    static_cast<ReStoreMPI::current_rank_t>(target)});
    }
    auto receiveMessages = context.SparseAllToAll(sendMessages);

    std::vector<ReStoreMPI::RecvMessage> receiveMessagesExpected;
    for (int source = (rank - 2 + 2 * size) % size; source != rank && source != (rank - 1 + 2 * size) % size;
         source     = (source - 2 + 2 * size) % size) {
        int                  intMes[] = {source, rank};
        auto                 uint8Mes = reinterpret_cast<uint8_t*>(intMes);
        std::vector<uint8_t> mes(uint8Mes, uint8Mes + 2 * sizeof(int));
        receiveMessagesExpected.emplace_back(ReStoreMPI::RecvMessage{mes, source});
    }

    std::sort(
        receiveMessages.begin(), receiveMessages.end(),
        [](const ReStoreMPI::RecvMessage& lhs, const ReStoreMPI::RecvMessage& rhs) {
            return lhs.srcRank < rhs.srcRank;
        });
    std::sort(
        receiveMessagesExpected.begin(), receiveMessagesExpected.end(),
        [](const ReStoreMPI::RecvMessage& lhs, const ReStoreMPI::RecvMessage& rhs) {
            return lhs.srcRank < rhs.srcRank;
        });

    ASSERT_EQ(receiveMessagesExpected.size(), receiveMessages.size());
    for (size_t i = 0; i < receiveMessages.size(); ++i) {
        auto received = receiveMessages[i];
        auto expected = receiveMessagesExpected[i];
        EXPECT_EQ(expected.srcRank, received.srcRank);
        EXPECT_EQ(expected.data.size(), received.data.size());
        EXPECT_EQ(2 * sizeof(int), received.data.size());
        int* dataReceived   = reinterpret_cast<int*>(received.data.data());
        int* dataExpected   = reinterpret_cast<int*>(expected.data.data());
        int  sourceExpected = dataExpected[0];
        int  sourceReceived = dataReceived[0];
        EXPECT_EQ(sourceExpected, sourceReceived);
        int targetExpected = dataExpected[1];
        int targetReceived = dataReceived[1];
        EXPECT_EQ(targetExpected, targetReceived);
    }
    for (auto message: sendMessages) {
        delete[] reinterpret_cast<const int*>(message.data);
    }
}


TEST(MPIContext, RankConversion) {
    int originalRank;
    int originalSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &originalRank);
    MPI_Comm_size(MPI_COMM_WORLD, &originalSize);
    ReStoreMPI::MPIContext context(MPI_COMM_WORLD);
    ASSERT_EQ(4, originalSize);
    for (int rank = 0; rank < originalSize; ++rank) {
        EXPECT_EQ(static_cast<ReStoreMPI::original_rank_t>(rank), context.getOriginalRank(rank));
        EXPECT_EQ(static_cast<ReStoreMPI::current_rank_t>(rank), context.getCurrentRank(rank));
    }
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, originalRank == 1 || originalRank == 2, originalRank, &comm);
    int currentSize;
    int currentRank;
    MPI_Comm_size(comm, &currentSize);
    MPI_Comm_rank(comm, &currentRank);
    assert(currentSize >= 1);
    context.updateComm(comm);

    EXPECT_EQ(originalSize, context.getOriginalSize());
    EXPECT_EQ(originalRank, context.getMyOriginalRank());
    EXPECT_EQ(currentSize, context.getCurrentSize());
    EXPECT_EQ(currentRank, context.getMyCurrentRank());

    std::vector<ReStoreMPI::original_rank_t> currentToOriginal;
    if (originalRank == 1 || originalRank == 2) {
        if (originalSize >= 3)
            currentToOriginal = {ReStoreMPI::original_rank_t(1), ReStoreMPI::original_rank_t(2)};
        else
            currentToOriginal = {ReStoreMPI::original_rank_t(1)};
    } else {
        currentToOriginal.emplace_back(ReStoreMPI::original_rank_t(0));
        for (int rank = 3; rank < originalSize; ++rank) {
            currentToOriginal.emplace_back(rank);
        }
    }

    for (int rank = 0; rank < currentSize; ++rank) {
        EXPECT_EQ(currentToOriginal[(size_t)rank], context.getOriginalRank(rank));
    }

    std::vector<std::optional<ReStoreMPI::current_rank_t>> originalToCurrent;
    if (originalRank == 1 || originalRank == 2) {
        originalToCurrent.emplace_back(std::nullopt);
        if (originalSize >= 2)
            originalToCurrent.emplace_back(ReStoreMPI::current_rank_t(0));
        if (originalSize >= 3)
            originalToCurrent.emplace_back(ReStoreMPI::current_rank_t(1));
        for (int rank = 3; rank < originalSize; ++rank) {
            originalToCurrent.emplace_back(std::nullopt);
        }
    } else {
        originalToCurrent.emplace_back(ReStoreMPI::current_rank_t(0));
        if (originalSize >= 2)
            originalToCurrent.emplace_back(std::nullopt);
        if (originalSize >= 3)
            originalToCurrent.emplace_back(std::nullopt);
        for (int rank = 3; rank < originalSize; ++rank) {
            originalToCurrent.emplace_back(rank - 2);
        }
    }

    for (int rank = 0; rank < originalSize; ++rank) {
        EXPECT_EQ(originalToCurrent[(size_t)rank], context.getCurrentRank(rank));
        if (originalToCurrent[(size_t)rank] == std::nullopt) {
            EXPECT_EQ(false, context.isAlive(rank));
        } else {
            EXPECT_EQ(true, context.isAlive(rank));
        }
    }

    std::vector<ReStoreMPI::original_rank_t> allRanksOriginal((size_t)originalSize);
    std::iota(allRanksOriginal.begin(), allRanksOriginal.end(), ReStoreMPI::original_rank_t(0));
    std::vector<ReStoreMPI::current_rank_t> allRanksCurrentExpected((size_t)currentSize);
    std::iota(allRanksCurrentExpected.begin(), allRanksCurrentExpected.end(), ReStoreMPI::current_rank_t(0));

    auto allRanksCurrent = context.getAliveCurrentRanks(allRanksOriginal);

    ASSERT_EQ(allRanksCurrentExpected.size(), allRanksCurrent.size());
    for (size_t i = 0; i < allRanksCurrentExpected.size(); ++i) {
        EXPECT_EQ(allRanksCurrentExpected[i], allRanksCurrent[i]);
    }

    auto allAliveRanks = context.getOnlyAlive(allRanksOriginal);

    ASSERT_EQ(currentToOriginal.size(), allAliveRanks.size());
    for (size_t i = 0; i < currentToOriginal.size(); ++i) {
        EXPECT_EQ(currentToOriginal[i], allAliveRanks[i]);
    }

    context.resetOriginalCommToCurrentComm();
    // Do it twice to make sure that freeing the original group does not also destroy the current group
    context.resetOriginalCommToCurrentComm();

    for (ReStoreMPI::current_rank_t rank = 0; rank < currentSize; ++rank) {
        EXPECT_EQ((ReStoreMPI::original_rank_t)rank, context.getOriginalRank(rank));
    }
}

int main(int argc, char** argv) {
    // Filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Add object that will finalize MPI on exit; Google Test owns this pointer
    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);

    // Get the event listener list.
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

    // Remove default listener: the default printer and the default XML printer
    ::testing::TestEventListener* l = listeners.Release(listeners.default_result_printer());

    // Adds MPI listener; Google Test owns this pointer
    listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l, MPI_COMM_WORLD));

    // Run tests, then clean up and exit. RUN_ALL_TESTS() returns 0 if all tests
    // pass and 1 if some test fails.
    int result = RUN_ALL_TESTS();

    return result;
}
