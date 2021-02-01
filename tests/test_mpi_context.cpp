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
    ReStoreMPI::MPIContext           context(comm);
    std::vector<ReStoreMPI::Message> sendMessages;
    for (int target = (rank + 2) % size; target != rank && target != (rank + 1) % size; target = (target + 2) % size) {
        sendMessages.emplace_back(
            ReStoreMPI::Message{std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(new int[2]{rank, target})),
                                2 * sizeof(int), static_cast<ReStoreMPI::current_rank_t>(target)});
    }
    auto receiveMessages = context.SparseAllToAll(sendMessages);

    std::vector<ReStoreMPI::Message> receiveMessagesExpected;
    for (int source = (rank - 2 + 2 * size) % size; source != rank && source != (rank - 1 + 2 * size) % size;
         source     = (source - 2 + 2 * size) % size) {
        receiveMessagesExpected.emplace_back(
            ReStoreMPI::Message{std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(new int[2]{source, rank})),
                                2 * sizeof(int), static_cast<ReStoreMPI::current_rank_t>(source)});
    }

    std::sort(
        receiveMessages.begin(), receiveMessages.end(),
        [](const ReStoreMPI::Message& lhs, const ReStoreMPI::Message& rhs) { return lhs.rank < rhs.rank; });
    std::sort(
        receiveMessagesExpected.begin(), receiveMessagesExpected.end(),
        [](const ReStoreMPI::Message& lhs, const ReStoreMPI::Message& rhs) { return lhs.rank < rhs.rank; });

    ASSERT_EQ(receiveMessagesExpected.size(), receiveMessages.size());
    for (size_t i = 0; i < receiveMessages.size(); ++i) {
        auto received = receiveMessages[i];
        auto expected = receiveMessagesExpected[i];
        EXPECT_EQ(expected.rank, received.rank);
        EXPECT_EQ(expected.size, received.size);
        EXPECT_EQ(2 * sizeof(int), received.size);
        int* dataReceived   = reinterpret_cast<int*>(received.data.get());
        int* dataExpected   = reinterpret_cast<int*>(expected.data.get());
        int  sourceExpected = dataExpected[0];
        int  sourceReceived = dataReceived[0];
        EXPECT_EQ(sourceExpected, sourceReceived);
        int targetExpected = dataExpected[1];
        int targetReceived = dataReceived[1];
        EXPECT_EQ(targetExpected, targetReceived);
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
    std::vector<ReStoreMPI::Message> sendMessages;
    for (int target = (rank + 2) % size; target != rank && target != (rank + 1) % size; target = (target + 2) % size) {
        sendMessages.emplace_back(
            ReStoreMPI::Message{std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(new int[2]{rank, target})),
                                2 * sizeof(int), static_cast<ReStoreMPI::current_rank_t>(target)});
    }
    auto receiveMessages = context.SparseAllToAll(sendMessages);

    std::vector<ReStoreMPI::Message> receiveMessagesExpected;
    for (int source = (rank - 2 + 2 * size) % size; source != rank && source != (rank - 1 + 2 * size) % size;
         source     = (source - 2 + 2 * size) % size) {
        receiveMessagesExpected.emplace_back(
            ReStoreMPI::Message{std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(new int[2]{source, rank})),
                                2 * sizeof(int), static_cast<ReStoreMPI::current_rank_t>(source)});
    }

    std::sort(
        receiveMessages.begin(), receiveMessages.end(),
        [](const ReStoreMPI::Message& lhs, const ReStoreMPI::Message& rhs) { return lhs.rank < rhs.rank; });
    std::sort(
        receiveMessagesExpected.begin(), receiveMessagesExpected.end(),
        [](const ReStoreMPI::Message& lhs, const ReStoreMPI::Message& rhs) { return lhs.rank < rhs.rank; });

    ASSERT_EQ(receiveMessagesExpected.size(), receiveMessages.size());
    for (size_t i = 0; i < receiveMessages.size(); ++i) {
        auto received = receiveMessages[i];
        auto expected = receiveMessagesExpected[i];
        EXPECT_EQ(expected.rank, received.rank);
        EXPECT_EQ(expected.size, received.size);
        EXPECT_EQ(2 * sizeof(int), received.size);
        int* dataReceived   = reinterpret_cast<int*>(received.data.get());
        int* dataExpected   = reinterpret_cast<int*>(expected.data.get());
        int  sourceExpected = dataExpected[0];
        int  sourceReceived = dataReceived[0];
        EXPECT_EQ(sourceExpected, sourceReceived);
        int targetExpected = dataExpected[1];
        int targetReceived = dataReceived[1];
        EXPECT_EQ(targetExpected, targetReceived);
    }
}

TEST(MPIContext, RankConversion) {
    int originalRank;
    int originalSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &originalRank);
    MPI_Comm_size(MPI_COMM_WORLD, &originalSize);
    ReStoreMPI::MPIContext context(MPI_COMM_WORLD);
    for (int rank = 0; rank < originalSize; ++rank) {
        EXPECT_EQ(
            static_cast<ReStoreMPI::original_rank_t>(rank),
            context.getOriginalRank(static_cast<ReStoreMPI::current_rank_t>(rank)));
        EXPECT_EQ(
            static_cast<ReStoreMPI::current_rank_t>(rank),
            context.getCurrentRank(static_cast<ReStoreMPI::original_rank_t>(rank)));
    }
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, originalRank == 1 || originalRank == 2, originalRank, &comm);
    int currentSize;
    MPI_Comm_size(comm, &currentSize);
    assert(currentSize >= 1);
    context.updateComm(comm);

    std::vector<ReStoreMPI::original_rank_t> currentToOriginal;
    if (originalRank == 1 || originalRank == 2) {
        if (originalSize >= 3)
            currentToOriginal = {ReStoreMPI::original_rank_t(1), ReStoreMPI::original_rank_t(2)};
        else
            currentToOriginal = {ReStoreMPI::original_rank_t(1)};
    } else {
        currentToOriginal.emplace_back(ReStoreMPI::original_rank_t(0));
        for (int rank = 3; rank < originalSize; ++rank) {
            currentToOriginal.emplace_back(static_cast<ReStoreMPI::original_rank_t>(rank));
        }
    }

    for (int rank = 0; rank < currentSize; ++rank) {
        EXPECT_EQ(currentToOriginal[rank], context.getOriginalRank(static_cast<ReStoreMPI::current_rank_t>(rank)));
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
            originalToCurrent.emplace_back(static_cast<ReStoreMPI::current_rank_t>(rank - 2));
        }
    }

    for (int rank = 0; rank < originalSize; ++rank) {
        EXPECT_EQ(originalToCurrent[rank], context.getCurrentRank(static_cast<ReStoreMPI::original_rank_t>(rank)));
        if (originalToCurrent[rank] == std::nullopt) {
            EXPECT_EQ(false, context.isAlive(static_cast<ReStoreMPI::original_rank_t>(rank)));
        } else {
            EXPECT_EQ(true, context.isAlive(static_cast<ReStoreMPI::original_rank_t>(rank)));
        }
    }

    std::vector<ReStoreMPI::original_rank_t> allRanksOriginal(originalSize);
    std::iota(allRanksOriginal.begin(), allRanksOriginal.end(), ReStoreMPI::original_rank_t(0));
    std::vector<ReStoreMPI::current_rank_t> allRanksCurrentExpected(currentSize);
    std::iota(allRanksCurrentExpected.begin(), allRanksCurrentExpected.end(), ReStoreMPI::current_rank_t(0));

    auto allRanksCurrent = context.getAliveCurrentRanks(allRanksOriginal);

    ASSERT_EQ(allRanksCurrentExpected.size(), allRanksCurrent.size());
    for (size_t i = 0; i < allRanksCurrentExpected.size(); ++i) {
        EXPECT_EQ(allRanksCurrentExpected[i], allRanksCurrent[i]);
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

    return 0;
}
