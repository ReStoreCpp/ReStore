#include <algorithm>
#include <cstddef>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <gtest/gtest_pred_impl.h>
#include <iterator>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "mocks.hpp"
#include "restore/block_retrieval.hpp"
#include "restore/common.hpp"
#include "restore/core.hpp"
#include "restore/mpi_context.hpp"

using namespace ::testing;

void testGetServingRanks(
    int numRanksToKill, size_t expectedLivingReplications, ReStore::block_id_t rangeStart, size_t rangeSize) {
    using BlockDistribution = ReStore::BlockDistribution<MPIContextMock>;
    using ReStore::OffsetMode;
    using ReStoreMPI::current_rank_t;
    using ReStoreMPI::RecvMessage;

    std::vector<ReStoreMPI::original_rank_t> deadRanks;
    for (int rank = 9; rank > 9 - numRanksToKill; --rank) {
        deadRanks.push_back(rank);
    }
    MPIContextMock mpiContext;
    EXPECT_CALL(mpiContext, getOnlyAlive(_))
        .WillRepeatedly([&deadRanks](std::vector<ReStoreMPI::original_rank_t> ranks) {
            return getAliveOnlyFake(deadRanks, ranks);
        });
    auto blockDistribution  = std::make_shared<BlockDistribution>(10, 100, 3, mpiContext);
    auto blockRange         = blockDistribution->rangeOfBlock(rangeStart);
    auto blockRangeExternal = std::pair<ReStore::block_id_t, size_t>(rangeStart, rangeSize);
    std::vector<ReStore::block_range_request_t> servingRanks;
    ReStore::getServingRanks(blockRange, blockRangeExternal, blockDistribution.get(), servingRanks);
    EXPECT_EQ(expectedLivingReplications, servingRanks.size());

    size_t avgSize = blockRangeExternal.second / servingRanks.size();
    std::sort(servingRanks.begin(), servingRanks.end());
    ReStore::block_id_t nextBlockId = rangeStart;
    for (const auto& servingRank: servingRanks) {
        EXPECT_LT(servingRank.second, 10 - numRanksToKill);
        auto size = servingRank.first.second;
        EXPECT_PRED2(
            [](size_t _avgSize, size_t _size) { return _size == _avgSize || _size == _avgSize + 1; }, avgSize, size);
        EXPECT_EQ(nextBlockId, servingRank.first.first);
        nextBlockId += servingRank.first.second;
    }
    EXPECT_EQ(rangeStart + rangeSize, nextBlockId);
}

TEST(BlockRetrievalTest, getServingRanks) {
    testGetServingRanks(0, 3, 1, 8);
}

TEST(BlockRetrievalTest, getServingRanksWithDeadRanks) {
    testGetServingRanks(4, 2, 1, 8);
}

TEST(BlockRetrievalTest, getServingRanksWithFewBlocks) {
    testGetServingRanks(0, 2, 1, 2);
}

TEST(BlockRetrievalTest, getServingRanksWithWholeBlockRange) {
    testGetServingRanks(0, 3, 0, 10);
}

TEST(BlockRetrievalTest, getServingRanksWithWholeBlockRangeAndDeadRanks) {
    testGetServingRanks(4, 2, 0, 10);
}

TEST(BlockRetrievalTest, getServingRanksDataLoss) {
    int                 numRanksToKill = 8;
    ReStore::block_id_t rangeStart     = 20;
    size_t              rangeSize      = 10;
    using BlockDistribution            = ReStore::BlockDistribution<MPIContextMock>;
    using ReStore::OffsetMode;
    using ReStoreMPI::current_rank_t;
    using ReStoreMPI::RecvMessage;

    std::vector<ReStoreMPI::original_rank_t> deadRanks;
    for (int rank = 9; rank > 9 - numRanksToKill; --rank) {
        deadRanks.push_back(rank);
    }
    MPIContextMock mpiContext;
    EXPECT_CALL(mpiContext, getOnlyAlive(_))
        .WillRepeatedly([&deadRanks](std::vector<ReStoreMPI::original_rank_t> ranks) {
            return getAliveOnlyFake(deadRanks, ranks);
        });
    auto blockDistribution  = std::make_shared<BlockDistribution>(10, 100, 3, mpiContext);
    auto blockRange         = blockDistribution->rangeOfBlock(rangeStart);
    auto blockRangeExternal = std::pair<ReStore::block_id_t, size_t>(rangeStart, rangeSize);
    std::vector<ReStore::block_range_request_t> servingRanks;
    EXPECT_THROW(
        ReStore::getServingRanks(blockRange, blockRangeExternal, blockDistribution.get(), servingRanks),
        ReStore::UnrecoverableDataLossException);
}

void testSendRecvBlockRanges(int numDeadRanks) {
    using BlockDistribution = ReStore::BlockDistribution<MPIContextMock>;
    using ReStore::OffsetMode;
    using ReStoreMPI::current_rank_t;
    using ReStoreMPI::RecvMessage;

    std::vector<ReStoreMPI::original_rank_t> deadRanks;
    for (int rank = 0; rank < numDeadRanks; ++rank) {
        deadRanks.push_back(rank);
    }

    MPIContextMock mpiContext;
    int            mySimulatedOriginalRank = numDeadRanks;
    int            mySimulatedCurrentRank  = 0;
    EXPECT_CALL(mpiContext, getOnlyAlive(_))
        .WillRepeatedly(
            [deadRanks](std::vector<ReStoreMPI::original_rank_t> ranks) { return getAliveOnlyFake(deadRanks, ranks); });
    EXPECT_CALL(mpiContext, getMyOriginalRank()).WillRepeatedly([mySimulatedOriginalRank]() noexcept {
        return mySimulatedOriginalRank;
    });
    EXPECT_CALL(mpiContext, getMyCurrentRank()).WillRepeatedly([mySimulatedCurrentRank]() noexcept {
        return mySimulatedCurrentRank;
    });
    EXPECT_CALL(mpiContext, getCurrentRank(_))
        .WillRepeatedly([numDeadRanks](ReStoreMPI::original_rank_t rank) noexcept { return rank - numDeadRanks; });
    EXPECT_CALL(mpiContext, getOriginalRank(_))
        .WillRepeatedly([numDeadRanks](ReStoreMPI::current_rank_t rank) noexcept { return rank + numDeadRanks; });

    ASSERT_EQ(mpiContext.getMyCurrentRank(), mpiContext.getCurrentRank(mySimulatedOriginalRank));
    ASSERT_EQ(mpiContext.getMyOriginalRank(), mpiContext.getOriginalRank(mySimulatedCurrentRank));

    auto blockDistribution = std::make_shared<BlockDistribution>(10, 100, 3, mpiContext);
    std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, current_rank_t>> requests = {
        std::make_pair(std::make_pair(1, 8), 0), std::make_pair(std::make_pair(1, 8), 1),
        std::make_pair(std::make_pair(10, 35), 0)};

    auto [sendBlockRanges, recvBlockRanges] =
        ReStore::getSendRecvBlockRanges(requests, blockDistribution.get(), mpiContext);

    for (const auto& sendBlockRange: sendBlockRanges) {
        EXPECT_EQ(
            blockDistribution->rangeOfBlock(sendBlockRange.first.first),
            blockDistribution->rangeOfBlock(sendBlockRange.first.first + sendBlockRange.first.second - 1));
        auto ranksWithBlock = blockDistribution->ranksBlockIsStoredOn(sendBlockRange.first.first);
        EXPECT_TRUE(
            std::find(ranksWithBlock.begin(), ranksWithBlock.end(), mySimulatedOriginalRank) != ranksWithBlock.end());
    }


    std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, current_rank_t>> recvRangesExpected;
    std::copy_if(
        requests.begin(), requests.end(), std::back_inserter(recvRangesExpected),
        [mySimulatedCurrentRank](const auto& item) { return item.second == mySimulatedCurrentRank; });
    EXPECT_EQ(2, recvRangesExpected.size());
    std::sort(recvBlockRanges.begin(), recvBlockRanges.end());
    size_t              indexInExpected = 0;
    ReStore::block_id_t nextBlockId     = recvRangesExpected[indexInExpected].first.first;
    for (const auto& recvBlockRange: recvBlockRanges) {
        ASSERT_LT(indexInExpected, recvRangesExpected.size());
        EXPECT_EQ(
            blockDistribution->rangeOfBlock(recvBlockRange.first.first),
            blockDistribution->rangeOfBlock(recvBlockRange.first.first + recvBlockRange.first.second - 1));
        auto ranksWithBlock = blockDistribution->ranksBlockIsStoredOn(recvBlockRange.first.first);
        EXPECT_TRUE(
            std::find(ranksWithBlock.begin(), ranksWithBlock.end(), mpiContext.getOriginalRank(recvBlockRange.second))
            != ranksWithBlock.end());
        EXPECT_EQ(nextBlockId, recvBlockRange.first.first);
        EXPECT_LE(
            recvBlockRange.first.first + recvBlockRange.first.second,
            recvRangesExpected[indexInExpected].first.first + recvRangesExpected[indexInExpected].first.second);
        nextBlockId += recvBlockRange.first.second;
        if (nextBlockId
            == recvRangesExpected[indexInExpected].first.first + recvRangesExpected[indexInExpected].first.second) {
            ++indexInExpected;
            if (indexInExpected < recvRangesExpected.size()) {
                nextBlockId = recvRangesExpected[indexInExpected].first.first;
            }
        }
    }
    EXPECT_EQ(nextBlockId, recvRangesExpected.back().first.first + recvRangesExpected.back().first.second);
}

TEST(BlockRetrievalTest, getSendRecvBlockRanges) {
    testSendRecvBlockRanges(0);
}

TEST(BlockRetrievalTest, getSendRecvBlockRangesWithDeadRanks) {
    testSendRecvBlockRanges(4);
}

TEST(BlockRetrievalTest, handleReceivedBlocks) {
    std::vector<ReStoreMPI::RecvMessage>                                                       recvMessages;
    std::vector<std::vector<int>>                                                              messageDataInt;
    std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, ReStoreMPI::current_rank_t>> recvBlockRanges;
    recvBlockRanges.emplace_back(std::make_pair(std::make_pair(5, 2), 0));
    recvBlockRanges.emplace_back(std::make_pair(std::make_pair(15, 3), 0));
    messageDataInt.push_back({-5, -6, -15, -16, -17});
    {
        std::vector<std::byte> messageDataByte(
            reinterpret_cast<std::byte*>(messageDataInt.back().data()),
            reinterpret_cast<std::byte*>(messageDataInt.back().data() + messageDataInt.back().size()));
        recvMessages.emplace_back(std::move(messageDataByte), 0);
    }

    recvBlockRanges.emplace_back(std::make_pair(std::make_pair(7, 2), 1));
    recvBlockRanges.emplace_back(std::make_pair(std::make_pair(20, 1), 1));
    messageDataInt.push_back({-7, -8, -20});
    {
        std::vector<std::byte> messageDataByte(
            reinterpret_cast<std::byte*>(messageDataInt.back().data()),
            reinterpret_cast<std::byte*>(messageDataInt.back().data() + messageDataInt.back().size()));
        recvMessages.emplace_back(std::move(messageDataByte), 1);
    }


    std::vector<int> allData;
    for (const auto& message: messageDataInt) {
        allData.insert(allData.end(), message.begin(), message.end());
    }
    size_t index = 0;
    ReStore::handleReceivedBlocks(
        recvMessages, recvBlockRanges, ReStore::OffsetMode::constant, sizeof(int),
        [&allData, &index](const std::byte* data, size_t size, ReStore::block_id_t id) {
            EXPECT_EQ(sizeof(int), size);
            const int* intData = reinterpret_cast<const int*>(data);
            int        intId   = static_cast<int>(id);
            EXPECT_EQ(-1 * intId, *intData);
            ASSERT_LT(index, allData.size());
            EXPECT_EQ(allData[index], *intData);
            ++index;
        },
        IdentityPermutation());
    EXPECT_EQ(allData.size(), index);
}
