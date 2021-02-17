#ifndef RESTORE_TEST_MOCKS_H
#define RESTORE_TEST_MOCKS_H

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "restore/mpi_context.hpp"

class MPIContextMock {
    using original_rank_t = ReStoreMPI::original_rank_t;
    using current_rank_t  = ReStoreMPI::current_rank_t;

    public:
    MOCK_METHOD(original_rank_t, getOriginalRank, (const current_rank_t), (const));
    MOCK_METHOD(current_rank_t, getCurrentRank, (const original_rank_t), (const));
    MOCK_METHOD(bool, isAlive, (const original_rank_t), (const));
    MOCK_METHOD(std::vector<original_rank_t>, getOnlyAlive, (const std::vector<original_rank_t>&), (const));
    MOCK_METHOD(ReStoreMPI::original_rank_t, numFailuresSinceReset, (), (const));
    MOCK_METHOD(
        std::vector<ReStoreMPI::RecvMessage>, SparseAllToAll, (const std::vector<ReStoreMPI::SendMessage>& messages),
        (const));
    MOCK_METHOD(
        std::vector<current_rank_t>, getAliveCurrentRanks, (const std::vector<original_rank_t>& originalRanks),
        (const));
};

std::vector<ReStoreMPI::original_rank_t>
getAliveOnlyFake(std::vector<ReStoreMPI::original_rank_t> deadRanks, std::vector<ReStoreMPI::original_rank_t> ranks) {
    std::vector<ReStoreMPI::original_rank_t> aliveRanks;

    std::sort(ranks.begin(), ranks.end());
    std::sort(deadRanks.begin(), deadRanks.end());
    std::set_difference(
        ranks.begin(), ranks.end(), deadRanks.begin(), deadRanks.end(), std::inserter(aliveRanks, aliveRanks.begin()));

    return aliveRanks;
}

#endif // Include guard