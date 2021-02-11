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


    /*
    std::vector<current_rank_t> getAliveCurrentRanks(const std::vector<original_rank_t>& originalRanks) const {
        return _rankManager.getAliveCurrentRanks(originalRanks);
    }

    std::vector<Message>
    SparseAllToAll(const std::vector<Message>& messages, const int tag = RESTORE_SPARSE_ALL_TO_ALL_TAG) const {
        return ReStoreMPI::SparseAllToAll(messages, _comm, tag);
    }
    */
};

#endif // Include guard