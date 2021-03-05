#ifndef RESTORE_TEST_MOCKS_H
#define RESTORE_TEST_MOCKS_H

#include <optional>
#include <unordered_set>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "restore/mpi_context.hpp"

// This is the default MPIContext mock which can be used for unit tests. When called hundreds of thousands times, it'll
// be too slow.
class MPIContextMock {
    using original_rank_t = ReStoreMPI::original_rank_t;
    using current_rank_t  = ReStoreMPI::current_rank_t;

    public:
    MOCK_METHOD(original_rank_t, getOriginalRank, (const current_rank_t), (const));
    MOCK_METHOD(std::optional<current_rank_t>, getCurrentRank, (const original_rank_t), (const));
    MOCK_METHOD(original_rank_t, getMyOriginalRank, (), (const));
    MOCK_METHOD(current_rank_t, getMyCurrentRank, (), (const));
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

// This is a simple to use and simple to write implementation to be used with unit tests. If you want to successively
// kill ranks as the failrue simulator does, this is way to slow. Look at MPIContextFake if that's your usecase.
std::vector<ReStoreMPI::original_rank_t>
getAliveOnlyFake(std::vector<ReStoreMPI::original_rank_t> deadRanks, std::vector<ReStoreMPI::original_rank_t> ranks) {
    std::vector<ReStoreMPI::original_rank_t> aliveRanks;

    std::sort(ranks.begin(), ranks.end());
    std::sort(deadRanks.begin(), deadRanks.end());
    std::set_difference(
        ranks.begin(), ranks.end(), deadRanks.begin(), deadRanks.end(), std::inserter(aliveRanks, aliveRanks.begin()));

    return aliveRanks;
}

// This is a faster implementation of a MPIContext. It does not expect calls and is very inflexible. But we can use it
// in the failure simulator because it's a lot faster than MpiC
class MPIContextFake {
    public:
    std::vector<ReStoreMPI::current_rank_t>
    getOnlyAlive(const std::vector<ReStoreMPI::original_rank_t>& originalRanks) const {
        std::vector<ReStoreMPI::current_rank_t> aliveRanks;
        for (auto rankId: originalRanks) {
            if (_deadRanks.find(rankId) == _deadRanks.end()) {
                aliveRanks.push_back(rankId);
            }
        }
        return aliveRanks;
    }

    void killRank(ReStoreMPI::original_rank_t rankId) {
        _deadRanks.insert(rankId);
    }

    void resurrectRanks() {
        _deadRanks.clear();
    }

    size_t numFailed() const {
        return _deadRanks.size();
    }

    private:
    std::unordered_set<ReStoreMPI::original_rank_t> _deadRanks;
};

#endif // Include guard
