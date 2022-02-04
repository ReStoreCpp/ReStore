#ifndef EQUAL_LOAD_BALANCER_H
#define EQUAL_LOAD_BALANCER_H

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "restore/block_retrieval.hpp"
#include "restore/helpers.hpp"
#include "restore/mpi_context.hpp"

namespace ReStore {
class EqualLoadBalancer {
    public:
    EqualLoadBalancer(
        const std::vector<std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::original_rank_t>>& blockRanges,
        const ReStoreMPI::original_rank_t                                                         numRanksOriginal)
        : _blockRanges(blockRanges),
          _ranksisAlive(asserting_cast<size_t>(numRanksOriginal), true),
          numAliveRanks(asserting_cast<size_t>(numRanksOriginal)) {
        assert(_ranksisAlive.size() == numAliveRanks);
        assert(asserting_cast<size_t>(std::count(_ranksisAlive.begin(), _ranksisAlive.end(), true)) == numAliveRanks);

        assert(std::is_sorted(blockRanges.begin(), blockRanges.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.second < rhs.second;
        }));

        _blockOffsetsPerRound.emplace_back(numRanksOriginal, std::numeric_limits<size_t>::max());

        for (size_t i = 0; i < blockRanges.size(); ++i) {
            size_t rank = asserting_cast<size_t>(_blockRanges[i].second);
            assert(_blockRanges[i].second < numRanksOriginal);
            assert(rank == 0 || _blockOffsetsPerRound.back()[rank - 1] != std::numeric_limits<size_t>::max());
            _blockOffsetsPerRound.back()[rank] = std::min(i, _blockOffsetsPerRound.back()[rank]);
        }
    }

    std::vector<std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::original_rank_t>>
    getNewBlocksAfterFailureForPushBlocks(const std::vector<ReStoreMPI::original_rank_t>& diedRanks) {
        if (diedRanks.empty()) {
            return {};
        }
        _previousReturnedBlockRanges.clear();
        _previousDiedRanksVector = diedRanks;
        for (const auto diedRank: diedRanks) {
            assert(_ranksisAlive[asserting_cast<size_t>(diedRank)] == true);
            _ranksisAlive[asserting_cast<size_t>(diedRank)] = false;
            --numAliveRanks;
        }

        // Store for each of the dead ranks how many blocks it used to hold and which blockRanges these corresponded to
        std::vector<size_t> blockRangeIndices;
        size_t              numBlocks = 0;
        for (size_t round = 0; round < _blockOffsetsPerRound.size(); ++round) {
            for (size_t i = 0; i < diedRanks.size(); ++i) {
                ReStoreMPI::original_rank_t diedRank = diedRanks[i];
                size_t blockRangeIndex               = _blockOffsetsPerRound[round][asserting_cast<size_t>(diedRank)];
                assert(
                    blockRangeIndex == std::numeric_limits<size_t>::max()
                    || _blockRanges[blockRangeIndex].second == diedRank);
                for (; blockRangeIndex < _blockRanges.size() && _blockRanges[blockRangeIndex].second == diedRank;
                     ++blockRangeIndex) {
                    // numBlocksOfRank[i] += blockRanges[blockRangeIndex].first.second;
                    numBlocks += _blockRanges[blockRangeIndex].first.second;
                    blockRangeIndices.emplace_back(blockRangeIndex);
                    // blockRangeIndices[i].first  = std::min(blockRangeIndex, blockRangeIndices[i].first);
                    // blockRangeIndices[i].second = std::max(blockRangeIndex, blockRangeIndices[i].second);
                }
                assert(blockRangeIndices.size() > 0);
                // There may be some edge cases with very few blocks on lots of ranks where this is not true.
                // But for both our applications, this should hold
                assert(_blockRanges[blockRangeIndices.back()].second == diedRank);
            }
        }

        assert(asserting_cast<size_t>(std::count(_ranksisAlive.begin(), _ranksisAlive.end(), true)) == numAliveRanks);
        // figure out how many blocks each alive PE should get
        block_id_t numBlocksPerRank       = numBlocks / numAliveRanks;
        int        numRanksWithMoreBlocks = asserting_cast<int>(numBlocks % numAliveRanks);

        using request_t = std::pair<block_range_external_t, ReStoreMPI::current_rank_t>;
        std::vector<request_t>      requests;
        ReStoreMPI::original_rank_t rankCounter                   = 0;
        size_t                      blockRangeIndexIndex          = 0;
        size_t                      numBlocksUsedFromCurrentRange = 0;
        for (int rank = 0; asserting_cast<size_t>(rank) < _ranksisAlive.size(); ++rank) {
            if (_ranksisAlive[asserting_cast<size_t>(rank)]) {
                const block_id_t lowerBound =
                    numBlocksPerRank * asserting_cast<block_id_t>(rankCounter)
                    + asserting_cast<block_id_t>(std::min(rankCounter, numRanksWithMoreBlocks));
                const block_id_t upperBound =
                    numBlocksPerRank * asserting_cast<block_id_t>(rankCounter + 1)
                    + asserting_cast<block_id_t>(std::min(rankCounter + 1, numRanksWithMoreBlocks));
                const block_id_t numBlocksForThisRank      = upperBound - lowerBound;
                size_t           numBlocksRemainingForRank = numBlocksForThisRank;

                while (numBlocksRemainingForRank > 0) {
                    auto blockRangeIndex = blockRangeIndices[blockRangeIndexIndex];
                    auto blockRange      = _blockRanges[blockRangeIndex];
                    assert(std::find(diedRanks.begin(), diedRanks.end(), blockRange.second) != diedRanks.end());
                    size_t numBlocksRemainingInBlockRange = blockRange.first.second - numBlocksUsedFromCurrentRange;
                    size_t numBlocksTakenFromRange =
                        std::min(numBlocksRemainingInBlockRange, numBlocksRemainingForRank);

                    auto startBlock = blockRange.first.first + numBlocksUsedFromCurrentRange;
                    requests.emplace_back(std::make_pair(
                        std::make_pair(
                            asserting_cast<block_id_t>(startBlock), asserting_cast<size_t>(numBlocksTakenFromRange)),
                        rank));

                    numBlocksRemainingForRank -= numBlocksTakenFromRange;
                    numBlocksUsedFromCurrentRange += numBlocksTakenFromRange;
                    assert(numBlocksUsedFromCurrentRange <= blockRange.first.second);
                    if (numBlocksUsedFromCurrentRange == blockRange.first.second) {
                        ++blockRangeIndexIndex;
                        numBlocksUsedFromCurrentRange = 0;
                    }
                }
                ++rankCounter;
            }
        }

        // Re-insert ranks as the user has not committed to the change yet
        for (const auto diedRank: diedRanks) {
            assert(_ranksisAlive[asserting_cast<size_t>(diedRank)] == false);
            _ranksisAlive[asserting_cast<size_t>(diedRank)] = true;
            ++numAliveRanks;
        }
        assert(std::is_sorted(requests.begin(), requests.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.second < rhs.second;
        }));
        _previousReturnedBlockRanges = requests;
        return requests;
    }

    std::vector<std::pair<block_id_t, size_t>> getNewBlocksAfterFailureForPullBlocks(
        const std::vector<ReStoreMPI::original_rank_t>& diedRanks, ReStoreMPI::original_rank_t myRank) {
        auto resultPushBlocks = getNewBlocksAfterFailureForPushBlocks(diedRanks);
        auto it = std::remove_if(resultPushBlocks.begin(), resultPushBlocks.end(), [myRank](auto request) {
            return request.second != myRank;
        });
        resultPushBlocks.erase(it, resultPushBlocks.end());
        std::vector<std::pair<block_id_t, size_t>> resultPullBlocks;
        resultPullBlocks.reserve(resultPushBlocks.size());
        std::transform(
            resultPushBlocks.begin(), resultPushBlocks.end(), std::back_inserter(resultPullBlocks),
            [](auto pushRequest) { return pushRequest.first; });
        return resultPullBlocks;
    }

    void commitToPreviousCall() {
        if (_previousDiedRanksVector.empty()) {
            assert(_previousReturnedBlockRanges.empty());
            return;
        }
        assert(!_previousReturnedBlockRanges.empty());
        // remove ranks from rank set
        for (const auto diedRank: _previousDiedRanksVector) {
            assert(_ranksisAlive[asserting_cast<size_t>(diedRank)] == true);
            _ranksisAlive[asserting_cast<size_t>(diedRank)] = false;
            --numAliveRanks;
        }

        size_t oldBlockRangesSize = _blockRanges.size();
        // Our suggestion is taken so we update the current block ranges
        _blockRanges.insert(
            _blockRanges.end(), _previousReturnedBlockRanges.begin(), _previousReturnedBlockRanges.end());


        assert(std::is_sorted(
            _previousReturnedBlockRanges.begin(), _previousReturnedBlockRanges.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; }));

        _blockOffsetsPerRound.emplace_back(_blockOffsetsPerRound.back().size(), std::numeric_limits<size_t>::max());

        for (size_t i = oldBlockRangesSize; i < _blockRanges.size(); ++i) {
            size_t rank                        = asserting_cast<size_t>(_blockRanges[i].second);
            _blockOffsetsPerRound.back()[rank] = std::min(i, _blockOffsetsPerRound.back()[rank]);
        }

        _previousDiedRanksVector.clear();
        _previousReturnedBlockRanges.clear();
    }

    private:
    std::vector<std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::original_rank_t>> _blockRanges;
    std::vector<bool>                                                                  _ranksisAlive;
    size_t                                                                             numAliveRanks;
    std::vector<std::vector<size_t>>                                                   _blockOffsetsPerRound;

    std::vector<std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::original_rank_t>> _previousReturnedBlockRanges;
    std::vector<ReStoreMPI::original_rank_t>                                           _previousDiedRanksVector;
};
} // namespace ReStore
#endif // EQUAL_LOAD_BALANCER_H
