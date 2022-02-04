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
          _ranksBitVector(asserting_cast<size_t>(numRanksOriginal), true),
          numAliveRanks(asserting_cast<size_t>(numRanksOriginal)) {
        assert(_ranksBitVector.size() == numAliveRanks);
        assert(
            asserting_cast<size_t>(std::count(_ranksBitVector.begin(), _ranksBitVector.end(), true)) == numAliveRanks);
    }

    std::vector<std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::original_rank_t>>
    getNewBlocksAfterFailureForPushBlocks(const std::vector<ReStoreMPI::original_rank_t>& diedRanks) {
        if (diedRanks.empty()) {
            return {};
        }
        _previousReturnedBlockRanges.clear();
        _previousDiedRanksVector = diedRanks;
        for (const auto diedRank: diedRanks) {
            assert(_ranksBitVector[asserting_cast<size_t>(diedRank)] == true);
            _ranksBitVector[asserting_cast<size_t>(diedRank)] = false;
            --numAliveRanks;
        }

        // Store for each of the dead ranks how many blocks it used to hold and which blockRanges these corresponded to
        std::vector<size_t> blockRangeIndices;
        size_t              numBlocks       = 0;
        size_t              blockRangeIndex = 0;
        while (blockRangeIndex < _blockRanges.size()) {
            for (size_t i = 0; i < diedRanks.size(); ++i) {
                ReStoreMPI::original_rank_t diedRank = diedRanks[i];
                for (; blockRangeIndex < _blockRanges.size() && _blockRanges[blockRangeIndex].second <= diedRank;
                     ++blockRangeIndex) {
                    if (_blockRanges[blockRangeIndex].second == diedRank) {
                        // numBlocksOfRank[i] += blockRanges[blockRangeIndex].first.second;
                        numBlocks += _blockRanges[blockRangeIndex].first.second;
                        blockRangeIndices.emplace_back(blockRangeIndex);
                        // blockRangeIndices[i].first  = std::min(blockRangeIndex, blockRangeIndices[i].first);
                        // blockRangeIndices[i].second = std::max(blockRangeIndex, blockRangeIndices[i].second);
                    }
                }
                assert(blockRangeIndices.size() > 0);
                assert(_blockRanges[blockRangeIndices.back()].second == diedRank);
            }
            assert(diedRanks.size() > 0);
            for (; blockRangeIndex < _blockRanges.size() && _blockRanges[blockRangeIndex].second >= diedRanks.back();
                 ++blockRangeIndex) {
                // Progress to next run of blockRanges
            }
        }

        assert(
            asserting_cast<size_t>(std::count(_ranksBitVector.begin(), _ranksBitVector.end(), true)) == numAliveRanks);
        // figure out how many blocks each alive PE should get
        block_id_t numBlocksPerRank       = numBlocks / numAliveRanks;
        int        numRanksWithMoreBlocks = asserting_cast<int>(numBlocks % numAliveRanks);

        using request_t = std::pair<block_range_external_t, ReStoreMPI::current_rank_t>;
        std::vector<request_t>      requests;
        ReStoreMPI::original_rank_t rankCounter                   = 0;
        size_t                      blockRangeIndexIndex          = 0;
        size_t                      numBlocksUsedFromCurrentRange = 0;
        for (int rank = 0; asserting_cast<size_t>(rank) < _ranksBitVector.size(); ++rank) {
            if (_ranksBitVector[asserting_cast<size_t>(rank)]) {
                const block_id_t lowerBound =
                    numBlocksPerRank * asserting_cast<block_id_t>(rankCounter)
                    + asserting_cast<block_id_t>(std::min(rankCounter, numRanksWithMoreBlocks));
                const block_id_t upperBound =
                    numBlocksPerRank * asserting_cast<block_id_t>(rankCounter + 1)
                    + asserting_cast<block_id_t>(std::min(rankCounter + 1, numRanksWithMoreBlocks));
                const block_id_t numBlocksForThisRank      = upperBound - lowerBound;
                size_t           numBlocksRemainingForRank = numBlocksForThisRank;

                while (numBlocksRemainingForRank > 0) {
                    blockRangeIndex = blockRangeIndices[blockRangeIndexIndex];
                    auto blockRange = _blockRanges[blockRangeIndex];
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
            assert(_ranksBitVector[asserting_cast<size_t>(diedRank)] == false);
            _ranksBitVector[asserting_cast<size_t>(diedRank)] = true;
            ++numAliveRanks;
        }
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
        // remove ranks from rank set
        for (const auto diedRank: _previousDiedRanksVector) {
            assert(_ranksBitVector[asserting_cast<size_t>(diedRank)] == true);
            _ranksBitVector[asserting_cast<size_t>(diedRank)] = false;
            --numAliveRanks;
        }

        // Our suggestion is taken so we update the current block ranges
        _blockRanges.insert(
            _blockRanges.end(), _previousReturnedBlockRanges.begin(), _previousReturnedBlockRanges.end());

        _previousDiedRanksVector.clear();
        _previousReturnedBlockRanges.clear();
    }

    private:
    std::vector<std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::original_rank_t>> _blockRanges;
    std::vector<bool>                                                                  _ranksBitVector;
    size_t                                                                             numAliveRanks;

    std::vector<std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::original_rank_t>> _previousReturnedBlockRanges;
    std::vector<ReStoreMPI::original_rank_t>                                           _previousDiedRanksVector;
};
} // namespace ReStore
#endif // EQUAL_LOAD_BALANCER_H
