#ifndef EQUAL_LOAD_BALANCER_H
#define EQUAL_LOAD_BALANCER_H

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "common.hpp"
#include "mpi_context.hpp"
#include "restore/helpers.hpp"

namespace ReStore {
class EqualLoadBalancer {
    public:
    EqualLoadBalancer(
        const std::vector<std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::original_rank_t>>& blockRanges,
        const int                                                                                 numRanksOriginal)
        : _blockRanges(blockRanges) {
        for (int rank = 0; rank < numRanksOriginal; ++rank) {
            _ranks.emplace(rank);
        }
    }

    std::vector<std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::original_rank_t>>
    getNewBlocksAfterFailure(const std::vector<ReStoreMPI::original_rank_t>& diedRanks) {
        for (const auto diedRank: diedRanks) {
            _ranks.erase(diedRank);
        }


        // First, sort by rank id so the following loop works
        std::sort(_blockRanges.begin(), _blockRanges.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.second < rhs.second;
        });

        // Store for each of the dead ranks how many blocks it used to hold and which blockRanges these corresponded to
        std::vector<size_t> blockRangeIndices;
        size_t              numBlocks       = 0;
        size_t              blockRangeIndex = 0;
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
        }

        // figure out how many blocks each alive PE should get
        block_id_t numBlocksPerRank       = numBlocks / _ranks.size();
        int        numRanksWithMoreBlocks = asserting_cast<int>(numBlocks % _ranks.size());

        std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, ReStoreMPI::current_rank_t>> requests;
        ReStoreMPI::original_rank_t                                                                rankCounter = 0;
        size_t blockRangeIndexIndex                                                                            = 0;
        size_t numBlocksUsedFromCurrentRange                                                                   = 0;
        for (const auto rank: _ranks) {
            const block_id_t lowerBound = numBlocksPerRank * asserting_cast<block_id_t>(rankCounter)
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
                size_t numBlocksTakenFromRange = std::min(numBlocksRemainingInBlockRange, numBlocksRemainingForRank);

                requests.emplace_back(std::make_pair(
                    std::make_pair(
                        asserting_cast<ReStore::block_id_t>(blockRange.first.first + numBlocksUsedFromCurrentRange),
                        asserting_cast<size_t>(numBlocksTakenFromRange)),
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
        // Remove old block ranges
        auto it = std::remove_if(_blockRanges.begin(), _blockRanges.end(), [this](auto blockRange) {
            return _ranks.find(blockRange.second) == _ranks.end();
        });
        _blockRanges.erase(it, _blockRanges.end());
        // Assume that our suggestion is taken and store them as current block ranges
        _blockRanges.insert(_blockRanges.end(), requests.begin(), requests.end());
        return requests;
    }

    private:
    std::vector<std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::original_rank_t>> _blockRanges;
    std::unordered_set<ReStoreMPI::original_rank_t>                                    _ranks;
};
} // namespace ReStore
#endif // EQUAL_LOAD_BALANCER_H
