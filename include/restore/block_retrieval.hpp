#ifndef RESTORE_BLOCK_RETRIEVAL_HPP
#define RESTORE_BLOCK_RETRIEVAL_HPP

#include <algorithm>
#include <cstddef>
#include <mpi.h>
#include <utility>
#include <vector>


#include "xxhash.h"

#include "restore/block_distribution.hpp"
#include "restore/block_serialization.hpp"
#include "restore/common.hpp"
#include "restore/helpers.hpp"
#include "restore/mpi_context.hpp"
#include "restore/pseudo_random_permutation.hpp"

namespace ReStore {
using block_range_external_t = std::pair<block_id_t, size_t>;
using block_range_request_t  = std::pair<block_range_external_t, ReStoreMPI::current_rank_t>;


// Returns blockRanges and originalRanks
template <class MPIContext>
inline void getServingRanks(
    const typename BlockDistribution<MPIContext>::BlockRange& blockRange,
    const block_range_external_t blockRangeExternal, const BlockDistribution<MPIContext>* _blockDistribution,
    std::vector<block_range_request_t>& result) {
    // TODO: Special case treatment for blocks that we have locally?
    assert(blockRange.contains(blockRangeExternal.first));
    assert(blockRange.contains(blockRangeExternal.first + blockRangeExternal.second - 1));
    auto ranksWithBlockRange = _blockDistribution->ranksBlockRangeIsStoredOn(blockRange);
    std::sort(ranksWithBlockRange.begin(), ranksWithBlockRange.end());
    if (ranksWithBlockRange.empty()) {
        throw UnrecoverableDataLossException();
    }
    const size_t numBlocksPerRank         = blockRangeExternal.second / ranksWithBlockRange.size();
    const size_t numRanksWithOneBlockMore = blockRangeExternal.second % ranksWithBlockRange.size();
    size_t       currentBlock             = blockRangeExternal.first;
    size_t       numSendingRanks          = std::min(ranksWithBlockRange.size(), blockRangeExternal.second);
    result.clear();
    result.reserve(numSendingRanks);
    for (size_t i = 0; i < numSendingRanks; ++i) {
        assert(currentBlock < blockRangeExternal.first + blockRangeExternal.second);
        assert(
            currentBlock + numBlocksPerRank + (i < numRanksWithOneBlockMore)
            <= blockRangeExternal.first + blockRangeExternal.second);
        result.emplace_back(
            block_range_external_t(currentBlock, numBlocksPerRank + (i < numRanksWithOneBlockMore)),
            ranksWithBlockRange[i]);
        assert(result.back().first.second > 0);
        // This might be possible without this additional variable but I didn't bother to figure it out.
        currentBlock += numBlocksPerRank + (i < numRanksWithOneBlockMore);
    }
    assert(currentBlock == blockRangeExternal.first + blockRangeExternal.second);
}

// Returns originalRank
template <class MPIContext>
inline ReStoreMPI::original_rank_t getServingRank(
    const typename BlockDistribution<MPIContext>::BlockRange& blockRange,
    const BlockDistribution<MPIContext>* _blockDistribution, ReStoreMPI::original_rank_t receivingRank) {
    // Special case treatment for blocks that we have locally
    if (_blockDistribution->isStoredOn(blockRange, receivingRank)) {
        return receivingRank;
    }

    // Select a random alive rank to serve this block range; use same PE for all requests of a PE for a given BlockRange.
    auto servingRank = _blockDistribution->randomAliveRankBlockRangeIsStoredOn(blockRange, asserting_cast<uint64_t>(receivingRank));

    if (servingRank == -1) {
        throw UnrecoverableDataLossException();
    }
    return servingRank;
}

// Project the block ids from the user ids to the internal ids. This means that the length of the requested
// block ranges change, too. If we are using the RangePermutation, we will still get some consecutive blocks
// ids. E.g. the requested range [0,100) might get translated to [0,10), [80, 90), [20, 30), ...
template <typename Permutation>
std::vector<std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::original_rank_t>>
projectBlockRequestsFromUserToPermutedIDs(
    std::vector<std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::current_rank_t>> userBlockRanges,
    Permutation                                                                       permutation) {
    std::vector<std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::current_rank_t>> internalBlockRanges;
    for (auto& userBlockRange: userBlockRanges) {
        const auto destinationRank = userBlockRange.second;

        auto userBlockId       = userBlockRange.first.first;
        auto lengthOfUserRange = userBlockRange.first.second;

        for (;;) {
            const auto internalBlockId       = permutation.f(userBlockId);
            const auto lastIdOfInternalRange = permutation.lastIdOfRange(internalBlockId);

            if (internalBlockId + lengthOfUserRange - 1 <= lastIdOfInternalRange) {
                // The remainder of the user-requested range is completely contained in the internal range.
                assert(lengthOfUserRange > 0);
                internalBlockRanges.push_back({{internalBlockId, lengthOfUserRange}, destinationRank});
                break;
            } else {
                // We have to split the user-requested range into into multiple internal ranges.
                assert(lastIdOfInternalRange >= internalBlockId);
                const auto consumedBlocks = lastIdOfInternalRange - internalBlockId + 1;
                assert(consumedBlocks >= 1);
                assert(consumedBlocks < lengthOfUserRange);

                internalBlockRanges.push_back({{internalBlockId, consumedBlocks}, destinationRank});
                lengthOfUserRange -= consumedBlocks;
                userBlockId += consumedBlocks;
            }
        }
    }
    return internalBlockRanges;
}

template <>
std::vector<std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::original_rank_t>>
projectBlockRequestsFromUserToPermutedIDs<IdentityPermutation>(
    std::vector<std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::current_rank_t>> userBlockRanges,
    IdentityPermutation                                                               permutation) {
    UNUSED(permutation);
    return userBlockRanges;
}

// Takes block range requests with current ranks
// Returns sendBlockRanges and recvBlockRanges with current ranks
template <class MPIContext>
inline std::pair<std::vector<block_range_request_t>, std::vector<block_range_request_t>> getSendRecvBlockRanges(
    const std::vector<block_range_request_t>& blockRanges, const BlockDistribution<MPIContext>* _blockDistribution,
    const MPIContext& _mpiContext) {
    std::vector<block_range_request_t> sendBlockRanges;
    std::vector<block_range_request_t> recvBlockRanges;
    for (const auto& blockRange: blockRanges) {
        const auto firstBlockIdInRange = blockRange.first.first;
        const auto lengthOfRange       = blockRange.first.second;
        assert(lengthOfRange > 0);
        const auto lastBlockIdInRange = firstBlockIdInRange + lengthOfRange - 1;
        const auto destinationRank    = blockRange.second;

        const auto firstBlockIdOfNextInternalRange = [&_blockDistribution](const block_id_t blockId) {
            return _blockDistribution->rangeOfBlock(blockId).start()
                   + _blockDistribution->rangeOfBlock(blockId).length();
        };

        // The requested block range might span over multiple internal block ranges. We therefore might need to split up
        // the request.
        for (block_id_t blockId = firstBlockIdInRange; blockId <= lastBlockIdInRange;
             blockId            = firstBlockIdOfNextInternalRange(blockId)) {
            const typename BlockDistribution<MPIContext>::BlockRange blockRangeInternal =
                _blockDistribution->rangeOfBlock(blockId);

            block_id_t end =
                std::min(blockRangeInternal.start() + blockRangeInternal.length(), firstBlockIdInRange + lengthOfRange);
            size_t size = end - blockId;
            assert(blockRangeInternal.contains(blockId));
            assert(blockRangeInternal.contains(blockId + size - 1));

            const auto servingRank =
                getServingRank(blockRangeInternal, _blockDistribution, _mpiContext.getOriginalRank(destinationRank));
            if (servingRank == _mpiContext.getMyOriginalRank()) {
                sendBlockRanges.emplace_back(block_range_external_t(blockId, size), destinationRank);
            }

            if (blockRange.second == _mpiContext.getMyCurrentRank()) {
                recvBlockRanges.emplace_back(
                    block_range_external_t(blockId, size), _mpiContext.getCurrentRank(servingRank).value());
            }
        }
    }
    auto sortByRankAndBegin = [](const block_range_request_t& lhs, const block_range_request_t& rhs) {
        bool ranksLess   = lhs.second < rhs.second;
        bool ranksEqual  = lhs.second == rhs.second;
        bool blockIdLess = lhs.first.first < rhs.first.first;
        return ranksLess || (ranksEqual && blockIdLess);
    };
    std::sort(sendBlockRanges.begin(), sendBlockRanges.end(), sortByRankAndBegin);
    std::sort(recvBlockRanges.begin(), recvBlockRanges.end(), sortByRankAndBegin);
    return std::make_pair(sendBlockRanges, recvBlockRanges);
}

// Takes recvBlockRanges with current ranks
template <typename HandleSerializedBlockFunction, typename Permutation>
inline void handleReceivedBlocks(
    const std::vector<ReStoreMPI::RecvMessage>& recvMessages, const std::vector<block_range_request_t>& recvBlockRanges,
    const OffsetMode _offsetMode, const size_t _constOffset, HandleSerializedBlockFunction handleSerializedBlock,
    const Permutation& blockIdPermuter) {
    static_assert(
        std::is_invocable<HandleSerializedBlockFunction, const std::byte*, size_t, block_id_t>(),
        "HandleSerializedBlockFunction must be invocable as (const std::byte*, size_t, "
        "block_id_t)");
    assert(std::is_sorted(
        recvBlockRanges.begin(), recvBlockRanges.end(),
        [](const block_range_request_t& lhs, const block_range_request_t& rhs) {
            bool ranksLess   = lhs.second < rhs.second;
            bool ranksEqual  = lhs.second == rhs.second;
            bool blockIdLess = lhs.first.first < rhs.first.first;
            return ranksLess || (ranksEqual && blockIdLess);
        }));
    assert(std::is_sorted(
        recvMessages.begin(), recvMessages.end(),
        [](const ReStoreMPI::RecvMessage& lhs, const ReStoreMPI::RecvMessage& rhs) {
            return lhs.srcRank < rhs.srcRank;
        }));

    // Iterate over all received messages and all block ranges contained in those messages.
    assert(_offsetMode == OffsetMode::constant); // TODO: Implement LUT mode
    UNUSED(_offsetMode);
    size_t idxRecvBlockRanges = 0; // Index into the recvBlockRanges vector
    // recvBlockRanges contains the block ranges we are expecting, recvMessages the block ranges we actually received.
    for (const ReStoreMPI::RecvMessage& recvMessage: recvMessages) {
        assert(recvMessage.data.size() > 0);
        assert(idxRecvBlockRanges < recvBlockRanges.size());
        assert(recvMessage.srcRank == recvBlockRanges[idxRecvBlockRanges].second);

        size_t idxRecvMessageData = 0; // Index into the recvMessages.data() vector
        while (idxRecvBlockRanges < recvBlockRanges.size()
               && recvBlockRanges[idxRecvBlockRanges].second == recvMessage.srcRank) {
            // For each block, call the user-provided deserialization function.
            const auto firstBlockId = recvBlockRanges[idxRecvBlockRanges].first.first;
            const auto numBlocks    = recvBlockRanges[idxRecvBlockRanges].first.second;
            const auto lastBlockId  = firstBlockId + numBlocks - 1;
            for (auto blockId = firstBlockId; blockId <= lastBlockId; ++blockId) {
                assert(idxRecvMessageData < recvMessage.data.size());
                assert(idxRecvMessageData + _constOffset <= recvMessage.data.size());

                // Provide the user with the de-permuted id (the one he specified when submitting the block).
                const auto userBlockId = blockIdPermuter.finv(blockId);
                handleSerializedBlock(&(recvMessage.data[idxRecvMessageData]), _constOffset, userBlockId);

                // Move onto the next block in recvMessage.data
                idxRecvMessageData += _constOffset;
            }

            idxRecvBlockRanges++;
        }
        // Has the whole message been consumed?
        assert(idxRecvMessageData == recvMessage.data.size());
    }
}

// Takes requests with current ranks
template <class MPIContext = ReStoreMPI::MPIContext>
inline std::vector<ReStoreMPI::RecvMessage> sparseAllToAll(
    const std::vector<block_range_request_t>& sendBlockRanges, const OffsetMode _offsetMode,
    const MPIContext& _mpiContext, const SerializedBlockStorage<MPIContext>* _serializedBlocks) {
    assert(std::is_sorted(
        sendBlockRanges.begin(), sendBlockRanges.end(),
        [](const block_range_request_t& lhs, const block_range_request_t& rhs) {
            bool ranksLess   = lhs.second < rhs.second;
            bool ranksEqual  = lhs.second == rhs.second;
            bool blockIdLess = lhs.first.first < rhs.first.first;
            return ranksLess || (ranksEqual && blockIdLess);
        }));

    std::vector<std::vector<std::byte>>  sendData;
    std::vector<ReStoreMPI::SendMessage> sendMessages;
    int                                  currentRank = MPI_UNDEFINED;
    for (const block_range_request_t& sendBlockRange: sendBlockRanges) {
        if (currentRank != sendBlockRange.second) {
            assert(currentRank < sendBlockRange.second);
            if (currentRank != MPI_UNDEFINED) {
                assert(sendData.size() > 0);
                sendMessages.emplace_back(sendData.back().data(), sendData.back().size(), currentRank);
            }
            sendData.emplace_back();
            currentRank = sendBlockRange.second;
        }
        // TODO Implement LUT mode
        assert(_offsetMode == OffsetMode::constant);
        UNUSED(_offsetMode);
        _serializedBlocks->forAllBlocks(sendBlockRange.first, [&sendData](const std::byte* ptr, size_t size) {
            sendData.back().insert(sendData.back().end(), ptr, ptr + size);
        });
    }
    if (!sendBlockRanges.empty()) {
        assert(currentRank != MPI_UNDEFINED);
        assert(sendData.size() > 0);
        sendMessages.emplace_back(sendData.back().data(), sendData.back().size(), currentRank);
    }

    auto result = _mpiContext.SparseAllToAll(sendMessages);
    std::sort(result.begin(), result.end(), [](const ReStoreMPI::RecvMessage& lhs, const ReStoreMPI::RecvMessage& rhs) {
        return lhs.srcRank < rhs.srcRank;
    });
    return result;
}


} // namespace ReStore
#endif // RESTORE_BLOCK_RETRIEVAL_HPP
