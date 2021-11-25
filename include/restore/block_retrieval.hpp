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
    const block_range_external_t blockRangeExternal, const BlockDistribution<MPIContext>* _blockDistribution,
    ReStoreMPI::original_rank_t receivingRank) {
    assert(blockRange.contains(blockRangeExternal.first));
    assert(blockRange.contains(blockRangeExternal.first + blockRangeExternal.second - 1));
    // Special case treatment for blocks that we have locally
    if (_blockDistribution->isStoredOn(blockRange, receivingRank)) {
        return receivingRank;
    }
    auto ranksWithBlockRange = _blockDistribution->ranksBlockRangeIsStoredOn(blockRange);
    std::sort(ranksWithBlockRange.begin(), ranksWithBlockRange.end());
    if (ranksWithBlockRange.empty()) {
        throw UnrecoverableDataLossException();
    }

    // Use same PE for all requests of a PE for a given BlockRange
    auto blockRangeReceiverPair = std::make_pair(blockRange.id(), receivingRank);
    // TODO Think about seed
    auto servingIndex = XXH64(&blockRangeReceiverPair, sizeof(blockRangeReceiverPair), 42) % ranksWithBlockRange.size();
    return ranksWithBlockRange[servingIndex];
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
        for (block_id_t blockId = blockRange.first.first; blockId < blockRange.first.first + blockRange.first.second;
             blockId            = _blockDistribution->rangeOfBlock(blockId).start()
                       + _blockDistribution->rangeOfBlock(blockId).length()) {
            const typename BlockDistribution<MPIContext>::BlockRange blockRangeInternal =
                _blockDistribution->rangeOfBlock(blockId);
            block_id_t end = std::min(
                blockRangeInternal.start() + blockRangeInternal.length(),
                blockRange.first.first + blockRange.first.second);
            size_t size        = end - blockId;
            auto   servingRank = getServingRank(
                blockRangeInternal, block_range_external_t(blockId, size), _blockDistribution,
                _mpiContext.getOriginalRank(blockRange.second));

            if (servingRank == _mpiContext.getMyOriginalRank()) {
                sendBlockRanges.emplace_back(block_range_external_t(blockId, size), blockRange.second);
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
template <class HandleSerializedBlockFunction>
inline void handleReceivedBlocks(
    const std::vector<ReStoreMPI::RecvMessage>& recvMessages, const std::vector<block_range_request_t>& recvBlockRanges,
    const OffsetMode _offsetMode, const size_t _constOffset, HandleSerializedBlockFunction handleSerializedBlock) {
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

    size_t currentIndexRecvBlockRanges = 0;
    for (const ReStoreMPI::RecvMessage& recvMessage: recvMessages) {
        assert(currentIndexRecvBlockRanges < recvBlockRanges.size());
        assert(recvMessage.srcRank == recvBlockRanges[currentIndexRecvBlockRanges].second);
        // TODO: Implement LUT mode
        assert(_offsetMode == OffsetMode::constant);
        UNUSED(_offsetMode);
        size_t currentIndexRecvMessage = 0;
        while (currentIndexRecvBlockRanges < recvBlockRanges.size()
               && recvBlockRanges[currentIndexRecvBlockRanges].second == recvMessage.srcRank) {
            for (block_id_t blockId = recvBlockRanges[currentIndexRecvBlockRanges].first.first;
                 blockId < recvBlockRanges[currentIndexRecvBlockRanges].first.first
                               + recvBlockRanges[currentIndexRecvBlockRanges].first.second;
                 ++blockId) {
                assert(currentIndexRecvMessage < recvMessage.data.size());
                assert(currentIndexRecvMessage + _constOffset <= recvMessage.data.size());
                handleSerializedBlock(&(recvMessage.data[currentIndexRecvMessage]), _constOffset, blockId);
                currentIndexRecvMessage += _constOffset;
            }

            currentIndexRecvBlockRanges++;
        }
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
