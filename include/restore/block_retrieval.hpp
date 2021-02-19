#ifndef RESTORE_BLOCK_RETRIEVAL_HPP
#define RESTORE_BLOCK_RETRIEVAL_HPP

#include <cstddef>
#include <mpi.h>

#include "restore/block_distribution.hpp"
#include "restore/block_serialization.hpp"
#include "restore/common.hpp"
#include "restore/mpi_context.hpp"

namespace ReStore {
using block_range_external_t = std::pair<block_id_t, size_t>;
using block_range_request_t  = std::pair<block_range_external_t, int>;


template <class MPIContext = ReStoreMPI::MPIContext>
inline std::pair<std::vector<block_range_request_t>, std::vector<block_range_request_t>> getSendRecvBlockRanges(
    const std::vector<block_range_request_t>& blockRanges, const BlockDistribution<MPIContext>* _blockDistribution,
    const MPIContext& _mpiContext) {
    std::vector<block_range_request_t> sendBlockRanges;
    std::vector<block_range_request_t> recvBlockRanges;
    for (const auto& blockRange: blockRanges) {
        for (block_id_t blockId = blockRange.first.first; blockId < blockRange.first.first + blockRange.first.second;
             blockId += _blockDistribution->rangeOfBlock(blockId).length()) {
            const auto                        blockRangeInternal = _blockDistribution->rangeOfBlock(blockId);
            const ReStoreMPI::original_rank_t servingRank        = getServingRank(blockRangeInternal);
            size_t                            size               = blockRangeInternal.length();
            if (blockRangeInternal.start() + blockRangeInternal.length()
                >= blockRange.first.first + blockRange.first.second) {
                size = blockRange.first.first + blockRange.first.second - blockRangeInternal.start();
            }
            if (servingRank == _mpiContext.getMyOriginalRank()) {
                sendBlockRanges.emplace_back(std::make_pair(blockRangeInternal.start(), size), blockRange.second);
            }
            if (blockRange.second == _mpiContext.getMyCurrentRank()) {
                recvBlockRanges.emplace_back(
                    std::make_pair(blockRangeInternal.start(), size), _mpiContext.getCurrentRank(servingRank));
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

template <class HandleSerializedBlockFunction>
inline void handleReceivedBlocks(
    const std::vector<ReStoreMPI::RecvMessage>& recvMessages, const std::vector<block_range_request_t>& recvBlockRanges,
    const OffsetMode _offsetMode, const size_t _constOffset, HandleSerializedBlockFunction handleSerializedBlock) {
    static_assert(
        std::is_invocable<HandleSerializedBlockFunction, const void*, size_t, block_id_t>(),
        "HandleSerializedBlockFunction must be invocable as (const uint8_t*, size_t, "
        "block_id_t)");
    size_t currentIndexRecvBlockRanges = 0;
    for (const ReStoreMPI::RecvMessage& recvMessage: recvMessages) {
        assert(currentIndexRecvBlockRanges < recvBlockRanges.size());
        assert(recvMessage.srcRank == recvBlockRanges[currentIndexRecvBlockRanges].second);
        // TODO: Implement LUT mode
        assert(_offsetMode == OffsetMode::constant);
        size_t currentIndexRecvMessage = 0;
        while (currentIndexRecvBlockRanges < recvBlockRanges.size()
               && recvBlockRanges[currentIndexRecvBlockRanges].second == recvMessage.srcRank) {
            for (block_id_t blockId = recvBlockRanges[currentIndexRecvBlockRanges].first.first;
                 blockId < recvBlockRanges[currentIndexRecvBlockRanges].first.first
                               + recvBlockRanges[currentIndexRecvBlockRanges].first.second;
                 ++blockId) {
                assert(currentIndexRecvMessage < recvMessage.data.size());
                assert(currentIndexRecvMessage + _constOffset < recvMessage.data.size());
                handleSerializedBlock(&(recvMessage.data[currentIndexRecvMessage]), _constOffset, blockId);
                currentIndexRecvMessage += _constOffset;
            }
            currentIndexRecvBlockRanges++;
        }
    }
}

template <class MPIContext = ReStoreMPI::MPIContext>
inline std::vector<ReStoreMPI::RecvMessage> sparseAllToAll(
    const std::vector<block_range_request_t>& sendBlockRanges, const OffsetMode _offsetMode,
    const MPIContext& _mpiContext, const SerializedBlockStorage<MPIContext>* _serializedBlocks) {
    std::vector<std::vector<uint8_t>>    sendData;
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
        _serializedBlocks->forAllBlocks(sendBlockRange.first, [&sendData](uint8_t* ptr, size_t size) {
            sendData.back().insert(sendData.back().end(), ptr, ptr + size);
        });
    }
    assert(currentRank != MPI_UNDEFINED);
    assert(sendData.size() > 0);
    sendMessages.emplace_back(sendData.back().data(), sendData.back().size(), currentRank);
    auto result = _mpiContext.SparseAllToAll(sendMessages);
    std::sort(result.begin(), result.end(), [](const ReStoreMPI::RecvMessage& lhs, const ReStoreMPI::RecvMessage& rhs) {
        return lhs.srcRank < rhs.srcRank;
    });
    return result;
}


} // namespace ReStore
#endif // RESTORE_BLOCK_RETRIEVAL_HPP
