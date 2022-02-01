#ifndef RESTORE_CORE_H
#define RESTORE_CORE_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

//#include "backward.hpp"
#include "mpi_context.hpp"
#include <mpi.h>

#include "restore/block_distribution.hpp"
#include "restore/block_retrieval.hpp"
#include "restore/block_serialization.hpp"
#include "restore/block_submission.hpp"
#include "restore/common.hpp"
#include "restore/helpers.hpp"

namespace ReStore {

template <class BlockType>
class ReStore {
    using Communication = BlockSubmissionCommunication<BlockType>;

    public:
    // Constructor
    //
    // mpiCommunicator: The MPI Communicator to use.
    // replicationLevel: The number of replicas to distribute among the different ranks.
    // offsetMode: When the serialized blocks are stored in memory, we need to know at which memory location
    //      each block starts. For large blocks, we can afford a look-up-table. This has the advantage that
    //      we can handle blocks with different lengths. For very small blocks, however, a look-up-table would
    //      incur too much of an memory overhead. Take for example block sizes of 4 bytes, resulting in millions
    //      or billions of blocks per rank. By using a constant offset, we can still store that many blocks without
    //      a look-up-table. The drawback is, that each block will take up constOffset _bytes_ of space.
    // constOffset: An upper bound for the number of _bytes_ a serialized block takes up. Has to be equal to 0
    //      when using look-up-table offset mode and greater than 0 when using consOffset mode.
    ReStore(
        MPI_Comm mpiCommunicator, uint16_t replicationLevel, OffsetMode offsetMode, size_t constOffset = 0,
        uint64_t blocksPerPermutationRange = 4096, uint64_t randomPermutationSeed = 0)
        : _replicationLevel(replicationLevel),
          _offsetMode(offsetMode),
          _constOffset(constOffset),
          _randomPermutationSeed(randomPermutationSeed),
          _blocksPerPermutationRange(blocksPerPermutationRange),
          _mpiContext(mpiCommunicator),
          _blockDistribution(nullptr), // Depends on the number of blocks which are submitted in submitBlocks.
          _serializedBlocks(nullptr) { // Depends on _blockDistribution
        if (offsetMode == OffsetMode::lookUpTable && constOffset != 0) {
            throw std::invalid_argument("Explicit offset mode set but the constant offset is not zero.");
        } else if (offsetMode == OffsetMode::constant && constOffset == 0) {
            throw std::invalid_argument("Constant offset mode requires a constOffset > 0.");
        } else if (replicationLevel == 0) {
            throw std::invalid_argument("What is a replication level of 0 supposed to mean?");
        } else if (mpiCommunicator == MPI_COMM_NULL) {
            throw std::invalid_argument("MPI Communicator is MPI_COMM_NULL.");
        } else if (blocksPerPermutationRange == 0) {
            throw std::invalid_argument("blocksPerPermutationRange must be greater than zero.");
        } else {
            _assertInvariants();
        }
    }

    // Copying a ReStore object does not really make sense. It would be really hard and probably not
    // what you want to deep copy the replicated blocks (including the remote ones?), too.
    ReStore(const ReStore& other) = delete;
    ReStore& operator=(const ReStore& other) = delete;

    // Moving a ReStore is disabled for now, because we do not need it and use const members
    ReStore(ReStore&& other) = delete;
    ReStore& operator=(ReStore&& other) = delete;

    // Destructor
    ~ReStore() = default;

    // replicationLevel()
    //
    // Get the replication level, that is how many copies of each block are scattered over the ranks.
    uint32_t replicationLevel() const noexcept {
        _assertInvariants();
        return this->_replicationLevel;
    }

    // offsetMode()
    //
    // Get the offset mode that defines how the serialized blocks are aligned in memory.
    OffsetModeDescriptor offsetMode() const noexcept {
        _assertInvariants();
        return OffsetModeDescriptor{this->_offsetMode, this->_constOffset};
    }

    void updateComm(MPI_Comm newComm) {
        _mpiContext.updateComm(newComm);
    }

    std::vector<ReStoreMPI::original_rank_t> getRanksDiedSinceLastCall() {
        return _mpiContext.getRanksDiedSinceLastCall();
    }

    // submitBlocks()
    //
    // Submits blocks to the replicated storage. They will be replicated among the ranks and can be
    // ReStored after a rank failure. Each rank has to call this function exactly once.
    // submitBlocks() also performs the replication and is therefore blocking until all ranks called it.
    // Even if there are multiple receivers for a single block, serialize will be called only once per block.
    //
    // serializeFunc: gets a reference to a block to serialize and a stream which can be used
    //      to append a flat representation of the current block to the serialized data's byte stream.
    // nextBlock: a generator function which should return <globalBlockId, const reference to block>
    //      on each call. If there are no more blocks getNextBlock should return {}.
    // totalNumberOfBlocks: The total number of blocks across all ranks.
    // canBeParallelized: Indicates if multiple serializeFunc calls can happen on different blocks
    //      concurrently. Also assumes that the blocks do not have to be serialized in the order they
    //      are emitted by nextBlock.
    // If a rank failure happens during this call, it will be propagated to the caller which can then handle it. This
    // includes updating the communicator of MPIContext.
    template <class SerializeBlockCallbackFunction, class NextBlockCallbackFunction>
    void submitBlocks(
        SerializeBlockCallbackFunction serializeFunc, NextBlockCallbackFunction nextBlock, size_t totalNumberOfBlocks,
        bool canBeParallelized = false // not supported yet
    ) {
        if (_offsetMode == OffsetMode::lookUpTable) {
            throw std::runtime_error("LUT mode is not implemented yet.");
        } else if (totalNumberOfBlocks == 1) {
            throw std::runtime_error("Cannot submit a single block, please use at least two blocks.");
        }
        static_assert(
            std::is_invocable<SerializeBlockCallbackFunction, const BlockType&, SerializedBlockStoreStream&>(),
            "serializeFunc must be invocable as _(const BlockType&, SerializedBlockStoreStream&");
        static_assert(
            std::is_invocable_r<std::optional<NextBlock<BlockType>>, NextBlockCallbackFunction>(),
            "nextBlock must be invocable as std::optional<ReStore::NextBlock<BlockType>>()");

        if (totalNumberOfBlocks == 0) {
            throw std::runtime_error("Invalid number of blocks: 0.");
        }

        // Initialize the block id permuter.
        const auto largestBlockId = totalNumberOfBlocks - 1;
        _blockIdPermuter.emplace(largestBlockId, _blocksPerPermutationRange, _randomPermutationSeed);
        assert(_blockIdPermuter);

        try { // Ranks failures might be detected during this block
            // We define original rank ids to be the rank ids during this function call
            _mpiContext.resetOriginalCommToCurrentComm();

            // Initialize the Block Distribution
            if (_blockDistribution) {
                throw std::runtime_error("You shall not call submitBlocks() twice!");
            }
            _blockDistribution = std::make_shared<BlockDistribution<>>(
                _mpiContext.getOriginalSize(), totalNumberOfBlocks, _replicationLevel, _mpiContext);
            _serializedBlocks =
                std::make_unique<SerializedBlockStorage<>>(_blockDistribution, _offsetMode, _constOffset);
            assert(_blockDistribution);
            assert(_serializedBlocks);
            assert(_mpiContext.getOriginalSize() == _mpiContext.getCurrentSize());

            // Initialize the Implementation object (as in PImpl)
            BlockSubmissionCommunication<BlockType> comm(_mpiContext, *_blockDistribution, offsetMode());

            // Allocate send buffers and serialize the blocks to be sent
            auto sendBuffers =
                comm.serializeBlocksForTransmission(serializeFunc, nextBlock, *_blockIdPermuter, canBeParallelized);

            // All blocks have been serialized, send & receive replicas
            auto receivedMessages = comm.exchangeData(sendBuffers);

            // Deallocate sendBuffers, they are no longer needed and take up replicationLevel * bytesPerRank memory.
            // By deallocating them now, before the received messages are stored into the serialized block storage,
            // we can reduce the peak memory consumption of this algorithm.
            sendBuffers = decltype(sendBuffers)();

            // Store the received blocks into our local block storage
            comm.parseAllIncomingMessages(
                receivedMessages, [this](
                                      block_id_t blockId, const std::byte* data, size_t lengthInBytes,
                                      ReStoreMPI::current_rank_t srcRank) {
                    UNUSED(lengthInBytes); // Currently, only constant offset mode is implemented
                    assert(lengthInBytes == _constOffset);
                    UNUSED(srcRank); // We simply do not need this right now
                    this->_serializedBlocks->writeBlock(blockId, data);
                });
        } catch (ReStoreMPI::FaultException& e) {
            // Reset BlockDistribution and SerializedBlockStorage
            _blockDistribution = nullptr;
            _serializedBlocks  = nullptr;
            throw e;
        }
    }

    // pullBlocks()
    //
    // Pulls blocks from other ranks in the replicated storage. That is, the caller provides the global // ids of those
    // blocks it wants but not from which rank to fetch them. This means that we have to perform an extra round of
    // communication compared with pushBlocks() to // request the blocks each rank wants.
    //
    // blockRanges: A list of ranges of global blck ids <firstId, numberOfBlocks> this rank wants
    // handleSerializedBlock: A function which takes a void * pointing to the start of the serialized
    //      byte stream, a length in bytes of this encoding and the global id of this block.
    // canBeParallelized: Indicates if multiple handleSerializedBlock calls can happen on different
    //      inputs concurrently.
    template <class HandleSerializedBlockFunction>
    void pullBlocks(
        std::vector<std::pair<block_id_t, size_t>> blockRanges, HandleSerializedBlockFunction handleSerializedBlock,
        bool canBeParallelized = false // not supported yet
    ) {
        UNUSED(canBeParallelized);

        // Transform to format used by functions already implemented for pushBlocks
        std::vector<std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::current_rank_t>> blockRangesWithReceiver;
        std::transform(
            blockRanges.begin(), blockRanges.end(), std::back_inserter(blockRangesWithReceiver),
            [&](const std::pair<block_id_t, size_t> blockRangeWithoutReceiver) {
                return std::make_pair(blockRangeWithoutReceiver, _mpiContext.getMyCurrentRank());
            });

        // Project the block ids from the user ids to the internal ids. This means that the length of the requested
        // block ranges change, too. If we are using the RangePermutation, we will still get some consecutive blocks
        // ids. E.g. the requested range [0,100) might get translated to [0,10), [80, 90), [20, 30), ...
        assert(_blockIdPermuter);
        const auto internalBlockRanges =
            projectBlockRequestsFromUserToPermutedIDs(blockRangesWithReceiver, *_blockIdPermuter);

        const auto [sendBlockRangesLocalRequests, recvBlockRanges] =
            getSendRecvBlockRanges(internalBlockRanges, _blockDistribution.get(), _mpiContext);


        auto sortByRankAndBegin = [](const block_range_request_t& lhs, const block_range_request_t& rhs) {
            bool ranksLess   = lhs.second < rhs.second;
            bool ranksEqual  = lhs.second == rhs.second;
            bool blockIdLess = lhs.first.first < rhs.first.first;
            return ranksLess || (ranksEqual && blockIdLess);
        };

        assert(std::is_sorted(recvBlockRanges.begin(), recvBlockRanges.end(), sortByRankAndBegin));


        using request_t = std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::current_rank_t>;
        std::vector<std::vector<request_t>>  sendData;
        std::vector<ReStoreMPI::SendMessage> sendMessagesRequests;
        int                                  currentRank = MPI_UNDEFINED;
        for (const auto& request: recvBlockRanges) {
            if (currentRank != request.second) {
                assert(currentRank < request.second);
                if (currentRank != MPI_UNDEFINED) {
                    assert(sendData.size() > 0);
                    sendMessagesRequests.emplace_back(
                        reinterpret_cast<std::byte*>(sendData.back().data()),
                        sendData.back().size() * sizeof(request_t), currentRank);
                }
                sendData.emplace_back();
                currentRank = request.second;
            }
            sendData.back().emplace_back(std::make_pair(request.first, _mpiContext.getMyCurrentRank()));
        }
        if (!recvBlockRanges.empty()) {
            assert(currentRank != MPI_UNDEFINED);
            assert(sendData.size() > 0);
            sendMessagesRequests.emplace_back(
                reinterpret_cast<std::byte*>(sendData.back().data()), sendData.back().size() * sizeof(request_t),
                currentRank);
        }

        auto recvMessagesRequests = _mpiContext.SparseAllToAll(sendMessagesRequests);
        // Not sure if needed. Used to avoid issues with overlapping sparse all to all calls
        ReStoreMPI::successOrThrowMpiCall([&]() { return MPI_Barrier(_mpiContext.getComm()); });

        std::vector<request_t> sendBlockRanges;

        for (const auto& receivedRequestMessage: recvMessagesRequests) {
            auto castedDataPtr           = reinterpret_cast<const request_t*>(receivedRequestMessage.data.data());
            auto castedPastTheEndDataPtr = reinterpret_cast<const request_t*>(
                receivedRequestMessage.data.data() + receivedRequestMessage.data.size());
            sendBlockRanges.insert(sendBlockRanges.end(), castedDataPtr, castedPastTheEndDataPtr);
        }

        std::sort(sendBlockRanges.begin(), sendBlockRanges.end(), sortByRankAndBegin);


        const auto recvMessages = sparseAllToAll(sendBlockRanges, _offsetMode, _mpiContext, _serializedBlocks.get());

        // Parse the received messages and call the user provided deserialization function.
        assert(_blockIdPermuter);
        handleReceivedBlocks(
            recvMessages, recvBlockRanges, _offsetMode, _constOffset, handleSerializedBlock, *_blockIdPermuter);
    }

    using block_range_external_t = std::pair<block_id_t, size_t>;
    using block_range_request_t  = std::pair<block_range_external_t, int>;

    // pushBlocks()
    //
    // Pushes blocks to other ranks in the replicated storage. That is, the caller provides the global
    // ids of those blocks it has to sent and where to send it to. For the receiver to know which of its
    // received blocks corresponds to which global id, the same information has to be provided on the
    // receiver side.
    // This function is for example useful if each rank knows the full result of the load balancer. In
    // this scenario, each rank knows which blocks each other rank needs. Compared to pullBlocks() we
    // therefore don't need to communicate the requests for block ranges over the network.
    //
    // blockRanges: A list of <blockRange, destinationRank> where a block range is a tuple of global
    //      block ids <firstId, numberOfBlocks>
    // handleSerializedBlock: A function which takes a void * pointing to the start of the serialized
    //      byte stream, a length in bytes of this encoding and the global id of this block.
    // canBeParallelized: Indicates if multiple handleSerializedBlock calls can happen on different
    //      inputs concurrently.
    template <class HandleSerializedBlockFunction>
    void pushBlocksCurrentRankIds(
        const std::vector<std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::current_rank_t>>& blockRanges,
        HandleSerializedBlockFunction                                                            handleSerializedBlock,
        bool canBeParallelized = false // not supported yet
    ) {
        if (_offsetMode == OffsetMode::lookUpTable) {
            throw std::runtime_error("LUT mode is not implemented yet.");
        }
        UNUSED(canBeParallelized);

        // Project the block ids from the user ids to the internal ids. This means that the length of the requested
        // block ranges change, too. If we are using the RangePermutation, we will still get some consecutive blocks
        // ids. E.g. the requested range [0,100) might get translated to [0,10), [80, 90), [20, 30), ...
        assert(_blockIdPermuter);
        const auto internalBlockRanges = projectBlockRequestsFromUserToPermutedIDs(blockRanges, *_blockIdPermuter);

        // Transfer the blocks over the network
        const auto [sendBlockRanges, recvBlockRanges] =
            getSendRecvBlockRanges(internalBlockRanges, _blockDistribution.get(), _mpiContext);
        const auto recvMessages = sparseAllToAll(sendBlockRanges, _offsetMode, _mpiContext, _serializedBlocks.get());

        // Parse the received messages and call the user provided deserialization function.
        assert(_blockIdPermuter);
        handleReceivedBlocks(
            recvMessages, recvBlockRanges, _offsetMode, _constOffset, handleSerializedBlock, *_blockIdPermuter);
    }

    // Warning! This changes the blockRanges destination rank from original ranks to current ranks!
    template <class HandleSerializedBlockFunction>
    void pushBlocksOriginalRankIds(
        std::vector<std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::original_rank_t>>& blockRanges,
        HandleSerializedBlockFunction                                                       handleSerializedBlock,
        bool canBeParallelized = false // not supported yet
    ) {
        std::transform(
            blockRanges.begin(), blockRanges.end(), blockRanges.begin(),
            [this](std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::current_rank_t> blockRange) {
                return std::make_pair(blockRange.first, _mpiContext.getCurrentRank(blockRange.second).value());
            });
        pushBlocksCurrentRankIds(blockRanges, handleSerializedBlock, canBeParallelized);
    }

    template <class HandleSerializedBlockFunction>
    void pushBlocksOriginalRankIds(
        const std::vector<std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::original_rank_t>>& blockRanges,
        HandleSerializedBlockFunction                                                             handleSerializedBlock,
        bool canBeParallelized = false // not supported yet
    ) {
        auto blockRangesCopy(blockRanges);
        pushBlocksOriginalRankIds(blockRangesCopy, handleSerializedBlock, canBeParallelized);
    }

    private:
#ifdef ID_RANDOMIZATION
    using BlockIdPermuter = RangePermutation<FeistelPseudoRandomPermutation>;
#else
    using BlockIdPermuter = IdentityPermutation;
#endif

    const uint16_t                            _replicationLevel;
    const OffsetMode                          _offsetMode;
    const size_t                              _constOffset;
    const uint64_t                            _randomPermutationSeed;
    const uint64_t                            _blocksPerPermutationRange;
    ReStoreMPI::MPIContext                    _mpiContext;
    std::shared_ptr<BlockDistribution<>>      _blockDistribution;
    std::unique_ptr<SerializedBlockStorage<>> _serializedBlocks;
    std::optional<BlockIdPermuter>            _blockIdPermuter;

    void _assertInvariants() const {
        assert(
            (_offsetMode == OffsetMode::constant && _constOffset > 0)
            || (_offsetMode == OffsetMode::lookUpTable && _constOffset == 0));
        assert(_replicationLevel > 0);
    }
}; // class ReStore

/*
Indended usage:

Storage storage;
// storage.setProcessMap(...) --- Skipped for now
storage.setReplication(k)
storage.setOffsetMode(constant|explicit, size_t c = 0)
storage.submitBlocks(...)

!! failure !!
pushPullBlocks(...) || pullBlocks(...)
*/

} // namespace ReStore
#endif // RESTORE_CORE_H
