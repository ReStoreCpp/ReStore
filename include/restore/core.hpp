#ifndef RESTORE_CORE_H
#define RESTORE_CORE_H

#include <cassert>
#include <cstdint>
#include <functional>
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

#include "helpers.hpp"
#include "restore/block_distribution.hpp"
#include "restore/block_serialization.hpp"

namespace ReStore {

template <class BlockType>
class ReStore {
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
    ReStore(MPI_Comm mpiCommunicator, uint16_t replicationLevel, OffsetMode offsetMode, size_t constOffset = 0)
        : _replicationLevel(replicationLevel),
          _offsetMode(offsetMode),
          _constOffset(constOffset),
          _mpiContext(mpiCommunicator),
          _blockDistribution(nullptr), // Depends on the number of blocks which are submitted in submitBlocks.
          _serializedBlocks(nullptr) { // Depends on _blockDistribution
        if (offsetMode == OffsetMode::lookUpTable && constOffset != 0) {
            throw std::runtime_error("Explicit offset mode set but the constant offset is not zero.");
        } else if (offsetMode == OffsetMode::constant && constOffset == 0) {
            throw std::runtime_error("Constant offset mode required a constOffset > 0.");
        } else if (replicationLevel == 0) {
            throw std::runtime_error("What is a replication level of 0 supposed to mean?");
        } else {
            _assertInvariants();
        }
    }

    // Copying a ReStore object does not really make sense. It would be really hard and probably not
    // what you want to deep copy the replicated blocks (including the remote ones?), too.
    ReStore(const ReStore& other) = delete;
    ReStore& operator=(const ReStore& other) = delete;

    // Moving a ReStore is fine
    ReStore(ReStore&& other) {
        // TODO Implement
    }

    ReStore& operator=(ReStore&& other) {
        // TODO implement
    }

    // Destructor
    ~ReStore() {
        // TODO Free all allocated storage allocated for blocks
    }

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
    std::pair<OffsetMode, size_t> offsetMode() const noexcept {
        _assertInvariants();
        return std::make_pair(this->_offsetMode, this->_constOffset);
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
    // totalNumberOfBlocks: The total number of blocks across all ranks. // TODO quickly discuss with Demian
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
        UNUSED(canBeParallelized);
        static_assert(
            std::is_invocable_r<size_t, SerializeBlockCallbackFunction, const BlockType&, SerializedBlockStoreStream>(),
            "serializeFunc must be invocable as size_t(const BlockType&, SerializedBlockStoreStream");
        static_assert(
            std::is_invocable_r<std::optional<std::pair<block_id_t, const BlockType&>>, NextBlockCallbackFunction>(),
            "serializeFunc must be invocable as std::optional<std::pair<block_id_t, const BlockType&>>()");

        if (totalNumberOfBlocks == 0) {
            throw std::runtime_error("Invalid number of blocks: 0.");
        }

        try { // Ranks failures might be detected during this block
            // We define original rank ids to be the rank ids during this function call
            _mpiContext.resetOriginalCommToCurrentComm();

            // Initialize the Block Distribution
            if (!_blockDistribution) {
                throw std::runtime_error("You shall not call submitBlocks twice!");
            }
            _blockDistribution = std::make_shared<BlockDistribution<>>(
                _mpiContext.getOriginalSize(), totalNumberOfBlocks, _replicationLevel, _mpiContext);
            _serializedBlocks =
                std::make_shared<SerializedBlockStorage<>>(_blockDistribution, _offsetMode, _constOffset);
            assert(_mpiContext.getOriginalSize() == _mpiContext.getCurrentSize());

            // Allocate one send buffer per destination rank
            // If the user did his homework and designed a BlockDistribution which requires few messages to be send
            // we do not want to allocate all those unneeded send buffers... that's why we use a map here instead
            // of a vector.
            auto sendBuffers = std::make_shared<std::map<ReStoreMPI::current_rank_t, std::vector<uint8_t>>>();

            bool noMoreBlocksToSerialize = false;
            // Loop over the nextBlock generator to fetch all block we need to serialize
            do {
                std::optional<std::pair<block_id_t, const BlockType&>> next = nextBlock();
                if (!next) {
                    noMoreBlocksToSerialize = true;
                } else {
                    block_id_t       blockId = next.value().first;
                    const BlockType& block   = next.value().second;

                    // Determine which ranks will get this block
                    assert(_blockDistribution);
                    auto ranks = std::make_shared<std::vector<ReStoreMPI::current_rank_t>>(
                        _mpiContext.getAliveCurrentRanks(_blockDistribution->ranksBlockIsStoredOn(blockId)));

                    // Create the proxy which the user defined serializer will write to. This proxy overloads the <<
                    // operator and automatically copies the written bytes to every destination rank's send buffer.
                    auto storeStream = SerializedBlockStoreStream(sendBuffers, ranks);

                    // Write the block's id to the stream
                    storeStream << blockId;
                    // TODO implement LUT mode

                    // Call the user-defined serialization function to serialize the block to a flat byte stream
                    serializeFunc(block, storeStream);
                }
            } while (!noMoreBlocksToSerialize);

            // All blocks have been serialized, send & receive replicas
            std::vector<ReStoreMPI::Message> sendMessages;

            for (auto&& [rankId, buffer]: *sendBuffers) {
                sendMessages.emplace_back(std::shared_ptr<uint8_t>(buffer.data()), buffer.size(), rankId);
            }
            auto receiveMessages = _mpiContext.SparseAllToAll(sendMessages);

            // Store the received blocks into our local block storage
            // TODO implement LUT mode

            assert(_mpiContext.getMyOriginalRank() == _mpiContext.getMyCurrentRank());
            _serializedBlocks->registerRanges(_blockDistribution->rangesStoredOnRank(_mpiContext.getMyOriginalRank()));

            for (auto&& message: receiveMessages) {
            }

        } catch (ReStoreMPI::FaultException& e) {
            // Reset BlockDistribution
            _blockDistribution = nullptr;
            throw e;
        }
    }

    // pullBlocks()
    //
    // Pulls blocks from other ranks in the replicated storage. That is, the caller provides the global
    // ids of those blocks it wants but not from which rank to fetch them.
    // This means that we have to perform an extra round of communication compared with pushBlocks() to
    // request the blocks each rank wants.
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
        // HandleSerializedBlockFunction void(SerializedBlockOutStream, size_t lengthOfStreamInBytes, block_id_t)
    }

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
    void pushBlocks(
        std::vector<std::pair<std::pair<block_id_t, size_t>, int>> blockRanges,
        std::function<void(void*, size_t, block_id_t)>             handleSerializedBlock,
        bool                                                       canBeParallelized = false // not supported yet
    ) {}

    private:
    const uint16_t                            _replicationLevel;
    const OffsetMode                          _offsetMode;
    const size_t                              _constOffset;
    ReStoreMPI::MPIContext                    _mpiContext;
    std::shared_ptr<BlockDistribution<>>      _blockDistribution;
    std::shared_ptr<SerializedBlockStorage<>> _serializedBlocks;

    void _assertInvariants() const {
        assert(
            (_offsetMode == OffsetMode::constant && _constOffset > 0)
            || (_offsetMode == OffsetMode::lookUpTable && _constOffset == 0));
        assert(_replicationLevel > 0);
    }
};

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
