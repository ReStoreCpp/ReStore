#ifndef RESTORE_CORE_H
#define RESTORE_CORE_H

#include <cassert>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mpi_context.hpp"
#include <mpi.h>


template <class BlockType>
class ReStore {
    public:
    // Defines how the serialized blocks are aligned in memory.
    // See the documentation for offsetMode() for details.
    enum class OffsetMode : uint8_t { constant, lookUpTable };

    // Global and local id. The global id is unique across all ranks, that is each copy of the
    // same block has the same global id on all ranks. We can use the global id to request a
    // block.
    // TODO Do we need local ids? If we do, describe the difference between global and local block ids

    typedef size_t global_id;

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
    ReStore(MPI_Comm mpiCommunicator, uint32_t replicationLevel, OffsetMode offsetMode, size_t constOffset = 0)
        : _replicationLevel(replicationLevel),
          _offsetMode(offsetMode),
          _constOffset(constOffset),
          _mpiContext(mpiCommunicator) {
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
    uint32_t replicationLevel() const {
        _assertInvariants();
        return this->_replicationLevel;
    }

    // offsetMode()
    //
    // Get the offset mode that defines how the serialized blocks are aligned in memory.
    std::pair<OffsetMode, size_t> offsetMode() const {
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
    // serializeFunc: gets a reference to a block to serialize and a void * pointing to the destination
    //      (where to write the serialized block). it should return the number of bytes written.
    // nextBlock: a generator function which should return <globalBlockId, const reference to block>
    //      on each call. If there are no more blocks getNextBlock should return {}
    // canBeParallelized: Indicates if multiple serializeFunc calls can happen on different blocks
    //      concurrently. Also assumes that the blocks do not have to be serialized in the order they
    //      are emitted by nextBlock.
    void submitBlocks(
        std::function<size_t(const BlockType&, void*)>                         serializeFunc,
        std::function<std::optional<std::pair<global_id, const BlockType&>>()> nextBlock,
        bool canBeParallelized = false // not supported yet
    ) {
        _assertInvariants();

        // Determine which rank will get which block range

        // Allocate one send buffer per block range. That is, those ranks which get the same blocks share a common
        // sendbuffer.

        // Loop over the nextBlock generator to fetch all block we need to serialize

        // Determine the receivers of copies of this block

        // Call serialize once, instructing it to write the serialization to one buffer

        // All blocks have been serialized, send & receive replicas
        std::vector<ReStoreMPI::Message> messages;
        _mpiContext.SparseAllToAll(messages);
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
    void pullBlocks(
        std::vector<std::pair<size_t, size_t>>     blockRanges,
        std::function<void(void*, size_t, size_t)> handleSerializedBlock,
        bool                                       canBeParallelized = false // not supported yet
    ) {}

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
        std::vector<std::pair<std::pair<size_t, size_t>, int>> blockRanges,
        std::function<void(void*, size_t, global_id)>          handleSerializedBlock,
        bool                                                   canBeParallelized = false // not supported yet
    ) {}

    private:
    const uint16_t         _replicationLevel;
    const OffsetMode       _offsetMode;
    const size_t           _constOffset;
    ReStoreMPI::MPIContext _mpiContext;

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

#endif // RESTORE_CORE_H
