#ifndef RESTORE_CORE_H
#define RESTORE_CORE_H

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "mpi.h"
#include "mpi_context.hpp"


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

    ReStore(uint32_t replicationLevel, OffsetMode offsetMode, size_t constOffset = 0)
        : _replicationLevel(replicationLevel),
          _offsetMode(offsetMode),
          _constOffset(constOffset),
          mpiContext(MPI_COMM_WORLD) {
        if (offsetMode == OffsetMode::lookUpTable && constOffset != 0) {
            throw std::runtime_error("Explicit offset mode set but the constant offset is not zero.");
        }
        if (offsetMode == OffsetMode::constant && constOffset == 0) {
            throw std::runtime_error("Constant offset mode required a constOffset > 0.");
        }
        if (replicationLevel == 0) { throw std::runtime_error("What is a replication level of 0 supposed to mean?"); }
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
    uint32_t replicationLevel() const { return this->_replicationLevel; }

    // offsetMode()
    //
    // Get the offset mode that defines how the serialized blocks are aligned in memory.
    std::pair<OffsetMode, size_t> offsetMode() const {
        assert(
            (_offsetMode == OffsetMode::constant && _constOffset > 0)
            || (_offsetMode == OffsetMode::lookUpTable && _constOffset == 0));

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
        // Allocate one send buffer per block range. That is, if blocks

        // Loop over the nextBlock generator to fetch all block we need to serialize

        // Determine the receivers of copies of this block

        // Call serialize once, instructing it to write the serialization at to one buffer

        // Copy the serialization to the other receiver's buffers

        // All blocks have been serialized, send & receive replicas
        std::vector<ReStoreMPIContext::Message<unsigned char>> messages;
        mpiContext.SparseAllToall(messages);
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
    const uint16_t    _replicationLevel;
    const OffsetMode  _offsetMode;
    const size_t      _constOffset;
    ReStoreMPIContext mpiContext;
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
