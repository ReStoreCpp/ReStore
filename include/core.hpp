#include <functional>
#include <utility>
#include <vector>
#include <optional>

template <class BlockType>
class ReStore
{
    public:
    // TODO Describe the difference between global and local block ids

    // Defines how the serialized blocks are aligned in memory.
    // See the documentation for offsetMode() for details.
    enum class OffsetMode : uint8_t { constant; explicit };

    ReStore(uint32_t replicationLevel, OffsetMode offsetMode, const_offset = 0) :
        replicationLevel(replicationLevel),
        offsetMode(offsetMode),
        const_offset(constnt_offset) {
        if (offsetMode == OffsetMode::explicit && const_offset != 0) {
            throw runtime_error("Explicit offset mode set but the constant offset is not zero.")
        }
    }

    // Copying a ReStore object does not really make sense. It would be really hard and probably not
    // what you want to deep copy the replicated blocks (including the remote ones?), too.
    ReStore(const ReStore& other) = delete;
    operator=(const ReStore& other) = delete;

    // Moving a ReStore is fine
    // TODO Implement
    ReStore& ReStore(ReStore&& other) {
    }

    ReStore& operator=(ReStore&& other) {
    }

    // Destructor
    ~ReStore() {
        // TODO Free all allocated storage allocated for blocks
    }

    // replicationLevel()
    //
    // Get the replication level, that is how many copies of each block are scattered over the ranks.
    uint32_t replicationLevel() const { return this->replicationLevel; }

    // offsetMode()
    //
    // Get the offset mode that defines how the serialized blocks are aligned in memory.
    std::pair<OffsetMode, size_t> offsetMode() const {
        return std::make_pair<OffsetMode, size_t>(this->offsetMode, this->const_offset);
    }

    // submitBlocks()
    //
    // Submits blocks to the replicated storage. They will be replicated among the ranks and can be
    // ReStored after a rank failure. Each rank has to call this function exactly once.
    // submitBlocks() also performs the replication and is therefore blocking until all ranks called it.
    // 
    // serializeFunc: gets a reference to a block to serialize and a void * pointing to the destination
    //      (where to write the serialized block). it should return the number of bytes written.
    // nextBlock: a generator function which should return <globalBlockId, const reference to block>
    //      on each call. If there are no more blocks getNextBlock should return {}
    // canBeParallelized: Indicates if multiple serializeFunc calls can happen on different blocks
    //      concurrently. Also assumes that the blocks do not have to be serialized in the order they
    //      are emitted by nextBlock.
    void submitBlocks(
        std::function<size_t(const BlockType&, void*)> serializeFunc,
        std::function<std::optional<std::pair<size_t globalId, const BlockType&>>()> nextBlock,
        bool canBeParallelized = false // not supported yet
    ) {

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
        std::vector<std::pair<size_t, size_t>> blockRanges,
        std::function<void(void*, size_t, size_t)> handleSerializedBlock,
        bool canBeParallelized = false // not supported yet
    );

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
    pushBlocks(
        std::vector<std::pair<std::pair<size_t, size_t>, int>> blockRanges,
        std::function<void(void*, length, globalId)> handleSerializedBlock,
        bool canBeParallelized = false // not supported yet
    );

    private:
    const uint16_t replicationLevel;
    const offsetMode offsetMode;
    const size_t const_offset;
}

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
