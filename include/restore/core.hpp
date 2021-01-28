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
    typedef size_t block_id_t;

    // BlockDistribution
    //
    // A class that models the distribution of blocks across the different ranks. For now, we only implement a single
    // strategy. However, the goal is to enable the user to choose his own distribution strategy.
    //
    // The strategy implemented by this class is as follows:
    //   Let k be the replication level
    //   Let s be the shift width
    //   - Blocks with adjacent global ids are grouped together into one block range.
    //   - The goal is to distrubute the blocks as evenly as possible among the ranks. Let's call the number of blocks
    //     the _range_ with the fewest blocks has bpr (blocks per range). No range has more than bpr+1 blocks.
    //   - It the first range, there are the blocks [0..bpr-1], in the second ranges there are the blocks [bpr..2bpr-1].
    //     If the number of blocks is not evenly dividable by the number of blocks, the last few ranges will get one
    //     block less than bpr. This way, all ranges will have bpr or bpr-1 blocks.
    //   - Each block range is stored on k different ranks. This means there are k ranges stored on each rank.
    //   - If the replication level is 3, and the rank id of the first rank which stores a particular block range is
    //     fid, the block is stored on fid, fid+s and fid+2s.
    class BlockDistribution {
        public:
        // BlockRange
        //
        // Represents a range of blocks. All blocks in this range have consecutive ids. For now, we store the ranges's
        // id, start block and number of blocks (length) explicitely. This uses more space than necessary+ but decreases
        // the dependency between BlockRange and BlockDistribution.
        // + We could compute the length and start from the id using the information in the corresponding
        // BlockDistribution object.
        struct BlockRange {
            block_id_t start;
            size_t     length;
            size_t     id;

            // Constructor
            //
            // Build a block range from the given block id. We need to know the number of blocks and the number ranges
            // to compute the starting block and number of block in this BlockRange.
            BlockRange(size_t range_id, size_t numBlocks, size_t numRanges) {
                assert(numRanges <= numBlocks);

                size_t blocksPerRange               = numBlocks / numRanges;
                size_t numRangesWithAdditionalBlock = numBlocks - blocksPerRange * numRanges;

                assert(blocksPerRange > 0);
                assert(blocksPerRange <= numBlocks);
                assert(blocksPerRange * numRanges + numRangesWithAdditionalBlock == numBlocks);

                // Do we - and all blocks with a lower id than us - have an additional block?
                if (range_id < numRangesWithAdditionalBlock) {
                    start  = range_id * (blocksPerRange + 1);
                    length = blocksPerRange + 1;
                } else {
                    start  = blocksPerRange * range_id + numRangesWithAdditionalBlock;
                    length = blocksPerRange;
                }

                assert(start < numBlocks);
                assert(length == blocksPerRange || length == blocksPerRange - 1);
            }

            // partOf()
            //
            // Returns true if the given block is part of this range; false otherwise
            bool includes(block_id_t block) { return block >= this->start && block < this->start + this->length; }
        };

        BlockDistribution(
            uint32_t numRanks, size_t numBlocks, uint16_t replicationLevel, ReStoreMPI::MPIContext& mpiContext)
            : _numRanks(numRanks),
              _numBlocks(numBlocks),
              _replicationLevel(replicationLevel),
              _numRanges(numRanks),
              _blocksPerRange(numBlocks / _numRanges),
              _numRangesWithAdditionalBlock(numBlocks - _blocksPerRange * _numRanges),
              _mpiContext(mpiContext),
              _shiftWidth(determineShiftWidth(numRanks, replicationLevel)) {
            if (numRanks <= 0) {
                throw std::runtime_error("There has to be at least one rank.");
            } else if (numBlocks == 0) {
                throw std::runtime_error("There has to be at least one block.");
            } else if (replicationLevel == 0) {
                throw std::runtime_error("A replication level of 0 is probably not what you want.");
            }
            assert(_numRanges > 0);
            assert(_blocksPerRange > 0);
            assert(_blocksPerRange <= _numBlocks);
            assert(_blocksPerRange * _numRanges + _numRangesWithAdditionalBlock == _numBlocks);
            assert(_shiftWidth > 0);
            assert(_shiftWidth * (replicationLevel - 1) < _numRanks);
        }

        // rangeForBlock()
        //
        // Computes the block range the given block is in.
        BlockRange rangeForBlock(block_id_t block) const {
            if (block < (_blocksPerRange + 1) * _numRangesWithAdditionalBlock) {
                return block / (_blocksPerRange + 1);
            } else {
                assert((block - (_blocksPerRange * _numRangesWithAdditionalBlock)) >= 0);
                return (block - (_blocksPerRange * _numRangesWithAdditionalBlock)) / _blocksPerRange;
            }
        }

        // ranksBlockIsStoredOn()
        //
        // Returns the ranks the given block is stored on. The ranks are identified by their original rank id.
        std::vector<ReStoreMPI::OriginalRank> ranksBlockIsStoredOn(block_id_t block) const {
            assert(block < _numBlocks);
            BlockRange range = rangeForBlock(block);
            assert(range.start.id < _numBlocks);
            assert(range.start.globalId + range.length < _numBlocks);
            assert(range.id < _numRanges);
            assert(range.id < _numRanks);

            // The range is located on the rank with the same id ...
            auto                     rankIds   = std::vector<ReStoreMPI::OriginalRank>();
            ReStoreMPI::OriginalRank firstRank = static_cast<ReStoreMPI::OriginalRank>(range.id);
            if (_mpiContext.isAlive(firstRank)) {
                rankIds.push_back(firstRank);
            }

            // ... and on <replication level> - 1 further ranks, all <shift width> apart.
            for (uint16_t replica = 1; replica < _replicationLevel; replica++) {
                ReStoreMPI::OriginalRank nextRank = static_cast<ReStoreMPI::OriginalRank>(
                    (static_cast<int>(firstRank) + _shiftWidth * replica) % _numRanks);
                assert(static_cast<int>(nextRank) < _numRanks);
                if (_mpiContext.isAlive(nextRank)) {
                    rankIds.push_back(nextRank);
                }
            }

            return rankIds;
        }

        // rangesStoredOnRank()
        //
        //  Returns the block ranges residing on the given rank
        std::vector<BlockRange> rangesStoredOnRank(ReStoreMPI::OriginalRank rankId) const {
            assert(static_cast<int>(rankId) >= 0);
            assert(static_cast<int>(rankId) < _numRanks);

            // The range with the same id as this rank is stored on this rank ...
            auto       rangeIds   = std::vector<BlockRange>();
            BlockRange firstRange = BlockRange(static_cast<size_t>(rankId));

            // ... as are <replication level> - 1 further ranges, all <shift width> apart
            for (uint16_t replica = 1; replica < _replicationLevel; replica++) {
                BlockRange nextRange = static_cast<BlockRange>((firstRange.id - _shiftWidth * replica) % _numRanks);
                assert(nextRange.id < _numRanges);
            }
        }

        // isStoredOn()
        //
        // Returns true if the given block or block range is stored on the given rank
        bool isStoredOn(BlockRange blockRange, ReStoreMPI::OriginalRank rankId) const {
            assert(static_cast<int>(rankId) >= 0);
            assert(static_cast<int>(rankId) < _numRanks);

            // I tried to find a closed form solution for this, it quickly grow to an angry beast.
            // Let's try this and think about a more clever solution once we actually _measure_ a performance
            // bottleneck.
            for (uint16_t replica = 1; replica < _replicationLevel; replica++) {
                int nextBlockId = (static_cast<int>(rankId) - _shiftWidth * replica) % _numRanges;
                assert(nextBlockId < _numBlocks);
                assert(nextBlockId >= 0);
                if (nextBlockId == blockRange.id) {
                    return true;
                }
            }
            return false;
        }

        bool isStoredOn(block_id_t block, ReStoreMPI::OriginalRank rankId) const {
            return isStoredOn(rangeForBlock(block), rankId);
        }

        private:
        uint32_t determineShiftWidth(uint32_t numRanks, uint16_t replicationLevel) const {
            return numRanks / replicationLevel;
        }

        const size_t           _numBlocks;
        const uint32_t         _numRanks;
        const uint16_t         _replicationLevel;
        const size_t           _blocksPerRange;
        const size_t           _numRangesWithAdditionalBlock;
        const size_t           _numRanges;
        const size_t           _shiftWidth;
        ReStoreMPI::MPIContext _mpiContext;
    };

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
        std::function<size_t(const BlockType&, void*)>                          serializeFunc,
        std::function<std::optional<std::pair<block_id_t, const BlockType&>>()> nextBlock,
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
    // TODO Rename Block, which contains only the globalId. As the struct does not really store the block
    // this might be confusing when appearing in the interface.
    void pushBlocks(
        std::vector<std::pair<std::pair<size_t, size_t>, int>> blockRanges,
        std::function<void(void*, size_t, block_id_t)>         handleSerializedBlock,
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
