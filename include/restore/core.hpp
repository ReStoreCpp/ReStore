#ifndef RESTORE_CORE_H
#define RESTORE_CORE_H

#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
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
    template <typename MPIContext = ReStoreMPI::MPIContext>
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
            BlockRange(size_t range_id, size_t numBlocks, size_t numRanges) : id(range_id) {
                if (numRanges > numBlocks) {
                    throw std::runtime_error("There cannot be more block ranges than blocks.");
                }

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

                if (start >= numBlocks) {
                    throw std::runtime_error("This range does not exists (id too large).");
                }
                assert(length == blocksPerRange || length == blocksPerRange + 1);
            }

            // Copying and copy assignment is fine ...
            BlockRange(const BlockRange&) = default;
            BlockRange& operator=(const BlockRange&) = default;

            // ... as is moving and move assignment
            BlockRange(BlockRange&&) = default;
            BlockRange& operator=(BlockRange&&) = default;

            // contains()
            //
            // Returns true if the given block is part of this range; false otherwise
            bool contains(block_id_t block) {
                return block >= this->start && block < this->start + this->length;
            }

            // Comparison Operator
            //
            // We assume that both block ranges belong to the same BlockDistribution.
            bool operator==(const BlockRange& that) const {
                assert(this->id != that.id || this->start == that.start);
                assert(this->id == that.id || this->start != that.start);
                return this->id == that.id;
            }
            bool operator!=(const BlockRange& that) const {
                return !(*this == that);
            }

            // How to print a BlockRange
            //
            // Writes "<BlockRange(id=0,start=1,length=2)>" to the ostream object.
            friend std::ostream& operator<<(std::ostream& os, const BlockRange& blockRange) {
                return os << "<BlockRange(id=" << blockRange.id << ",start=" << blockRange.start
                          << ",length=" << blockRange.length << ")>";
            }
        };

        BlockDistribution(uint32_t numRanks, size_t numBlocks, uint16_t replicationLevel, const MPIContext& mpiContext)
            : _constructorArgumentsValid(
                validateConstructorArguments(numRanks, numBlocks, replicationLevel, mpiContext)),
              _numBlocks(numBlocks),
              _numRanks(numRanks),
              _replicationLevel(replicationLevel),
              _numRanges(static_cast<size_t>(numRanks)),
              _blocksPerRange(numBlocks / _numRanges),
              _numRangesWithAdditionalBlock(numBlocks - _blocksPerRange * _numRanges),
              _shiftWidth(determineShiftWidth(numRanks, replicationLevel)),
              _mpiContext(mpiContext) {
            assert(_numRanges > 0);
            assert(_blocksPerRange > 0);
            assert(_blocksPerRange <= _numBlocks);
            assert(_blocksPerRange * _numRanges + _numRangesWithAdditionalBlock == _numBlocks);
            assert(_shiftWidth > 0);
            assert(_shiftWidth * (replicationLevel - 1u) < _numRanks);
        }

        // Copying is not okay, because we cannot copy the MPIContext class
        BlockDistribution(const BlockDistribution&) = delete;
        BlockDistribution& operator=(const BlockDistribution&) = delete;

        // Moving is fine
        BlockDistribution(BlockDistribution&&) = default;
        BlockDistribution& operator=(BlockDistribution&&) = default;

        // blockRangeById()
        //
        // A factory method to build a BlockRange by it's id.
        BlockRange blockRangeById(size_t rangeId) const {
            return BlockRange(rangeId, _numBlocks, _numRanges);
        }

        // rangeOfBlock()
        //
        // Computes the block range the given block is in.
        BlockRange rangeOfBlock(block_id_t block) const {
            if (block >= _numBlocks) {
                throw std::runtime_error("Block id is greater than (or equal to) the number of blocks.");
            }
            assert(_blocksPerRange > 0);
            assert(_blocksPerRange < _numBlocks);
            assert(_numRangesWithAdditionalBlock < _numRanges);

            if (block < (_blocksPerRange + 1) * _numRangesWithAdditionalBlock) {
                size_t blockId = block / (_blocksPerRange + 1);
                return blockRangeById(blockId);
            } else {
                assert(block >= _blocksPerRange * _numRangesWithAdditionalBlock);
                size_t rangeId = _numRangesWithAdditionalBlock
                                 + (block - ((_blocksPerRange + 1) * _numRangesWithAdditionalBlock)) / _blocksPerRange;
                return blockRangeById(rangeId);
            }
        }

        // ranksBlockIsStoredOn()
        //
        // Returns the ranks the given block is stored on. The ranks are identified by their original rank id.
        // Dead ranks are filtered from the result list.
        std::vector<ReStoreMPI::original_rank_t> ranksBlockIsStoredOn(block_id_t block) const {
            assert(block < _numBlocks);
            BlockRange range = rangeOfBlock(block);
            assert(range.start < _numBlocks);
            assert(range.start + range.length <= _numBlocks);
            assert(range.id < _numRanges);
            assert(range.id < _numRanks);

            // The range is located on the rank with the same id ...
            auto rankIds = std::vector<ReStoreMPI::original_rank_t>();
            assert(range.id <= std::numeric_limits<ReStoreMPI::original_rank_t>::max());
            assert(range.id >= 0);
            ReStoreMPI::original_rank_t firstRank = static_cast<ReStoreMPI::original_rank_t>(range.id);
            rankIds.push_back(firstRank);

            // ... and on <replication level> - 1 further ranks, all <shift width> apart.
            for (uint16_t replica = 1; replica < _replicationLevel; replica++) {
                assert(firstRank >= 0);
                ReStoreMPI::original_rank_t nextRank = static_cast<ReStoreMPI::original_rank_t>(
                    (static_cast<uint64_t>(firstRank) + _shiftWidth * replica) % _numRanks);
                assert(nextRank >= 0);
                assert(static_cast<size_t>(nextRank) < _numRanks);
                rankIds.push_back(nextRank);
            }

            return _mpiContext.getOnlyAlive(rankIds);
        }

        // rangesStoredOnRank()
        //
        // Returns the block ranges residing on the given rank.
        // If the given rank is dead, this returns an empty vector.
        std::vector<BlockRange> rangesStoredOnRank(ReStoreMPI::original_rank_t rankId) const {
            if (!_mpiContext.isAlive(rankId)) {
                return std::vector<BlockRange>();
            }

            if (rankId < 0) {
                throw std::runtime_error("Invalid rank id: Less than zero.");
            } else if (static_cast<size_t>(rankId) > _numRanks) {
                throw std::runtime_error("Invalid rank id: lower than the number of ranks.");
            }

            // The range with the same id as this rank is stored on this rank ...
            auto rangeIds = std::vector<BlockRange>();
            assert(rankId >= 0);
            BlockRange firstRange = blockRangeById(static_cast<size_t>(rankId));
            rangeIds.push_back(firstRange);

            // ... as are <replication level> - 1 further ranges, all <shift width> apart
            for (uint16_t replica = 1; replica < _replicationLevel; replica++) {
                assert(firstRange.id <= std::numeric_limits<int64_t>::max());
                assert(_shiftWidth * replica <= std::numeric_limits<int64_t>::max());
                assert(_numRanges < std::numeric_limits<int64_t>::max());
                assert(_numRanges > 0);

                assert(_shiftWidth * replica <= std::numeric_limits<int64_t>::max());
                assert(_shiftWidth <= std::numeric_limits<int64_t>::max());
                static_assert(std::numeric_limits<decltype(replica)>::max() <= std::numeric_limits<int32_t>::max());
                int64_t rangeId = static_cast<int64_t>(firstRange.id)
                                  - static_cast<int64_t>(_shiftWidth) * static_cast<int32_t>(replica);
                if (rangeId < 0) {
                    assert(_numRanges < std::numeric_limits<int64_t>::max());
                    rangeId = static_cast<int64_t>(_numRanges)
                              + static_cast<int64_t>(rangeId % static_cast<int64_t>(_numRanges));
                }
                assert(rangeId >= 0);
                BlockRange nextRange = blockRangeById(static_cast<size_t>(rangeId));
                assert(nextRange.id < _numRanges);
                rangeIds.push_back(nextRange);
            }

            return rangeIds;
        }

        // isStoredOn()
        //
        // Returns true if the given block or block range is stored on the given rank.
        // If the given rank is dead, this will return false.
        bool isStoredOn(BlockRange blockRange, ReStoreMPI::original_rank_t rankId) const {
            if (rankId < 0) {
                throw std::runtime_error("A rank id cannot be negative.");
            } else if (static_cast<size_t>(rankId) >= _numRanks) {
                throw std::runtime_error("Rank id larger than (or equal to) the number of ranks.");
            } else if (blockRange.id > _numRanges) {
                throw std::runtime_error("The given block range's id is too large.");
            }

            // If the given rank is dead, return false (it does not store anything).
            if (!_mpiContext.isAlive(rankId)) {
                return false;
            }

            // I tried to find a closed form solution for this, it quickly grow to an angry beast.
            // Let's try this and think about a more clever solution once we actually _measure_ a performance
            // bottleneck.
            for (uint16_t replica = 0; replica < _replicationLevel; replica++) {
                assert(_shiftWidth * replica <= std::numeric_limits<int64_t>::max());
                assert(_numRanges < std::numeric_limits<int64_t>::max());
                assert(_numRanges > 0);

                assert(_shiftWidth * replica <= std::numeric_limits<int64_t>::max());
                assert(_shiftWidth <= std::numeric_limits<int64_t>::max());
                static_assert(std::numeric_limits<decltype(replica)>::max() <= std::numeric_limits<int32_t>::max());
                int64_t rangeId = rankId - static_cast<int64_t>(_shiftWidth) * static_cast<int32_t>(replica);
                if (rangeId < 0) {
                    assert(_numRanges < std::numeric_limits<int64_t>::max());
                    rangeId = static_cast<int64_t>(_numRanges)
                              + static_cast<int64_t>(rangeId % static_cast<int64_t>(_numRanges));
                }
                assert(rankId >= 0);
                if (static_cast<size_t>(rangeId) == blockRange.id) {
                    return true;
                }
            }
            return false;
        }

        bool isStoredOn(block_id_t block, ReStoreMPI::original_rank_t rankId) const {
            return isStoredOn(rangeOfBlock(block), rankId);
        }

        // shiftWidth()
        //
        // Return the shift width. That is, if a block range is stored on rank i, it is also stored on rank i = shift
        // width (until the replication level is reached).
        size_t shiftWidth() const {
            assert(_shiftWidth < _numRanks);
            return _shiftWidth;
        }

        // numBlocks()
        //
        // Return the number of blocks.
        size_t numBlocks() const {
            return _numBlocks;
        }

        // numRanks()
        //
        // Return the number of ranks. This is *not* the number of ranks which are still alive!
        size_t numRanks() const {
            return _numRanks;
        }

        // replicationLevel()
        //
        // Return the replicationLevel. This is *not* the current replication level (including failed ranks)!
        uint16_t replicationLevel() const {
            return _replicationLevel;
        }

        // blocksPerRange()
        //
        // Return the number of blocks in each range. The first numRangesWithAdditionalBlock() ranges will have a single
        // additional range.
        size_t blocksPerRange() const {
            return _blocksPerRange;
        }

        // numRangesWithAdditionalBlock()
        //
        // Return the number of ranges that will have one block more than the what blocksPerRange() returns.
        size_t numRangesWithAdditionalBlock() const {
            return _numRangesWithAdditionalBlock;
        }

        // numRanges()
        //
        // Return the number of block ranges.
        size_t numRanges() const {
            return _numRanges;
        }

        private:
        uint32_t determineShiftWidth(uint32_t numRanks, uint16_t replicationLevel) const {
            assert(numRanks > 0);
            assert(replicationLevel > 0);
            assert(replicationLevel <= numRanks);
            return static_cast<uint32_t>(numRanks) / replicationLevel;
        }

        bool validateConstructorArguments(
            uint32_t numRanks, size_t numBlocks, uint16_t replicationLevel, const MPIContext& mpiContext) const {
            if (numRanks <= 0) {
                throw std::runtime_error("There has to be at least one rank.");
            } else if (numBlocks == 0) {
                throw std::runtime_error("There has to be at least one block.");
            } else if (replicationLevel == 0) {
                throw std::runtime_error("A replication level of 0 is probably not what you want.");
            } else if (replicationLevel > numRanks) {
                throw std::runtime_error(
                    "A replication level that is greater than the number of ranks cannot be fulfilled.");
            } else if (numBlocks < numRanks) {
                throw std::runtime_error("Having less blocks than ranks is unsupported.");
            }
            UNUSED(mpiContext);

            return true;
        }

        const bool        _constructorArgumentsValid;
        const size_t      _numBlocks;
        const uint32_t    _numRanks;
        const uint16_t    _replicationLevel;
        const size_t      _numRanges;
        const size_t      _blocksPerRange;
        const size_t      _numRangesWithAdditionalBlock;
        const size_t      _shiftWidth;
        const MPIContext& _mpiContext;
    };

    class SerializedBlockStoreStream {
        public:
        SerializedBlockStoreStream(
            std::shared_ptr<std::map<ReStoreMPI::current_rank_t, std::vector<uint8_t>>> buffers,
            std::shared_ptr<std::vector<ReStoreMPI::current_rank_t>>                    ranks)
            : _buffers(buffers),
              _ranks(ranks),
              _bytesWritten(0) {
            if (!buffers || !ranks) {
                throw std::runtime_error("buffers and ranks have to point to a valid object.");
            }
        }

        template <class T>
        SerializedBlockStoreStream& operator<<(const T& value) {
            static_assert(std::is_pod<T>(), "You may only serialize a POD this way.");
            assert(_buffers);
            assert(_ranks);

            auto src = reinterpret_cast<const uint8_t*>(&value);
            for (auto rank: *_ranks) {
                if (_buffers->find(rank) == _buffers->end()) {
                    (*_buffers)[rank] = std::vector<uint8_t>();
                }
                assert(rank > 0);
                assert(static_cast<size_t>(rank) < _buffers->size());
                (*_buffers)[rank].insert((*_buffers)[static_cast<size_t>(rank)].end(), src, src + sizeof(T));
            }
            _bytesWritten += sizeof(T);

            return *this;
        }

        size_t bytesWritten() const noexcept {
            return _bytesWritten;
        }

        private:
        std::shared_ptr<std::map<ReStoreMPI::current_rank_t, std::vector<uint8_t>>> _buffers; // One buffer per rank
        std::shared_ptr<std::vector<ReStoreMPI::current_rank_t>>                    _ranks;   // Which ranks to send to
        size_t                                                                      _bytesWritten;
    };

    class SerializedBlockLoadStream {
        public:
        std::vector<int>           data;
        SerializedBlockLoadStream& operator<<(int val) {
            data.push_back(val);
            return *this;
        }
    };

    class SerializedBlockStorage {
        public:
        SerializedBlockStorage(OffsetMode offsetMode, BlockDistribution<>& blockDistribution, size_t constOffset = 0)
            : _offsetMode(offsetMode),
              _constOffset(constOffset),
              _blockDistribution(blockDistribution) {
            if (_offsetMode == OffsetMode::constant && _constOffset == 0) {
                throw std::runtime_error("If constant offset mode is used, the offset has to be greater than 0.");
            } else if (_offsetMode == OffsetMode::lookUpTable && constOffset != 0) {
                throw std::runtime_error("You've specified LUT as the offset mode and an constant offset.");
            }
        }

        void registerRanges(std::vector<typename BlockDistribution<>::BlockRange>&& ranges) {
            if (ranges.size() == 0) {
                throw std::runtime_error("You have to register some ranges.");
            }

            _ranges = std::move(ranges);
            _data   = std::vector<std::vector<uint8_t>>(_ranges.size());

            for (size_t index = 0; index < _ranges.size(); index++) {
                _rangeIndices[_ranges[index].id] = index;
            }

            // TODO implement LUT mode
            assert(_ranges.size() == _data.size());
            assert(_offsets.size() == 0);
            assert(_rangeIndices.size() == _ranges.size());
        }

        void writeBlock(block_id_t blockId, uint8_t* data) {
            // TODO implement LUT mode
            assert(_offsetMode == OffsetMode::constant);

            if (data == nullptr) {
                throw std::runtime_error("The data argument might not be a nullptr.");
            }

            auto  rangeOfBlock = _blockDistribution.rangeBlockIsStoredIn(blockId);
            auto& rangeData    = _data[indexOf(rangeOfBlock)];
            assert(_constOffset > 0);
            _data[rangeData].insert(rangeData.end(), data, data + _constOffset);
        }

        template <class HandleBlockFunction>
        void forAllBlocks(std::pair<block_id_t, size_t> blockRange, HandleBlockFunction handleBlock);

        private:
        using BlockRange = typename BlockDistribution<>::BlockRange;

        const OffsetMode                  _offsetMode;
        const size_t                      _constOffset;  // only in ConstOffset mode
        std::map<size_t, size_t>          _rangeIndices; // Maps a rangeId to its indices in following vectors
        std::vector<BlockRange>           _ranges;       // For all outer vectors, the indices correspond
        std::vector<std::vector<size_t>>  _offsets;      // A sentinel points to last elem + 1; only in LUT mode
        std::vector<std::vector<uint8_t>> _data;
        const BlockDistribution<>&        _blockDistribution;

        // Return the index this range has in the outer vectors
        void indexOf(BlockRange blockRange) const {
            // If we want to get rid of this map, we could sort the _ranges vector and use a binary_search instead
            auto index = _rangeIndices[blockRange.id];
            assert(index < _data.size());
            assert(_ranges.size() == _data.size());
        }
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
          _mpiContext(mpiCommunicator),
          _blockDistribution(nullptr),
          _serializedBlocks(offsetMode, *_blockDistribution, constOffset) { // TODO obviously not a good idea
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
            _serializedBlocks.registerRanges(_blockDistribution->rangesStoredOnRank(_mpiContext.getMyOriginalRank()));

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
    const uint16_t                       _replicationLevel;
    const OffsetMode                     _offsetMode;
    const size_t                         _constOffset;
    ReStoreMPI::MPIContext               _mpiContext;
    std::shared_ptr<BlockDistribution<>> _blockDistribution;
    SerializedBlockStorage               _serializedBlocks;

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
