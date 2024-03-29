#ifndef RESTORE_BLOCK_DISTRIBUTION_H
#define RESTORE_BLOCK_DISTRIBUTION_H

#include <iostream>
#include <memory>

#include "restore/common.hpp"
#include "restore/helpers.hpp"
#include "restore/mpi_context.hpp"

namespace ReStore {

// BlockDistribution
//
// A class that models the distribution of blocks across the different ranks. For now, we only implement a single
// strategy. However, the goal is to enable the user to choose his own distribution strategy.
//
// The strategy implemented by this class is as follows:
//   Let k be the replication level
//   Let s be the shift width
//   - Blocks with adjacent global ids are grouped together into one block range.
//   - The goal is to distribute the blocks as evenly as possible among the ranks. Let's call the number of blocks
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
    // Represents a range of blocks. All blocks in this range have consecutive ids. We store only the range's id and a
    // reference to its correpsonding BlockDistribution object. We can then compute the start block and number of blocks
    // from that.
    class BlockRange {
        public:
        static const size_t INVALID_RANGE = std::numeric_limits<size_t>::max();

        // Constructor
        //
        // Build a block range from the given block id. We need to know the number of blocks and the number ranges
        // to compute the starting block and number of block in this BlockRange.
        BlockRange(size_t range_id, const BlockDistribution& blockDistribution)
            : _id(range_id),
              _numRangesWithAdditionalBlock(blockDistribution.numRangesWithAdditionalBlock()),
              _blocksPerRange(blockDistribution.blocksPerRange()),
              _numRanges(blockDistribution.numRanges()),
              _numBlocks(blockDistribution.numBlocks()) {
            if (range_id > _numRanges) {
                throw std::invalid_argument("This range does not exist (id too large).");
            }
            updateCachedStartLengthLast();
        }

        // The default constructor creates an invalid range.
        BlockRange() noexcept
            : _id(INVALID_RANGE),
              _numRangesWithAdditionalBlock(0),
              _blocksPerRange(0),
              _numRanges(0),
              _numBlocks(0) {}

        // Copying and copy assignment is fine ...
        BlockRange(const BlockRange& that) noexcept
            : _id(that._id),
              _numRangesWithAdditionalBlock(that._numRangesWithAdditionalBlock),
              _blocksPerRange(that._blocksPerRange),
              _numRanges(that._numRanges),
              _numBlocks(that._numBlocks) {
            assert(_id <= _numRanges);
            updateCachedStartLengthLast();
        }

        BlockRange& operator=(const BlockRange& that) {
            this->_id                           = that._id;
            this->_numRangesWithAdditionalBlock = that._numRangesWithAdditionalBlock;
            this->_blocksPerRange               = that._blocksPerRange;
            this->_numRanges                    = that._numRanges;
            this->_numBlocks                    = that._numBlocks;
            assert(_id <= _numRanges);
            updateCachedStartLengthLast();
            return *this;
        }

        // ... as is moving and move assignment
        BlockRange(BlockRange&& that) noexcept
            : _id(that._id),
              _numRangesWithAdditionalBlock(that._numRangesWithAdditionalBlock),
              _blocksPerRange(that._blocksPerRange),
              _numRanges(that._numRanges),
              _numBlocks(that._numBlocks) {
            assert(_id <= _numRanges);
            updateCachedStartLengthLast();
        }

        BlockRange& operator=(BlockRange&& that) noexcept {
            this->_id                           = that._id;
            this->_numRangesWithAdditionalBlock = that._numRangesWithAdditionalBlock;
            this->_blocksPerRange               = that._blocksPerRange;
            this->_numRanges                    = that._numRanges;
            this->_numBlocks                    = that._numBlocks;
            assert(_id <= _numRanges);
            updateCachedStartLengthLast();
            return *this;
        }

        block_id_t start() const {
            assert(isValid());
            return _start;
        }

        block_id_t last() const {
            assert(isValid());
            return _last;
        }

        size_t length() const {
            assert(isValid());
            return _length;
        }

        size_t id() const noexcept {
            assert(isValid());
            return _id;
        }

        // contains()
        //
        // Returns true if the given block is part of this range; false otherwise
        bool contains(block_id_t block) const {
            assert(isValid());
            return block >= this->start() && block <= this->last();
        }

        // Comparison Operator
        //
        // We assume that both block ranges belong to the same BlockDistribution.
        bool operator==(const BlockRange& that) const {
#ifndef NDEBUG
            assert(isValid());
            if (this->_id == that._id) {
                assert(this->_numRangesWithAdditionalBlock == that._numRangesWithAdditionalBlock);
                assert(this->_blocksPerRange == that._blocksPerRange);
                assert(this->_numRanges == that._numRanges);
                assert(this->_numBlocks == that._numBlocks);
                assert(this->_start == that._start);
                assert(this->_length == that._length);
                assert(this->_last == that._last);
            }
#endif
            return this->_id == that._id;
        }

        bool operator!=(const BlockRange& that) const {
            return !(*this == that);
        }

        // How to print a BlockRange
        //
        // Writes "<BlockRange(id=0,start=1,length=2)>" to the ostream object.
        friend std::ostream& operator<<(std::ostream& os, const BlockRange& blockRange) {
            return os << "<BlockRange(id=" << blockRange.id() << ",start=" << blockRange.start()
                      << ",length=" << blockRange.length() << ")>";
        }

        bool isValid() const {
            return _id != INVALID_RANGE;
        }

        private:
        size_t _id = std::numeric_limits<size_t>::max();
        // We're caching the values of the block distribution's members for decoupling.
        block_id_t _numRangesWithAdditionalBlock;
        block_id_t _blocksPerRange;
        size_t     _numRanges;
        block_id_t _numBlocks;
        // Pre-compute these values for increased performance.
        block_id_t _start;
        block_id_t _last;
        block_id_t _length;

        void updateCachedStartLengthLast() {
            assert(isValid());
            assert(_blocksPerRange > 0);
            assert(_blocksPerRange <= _numBlocks);
            assert(_blocksPerRange * _numRanges + _numRangesWithAdditionalBlock == _numBlocks);

            // Do we - and all blocks with a lower id than us - have an additional block?
            if (_id < _numRangesWithAdditionalBlock) {
                _start  = _id * (_blocksPerRange + 1);
                _length = _blocksPerRange + 1;
            } else {
                _start  = _blocksPerRange * _id + _numRangesWithAdditionalBlock;
                _length = _blocksPerRange;
            }
            assert(_length > 0);
            assert(_length == _blocksPerRange || _length == _blocksPerRange + 1);
            assert(_start <= _numBlocks);

            _last = _start + _length - 1;

            assert(_last <= _numBlocks);
            assert(_start <= _last);
            assert(_last >= _length - 1);
        }
    };

    BlockDistribution(uint32_t numRanks, size_t numBlocks, uint16_t replicationLevel, const MPIContext& mpiContext)
        : _constructorArgumentsValid(validateConstructorArguments(numRanks, numBlocks, replicationLevel, mpiContext)),
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

    // Comparison operators
    bool operator==(const BlockDistribution& that) const noexcept {
        return this->_numBlocks == that._numBlocks && this->_numRanks == that._numRanks
               && this->_replicationLevel == that._replicationLevel && this->_numRanges == that._numRanges
               && this->_blocksPerRange == that._blocksPerRange
               && this->_numRangesWithAdditionalBlock == that._numRangesWithAdditionalBlock
               && this->_shiftWidth == that._shiftWidth;
    }

    bool operator!=(const BlockDistribution& that) const noexcept {
        return !(*this == that);
    }

    // blockRangeById()
    //
    // A factory method to build a BlockRange by its id.
    BlockRange blockRangeById(size_t rangeId) const {
        return BlockRange(rangeId, *this);
    }

    // rangeOfBlock()
    //
    // Computes the block range the given block is in.
    BlockRange rangeOfBlock(block_id_t block) const {
        if (block >= _numBlocks) {
            throw std::runtime_error(
                "Block id " + std::to_string(block) + " is greater than (or equal to) the number of blocks ("
                + std::to_string(_numBlocks) + ").");
        }
        assert(_blocksPerRange > 0);
        assert(_blocksPerRange <= _numBlocks);
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
        return ranksBlockRangeIsStoredOn(range);
    }

    // ranksBlockRangeIsStoredOn()
    //
    // Returns the ranks the given block is stored on. The ranks are identified by their original rank id.
    // Dead ranks are filtered from the result list.
    std::vector<ReStoreMPI::original_rank_t> ranksBlockRangeIsStoredOn(const BlockRange& range) const {
        assert(range.start() < _numBlocks);
        assert(range.start() + range.length() <= _numBlocks);
        assert(range.id() < _numRanges);
        assert(range.id() < _numRanks);

        // The range is located on the rank with the same id ...
        auto rankIds = std::vector<ReStoreMPI::original_rank_t>();
        assert(range.id() <= static_cast<size_t>(std::numeric_limits<ReStoreMPI::original_rank_t>::max()));
        ReStoreMPI::original_rank_t firstRank = static_cast<ReStoreMPI::original_rank_t>(range.id());
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

    // randomRankBlockRangeIsStoredOn()
    //
    // This is an a version of ranksBlockRangeIsStoredOn() which does not generate the whole vector of ranks but only
    // a single rank. You can seed the random distribution with \c seed.
    // If no alive rank is found on which the block range is stored, returns -1.
    ReStoreMPI::original_rank_t
    randomAliveRankBlockRangeIsStoredOn(const BlockRange& range, const uint64_t seed) const {
        assert(range.start() < _numBlocks);
        assert(range.start() + range.length() <= _numBlocks);
        assert(range.id() < _numRanges);
        assert(range.id() < _numRanks);
        assert(range.id() <= static_cast<size_t>(std::numeric_limits<ReStoreMPI::original_rank_t>::max()));

        const XXH64_hash_t baseSeed = 0xA807A1286AC9D67E;

        // The range is located on the rank with the same id and on <replication level> - 1 further ranks, all <shift
        // width> apart.
        ReStoreMPI::original_rank_t firstRank = static_cast<ReStoreMPI::original_rank_t>(range.id());
        assert(firstRank >= 0);

        const auto hash         = xxhash(seed, baseSeed);
        uint64_t   nthAliveRank = hash % _replicationLevel;
        bool       secondRound  = false;
        do {
            uint64_t aliveRanksSeen = 0;
            for (uint64_t replica = 0; replica < _replicationLevel; replica++) {
                const auto rank = static_cast<ReStoreMPI::original_rank_t>(
                    (static_cast<uint64_t>(firstRank) + _shiftWidth * replica) % _numRanks);
                assert(rank >= 0);

                if (_mpiContext.isAlive(rank)) {
                    if (aliveRanksSeen == nthAliveRank) {
                        return rank;
                    }
                    aliveRanksSeen++;
                }
            }

            // No alive rank which stores this block range was found.
            if (aliveRanksSeen == 0) {
                assert(!secondRound); // In the second round, there is at least one alive rank.
                return static_cast<ReStoreMPI::original_rank_t>(-1);
            } else {
                assert(!secondRound); // In the second round, we should always find the nthAliveRank.
                // We found alive ranks but less than nthAliveRank. Redo the search with a random number between 0 and
                // aliveRanksSeen.
                nthAliveRank = hash % aliveRanksSeen;
                assert(nthAliveRank < aliveRanksSeen);
                aliveRanksSeen = 0;
                secondRound    = true;
            }
        } while (secondRound);
        assert(false);
        return -2; // Silence compiler warning.
    }

    // rangesStoredOnRank()
    //
    // Returns the block ranges residing on the given rank.
    // If the given rank is dead, this returns an empty vector.
    // std::vector<BlockRange> rangesStoredOnRank(ReStoreMPI::original_rank_t rankId) const {
    //     if (!_mpiContext.isAlive(rankId)) {
    //         return std::vector<BlockRange>();
    //     }

    //     if (rankId < 0) {
    //         throw std::runtime_error("Invalid rank id: Less than zero.");
    //     } else if (static_cast<size_t>(rankId) > _numRanks) {
    //         throw std::runtime_error("Invalid rank id: lower than the number of ranks.");
    //     }

    //     // The range with the same id as this rank is stored on this rank ...
    //     auto rangeIds = std::vector<BlockRange>();
    //     assert(rankId >= 0);
    //     BlockRange firstRange = blockRangeById(static_cast<size_t>(rankId));
    //     rangeIds.push_back(firstRange);

    //     // ... as are <replication level> - 1 further ranges, all <shift width> apart
    //     for (uint16_t replica = 1; replica < _replicationLevel; replica++) {
    //         assert(firstRange.id() <= static_cast<size_t>(std::numeric_limits<int64_t>::max()));
    //         assert(_shiftWidth * replica <= static_cast<size_t>(std::numeric_limits<int64_t>::max()));
    //         assert(in_range<int64_t>(_numRanges));
    //         assert(_numRanges > 0);

    //         assert(in_range<int64_t>(_shiftWidth));
    //         assert(in_range<int64_t>(_shiftWidth * replica));
    //         static_assert(std::numeric_limits<decltype(replica)>::max() <= std::numeric_limits<int32_t>::max());
    //         int64_t rangeId = static_cast<int64_t>(firstRange.id())
    //                           - static_cast<int64_t>(_shiftWidth) * static_cast<int32_t>(replica);
    //         if (rangeId < 0) {
    //             assert(in_range<int64_t>(_numRanges));
    //             rangeId =
    //                 static_cast<int64_t>(_numRanges) + static_cast<int64_t>(rangeId % static_cast<int64_t>(_numRanges));
    //         }
    //         assert(rangeId >= 0);
    //         BlockRange nextRange = blockRangeById(static_cast<size_t>(rangeId));
    //         assert(nextRange.id() < _numRanges);
    //         rangeIds.push_back(nextRange);
    //     }

    //     return rangeIds;
    // }

    // isStoredOn()
    //
    // Returns true if the given block or block range is stored on the given rank.
    // If the given rank is dead, this will return false.
    bool isStoredOn(BlockRange blockRange, ReStoreMPI::original_rank_t rankId) const {
        if (rankId < 0) {
            throw std::runtime_error("A rank id cannot be negative.");
        } else if (static_cast<size_t>(rankId) >= _numRanks) {
            throw std::runtime_error("Rank id larger than (or equal to) the number of ranks.");
        } else if (blockRange.id() > _numRanges) {
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
            assert(in_range<int64_t>(_numRanges));
            assert(in_range<int64_t>(_shiftWidth));
            assert(in_range<int64_t>(_shiftWidth * replica));
            assert(_numRanges > 0);

            static_assert(std::numeric_limits<decltype(replica)>::max() <= std::numeric_limits<int32_t>::max());
            int64_t rangeId = rankId - static_cast<int64_t>(_shiftWidth) * static_cast<int32_t>(replica);
            if (rangeId < 0) {
                assert(in_range<int64_t>(_numRanges));
                rangeId =
                    static_cast<int64_t>(_numRanges) + static_cast<int64_t>(rangeId % static_cast<int64_t>(_numRanges));
            }
            assert(rankId >= 0);
            if (static_cast<size_t>(rangeId) == blockRange.id()) {
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
        if (numBlocks == 0) {
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

} // namespace ReStore

#endif // Include guard
