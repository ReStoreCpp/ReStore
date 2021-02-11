#ifndef RESTORE_BLOCK_SERIALIZATION_H
#define RESTORE_BLOCK_SERIALIZATION_H

#include <algorithm>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "restore/block_distribution.hpp"
#include "restore/common.hpp"
#include "restore/mpi_context.hpp"

namespace ReStore {

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
    ReStore::SerializedBlockStoreStream& operator<<(const T& value) {
        static_assert(std::is_pod<T>(), "You may only serialize a POD this way.");
        assert(_buffers);
        assert(_ranks);

        auto src = reinterpret_cast<const uint8_t*>(&value);
        for (auto rank: *_ranks) {
            if (_buffers->find(rank) == _buffers->end()) {
                (*_buffers)[rank] = std::vector<uint8_t>();
            }
            assert(rank >= 0);
            assert(static_cast<size_t>(rank) < _buffers->size());
            (*_buffers)[rank].insert((*_buffers)[rank].end(), src, src + sizeof(T));
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

template <typename MPIContext = ReStoreMPI::MPIContext>
class SerializedBlockStorage {
    public:
    SerializedBlockStorage(
        std::shared_ptr<const BlockDistribution<MPIContext>> blockDistribution, OffsetMode offsetMode,
        size_t constOffset = 0)
        : _offsetMode(offsetMode),
          _constOffset(constOffset),
          _blockDistribution(blockDistribution) {
        if (_offsetMode == OffsetMode::constant && _constOffset == 0) {
            throw std::runtime_error("If constant offset mode is used, the offset has to be greater than 0.");
        } else if (_offsetMode == OffsetMode::lookUpTable && constOffset != 0) {
            throw std::runtime_error("You've specified LUT as the offset mode and an constant offset.");
        }
    }

    void registerRanges(std::vector<typename BlockDistribution<MPIContext>::BlockRange>&& ranges) {
        if (ranges.size() == 0) {
            throw std::runtime_error("You have to register some ranges.");
        }

        _ranges = std::move(ranges);
        _data   = std::vector<std::vector<uint8_t>>(_ranges.size());

        for (size_t index = 0; index < _ranges.size(); index++) {
            _rangeIndices[_ranges[index].id()] = index;
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

        auto  rangeOfBlock = _blockDistribution->rangeOfBlock(blockId);
        auto& rangeData    = _data[indexOf(rangeOfBlock)];
        assert(_constOffset > 0);
        rangeData.insert(rangeData.end(), data, data + _constOffset);
    }

    template <class HandleBlockFunction>
    void forAllBlocks(const std::pair<block_id_t, size_t> blockRange, HandleBlockFunction handleBlock) {
        static_assert(
            std::is_invocable<HandleBlockFunction, const uint8_t*, size_t>(),
            "HandleBlockFunction must be invocable as (const uint8_t*, size_t)");
        block_id_t currentBlockId = blockRange.first;
        while (currentBlockId < blockRange.first + blockRange.second) {
            const BlockRange blockRangeInternal = _blockDistribution->rangeOfBlock(currentBlockId);
            assert(blockRangeInternal.contains(currentBlockId));
            assert(currentBlockId >= blockRangeInternal.start());
            const size_t blockRangeIndex = indexOf(blockRangeInternal);
            assert(blockRangeIndex < _ranges.size());
            assert(blockRangeIndex < _data.size());
            assert(_offsetMode == OffsetMode::constant || blockRangeIndex < _offsets.size());
            assert(
                _offsetMode == OffsetMode::constant
                || _offsets[blockRangeIndex].size() == blockRangeInternal.length() + 1);
            assert(
                _offsetMode == OffsetMode::lookUpTable
                || _data[blockRangeIndex].size() == blockRangeInternal.length() * _constOffset);
            const size_t beginIndexInBlockRange = currentBlockId - blockRangeInternal.start();
            assert(beginIndexInBlockRange < blockRangeInternal.length());
            const size_t lenghtRemaining = blockRange.second - (currentBlockId - blockRange.first);
            const size_t endIndexInBlockRange =
                std::min(beginIndexInBlockRange + lenghtRemaining, blockRangeInternal.length());
            assert(endIndexInBlockRange <= blockRangeInternal.length());
            for (size_t currentIndexInBlockRange = beginIndexInBlockRange;
                 currentIndexInBlockRange < endIndexInBlockRange; ++currentIndexInBlockRange) {
                const size_t begin = _offsetMode == OffsetMode::constant
                                         ? currentIndexInBlockRange * _constOffset
                                         : _offsets[blockRangeIndex][currentIndexInBlockRange];
                assert(begin < _data[blockRangeIndex].size());
                const size_t length = _offsetMode == OffsetMode::constant
                                          ? _constOffset
                                          : _offsets[blockRangeIndex][currentIndexInBlockRange + 1];
                assert(begin + length <= _data[blockRangeIndex].size());
                handleBlock(&(_data[blockRangeIndex][begin]), length);
            }
            currentBlockId += (endIndexInBlockRange - beginIndexInBlockRange);
        }
    }

    private:
    using BlockRange = typename BlockDistribution<MPIContext>::BlockRange;

    const OffsetMode                  _offsetMode;
    const size_t                      _constOffset;  // only in ConstOffset mode
    std::map<size_t, size_t>          _rangeIndices; // Maps a rangeId to its indices in following vectors
    std::vector<BlockRange>           _ranges;       // For all outer vectors, the indices correspond
    std::vector<std::vector<size_t>>  _offsets;      // A sentinel points to last elem + 1; only in LUT mode
    std::vector<std::vector<uint8_t>> _data;
    const std::shared_ptr<const BlockDistribution<MPIContext>> _blockDistribution;

    // Return the index this range has in the outer vectors
    size_t indexOf(BlockRange blockRange) {
        // If we want to get rid of this map, we could sort the _ranges vector and use a binary_search instead
        if (_rangeIndices.find(blockRange.id()) == _rangeIndices.end()) {
            throw std::invalid_argument("BlockRange not stored");
        }
        size_t index = _rangeIndices[blockRange.id()];
        assert(index < _data.size());
        assert(_ranges.size() == _data.size());
        return index;
    }
};
} // end of namespace ReStore
#endif // Include guard
