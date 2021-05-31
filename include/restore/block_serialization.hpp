#ifndef RESTORE_BLOCK_SERIALIZATION_H
#define RESTORE_BLOCK_SERIALIZATION_H

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "restore/block_distribution.hpp"
#include "restore/common.hpp"
#include "restore/mpi_context.hpp"

namespace ReStore {

class SerializedBlockStoreStream {
    public:
    using original_rank_t = ReStoreMPI::original_rank_t;
    using current_rank_t  = ReStoreMPI::current_rank_t;

    SerializedBlockStoreStream(std::unordered_map<current_rank_t, std::vector<std::byte>>& buffers)
        : _bytesWritten(0),
          _buffers(buffers) {}

    void setDestinationRanks(std::vector<current_rank_t> ranks) {
        if (ranks.size() == 0) {
            throw std::runtime_error("The ranks array is empty.");
        }

        _outputBuffers.clear();
        _outputBuffers.reserve(ranks.size());

        for (auto rank: ranks) {
            assert(rank >= 0);
            auto buffer = _buffers.find(rank);
            if (buffer == _buffers.end()) {
                auto& bufferRef = _buffers[rank] = std::vector<std::byte>();
                _outputBuffers.push_back(&bufferRef);
            } else {
                _outputBuffers.push_back(&(buffer->second));
            }
        }
        assert(_outputBuffers.size() == ranks.size());

        _outputBuffers.reserve(ranks.size());
    }

    // Keep the user from copying or moving our SerializedBlockStore object we pass to him.
    SerializedBlockStoreStream(const SerializedBlockStoreStream&) = delete;
    SerializedBlockStoreStream(SerializedBlockStoreStream&&)      = delete;
    SerializedBlockStoreStream& operator=(const SerializedBlockStoreStream&) = delete;
    SerializedBlockStoreStream& operator=(SerializedBlockStoreStream&&) = delete;

    // reserve()
    //
    // Reserve enough space to write n bytes without reallocating the buffers.
    void reserve(size_t n) {
        for (auto& buffer: _outputBuffers) {
            buffer->reserve(n);
        }
    }

    // operator<<
    //
    // Can be used to serialize a plain old datatype (pod)
    template <class T>
    SerializedBlockStoreStream& operator<<(const T& value) {
        static_assert(std::is_pod<T>(), "You may only serialize a POD this way.");

        auto src = reinterpret_cast<const std::byte*>(&value);
        for (auto buffer: _outputBuffers) {
            buffer->insert(buffer->end(), src, src + sizeof(T));
        }
        _bytesWritten += sizeof(T);

        return *this;
    }

    // writeBytes()
    //
    // Copy n bytes, starting at begin to the buffers.
    void writeBytes(const std::byte* begin, size_t n) {
        for (auto buffer: _outputBuffers) {
            buffer->insert(buffer->end(), begin, begin + n);
        }
        _bytesWritten += n;
    }

    // Return the number of bytes written to the buffers.
    size_t bytesWritten(current_rank_t rank) const {
        auto bufferIt = _buffers.find(rank);
        if (bufferIt == _buffers.end()) {
            throw new std::runtime_error("No buffer for this rank.");
        }

        return bufferIt->second.size();
    }

    // This handle can be used to alter parts of the stream that are not at the current write front. Use
    // reserveBytesForWriting() to get one. Pass it to writeToReservedBytes() to write to the reserved position.
    struct WritableStreamPosition {
        public:
        size_t bytesLeft() const {
            return length - written;
        }

        size_t currentPosition() const {
            return index + written;
        }

        private:
        WritableStreamPosition(current_rank_t _rank, size_t _index, size_t _length)
            : rank(_rank),
              index(_index),
              length(_length),
              written(0) {}

        void writeBytes(const size_t numBytes) {
            written += numBytes;
        }

        current_rank_t rank;
        size_t         index;
        size_t         length;
        size_t         written;

        friend SerializedBlockStoreStream;
    };

    // Reserve n bytes at the current stream position. We can use the returned handle to later write to these bytes.
    WritableStreamPosition reserveBytesForWriting(ReStoreMPI::original_rank_t rank, size_t n) {
        if (rank < 0) {
            throw new std::runtime_error("Negative rank not allowed.");
        }

        auto bufferIt = _buffers.find(rank);
        if (bufferIt == _buffers.end()) {
            throw new std::runtime_error("There is no buffer for this rank.");
        }
        auto& buffer = bufferIt->second;
        assert(bufferIt->first == rank);

        WritableStreamPosition position(rank, bytesWritten(rank), n);

        // Write dummy data to the stream.
        buffer.resize(buffer.size() + n);

        assert(position.index + position.length == bytesWritten(rank));
        return position;
    }

    // Write to previously reserved bytes in the stream.
    void writeToReservedBytes(WritableStreamPosition& position, const std::byte* begin, size_t length = 0) {
        auto& buffer       = _buffers[position.rank];
        auto  bytesToWrite = length != 0 ? length : position.bytesLeft();

        if (bytesToWrite > position.bytesLeft()) {
            throw new std::runtime_error("Trying to write more bytes than there are left for this handle.");
        }

        std::copy(
            begin, begin + bytesToWrite,
            buffer.begin() + throwing_cast<std::vector<std::byte>::difference_type>(position.currentPosition()));

        position.writeBytes(bytesToWrite);
    }

    template <class T>
    void writeToReservedBytes(WritableStreamPosition& position, const T& value) {
        static_assert(std::is_pod<T>(), "You may only serialize a POD this way.");
        assert(sizeof(T) <= position.length);

        auto src = reinterpret_cast<const std::byte*>(&value);
        writeToReservedBytes(position, src, sizeof(T));
    }

    private:
    size_t                                                      _bytesWritten;
    std::vector<std::vector<std::byte>*>                        _outputBuffers;
    std::unordered_map<current_rank_t, std::vector<std::byte>>& _buffers;
};

template <typename MPIContext = ReStoreMPI::MPIContext>
class SerializedBlockStorage {
    using BlockRange = typename BlockDistribution<MPIContext>::BlockRange;

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

    // numBlocks()
    //
    // Returns the number of block that are stored in this object.
    size_t numBlocks() const {
        // TODO Implement LUT mode
        size_t numBlocks = 0;
        for (auto&& rangeData: _data) {
            assert(_constOffset > 0);
            numBlocks += rangeData.size() / _constOffset;
        }
        return numBlocks;
    }

    // writeBlock()
    //
    // Writes the data associated with that block to the storage.
    // blockId and data: The id and data of the block to be written. In this overload we know the length of the data
    //      because it is equal to the constant offset.
    void writeBlock(block_id_t blockId, const std::byte* data) {
        // TODO implement LUT mode
        assert(_offsetMode == OffsetMode::constant);
        // if (_offsetMode == OffsetMode::constant && blockId != numBlocks()) {
        //    throw std::
        //}

        if (data == nullptr) {
            throw std::runtime_error("The data argument might not be a nullptr.");
        }

        auto rangeOfBlock = _blockDistribution->rangeOfBlock(blockId);
        if (!hasRange(rangeOfBlock)) {
            registerRange(rangeOfBlock);
        }
        auto& rangeData = _data[indexOf(rangeOfBlock)];
        assert(_constOffset > 0);
        assert(_data[indexOf(rangeOfBlock)].size() == rangeOfBlock.length() * _constOffset);
        assert(blockId >= rangeOfBlock.start());
        auto offsetInBlockRange = (blockId - rangeOfBlock.start()) * _constOffset;
        auto blockDest          = rangeData.begin()
                         + asserting_cast<typename decltype(rangeData.begin())::difference_type>(offsetInBlockRange);
        std::copy(data, data + _constOffset, blockDest);
    }

    template <class HandleBlockFunction>
    void forAllBlocks(const std::pair<block_id_t, size_t> blockRange, HandleBlockFunction handleBlock) const {
        static_assert(
            std::is_invocable<HandleBlockFunction, const std::byte*, size_t>(),
            "HandleBlockFunction must be invocable as (const std::byte*, size_t)");
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
            const size_t lengthRemaining = blockRange.second - (currentBlockId - blockRange.first);
            const size_t endIndexInBlockRange =
                std::min(beginIndexInBlockRange + lengthRemaining, blockRangeInternal.length());
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
    const OffsetMode                    _offsetMode;
    const size_t                        _constOffset;  // only in ConstOffset mode
    std::unordered_map<size_t, size_t>  _rangeIndices; // Maps a rangeId to its indices in following vectors
    std::vector<BlockRange>             _ranges;       // For all outer vectors, the indices correspond
    std::vector<std::vector<size_t>>    _offsets;      // A sentinel points to last elem + 1; only in LUT mode
    std::vector<std::vector<std::byte>> _data;
    const std::shared_ptr<const BlockDistribution<MPIContext>> _blockDistribution;

    // Return the index this range has in the outer vectors
    size_t indexOf(BlockRange blockRange) const {
        // If we want to get rid of this map, we could sort the _ranges vector and use a binary_search instead
        auto indexIt = _rangeIndices.find(blockRange.id());
        if (indexIt == _rangeIndices.end()) {
            throw std::invalid_argument("BlockRange not stored");
        }
        size_t index = indexIt->second;
        assert(index < _data.size());
        assert(_ranges.size() == _data.size());
        return index;
    }

    // hasRange()
    //
    // Returns true if at least one block of the given range has been written to this storage.
    bool hasRange(const BlockRange& blockRange) const {
        return hasRange(blockRange.id());
    }

    bool hasRange(size_t blockId) const {
        return _rangeIndices.find(blockId) != _rangeIndices.end();
    }

    // registerRange()
    //
    // Registers the block ranges to be stored in this object. May only be called once per range during the lifetime of
    // this object.
    // ranges: The block range we should reserve storage for
    void registerRange(const BlockRange& range) {
        if (_rangeIndices.find(range.id()) != _rangeIndices.end()) {
            throw std::runtime_error("This range already exists.");
        }

        size_t numRanges = this->numRanges();
        _ranges.push_back(range);
        _data.emplace_back(range.length() * _constOffset);
        _rangeIndices[range.id()] = numRanges;
        // TODO implement LUT mode

        assert(_ranges.size() == _data.size());
        assert(_offsets.size() == 0);
        assert(_rangeIndices.size() == _ranges.size());
        assert(_data[indexOf(range)].size() == range.length() * _constOffset);
    }

    // numRanges()
    //
    // Returns the number of block ranges that are stored in this object.
    size_t numRanges() const noexcept {
        // TODO implement LUT mode
        assert(_ranges.size() == _data.size());
        assert(_offsets.size() == 0);
        assert(_rangeIndices.size() == _ranges.size());
        return _ranges.size();
    }
};
} // end of namespace ReStore
#endif // Include guard
