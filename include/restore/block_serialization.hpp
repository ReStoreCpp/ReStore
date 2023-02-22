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

    SerializedBlockStoreStream(std::vector<std::vector<std::byte>>& buffers, ReStoreMPI::original_rank_t numRanks)
        : _bytesWritten(0),
          _buffers(buffers),
          _numWritableStreamPositionsWithBytesLeft(0) {
        if (numRanks <= 0) {
            throw std::runtime_error("numRanks might not be less than or equal to zero");
        }
        if (buffers.size() != asserting_cast<size_t>(numRanks)) {
            throw std::runtime_error("The send buffers are not allocated.");
        }
    }

    void setDestinationRanks(std::vector<current_rank_t> ranks) {
        if (ranks.size() == 0) {
            throw std::runtime_error("The ranks array is empty.");
        }

        _outputBuffers.clear();
        _outputBuffers.reserve(ranks.size());

        for (auto rank: ranks) {
            assert(rank >= 0);
            assert(asserting_cast<size_t>(rank) < _buffers.size());
            _outputBuffers.push_back(&(_buffers[asserting_cast<size_t>(rank)]));
        }
        assert(_outputBuffers.size() == ranks.size());
    }

    // Keep the user from copying or moving our SerializedBlockStore object we pass to him.
    SerializedBlockStoreStream(const SerializedBlockStoreStream&)            = delete;
    SerializedBlockStoreStream(SerializedBlockStoreStream&&)                 = delete;
    SerializedBlockStoreStream& operator=(const SerializedBlockStoreStream&) = delete;
    SerializedBlockStoreStream& operator=(SerializedBlockStoreStream&&)      = delete;

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
    // Can be used to serialize a trivially copyable type
    template <class T>
    inline SerializedBlockStoreStream& operator<<(const T& value) {
        static_assert(std::is_trivially_copyable<T>(), "You may only serialize a trivially copyable type this way.");

        auto src = reinterpret_cast<const std::byte*>(&value);
        for (auto buffer: _outputBuffers) {
            buffer->insert(buffer->end(), src, src + sizeof(T));
        }
        _bytesWritten += sizeof(T);

        return *this;
    }

    // writeBytes()
    //
    // Copy n bytes to the buffers, starting at begin.
    inline void writeBytes(const std::byte* begin, size_t n) {
        for (auto buffer: _outputBuffers) {
            buffer->insert(buffer->end(), begin, begin + n);
        }
        _bytesWritten += n;
    }

    // Return the number of bytes written to the buffers.
    size_t bytesWritten(current_rank_t rank) const {
        if (rank < 0 || throwing_cast<size_t>(rank) >= _buffers.size()) {
            throw std::runtime_error("Invalid rank id.");
        }
        return _buffers[asserting_cast<size_t>(rank)].size();
    }

    // This handle can be used to alter parts of the stream that are not at the current write front. Use
    // reserveBytesForWriting() to get one. Pass it to writeToReservedBytes() to write to the reserved position.
    struct WritableStreamPosition {
        public:
        size_t bytesLeft() const {
            assert(written <= length);
            return length - written;
        }

        size_t currentPosition() const {
            assert(written <= length);
            return index + written;
        }

        private:
        WritableStreamPosition(current_rank_t _rank, size_t _index, size_t _length)
            : rank(_rank),
              index(_index),
              length(_length),
              written(0) {
            assert(rank >= 0);
        }

        void advance(const size_t numBytes) {
            assert(numBytes <= bytesLeft());
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
            throw std::runtime_error("Negative rank not allowed.");
        } else if (asserting_cast<size_t>(rank) >= _buffers.size()) {
            throw std::runtime_error("Rank ID larger or equal to the number of ranks.");
        }

        auto&                  buffer = _buffers[asserting_cast<size_t>(rank)];
        WritableStreamPosition position(rank, bytesWritten(rank), n);
        _numWritableStreamPositionsWithBytesLeft++;

        // Write dummy data to the stream.
        buffer.resize(buffer.size() + n);

        assert(position.index + position.length == bytesWritten(rank));
        return position;
    }

    // Write to previously reserved bytes in the stream.
    void writeToReservedBytes(WritableStreamPosition& position, const std::byte* begin, size_t length = 0) {
        assert(asserting_cast<size_t>(position.rank) < _buffers.size());
        auto& buffer       = _buffers[asserting_cast<size_t>(position.rank)];
        auto  bytesToWrite = length != 0 ? length : position.bytesLeft();

        if (bytesToWrite > position.bytesLeft()) {
            throw std::runtime_error("Trying to write more bytes than there are left for this handle.");
        }

        std::copy(
            begin, begin + bytesToWrite,
            buffer.begin() + throwing_cast<std::vector<std::byte>::difference_type>(position.currentPosition()));

        position.advance(bytesToWrite);

        if (position.bytesLeft() == 0) {
            _numWritableStreamPositionsWithBytesLeft--;
        }
    }

    template <class T>
    void writeToReservedBytes(WritableStreamPosition& position, const T& value) {
        static_assert(std::is_trivially_copyable<T>(), "You may only serialize a trivially copyable type this way.");
        assert(sizeof(T) <= position.length);

        auto src = reinterpret_cast<const std::byte*>(&value);
        writeToReservedBytes(position, src, sizeof(T));
    }

    // Return the number of WritableStreamPosition objects that have bytes left to write.
    size_t numWritableStreamPositionsWithBytesLeft() const {
        return _numWritableStreamPositionsWithBytesLeft;
    }

    private:
    size_t                               _bytesWritten;
    std::vector<std::vector<std::byte>*> _outputBuffers;
    std::vector<std::vector<std::byte>>& _buffers;
    size_t                               _numWritableStreamPositionsWithBytesLeft;
};

template <typename MPIContext = ReStoreMPI::MPIContext>
class SerializedBlockStorage {
    using BlockRange = typename BlockDistribution<MPIContext>::BlockRange;

    public:
    SerializedBlockStorage(
        const BlockDistribution<MPIContext>& blockDistribution, OffsetMode offsetMode, size_t constOffset = 0)
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
        assert(data != nullptr);

        if (!_writingState || !(_writingState->range.contains(blockId))) {
            auto range = _blockDistribution.rangeOfBlock(blockId);
            if (!hasRange(range)) {
                registerRange(range);
            }
            auto indexOpt = indexOf(range);
            assert(indexOpt);
            _writingState.emplace(range, _data[*indexOpt]);
        }

        auto& rangeOfBlock = _writingState->range;
        auto& dest         = _writingState->data;
        assert(_constOffset > 0);
        assert(dest.size() == rangeOfBlock.length() * _constOffset);
        assert(blockId >= rangeOfBlock.start());

        auto offsetInBlockRange = (blockId - rangeOfBlock.start()) * _constOffset;
        using offset_type       = typename decltype(dest.begin())::difference_type;
        auto blockDest          = dest.begin() + asserting_cast<offset_type>(offsetInBlockRange);
        std::copy(data, data + _constOffset, blockDest);
    }

    // writeConsecutiveBlocks()
    //
    // Writes the data associated with that block to the storage.
    // blockId and data: The id and data of the block to be written. In this overload we know the length of the data
    //      because it is equal to the constant offset.
    // TODO Maybe also use these optimizations in writeBlock? Completely remove writeBlock? Forward calls to writeBlock
    // to this function?
    void writeConsecutiveBlocks(block_id_t firstBlockId, block_id_t lastBlockId, const std::byte* data) {
        // TODO implement LUT mode
        assert(_offsetMode == OffsetMode::constant);
        assert(data != nullptr);
        assert(_constOffset > 0);
        const auto numTotalBlocksToCopy = lastBlockId - firstBlockId + 1;
        const auto assumedEndOfInput    = data + numTotalBlocksToCopy * _constOffset;
        UNUSED(numTotalBlocksToCopy); // Only used in assertions.
        UNUSED(assumedEndOfInput);    // Only used in assertions.

        auto getAndRegisterRangeOfBlock =
            [this](block_id_t blockId, BlockRange& range_out, std::byte*& destBase_out) -> void {
            // Get the BlockRange of the given block.
            range_out = _blockDistribution.rangeOfBlock(blockId);

            // If we do not have the output buffer for this block range, register it.
            if (!hasRange(range_out)) {
                registerRange(range_out);
            }

            // Get the output buffer associated with the the BlockRange and return a pointer to it via the destBase_out
            // parameter.
            assert(indexOf(range_out));
            auto indexOpt = indexOf(range_out);
            assert(indexOpt);
            destBase_out = _data[*indexOpt].data();
            assert(destBase_out != nullptr);
            // The vectors in _data are already resized to be able to hold the data for all the blocks in that range.
            assert(_data[*indexOpt].size() == range_out.length() * _constOffset);
        };

        // Which blocks to copy?
        auto       firstBlockToCopy = firstBlockId;
        BlockRange rangeOfBlock;
        std::byte* destBase = nullptr;
        // Compute the pointers to the blocks to copy.
        auto srcPtrBegin = data;
        assert(srcPtrBegin != nullptr);
        do {
            // Which blocks to copy?
            getAndRegisterRangeOfBlock(firstBlockToCopy, rangeOfBlock, destBase);
            assert(destBase != nullptr);
            assert(rangeOfBlock.isValid());
            assert(rangeOfBlock.contains(firstBlockToCopy));
            const auto lastBlockToCopy = std::min(lastBlockId, rangeOfBlock.last());
            assert(destBase != nullptr);
            assert(rangeOfBlock.contains(lastBlockToCopy));
            const auto numBlocksToCopy = lastBlockToCopy - firstBlockToCopy + 1;

            assert(firstBlockToCopy >= firstBlockId);
            assert(lastBlockToCopy <= lastBlockId);
            assert(firstBlockToCopy <= lastBlockToCopy); // Copy at least one block
            assert(numBlocksToCopy > 0);

            // Compute the pointers to the blocks to copy.
            const auto numBytesToCopy = numBlocksToCopy * _constOffset;
            const auto srcPtrEnd      = srcPtrBegin + numBytesToCopy;
            assert(srcPtrEnd != nullptr);

            // Where to copy the blocks to?
            const size_t destOffset = (firstBlockToCopy - rangeOfBlock.start()) * _constOffset;
            assert(destOffset < _data[*indexOf(rangeOfBlock)].size());
            std::byte* destPtr = destBase + destOffset;
            assert(destPtr != nullptr);

            assert(srcPtrBegin != nullptr);
            assert(srcPtrEnd != nullptr);
            assert(destPtr != nullptr);
            assert(srcPtrBegin >= data);
            assert(srcPtrBegin <= assumedEndOfInput - 1);
            assert(srcPtrEnd > data);
            assert(srcPtrEnd <= assumedEndOfInput);
            assert(srcPtrBegin < srcPtrEnd);
            std::copy(srcPtrBegin, srcPtrEnd, destPtr);

            // Advance iterators
            firstBlockToCopy = lastBlockToCopy + 1;
            srcPtrBegin      = srcPtrEnd;
        } while (firstBlockToCopy <= lastBlockId);
        assert(firstBlockToCopy > lastBlockId);
    }

    template <class HandleBlockFunction>
    void forAllBlocks(const std::pair<block_id_t, size_t> blockRange, HandleBlockFunction handleBlock) const {
        static_assert(
            std::is_invocable<HandleBlockFunction, const std::byte*, size_t>(),
            "HandleBlockFunction must be invocable as (const std::byte*, size_t)");
        block_id_t currentBlockId = blockRange.first;
        while (currentBlockId < blockRange.first + blockRange.second) {
            const BlockRange blockRangeInternal = _blockDistribution.rangeOfBlock(currentBlockId);
            assert(blockRangeInternal.contains(currentBlockId));
            assert(currentBlockId >= blockRangeInternal.start());
            auto indexOpt = indexOf(blockRangeInternal);
            if (!indexOpt) {
                throw std::invalid_argument("Range does not exist in the serialization buffer.");
            }
            const size_t blockRangeIndex = *indexOpt;
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
    const OffsetMode                       _offsetMode;
    const size_t                           _constOffset;  // only in ConstOffset mode
    std::vector<std::pair<size_t, size_t>> _rangeIndices; // Maps rangeId to its indices in following vectors; unordered
    std::vector<BlockRange>                _ranges;       // For all outer vectors, the indices correspond
    std::vector<std::vector<size_t>>       _offsets;      // A sentinel points to last elem + 1; only in LUT mode
    std::vector<std::vector<std::byte>>    _data;
    const BlockDistribution<MPIContext>&   _blockDistribution;

    struct WritingState {
        WritingState(BlockRange _range, std::vector<std::byte>& _data) noexcept : range(_range), data(_data) {}

        BlockRange              range;
        std::vector<std::byte>& data;
    };
    std::optional<WritingState> _writingState;

    // Return the index this range has in the outer vectors
    std::optional<size_t> indexOf(BlockRange blockRange) const {
        return indexOf(blockRange.id());
    }

    // Return the index this range has in the outer vectors
    std::optional<size_t> indexOf(size_t rangeId) const {
        auto indexIt = find_if(_rangeIndices.begin(), _rangeIndices.end(), [rangeId](std::pair<size_t, size_t> kv) {
            return kv.first == rangeId;
        });

        if (indexIt == _rangeIndices.end()) {
            return std::nullopt;
        } else {
            assert(_ranges.size() == _data.size());
            assert(indexIt->second < _data.size());
            return std::make_optional(indexIt->second);
        }
    }

    // hasRange()
    //
    // Returns true if at least one block of the given range has been written to this storage.
    bool hasRange(const BlockRange& blockRange) const {
        return hasRange(blockRange.id());
    }

    bool hasRange(size_t blockId) const {
        return indexOf(blockId).has_value();
    }

    // registerRange()
    //
    // Registers the block ranges to be stored in this object. May only be called once per range during the lifetime of
    // this object.
    // ranges: The block range we should reserve storage for
    void registerRange(const BlockRange& range) {
        if (hasRange(range)) {
            throw std::runtime_error("This range already exists.");
        }

        size_t numRanges = this->numRanges();
        _ranges.push_back(range);
        _data.emplace_back(range.length() * _constOffset);
        _rangeIndices.emplace_back(range.id(), numRanges);
        // TODO implement LUT mode

        assert(_ranges.size() == _data.size());
        assert(_offsets.size() == 0);
        assert(_rangeIndices.size() == _ranges.size());
        assert(_data[*indexOf(range)].size() == range.length() * _constOffset);
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
