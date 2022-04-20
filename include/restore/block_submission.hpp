#ifndef RESTORE_BLOCK_SUBMISSION_H
#define RESTORE_BLOCK_SUBMISSION_H

#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "restore/block_distribution.hpp"
#include "restore/block_serialization.hpp"
#include "restore/common.hpp"
#include "restore/helpers.hpp"
#include "restore/mpi_context.hpp"
#include "restore/pseudo_random_permutation.hpp"

namespace ReStore {

// Describes a list of blocks with contiguous block ids which are serialized in consecutive storage. Used by the user to
// provide submit already serialized blocks.
struct SerializedBlocksDescriptor {
    SerializedBlocksDescriptor(block_id_t _blockIdBegin, block_id_t _blockIdEnd, const std::byte* _data)
        : blockIdBegin(_blockIdBegin),
          blockIdEnd(_blockIdEnd),
          data(_data) {}

    block_id_t       blockIdBegin;
    block_id_t       blockIdEnd;
    const std::byte* data;
};

// This class implements helper functions for communication during ReStore::submitBlocks()
template <class BlockType, class MPIContext = ReStoreMPI::MPIContext>
class BlockSubmissionCommunication {
    public:
    // If the user did his homework and designed a BlockDistribution which requires few messages to be send
    // we do not want to allocate all those unneeded send buffers... that's why we use a map here instead
    // of a vector.
    using SendBuffers = std::vector<std::vector<std::byte>>;

    using BlockDistr = BlockDistribution<MPIContext>;

    // BlockIDSerialization
    //
    // This class abstracts away the serialization of the block IDs. This enables us to save bytes when storing
    // consecutive block IDs by not writing them out.
    template <typename IDType, class = std::enable_if_t<std::is_unsigned_v<IDType>>>
    class BlockIDSerialization {
        public:
        enum class BlockIDMode : uint8_t { EVERY_ID, RANGES };
        struct BlockIDRange {
            explicit BlockIDRange(IDType id) : first(id), last(id) {}
            BlockIDRange(IDType _first, IDType _last) : first(_first), last(_last) {}
            BlockIDRange() = delete;

            IDType first;
            IDType last; // This is always initalized, but gcc doesn't get it.
        };

        BlockIDSerialization(BlockIDMode mode, SerializedBlockStoreStream& stream, ReStoreMPI::original_rank_t numRanks)
            : _mode(mode),
              _stream(stream),
              _currentRanges(asserting_cast<size_t>(numRanks), std::nullopt) {
            if (mode != BlockIDMode::RANGES) {
                throw std::runtime_error("Currently, only the RANGES mode is supported.");
            }
            assert(_currentRanges.size() == asserting_cast<size_t>(numRanks));
            assert(numRanks > 0);
        }

        void writeId(IDType id, std::vector<ReStoreMPI::original_rank_t> ranks) {
            assert(!ranks.empty());
            for (const auto rank: ranks) {
                writeId(id, rank);
            }
        }

        void writeId(IDType firstId, IDType lastId, std::vector<ReStoreMPI::original_rank_t> ranks) {
            assert(!ranks.empty());
            for (const ReStoreMPI::original_rank_t rank: ranks) {
                writeId(firstId, lastId, rank);
            }
        }

        void writeId(IDType id, ReStoreMPI::original_rank_t rank) {
            // Get the range state of this rank
            assert(rank >= 0);
            assert(rank < asserting_cast<decltype(rank)>(_currentRanges.size()));

            auto& rangeState = _currentRanges[asserting_cast<size_t>(rank)];
            // Is this the first id?
            if (!rangeState) {
                BlockIDRange range(id);
                const auto   positionInStream = _stream.reserveBytesForWriting(rank, 2 * sizeof(IDType));
                rangeState.emplace(range, positionInStream);
            } else { // This is not the first id
                // Is this a consecutive id range?
                if (rangeState->range.last == id - 1) {
                    rangeState->range.last++;
                } else { // Not a consecutive id
                    // close and write out previous range
                    serializeBlockIDRange(rangeState->positionInStream, rangeState->range);

                    // start a new range
                    rangeState->range.first = rangeState->range.last = id;
                    rangeState->positionInStream = _stream.reserveBytesForWriting(rank, 2 * sizeof(IDType));
                }
            }
        }

        void writeId(IDType firstId, IDType lastId, ReStoreMPI::original_rank_t rank) {
            // Get the range state of this rank
            assert(rank >= 0);
            assert(rank < asserting_cast<decltype(rank)>(_currentRanges.size()));

            auto& rangeState = _currentRanges[asserting_cast<size_t>(rank)];
            // Is this the first id?
            if (!rangeState) {
                BlockIDRange range(firstId, lastId);
                const auto   positionInStream = _stream.reserveBytesForWriting(rank, 2 * sizeof(IDType));
                rangeState.emplace(range, positionInStream);
            } else { // This is not the first id
                // Is this a consecutive id range?
                if (rangeState->range.last == firstId - 1) {
                    rangeState->range.last = lastId;
                } else { // Not a consecutive id
                    // close and write out previous range
                    serializeBlockIDRange(rangeState->positionInStream, rangeState->range);

                    // start a new range
                    rangeState->range.first      = firstId;
                    rangeState->range.last       = lastId;
                    rangeState->positionInStream = _stream.reserveBytesForWriting(rank, 2 * sizeof(IDType));
                }
            }
        }

        // Closes up the current range and writes it to the stream.
        void finalize() {
            if (!finalized) {
                for (auto& rangeState: _currentRanges) {
                    if (rangeState) {
                        serializeBlockIDRange(rangeState->positionInStream, rangeState->range);
                    }
                }
            }
            finalized = true;
            assert(_stream.numWritableStreamPositionsWithBytesLeft() == 0);
        }

        ~BlockIDSerialization() {
            // Write out the last range
            finalize();
        }

        private:
        using WritableStreamPosition = SerializedBlockStoreStream::WritableStreamPosition;
        struct IDRangeState {
            IDRangeState(BlockIDRange _range, WritableStreamPosition _positionInStream)
                : range(_range),
                  positionInStream(_positionInStream) {}
            BlockIDRange           range;
            WritableStreamPosition positionInStream;
        };

        // Serialize a BlockIDRange
        void serializeBlockIDRange(WritableStreamPosition& positionInStream, BlockIDRange range) {
            assert(positionInStream.bytesLeft() == sizeof(decltype(range.first)) + sizeof(decltype(range.last)));
            _stream.writeToReservedBytes(positionInStream, range.first); // Advances positionInStream.
            assert(positionInStream.bytesLeft() == sizeof(decltype(range.first)));
            _stream.writeToReservedBytes(positionInStream, range.last); // Advances positionInStream.
            assert(positionInStream.bytesLeft() == 0);
        }

        const BlockIDMode                        _mode;
        SerializedBlockStoreStream&              _stream;
        std::vector<std::optional<IDRangeState>> _currentRanges; // One range per rank
        bool                                     finalized = false;
    };

    template <typename IDType, class = std::enable_if_t<std::is_unsigned_v<IDType>>>
    class BlockIDDeserialization {
        public:
        using BlockIDMode  = typename BlockIDSerialization<IDType>::BlockIDMode;
        using BlockIDRange = typename BlockIDSerialization<IDType>::BlockIDRange;
        using DataStream   = std::vector<std::byte>;

        BlockIDDeserialization(BlockIDMode mode, const DataStream& dataStream)
            : _mode(mode),
              _dataStream(dataStream),
              _currentRange(std::nullopt),
              _lastId(std::numeric_limits<IDType>::max()) {
            // We have to read at least one descriptor from the data stream.
            assert(_dataStream.size() >= DESCRIPTOR_SIZE);
        }

        BlockIDDeserialization()                              = delete;
        BlockIDDeserialization(BlockIDDeserialization&&)      = delete;
        BlockIDDeserialization(const BlockIDDeserialization&) = delete;
        BlockIDDeserialization& operator=(BlockIDDeserialization&&) = delete;
        BlockIDDeserialization& operator=(const BlockIDDeserialization&) = delete;

        // Returns the next block. If the current range still hast ids left, returns the next one from that
        // range. If there are no more ids in the current range, reads the descriptor of the next range from the
        // given position in the stream. Returns (bytesConsumed,blockID) where bytesConsumed is the number of bytes
        // read from the data stream and blockID is the id of the next block.
        std::pair<size_t, IDType> readId(size_t position) {
// GCC throws a false-positive warning here.
#pragma GCC diagnostic push
#if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
            if (_currentRange && _lastId < _currentRange->last) {
#pragma GCC diagnostic pop
                return std::make_pair(0, ++_lastId);
            } else {
                return std::make_pair(DESCRIPTOR_SIZE, _deserializeId(position));
            }
        }

        private:
        IDType _deserializeId(size_t position) {
            const auto startOfDescriptor = position;
            if (startOfDescriptor + DESCRIPTOR_SIZE >= _dataStream.size()) {
                throw std::runtime_error("Trying to read the id descriptor past the end of the data stream.");
            }

            assert(position + sizeof(decltype(BlockIDRange::first)) < _dataStream.size());
            auto first = *reinterpret_cast<const decltype(BlockIDRange::first)*>(&(_dataStream[position]));
            position += sizeof(BlockIDRange::first);

            assert(position + sizeof(decltype(BlockIDRange::first)) < _dataStream.size());
            assert(&_dataStream[position] + sizeof(decltype(BlockIDRange::last)) <= &_dataStream.back());
            auto last = *reinterpret_cast<const IDType*>(&(_dataStream[position]));
            assert(position + sizeof(BlockIDRange::last) == startOfDescriptor + DESCRIPTOR_SIZE);

            assert(first <= last);
            if (!_currentRange) {
                _currentRange.emplace(first, last);
            } else {
                _currentRange->first = first;
                _currentRange->last  = last;
            }

            _lastId = _currentRange->first;
            return _lastId;
        }

        static constexpr size_t     DESCRIPTOR_SIZE = sizeof(BlockIDRange::first) + sizeof(BlockIDRange::last);
        BlockIDMode                 _mode;
        const DataStream&           _dataStream;
        std::optional<BlockIDRange> _currentRange;
        IDType                      _lastId;
    }; // namespace ReStore

    BlockSubmissionCommunication(
        const MPIContext& mpiContext, const BlockDistr& blockDistribution, OffsetModeDescriptor offsetModeDescriptor)
        : _mpiContext(mpiContext),
          _blockDistribution(blockDistribution),
          _offsetModeDescriptor(offsetModeDescriptor) {}


    SendBuffers copySerializedBlocksToSendBuffers(
        const std::vector<SerializedBlocksDescriptor>& blocks, const size_t localNumberOfBlocks) {
        const auto  bytesPerBlock = _offsetModeDescriptor.constOffset;
        SendBuffers sendBuffers;

        assert(_mpiContext.getOriginalSize() == _mpiContext.getCurrentSize());
        sendBuffers.resize(asserting_cast<size_t>(_mpiContext.getOriginalSize()));
        assert(std::for_each(
            sendBuffers.begin(), sendBuffers.end(), [](const std::vector<std::byte>& b) { return b.empty(); }));

        // Create the object which represents the store stream
        auto storeStream = SerializedBlockStoreStream(sendBuffers, _mpiContext.getOriginalSize());
        // This is too much, but let's see if we can get away with it.
        storeStream.reserve(bytesPerBlock * localNumberOfBlocks);

        // Create the object resposible for serializing the range ids
        BlockIDSerialization<block_id_t> blockIDSerializationManager(
            BlockIDSerialization<block_id_t>::BlockIDMode::RANGES, storeStream, _mpiContext.getOriginalSize());

        // Loop over the blocks and copy them to the respective send buffers.
        assert(blocks.size() >= 1);
        auto currentRange = _blockDistribution.rangeOfBlock(blocks[0].blockIdBegin);
        auto ranks        = _blockDistribution.ranksBlockRangeIsStoredOn(currentRange);
        // Create the proxy to write to; it will automatically copy the written bytes to every destination
        // rank's send buffer.
        storeStream.setDestinationRanks(ranks);
        for (const auto& block: blocks) {
            assert(block.blockIdEnd > block.blockIdBegin); // Empty ranges are not allowed, do not specify the range.
            const auto firstBlock = block.blockIdBegin;
            const auto lastBlock  = block.blockIdEnd - 1;
            auto       dataPtr    = block.data;
            assert(firstBlock <= lastBlock);
            assert(firstBlock < _blockDistribution.numBlocks());
            assert(lastBlock < _blockDistribution.numBlocks());

            auto nextBlock = firstBlock;

            do {
                // Determine which ranks will get the next block(s); assume that no failures occurred.
                assert(_mpiContext.numFailuresSinceReset() == 0);
                // Only recompute the destination ranks if they have changed.
                if (!currentRange.contains(nextBlock)) {
                    currentRange = _blockDistribution.rangeOfBlock(nextBlock);
                    ranks        = _blockDistribution.ranksBlockRangeIsStoredOn(currentRange);

                    // Create the proxy to write to; it will automatically copy the written bytes to every destination
                    // rank's send buffer.
                    storeStream.setDestinationRanks(ranks);
                }
                const auto firstBlockToWrite = nextBlock;
                const auto lastBlockToWrite  = std::min(lastBlock, currentRange.last());
                const auto numBlocksToWrite  = lastBlockToWrite - firstBlockToWrite + 1;
                assert(!ranks.empty());
                assert(currentRange.contains(firstBlockToWrite));
                assert(currentRange.contains(lastBlockToWrite));

                // Write the block's id to the stream.
                blockIDSerializationManager.writeId(firstBlockToWrite, lastBlockToWrite, ranks);

                // Copy the already serialized bytes to the send buffers.
                const auto bytesWrittenBeforeSerialization = storeStream.bytesWritten(ranks[0]);
                const auto numBytesToWrite                 = bytesPerBlock * numBlocksToWrite;
                storeStream.writeBytes(dataPtr, numBytesToWrite);
                assert(_offsetModeDescriptor.mode == OffsetMode::constant);
                [[maybe_unused]] const auto bytesWrittenDuringSerialization =
                    storeStream.bytesWritten(ranks[0]) - bytesWrittenBeforeSerialization;
                assert(bytesWrittenDuringSerialization == numBytesToWrite);

                // Forward nextBlock and the currentDataPtr
                nextBlock += numBlocksToWrite;
                assert(numBytesToWrite == bytesPerBlock * numBlocksToWrite);
                dataPtr += numBytesToWrite;
            } while (nextBlock <= lastBlock);
            assert(nextBlock == lastBlock + 1);
        }

        // Write out the remaining open id ranges
        blockIDSerializationManager.finalize();

        return sendBuffers;
    }

    // serializeBlocksForTransmission()
    //
    // Serializes all blocks enumerated by repeated calls to nextBlock() using the provided serializeFunc()
    // callback into one send buffer for each rank which gets at least one block.
    // serializeFunc and nextBlock are identical to the parameters described in submitBlocks
    // Allocates and returns the send buffers.
    // Assumes that no ranks have failed since the creation of BlockDistribution and reset of MPIContext.
    template <class SerializeBlockCallbackFunction, class NextBlockCallbackFunction, typename Permutation>
    SendBuffers serializeBlocksForTransmission(
        SerializeBlockCallbackFunction serializeFunc, NextBlockCallbackFunction nextBlock,
        const Permutation& blockIdPermuter, bool canBeParallelized = false // not supported yet
    ) {
        UNUSED(canBeParallelized);

        // Allocate one send buffer per destination rank
        SendBuffers sendBuffers;
        assert(_mpiContext.getOriginalSize() == _mpiContext.getCurrentSize());
        sendBuffers.resize(asserting_cast<size_t>(_mpiContext.getOriginalSize()));
        assert(std::for_each(
            sendBuffers.begin(), sendBuffers.end(), [](const std::vector<std::byte>& b) { return b.empty(); }));

        // Create the object which represents the store stream
        auto storeStream = SerializedBlockStoreStream(sendBuffers, _mpiContext.getOriginalSize());
        // storeStream.reserve(_blockDistribution.numBlocks());
        // TODO storeStream.reserve(...);

        // Create the object resposible for serializing the range ids
        BlockIDSerialization<block_id_t> blockIDSerializationManager(
            BlockIDSerialization<block_id_t>::BlockIDMode::RANGES, storeStream, _mpiContext.getOriginalSize());

        // Loop over the nextBlock generator to fetch all blocks we need to serialize.
        std::optional<typename BlockDistr::BlockRange> currentRange = std::nullopt;
        std::vector<ReStoreMPI::original_rank_t>       ranks;
        for (;;) {
            std::optional<NextBlock<BlockType>> next = nextBlock();
            if (!next.has_value()) {
                break; // No more blocks, we are done serializing.
            }

            block_id_t       blockId = next->blockId;
            const BlockType& block   = next->block;
            if (blockId >= _blockDistribution.numBlocks()) {
                throw std::runtime_error(
                    "The block id (" + std::to_string(blockId) + ") is bigger than the number of blocks ("
                    + std::to_string(_blockDistribution.numBlocks())
                    + "). Have you passed the number of block *in total* (not only on this rank)?");
            }

            // Use a pseudorandom bijection to break patterns in the distribution of the block id's over the ranks.
            blockId = blockIdPermuter.f(blockId);

            // Determine which ranks will get this block; assume that no failures occurred
            assert(_mpiContext.numFailuresSinceReset() == 0);
            // Only recompute the destination ranks if they have changed
            if (!currentRange || !(currentRange->contains(blockId))) {
                currentRange = _blockDistribution.rangeOfBlock(blockId);
                ranks        = _blockDistribution.ranksBlockRangeIsStoredOn(*currentRange);

                // Create the proxy which the user defined serializer will write to. This proxy overloads the <<
                // operator and automatically copies the written bytes to every destination rank's send buffer.
                storeStream.setDestinationRanks(ranks);
            }
            assert(!ranks.empty());
            assert(currentRange);

            // Write the block's id to the stream
            blockIDSerializationManager.writeId(blockId, ranks);

            // Call the user-defined serialization function to serialize the block to a flat byte stream
            auto bytesWrittenBeforeSerialization = storeStream.bytesWritten(ranks[0]);
            serializeFunc(block, storeStream);
            assert(_offsetModeDescriptor.mode == OffsetMode::constant);
            auto bytesWrittenDuringSerialization = storeStream.bytesWritten(ranks[0]) - bytesWrittenBeforeSerialization;
            if (bytesWrittenDuringSerialization != _offsetModeDescriptor.constOffset) {
                throw std::runtime_error(
                    "You wrote too many or too few bytes (" + std::to_string(bytesWrittenDuringSerialization)
                    + ") during serialization of block " + std::to_string(blockId) + ". Is your constant offset ("
                    + std::to_string(_offsetModeDescriptor.constOffset) + ") set correctly?");
            }
        }

        // Write out the remaining open id ranges
        blockIDSerializationManager.finalize();

        return sendBuffers;
    }

    // parseIncomingMessage()
    //
    // This functions iterates over a received message and calls handleBlockData() for each block stored in the
    // messages payload. This functions responsabilities are figuring out where a block starts and ends as well as
    // which id it has.
    //      message: The message to be parsed.
    //      handleBlockData(block_id_t blockId, const std::byte* data, size_t lengthInBytes): the callback function
    //      to call for each detected block.
    // If an empty message is passed, nothing will be done.
    // TODO implement LUT mode
    template <class HandleBlockDataFunc>
    void parseIncomingMessage(const ReStoreMPI::RecvMessage& message, HandleBlockDataFunc handleBlockData) {
        static_assert(
            std::is_invocable<HandleBlockDataFunc, block_id_t, const std::byte*, size_t, ReStoreMPI::current_rank_t>(),
            "handleBlockData has to be invocable as _(block_id_t, const std::byte*, size_t, current_rank_t)");

        assert(_offsetModeDescriptor.mode == OffsetMode::constant);
        assert(message.data.size() > 0);

        BlockIDDeserialization<block_id_t> blockIdDeserializer(
            BlockIDSerialization<block_id_t>::BlockIDMode::RANGES, message.data);

        decltype(message.data.size()) consumedBytes = 0;
        while (consumedBytes < message.data.size()) {
            auto [bytesRead, currentBlockId] = blockIdDeserializer.readId(consumedBytes);
            consumedBytes += bytesRead;

            auto startOfPayload = consumedBytes;
            assert(startOfPayload < message.data.size());

            // Handle the block data
            const auto payloadSize = _offsetModeDescriptor.constOffset;
            handleBlockData(currentBlockId, message.data.data() + startOfPayload, payloadSize, message.srcRank);
            consumedBytes += payloadSize;
        }
    }

    // parseAllIncomingMessages()
    //
    // Iterates over the given messages and calls parseIncomingMessage() for each of them.
    template <class HandleBlockDataFunc>
    void parseAllIncomingMessages(
        const std::vector<ReStoreMPI::RecvMessage>& messages, HandleBlockDataFunc handleBlockData) {
        for (auto&& message: messages) {
            assert(message.data.size() > 0);
            assert(message.srcRank < _mpiContext.getCurrentSize());
            parseIncomingMessage(message, handleBlockData);
        }
    }

    // exchangeData()
    //
    // A wrapper around a SparseAllToAll. Transmits the sendBuffers' content and receives data addressed to us.
    // Sending no data is fine, you may still retrieve data.
    std::vector<ReStoreMPI::RecvMessage> exchangeData(const SendBuffers& sendBuffers) {
#ifndef DENSE_ALL_TO_ALL_IN_SUBMIT_BLOCKS
        std::vector<ReStoreMPI::SendMessage> sendMessages;
        for (ReStoreMPI::current_rank_t rankId = 0; asserting_cast<size_t>(rankId) < sendBuffers.size(); rankId++) {
            auto& buffer = sendBuffers[asserting_cast<size_t>(rankId)];
            // Skip empty messages
            if (buffer.size() > 0) {
                sendMessages.emplace_back(ReStoreMPI::SendMessage{buffer.data(), (int)buffer.size(), rankId});
            }
        }

        return _mpiContext.SparseAllToAll(sendMessages);
#else
        assert(sendBuffers.size() == asserting_cast<size_t>(_mpiContext.getCurrentSize()));
        std::vector<std::byte> sendData;
        std::vector<int>       sendCounts(sendBuffers.size());
        std::vector<int>       sendDispls(sendBuffers.size());
        size_t                 numBytesToSend = 0;
        for (ReStoreMPI::current_rank_t rankId = 0; asserting_cast<size_t>(rankId) < sendBuffers.size(); rankId++) {
            const auto rankIdAsSizeT = asserting_cast<size_t>(rankId);
            auto&      buffer        = sendBuffers[rankIdAsSizeT];
            numBytesToSend += buffer.size();
            sendCounts[rankIdAsSizeT] = asserting_cast<int>(buffer.size());
            sendDispls[rankIdAsSizeT] = rankId == 0 ? 0 : sendDispls[rankIdAsSizeT - 1] + sendCounts[rankIdAsSizeT - 1];
        }
        sendData.reserve(numBytesToSend);
        for (ReStoreMPI::current_rank_t rankId = 0; asserting_cast<size_t>(rankId) < sendBuffers.size(); rankId++) {
            const auto rankIdAsSizeT = asserting_cast<size_t>(rankId);
            auto&      buffer        = sendBuffers[rankIdAsSizeT];
            assert(sendCounts[rankIdAsSizeT] == asserting_cast<int>(buffer.size()));
            assert(sendDispls[rankIdAsSizeT] == asserting_cast<int>(sendData.size()));
            sendData.insert(sendData.end(), buffer.begin(), buffer.end());
        }

        std::vector<int> recvCounts(sendBuffers.size());
        _mpiContext.alltoall(sendCounts, recvCounts, 1);
        std::vector<int> recvDispls(sendBuffers.size());
        std::exclusive_scan(recvCounts.begin(), recvCounts.end(), recvDispls.begin(), 0);


        std::vector<std::byte> recvData(asserting_cast<size_t>(recvDispls.back() + recvCounts.back()));
        _mpiContext.alltoallv(sendData, sendCounts, sendDispls, recvData, recvCounts, recvDispls);

        std::vector<ReStoreMPI::RecvMessage> result;
        result.reserve(sendBuffers.size());
        for (ReStoreMPI::current_rank_t rankId = 0; asserting_cast<size_t>(rankId) < sendBuffers.size(); rankId++) {
            const auto rankIdAsSizeT = asserting_cast<size_t>(rankId);
            const auto displ         = recvDispls[rankIdAsSizeT];
            const auto count         = recvCounts[rankIdAsSizeT];
            if (count > 0) {
                result.emplace_back(
                    std::vector<std::byte>(recvData.begin() + displ, recvData.begin() + displ + count), rankId);
            }
        }

        return result;
#endif
    }

    private:
    const MPIContext&          _mpiContext;
    const BlockDistr&          _blockDistribution;
    const OffsetModeDescriptor _offsetModeDescriptor;
}; // namespace ReStore
} // end of namespace ReStore
#endif // Include guard
