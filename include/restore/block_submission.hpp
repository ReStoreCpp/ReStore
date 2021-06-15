#ifndef RESTORE_BLOCK_SUBMISSION_H
#define RESTORE_BLOCK_SUBMISSION_H

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

namespace ReStore {

// This class implements helper functions for communication during ReStore::submitBlocks()
template <class BlockType, class MPIContext = ReStoreMPI::MPIContext>
class BlockSubmissionCommunication {
    public:
    // If the user did his homework and designed a BlockDistribution which requires few messages to be send
    // we do not want to allocate all those unneeded send buffers... that's why we use a map here instead
    // of a vector.
    using SendBuffers = std::unordered_map<ReStoreMPI::current_rank_t, std::vector<std::byte>>;

    using BlockDistr = BlockDistribution<MPIContext>;

    // BlockIDSerialization
    //
    // This class abstracts away the serialization of the block IDs. This enables us to save bytes when storing
    // consecutive block IDs by not writing them out.
    template <typename IDType, class = std::enable_if<std::is_unsigned_v<IDType>>>
    class BlockIDSerialization {
        public:
        enum class BlockIDMode : uint8_t { EVERY_ID, RANGES };
        struct BlockIDRange {
            explicit BlockIDRange(IDType id) : first(id), last(id) {}
            BlockIDRange(IDType _first, IDType _last) : first(_first), last(_last) {}
            IDType first;
            IDType last;
        };

        BlockIDSerialization(BlockIDMode mode, SerializedBlockStoreStream& stream) : _mode(mode), _stream(stream) {
            if (mode != BlockIDMode::RANGES) {
                throw std::runtime_error("Currently, only the RANGES mode is supported.");
            }
        }

        void writeId(IDType id, std::vector<ReStoreMPI::original_rank_t> ranks) {
            for (auto rank: ranks) {
                writeId(id, rank);
            }
        }

        void writeId(IDType id, ReStoreMPI::original_rank_t rank) {
            // Get the range state of this rank
            auto iter = _currentRanges.find(rank);

            // Is this the first id?
            if (iter == _currentRanges.end()) {
                BlockIDRange range(id);
                auto         positionInStream = _stream.reserveBytesForWriting(rank, 2 * sizeof(IDType));
                auto [kv, inserted]           = _currentRanges.try_emplace(rank, range, positionInStream);
                UNUSED(inserted);
                UNUSED(kv);
                assert(inserted);
            } else { // This is not the first id
                auto& rangeState = iter->second;
                // Is this a consecutive id range?
                if (rangeState.range.last == id - 1) {
                    rangeState.range.last++;
                } else { // Not a consecutive id
                    // close and write out previous range
                    serializeBlockIDRange(rangeState.positionInStream, rangeState.range);
                    rangeState.range.first = rangeState.range.last = id;

                    // start a new range
                    rangeState.positionInStream = _stream.reserveBytesForWriting(rank, 2 * sizeof(IDType));
                }
            }
        }

        // Closes up the current range and writes it to the stream.
        void finalize() {
            if (!finalized) {
                for (auto& [rankId, rangeState]: _currentRanges) {
                    serializeBlockIDRange(rangeState.positionInStream, rangeState.range);
                }
            }
            finalized = true;
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
            _stream.writeToReservedBytes(positionInStream, range.first);
            assert(positionInStream.bytesLeft() == sizeof(decltype(range.first)));
            _stream.writeToReservedBytes(positionInStream, range.last);
            assert(positionInStream.bytesLeft() == 0);
        }

        const BlockIDMode                                             _mode;
        SerializedBlockStoreStream&                                   _stream;
        std::unordered_map<ReStoreMPI::original_rank_t, IDRangeState> _currentRanges;
        bool                                                          finalized = false;
    };

    template <typename IDType, class = std::enable_if<std::is_unsigned_v<IDType>>>
    class BlockIDDeserialization {
        public:
        using BlockIDMode  = typename BlockIDSerialization<IDType>::BlockIDMode;
        using BlockIDRange = typename BlockIDSerialization<IDType>::BlockIDRange;
        using DataStream   = std::vector<std::byte>;

        BlockIDDeserialization(BlockIDMode mode, const DataStream& dataStream)
            : _mode(mode),
              _dataStream(dataStream),
              _currentRange(std::nullopt) {}

        // Returns the next block if. If the current range still hast ids left, returns the next one from that range. If
        // there are no more ids in the current range, reads the descriptor of the next range form the given position in
        // the stream.
        // Returns (bytesConsumed,blockID) where bytesConsumed is the number of bytes read from the data stream and
        // blockID is the id of the next block.
        std::pair<size_t, IDType> readId(size_t position) {
            if (_currentRange.has_value() && _lastId < _currentRange->last) {
                return std::make_pair(0, ++_lastId);
            } else {
                return std::make_pair(DESCRIPTOR_SIZE, deserializeId(position));
            }
        }

        private:
        IDType deserializeId(size_t position) {
            const auto startOfDescriptor = position;
            if (startOfDescriptor + DESCRIPTOR_SIZE >= _dataStream.size()) {
                throw new std::runtime_error("Trying to read the id descriptor past the end of the data stream.");
            }

            UNUSED(position);

            auto first = *reinterpret_cast<const IDType*>(&(_dataStream[position]));
            position += sizeof(BlockIDRange::first);

            auto last = *reinterpret_cast<const IDType*>(&(_dataStream[position]));
            assert(position + sizeof(BlockIDRange::last) == startOfDescriptor + DESCRIPTOR_SIZE);

            if (!_currentRange.has_value()) {
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
    };

    BlockSubmissionCommunication(
        const MPIContext& mpiContext, const BlockDistr& blockDistribution, OffsetModeDescriptor offsetModeDescriptor)
        : _mpiContext(mpiContext),
          _blockDistribution(blockDistribution),
          _offsetModeDescriptor(offsetModeDescriptor) {}

    // serializeBlocksForTransmission()
    //
    // Serializes all blocks enumerated by repeated calls to nextBlock() using the provided serializeFunc()
    // callback into one send buffer for each rank which gets at least one block.
    // serializeFunc and nextBlock are identical to the parameters described in submitBlocks
    // Allocates and returns the send buffers.
    // Assumes that no ranks have failed since the creation of BlockDistribution and reset of MPIContext.
    template <class SerializeBlockCallbackFunction, class NextBlockCallbackFunction>
    SendBuffers serializeBlocksForTransmission(
        SerializeBlockCallbackFunction serializeFunc, NextBlockCallbackFunction nextBlock,
        bool canBeParallelized = false // not supported yet
    ) {
        UNUSED(canBeParallelized);

        // Allocate one send buffer per destination rank
        SendBuffers sendBuffers;

        // Create the object which represents the store stream
        auto storeStream = SerializedBlockStoreStream(sendBuffers);
        // TODO storeStream.reserve(...);

        // Create the object resposible for serializing the range ids
        BlockIDSerialization<block_id_t> blockIDSerializationManager(
            BlockIDSerialization<block_id_t>::BlockIDMode::RANGES, storeStream);

        bool doneSerializingBlocks = false;
        // Loop over the nextBlock generator to fetch all block we need to serialize
        do {
            std::optional<NextBlock<BlockType>> next = nextBlock();
            if (!next.has_value()) {
                doneSerializingBlocks = true;
            } else {
                block_id_t       blockId = next.value().blockId;
                const BlockType& block   = next.value().block;
                if (blockId >= _blockDistribution.numBlocks()) {
                    throw std::runtime_error("The block id is bigger than the number of blocks. Have you passed "
                                             "the number of block *in total* (not only on this rank)?");
                }

                // Determine which ranks will get this block; assume that no failures occurred
                assert(_mpiContext.numFailuresSinceReset() == 0);
                auto ranks = _blockDistribution.ranksBlockIsStoredOn(blockId);

                // Create the proxy which the user defined serializer will write to. This proxy overloads the <<
                // operator and automatically copies the written bytes to every destination rank's send buffer.
                storeStream.setDestinationRanks(ranks);

                // Write the block's id to the stream
                blockIDSerializationManager.writeId(blockId, ranks);

                // Call the user-defined serialization function to serialize the block to a flat byte stream
                auto bytesWrittenBeforeSerialization = storeStream.bytesWritten(ranks[0]);
                serializeFunc(block, storeStream);
                assert(_offsetModeDescriptor.mode == OffsetMode::constant);
                auto bytesWrittenDuringSerialization =
                    storeStream.bytesWritten(ranks[0]) - bytesWrittenBeforeSerialization;
                if (bytesWrittenDuringSerialization != _offsetModeDescriptor.constOffset) {
                    throw std::runtime_error(
                        "You wrote too many or too few bytes (" + std::to_string(bytesWrittenDuringSerialization)
                        + ") during serialization of block " + std::to_string(blockId) + ". Is your constant offset ("
                        + std::to_string(_offsetModeDescriptor.constOffset) + ") set correctly?");
                }
            }
        } while (!doneSerializingBlocks);

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
    //      handleBlockData(block_id_t blockId, const std::byte* data, size_t lengthInBytes): the callback function to
    //      call for each detected block.
    // If an empty message is passed, nothing will be done.
    // TODO implement LUT mode
    template <class HandleBlockDataFunc>
    void parseIncomingMessage(const ReStoreMPI::RecvMessage& message, HandleBlockDataFunc handleBlockData) {
        static_assert(
            std::is_invocable<HandleBlockDataFunc, block_id_t, const std::byte*, size_t, ReStoreMPI::current_rank_t>(),
            "handleBlockData has to be invocable as _(block_id_t, const std::byte*, size_t, current_rank_t)");

        assert(_offsetModeDescriptor.mode == OffsetMode::constant);

        if (message.data.size() == 0) {
            return;
        }

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
            parseIncomingMessage(message, handleBlockData);
        }
    }

    // exchangeData()
    //
    // A wrapper around a SparseAllToAll. Transmits the sendBuffers' content and receives data addressed to us.
    // Sending no data is fine, you may still retreive data.
    std::vector<ReStoreMPI::RecvMessage> exchangeData(const SendBuffers& sendBuffers) {
        std::vector<ReStoreMPI::SendMessage> sendMessages;
        for (auto&& [rankId, buffer]: sendBuffers) {
            sendMessages.emplace_back(ReStoreMPI::SendMessage{buffer.data(), (int)buffer.size(), rankId});
        }

        return _mpiContext.SparseAllToAll(sendMessages);
    }

    private:
    const MPIContext&          _mpiContext;
    const BlockDistr&          _blockDistribution;
    const OffsetModeDescriptor _offsetModeDescriptor;
}; // namespace ReStore
} // end of namespace ReStore
#endif // Include guard