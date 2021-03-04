#ifndef RESTORE_BLOCK_SUBMISSION_H
#define RESTORE_BLOCK_SUBMISSION_H

#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "restore/block_distribution.hpp"
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

    BlockSubmissionCommunication(MPIContext& mpiContext, const BlockDistr& blockDistribution)
        : _mpiContext(mpiContext),
          _blockDistribution(blockDistribution) {}

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

        bool doneSerializingBlocks = false;
        // Loop over the nextBlock generator to fetch all block we need to serialize
        do {
            std::optional<NextBlock<BlockType>> next = nextBlock();
            if (!next.has_value()) {
                doneSerializingBlocks = true;
            } else {
                block_id_t       blockId = next.value().blockId;
                const BlockType& block   = next.value().block;
                assert(blockId < _blockDistribution.numBlocks());

                // Determine which ranks will get this block; assume that no failures occurred
                assert(_mpiContext.numFailuresSinceReset() == 0);
                auto ranks = _blockDistribution.ranksBlockIsStoredOn(blockId);

                // Create the proxy which the user defined serializer will write to. This proxy overloads the <<
                // operator and automatically copies the written bytes to every destination rank's send buffer.
                auto storeStream = SerializedBlockStoreStream(sendBuffers, ranks);

                // Write the block's id to the stream
                storeStream << blockId;
                // TODO implement efficient storing of continuous blocks

                // Call the user-defined serialization function to serialize the block to a flat byte stream
                serializeFunc(block, storeStream);
            }
        } while (!doneSerializingBlocks);

        return sendBuffers;
    }

    // parseIncomingMessage()
    //
    // This functions iterates over a received message and calls handleBlockData() for each block stored in the
    // messages payload. This functions responsabilities are figuring out where a block starts and ends as well as
    // which id it has. message: The message to be parsed. handleBlockData(block_id_t blockId, const std::byte* data,
    // size_t lengthInBytes) the callback function to call for each detected block.
    // TODO implement LUT mode
    template <class HandleBlockDataFunc>
    void parseIncomingMessage(
        const ReStoreMPI::RecvMessage& message, HandleBlockDataFunc handleBlockData,
        const std::pair<OffsetMode, size_t>& offsetModeDescriptor) {
        static_assert(
            std::is_invocable<HandleBlockDataFunc, block_id_t, const std::byte*, size_t, ReStoreMPI::current_rank_t>(),
            "handleBlockData has to be invocable as _(block_id_t, const std::byte*, size_t, current_rank_t)");

        assert(offsetModeDescriptor.first == OffsetMode::constant);
        auto constOffset = offsetModeDescriptor.second;

        block_id_t currentBlockId;

        size_t bytesPerBlock = sizeof(block_id_t) + constOffset;
        assert(bytesPerBlock > 0);
        assert(message.data.size() % bytesPerBlock == 0);

        size_t numBlocksInThisMessage = message.data.size() / bytesPerBlock;
        assert(numBlocksInThisMessage > 0);

        for (size_t blockIndex = 0; blockIndex < numBlocksInThisMessage; ++blockIndex) {
            auto startOfBlockId = blockIndex * bytesPerBlock;
            auto startOfPayload = startOfBlockId + sizeof(decltype(currentBlockId));
            assert(startOfBlockId < message.data.size());
            assert(startOfPayload < message.data.size());
            assert(startOfBlockId < startOfPayload);

            // Get the block id from the data stream
            // This const kind of feels like a Cola light with a burger menu ...
            currentBlockId = *reinterpret_cast<const decltype(currentBlockId)*>(message.data.data() + startOfBlockId);

            // Handle the block data
            handleBlockData(currentBlockId, message.data.data() + startOfPayload, constOffset, message.srcRank);
        }
    }

    // parseAllIncomingMessages()
    //
    // Iterates over the given messages and calls parseIncomingMessage() for each of them.
    template <class HandleBlockDataFunc>
    void parseAllIncomingMessages(
        const std::vector<ReStoreMPI::RecvMessage>& messages, HandleBlockDataFunc handleBlockData,
        const std::pair<OffsetMode, size_t>& offsetModeDescriptor) {
        for (auto&& message: messages) {
            parseIncomingMessage(message, handleBlockData, offsetModeDescriptor);
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
    const MPIContext& _mpiContext;
    const BlockDistr& _blockDistribution;
};
} // end of namespace ReStore
#endif // Include guard
