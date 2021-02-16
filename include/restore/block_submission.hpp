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
    using SendBuffers = std::unordered_map<ReStoreMPI::current_rank_t, std::vector<uint8_t>>;

    using BlockDistr = BlockDistribution<MPIContext>;

    BlockSubmissionCommunication(MPIContext& mpiContext, const BlockDistr& blockDistribution)
        : _mpiContext(mpiContext),
          _blockDistribution(blockDistribution) {}

    // serializeBlocksForTransmission()
    //
    // Serializes all blocks enumerated by repeated calls to nextBlock() using the provided serializeFunc()
    // callback
    // into one send buffer for each rank which gets at least one block.
    // serializeFunc, nextBlock, totalNUmberOfBlocks are identical to the parameters described in submitBlocks
    // Allocates and returns the send buffers.
    template <class SerializeBlockCallbackFunction, class NextBlockCallbackFunction>
    SendBuffers serializeBlocksForTransmission(
        SerializeBlockCallbackFunction serializeFunc, NextBlockCallbackFunction nextBlock, size_t totalNumberOfBlocks,
        bool canBeParallelized = false // not supported yet
    ) {
        UNUSED(canBeParallelized);

        // Allocate one send buffer per destination rank
        SendBuffers sendBuffers;

        bool doneSerializingBlocks = false;
        // Loop over the nextBlock generator to fetch all block we need to serialize
        do {
            std::optional<std::pair<block_id_t, const BlockType&>> next = nextBlock();
            if (!next.has_value()) {
                doneSerializingBlocks = true;
            } else {
                block_id_t       blockId = next.value().first;
                const BlockType& block   = next.value().second;
                assert(blockId < totalNumberOfBlocks);

                // Determine which ranks will get this block
                auto ranks = _mpiContext.getAliveCurrentRanks(_blockDistribution.ranksBlockIsStoredOn(blockId));

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
    // which id it has. message: The message to be parsed. handleBlockData(block_id_t blockId, const uint8_t* data,
    // size_t lengthInBytes) the callback function to call for
    //      each detected block.
    // TODO implement LUT mode
    template <class HandleBlockDataFunc>
    void parseIncomingMessage(
        const ReStoreMPI::RecvMessage& message, HandleBlockDataFunc handleBlockData,
        const std::pair<OffsetMode, size_t>& offsetModeDescriptor) {
        static_assert(
            std::is_invocable<HandleBlockDataFunc, block_id_t, const uint8_t*, size_t>(),
            "handleBlockData has to be invocable as _(block_id_t, const uint8_t*, size_t)");

        assert(offsetModeDescriptor.first == OffsetMode::constant);
        auto constOffset = offsetModeDescriptor.second;

        block_id_t currentBlockId;

        size_t bytesPerBlock = sizeof(block_id_t) + constOffset;
        assert(bytesPerBlock > 0);
        assert(message.data.size() % bytesPerBlock == 0);

        size_t numBlocksInThisMessage = message.data.size() / bytesPerBlock;
        assert(numBlocksInThisMessage > 0);

        for (size_t blockIndex = 0; blockIndex < numBlocksInThisMessage; blockIndex++) {
            auto startOfBlock = blockIndex * bytesPerBlock;
            auto endOfId      = startOfBlock + sizeof(decltype(currentBlockId));
            assert(startOfBlock < message.data.size());
            assert(endOfId < message.data.size());
            assert(startOfBlock > endOfId);
            assert(endOfId <= std::numeric_limits<long>::max());

            // Get the block id from the data stream
            std::copy(
                message.data.begin() + static_cast<long>(startOfBlock),
                message.data.begin() + static_cast<long>(endOfId), &currentBlockId);

            // Handle the block data
            handleBlockData(currentBlockId, message.data.data() + endOfId, constOffset);
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
    const MPIContext&        _mpiContext;
    const BlockDistr&        _blockDistribution;
};
} // end of namespace ReStore
#endif // Include guard