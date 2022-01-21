#include <cstddef>
#include <cstdint>
#include <vector>

#include "restore/core.hpp"
#include "restore/helpers.hpp"
#include "restore/mpi_context.hpp"


namespace ReStore {

template <class data_t>
class ReStoreVector {
    public:
    using BlockRangeToRestore            = std::pair<block_id_t, size_t>;
    using BlockRangeRequestToRestore     = std::pair<BlockRangeToRestore, ReStoreMPI::original_rank_t>;
    using BlockRangeRequestToRestoreList = std::vector<BlockRangeRequestToRestore>;
    using BlockRangeToRestoreList        = std::vector<BlockRangeToRestore>;

    ReStoreVector(
        size_t blockSize, MPI_Comm comm, uint16_t replicationLevel, uint64_t blocksPerPermutationRange,
        data_t paddingValue = data_t())
        : _nativeBlockSize(blockSize),
          _reStore(comm, replicationLevel, OffsetMode::constant, _bytesPerBlock(), blocksPerPermutationRange),
          _mpiContext(comm),
          _paddingValue(paddingValue),
          _isPadded(false) {
        if (blockSize == 0) {
            throw std::invalid_argument("The block size must not be 0.");
        } else if (comm == MPI_COMM_NULL) {
            throw std::invalid_argument("Invalid MPI communicator.");
        } else if (replicationLevel == 0) {
            throw std::invalid_argument("The replication level must not be 0.");
        }
    }

    // Submits the data to the ReStore. If a rank failure occurs, this will throw a ReStoreMPI::FaultException.
    size_t submitData(const std::vector<data_t>& data) {
        if (data.empty()) {
            throw std::invalid_argument("The data vector does not contain any elements.");
        }
        // TODO Maybe serialize multiple data points at once?
        _isPadded = data.size() % _nativeBlockSize != 0;
        // Round up to the next multiple of _nativeBlockSize
        auto numBlocksLocal = (data.size() + _nativeBlockSize - 1) / _nativeBlockSize;
        // Will throw on rank failure.
        auto idOfMyFirstBlock = _mpiContext.exclusive_scan(numBlocksLocal, MPI_SUM);
        auto numBlocksGlobal  = _mpiContext.allreduce(numBlocksLocal, MPI_SUM);

        BlockProxy currentProxy   = nullptr;
        block_id_t currentBlockId = idOfMyFirstBlock;
        auto       blockIt        = data.begin();

        assert(data.size() * sizeof(data_t) == numBlocksLocal * _bytesPerBlock() || _isPadded);
        // Will throw on Rank failure.
        _reStore.submitBlocks(
            [this, &data](const BlockProxy& blockProxy, SerializedBlockStoreStream& stream) {
                assert(blockProxy != nullptr);
                if (blockProxy + _nativeBlockSize > data.data() + data.size()) {
                    // Only write as many bytes as there are left in data
                    size_t bytesToWrite = asserting_cast<size_t>(
                        reinterpret_cast<const std::byte*>(data.data() + data.size())
                        - reinterpret_cast<const std::byte*>(blockProxy));
                    stream.writeBytes(reinterpret_cast<const std::byte*>(blockProxy), bytesToWrite);
                    // Pad to the next multiple of _bytesPerBlock();
                    // This could probably be done without allocating a new vector but it only happens once, so it
                    // shouldn't really matter.
                    const size_t numPaddingBytes  = asserting_cast<size_t>(_bytesPerBlock() - bytesToWrite);
                    const size_t numPaddingValues = numPaddingBytes / sizeof(data_t);
                    assert(numPaddingBytes % sizeof(data_t) == 0);
                    std::vector<data_t> padding(numPaddingValues, _paddingValue);
                    stream.writeBytes(reinterpret_cast<std::byte*>(padding.data()), numPaddingBytes);
                } else {
                    // Enough data left. Just write it.
                    stream.writeBytes(reinterpret_cast<const std::byte*>(blockProxy), _bytesPerBlock());
                }
            },
            [&blockIt, &data, &currentProxy, &currentBlockId, idOfMyFirstBlock, numBlocksLocal, numBlocksGlobal,
             this]() {
                std::optional<NextBlock<BlockProxy>> nextBlock = std::nullopt;
                if (blockIt != data.end()) {
                    assert(currentBlockId >= idOfMyFirstBlock);
                    assert(currentBlockId < idOfMyFirstBlock + numBlocksLocal);
                    assert(currentBlockId < numBlocksGlobal);
                    UNUSED(idOfMyFirstBlock);
                    UNUSED(numBlocksLocal);
                    UNUSED(numBlocksGlobal);
                    currentProxy = BlockProxy{&(*blockIt)};
                    assert(currentProxy != nullptr);
                    nextBlock.emplace(currentBlockId, currentProxy);
                    blockIt += std::min(asserting_cast<long>(_nativeBlockSize), data.end() - blockIt);
                    currentBlockId++;
                }
                return nextBlock;
            },
            numBlocksGlobal);
        return numBlocksLocal;
    }

    // Update the communicator.
    void updateComm(MPI_Comm comm) {
        if (comm == MPI_COMM_NULL) {
            throw std::invalid_argument("Invalid communicator.");
        }
        _mpiContext.updateComm(comm);
        _reStore.updateComm(comm);
    }

    void restoreDataAppendPushBlocks(
        std::vector<data_t>& data, const BlockRangeRequestToRestoreList& newBlocksPerOriginalRank) {
        // How many new blocks is this rank getting?
        size_t numBlocksForMe = 0;
        auto   myOriginalRank = _mpiContext.getMyOriginalRank();
        for (auto range: newBlocksPerOriginalRank) {
            if (range.second == myOriginalRank) {
                numBlocksForMe += range.first.second;
            }
        }

        // Reserve the appropriate amount of space for the current plus the new elements in the data vector.
        auto sizeBeforeExpansion = data.size();
        data.resize(data.size() + numBlocksForMe * _nativeBlockSize);

        // Append the new blocks to the data vector
        auto nextBlockPtr = reinterpret_cast<std::byte*>(data.data() + sizeBeforeExpansion);
        _reStore.pushBlocksOriginalRankIds(
            newBlocksPerOriginalRank,
            [this, &nextBlockPtr](const std::byte* dataPtr, size_t dataSize, block_id_t blockId) {
                UNUSED(blockId);
                assert(_bytesPerBlock() == dataSize);
                UNUSED(this);
                std::copy(dataPtr, dataPtr + dataSize, nextBlockPtr);
                nextBlockPtr += dataSize;
                if (_isPadded) {
                    // Remove all padded values
                    while (*reinterpret_cast<data_t*>(nextBlockPtr - sizeof(data_t)) == _paddingValue) {
                        nextBlockPtr -= sizeof(data_t);
                    }
                }
            });
        // After removing padded values, we have to adjust the size
        data.resize(asserting_cast<size_t>(reinterpret_cast<data_t*>(nextBlockPtr) - data.data()));
    }

    void restoreDataAppendPullBlocks(std::vector<data_t>& data, const BlockRangeToRestoreList& newBlocks) {
        // How many new blocks is this rank getting?
        size_t numBlocksForMe = 0;
        for (auto range: newBlocks) {
            numBlocksForMe += range.second;
        }

        // Reserve the appropriate amount of space for the current plus the new elements in the data vector.
        auto sizeBeforeExpansion = data.size();
        data.resize(data.size() + numBlocksForMe * _nativeBlockSize);

        // Append the new blocks to the data vector
        auto nextBlockPtr = reinterpret_cast<std::byte*>(data.data() + sizeBeforeExpansion);
        _reStore.pullBlocks(
            newBlocks, [this, &nextBlockPtr](const std::byte* dataPtr, size_t dataSize, block_id_t blockId) {
                UNUSED(blockId);
                assert(_bytesPerBlock() == dataSize);
                UNUSED(this);
                std::copy(dataPtr, dataPtr + dataSize, nextBlockPtr);
                nextBlockPtr += dataSize;
                if (_isPadded) {
                    // Remove all padded values
                    while (*reinterpret_cast<data_t*>(nextBlockPtr - sizeof(data_t)) == _paddingValue) {
                        nextBlockPtr -= sizeof(data_t);
                    }
                }
            });
        // After removing padded values, we have to adjust the size
        data.resize(asserting_cast<size_t>(reinterpret_cast<data_t*>(nextBlockPtr) - data.data()));
    }


    std::vector<ReStoreMPI::original_rank_t> getRanksDiedSinceLastCall() {
        return _reStore.getRanksDiedSinceLastCall();
    }

    private:
    using BlockProxy = const data_t*;

    size_t _bytesPerBlock() const {
        return _nativeBlockSize * sizeof(data_t);
    }

    size_t _nativeBlockSize; // The block size as specified by the user.
    // size_t serializationBlockSize; // The block size for serialization, must be a multiple of nativeBlockSize.
    ReStore<BlockProxy>    _reStore;
    ReStoreMPI::MPIContext _mpiContext;
    data_t                 _paddingValue;
    bool                   _isPadded;
};

} // namespace ReStore
