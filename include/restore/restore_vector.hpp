#include <cstddef>
#include <cstdint>
#include <vector>

#include "restore/core.hpp"
#include "restore/mpi_context.hpp"

namespace ReStore {

template <class data_t>
class ReStoreVector {
    public:
    using BlockRangeToRestore     = std::pair<std::pair<block_id_t, size_t>, ReStoreMPI::original_rank_t>;
    using BlockRangeToRestoreList = std::vector<BlockRangeToRestore>;

    ReStoreVector(size_t blockSize, MPI_Comm comm, uint16_t replicationLevel)
        : _nativeBlockSize(blockSize),
          _reStore(comm, replicationLevel, OffsetMode::constant, _bytesPerBlock()),
          _mpiContext(comm) {
        if (blockSize == 0) {
            throw std::invalid_argument("The block size must not be 0.");
        } else if (comm == MPI_COMM_NULL) {
            throw std::invalid_argument("Invalid MPI communicator.");
        } else if (replicationLevel == 0) {
            throw std::invalid_argument("The replication level must not be 0.");
        }
    }

    // Submits the data to the ReStore. If a rank failure occurs, this will throw a ReStoreMPI::FaultException.
    void submitData(const std::vector<data_t>& data) {
        if (data.empty()) {
            throw std::invalid_argument("The data vector does not contain any elements.");
        } else if (data.size() % _nativeBlockSize != 0) {
            throw std::invalid_argument("The data vector size is not a multiple of the block size.");
        }
        // TODO Maybe serialize multiple data points at once?

        auto numBlocksLocal = data.size() / _nativeBlockSize;
        // Will throw on rank failure.
        auto idOfMyFirstBlock = _mpiContext.exclusive_scan(numBlocksLocal, MPI_SUM);
        auto numBlocksGlobal  = _mpiContext.allreduce(numBlocksLocal, MPI_SUM);

        BlockProxy currentProxy   = nullptr;
        block_id_t currentBlockId = idOfMyFirstBlock;
        auto       blockIt        = data.begin();

        assert(data.size() * sizeof(data_t) == numBlocksLocal * _bytesPerBlock());
        // Will throw on Rank failure.
        _reStore.submitBlocks(
            [this](const BlockProxy& blockProxy, SerializedBlockStoreStream& stream) {
                assert(blockProxy != nullptr);
                stream.writeBytes(reinterpret_cast<const std::byte*>(blockProxy), _bytesPerBlock());
            },
            [&blockIt, &data, &currentProxy, &currentBlockId, idOfMyFirstBlock, numBlocksLocal, numBlocksGlobal, this]() {
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
                    blockIt += asserting_cast<long>(_nativeBlockSize);
                    currentBlockId++;
                }
                return nextBlock;
            },
            numBlocksGlobal);
    }

    // Update the communicator.
    void updateComm(MPI_Comm comm) {
        if (comm == MPI_COMM_NULL) {
            throw std::invalid_argument("Invalid communicator.");
        }
        _mpiContext.updateComm(comm);
        _reStore.updateComm(comm);
    }

    void restoreDataAppend(std::vector<data_t>& data, const BlockRangeToRestoreList& newBlocksPerOriginalRank) {
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
            });
    }

    private:
    using BlockProxy = const data_t*;

    size_t _bytesPerBlock() const {
        return _nativeBlockSize * sizeof(data_t);
    }

    size_t _nativeBlockSize; // The block size as specified by the user. E.g. the number of dimensions per data point.
    // size_t serializationBlockSize; // The block size for serialization, must be a multiple of nativeBlockSize.
    ReStore<BlockProxy>    _reStore;
    ReStoreMPI::MPIContext _mpiContext;
};

} // namespace ReStore
