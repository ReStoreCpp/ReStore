#include <algorithm>
#include <cassert>
#include <chrono>
#include <cppitertools/range.hpp>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mpi.h>
#include <random>
#include <restore/common.hpp>
#include <utility>

#if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wsuggest-override"
    #include <benchmark/benchmark.h>
    #pragma GCC diagnostic pop
#else
    #include <benchmark/benchmark.h>
#endif

#include <../tests/mpi_helpers.hpp>
#include <restore/core.hpp>
#include <restore/helpers.hpp>

using iter::range;

static void BM_submitBlocks(benchmark::State& state) {
    // Parse arguments
    auto blockSize        = throwing_cast<size_t>(state.range(0));
    auto replicationLevel = throwing_cast<uint16_t>(state.range(1));
    auto bytesPerRank     = throwing_cast<size_t>(state.range(2));

    assert(bytesPerRank % blockSize == 0);
    size_t blocksPerRank = bytesPerRank / blockSize;

    using ElementType = uint8_t;
    using BlockType   = std::vector<ElementType>;

    // Measurement
    for (auto _: state) {
        UNUSED(_);

        // Setup
        auto rankId    = asserting_cast<uint64_t>(myRankId());
        auto numBlocks = static_cast<size_t>(numRanks()) * bytesPerRank / blockSize;

        std::vector<BlockType> data;
        for (uint64_t base: range(blocksPerRank * rankId, blocksPerRank * rankId + blocksPerRank)) {
            data.emplace_back();
            data.back().reserve(blockSize);
            for (uint64_t increment: range(0ul, blockSize)) {
                data.back().push_back(
                    static_cast<ElementType>((base - increment) % std::numeric_limits<uint8_t>::max()));
            }
            assert(data.back().size() == blockSize);
        }
        assert(data.size() == blocksPerRank);

        ReStore::ReStore<BlockType> store(
            MPI_COMM_WORLD, replicationLevel, ReStore::OffsetMode::constant, sizeof(uint8_t) * blockSize);

        // Ensure, that all ranks start into the times section at about the same time. This prevens faster ranks from
        // having to wait for the slower ranks in the timed section. This ist also a workaround for a bug in the
        // SparseAllToAll implementation which will sometimes allow messages spilling over into the next SparseAllToAll
        // round.
        MPI_Barrier(MPI_COMM_WORLD);

        // Start measurement
        auto start = std::chrono::high_resolution_clock::now();

        unsigned counter = 0;
        store.submitBlocks(
            [](const BlockType& range, ReStore::SerializedBlockStoreStream& stream) {
                stream.writeBytes(reinterpret_cast<const std::byte*>(range.data()), range.size() * sizeof(ElementType));
            },
            [&counter, &data]() -> std::optional<ReStore::NextBlock<BlockType>> {
                auto ret = data.size() == counter
                               ? std::nullopt
                               : std::make_optional(ReStore::NextBlock<BlockType>(
                                   {counter + static_cast<size_t>(myRankId()) * data.size(), data[counter]}));
                counter++; // We cannot put this in the above line, as we can't assume if the first argument of the pair
                           // is bound before or after the increment.
                return ret;
            },
            numBlocks);
        assert(counter == data.size() + 1);

        // End and register measurement
        auto end            = std::chrono::high_resolution_clock::now();
        auto elapsedSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        MPI_Allreduce(MPI_IN_PLACE, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(elapsedSeconds);
    }
}

static void BM_pushBlocksRedistribute(benchmark::State& state) {
    // Each rank submits different data. The replication level is set to 3.

    // Parse arguments
    auto blockSize        = throwing_cast<size_t>(state.range(0));
    auto replicationLevel = throwing_cast<uint16_t>(state.range(1));
    auto bytesPerRank     = throwing_cast<size_t>(state.range(2));

    assert(bytesPerRank % blockSize == 0);
    auto blocksPerRank = bytesPerRank / blockSize;

    using ElementType = uint8_t;
    using BlockType   = std::vector<ElementType>;

    // Setup
    uint64_t rankId    = asserting_cast<uint64_t>(myRankId());
    size_t   numBlocks = static_cast<size_t>(numRanks()) * bytesPerRank / blockSize;

    std::vector<BlockType> data;
    for (uint64_t base: range(blocksPerRank * rankId, blocksPerRank * rankId + blocksPerRank)) {
        data.emplace_back();
        data.back().reserve(blockSize);
        for (uint64_t increment: range(0ul, blockSize)) {
            data.back().push_back(static_cast<ElementType>((base - increment) % std::numeric_limits<uint8_t>::max()));
        }
        assert(data.back().size() == blockSize);
    }
    assert(data.size() == blocksPerRank);

    ReStore::ReStore<BlockType> store(
        MPI_COMM_WORLD, replicationLevel, ReStore::OffsetMode::constant, sizeof(uint8_t) * blockSize);

    unsigned counter = 0;

    store.submitBlocks(
        [](const BlockType& range, ReStore::SerializedBlockStoreStream& stream) {
            stream.writeBytes(reinterpret_cast<const std::byte*>(range.data()), range.size() * sizeof(ElementType));
        },
        [&counter, &data]() -> std::optional<ReStore::NextBlock<BlockType>> {
            auto ret = data.size() == counter
                           ? std::nullopt
                           : std::make_optional(ReStore::NextBlock<BlockType>(
                               {counter + static_cast<size_t>(myRankId()) * data.size(), data[counter]}));
            counter++; // We cannot put this in the above line, as we can't assume if the first argument of the pair
                       // is bound before or after the increment.
            return ret;
        },
        numBlocks);
    assert(counter == data.size() + 1);

    std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, int>> blockRanges;
    for (int rank: range(numRanks())) {
        // Get data that was originally on the next rank
        int                 nextRank   = (rank + 1) % numRanks();
        ReStore::block_id_t startBlock = static_cast<size_t>(nextRank) * blocksPerRank;
        blockRanges.emplace_back(std::make_pair(startBlock, blocksPerRank), rank);
    }
    auto myStartBlock = static_cast<size_t>((myRankId() + 1) % numRanks()) * blocksPerRank;

    std::vector<BlockType> recvData(blocksPerRank, BlockType(blockSize));
    // Measurement
    for (auto _: state) {
        UNUSED(_);

        // Ensure, that all ranks start into the times section at about the same time. This prevens faster ranks from
        // having to wait for the slower ranks in the timed section. This ist also a workaround for a bug in the
        // SparseAllToAll implementation which will sometimes allow messages spilling over into the next SparseAllToAll
        // round.
        MPI_Barrier(MPI_COMM_WORLD);

        auto start = std::chrono::high_resolution_clock::now();
        store.pushBlocksCurrentRankIds(
            blockRanges, [&recvData, myStartBlock](const std::byte* buffer, size_t size, ReStore::block_id_t blockId) {
                assert(blockId >= myStartBlock);
                auto index = blockId - myStartBlock;
                assert(index < recvData.size());
                // assert(recvData[index].size() == 0);
                recvData[index].clear();
                recvData[index].insert(
                    recvData[index].end(), reinterpret_cast<const ElementType*>(buffer),
                    reinterpret_cast<const ElementType*>(buffer + size));
            });
        benchmark::DoNotOptimize(recvData.data());
        benchmark::ClobberMemory();
        assert(std::all_of(recvData.begin(), recvData.end(), [blockSize](const BlockType& block) {
            return block.size() == blockSize;
        }));
        auto end            = std::chrono::high_resolution_clock::now();
        auto elapsedSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        MPI_Allreduce(MPI_IN_PLACE, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(elapsedSeconds);
    }
}

static void BM_pullBlocksSmallRange(benchmark::State& state) {
    // Each rank submits different data. The replication level is set to 3.

    // Parse arguments
    auto blockSize                 = throwing_cast<size_t>(state.range(0));
    auto replicationLevel          = throwing_cast<uint16_t>(state.range(1));
    auto bytesPerRank              = throwing_cast<size_t>(state.range(2));
    auto numRanksDataLoss          = throwing_cast<size_t>(state.range(3));
    auto blocksPerRangePermutation = throwing_cast<size_t>(state.range(4));

    assert(bytesPerRank % blockSize == 0);
    auto blocksPerRank = bytesPerRank / blockSize;

    auto recvBlocksPerRank = blocksPerRank * numRanksDataLoss / asserting_cast<size_t>(numRanks());

    if (recvBlocksPerRank == 0) {
        state.SkipWithError("Parameters set such that no one receives anything!");
        return;
    }

    using ElementType = uint8_t;
    using BlockType   = std::vector<ElementType>;

    // Setup
    uint64_t rankId    = asserting_cast<uint64_t>(myRankId());
    size_t   numBlocks = static_cast<size_t>(numRanks()) * bytesPerRank / blockSize;

    std::vector<BlockType> data;
    for (uint64_t base: range(blocksPerRank * rankId, blocksPerRank * rankId + blocksPerRank)) {
        data.emplace_back();
        data.back().reserve(blockSize);
        for (uint64_t increment: range(0ul, blockSize)) {
            data.back().push_back(static_cast<ElementType>((base - increment) % std::numeric_limits<uint8_t>::max()));
        }
        assert(data.back().size() == blockSize);
    }
    assert(data.size() == blocksPerRank);

    ReStore::ReStore<BlockType> store(
        MPI_COMM_WORLD, replicationLevel, ReStore::OffsetMode::constant, sizeof(uint8_t) * blockSize,
        blocksPerRangePermutation);

    unsigned counter = 0;

    store.submitBlocks(
        [](const BlockType& range, ReStore::SerializedBlockStoreStream& stream) {
            stream.writeBytes(reinterpret_cast<const std::byte*>(range.data()), range.size() * sizeof(ElementType));
        },
        [&counter, &data]() -> std::optional<ReStore::NextBlock<BlockType>> {
            auto ret = data.size() == counter
                           ? std::nullopt
                           : std::make_optional(ReStore::NextBlock<BlockType>(
                               {counter + static_cast<size_t>(myRankId()) * data.size(), data[counter]}));
            counter++; // We cannot put this in the above line, as we can't assume if the first argument of the pair
                       // is bound before or after the increment.
            return ret;
        },
        numBlocks);
    assert(counter == data.size() + 1);

    std::vector<std::pair<ReStore::block_id_t, size_t>> blockRanges;
    blockRanges.reserve(asserting_cast<size_t>(numRanks()));
    ReStore::block_id_t myStartBlock = std::numeric_limits<ReStore::block_id_t>::max();

    // Use a random Range of numRanksDataLoss PEs whose data we redistribute
    std::mt19937                                             rng(42);
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, asserting_cast<unsigned long>(numRanks()) - numRanksDataLoss);

    std::vector<BlockType> recvData(recvBlocksPerRank, BlockType(blockSize));
    // Measurement
    for (auto _: state) {
        UNUSED(_);

        assert(myStartBlock == std::numeric_limits<ReStore::block_id_t>::max());

        // Build the data structure specifying which block to transfer to which rank.
        blockRanges.clear();
        ReStore::block_id_t startBlockId = dist(rng) * blocksPerRank;
        myStartBlock = startBlockId + recvBlocksPerRank * asserting_cast<ReStore::block_id_t>(myRankId());
        blockRanges.emplace_back(myStartBlock, recvBlocksPerRank);
        myStartBlock = startBlockId;
        assert(myStartBlock != std::numeric_limits<ReStore::block_id_t>::max());

        // Ensure, that all ranks start into the times section at about the same time. This prevens faster ranks from
        // having to wait for the slower ranks in the timed section. This ist also a workaround for a bug in the
        // SparseAllToAll implementation which will sometimes allow messages spilling over into the next SparseAllToAll
        // round.
        MPI_Barrier(MPI_COMM_WORLD);

        auto start = std::chrono::high_resolution_clock::now();
        store.pullBlocks(
            blockRanges, [&recvData, myStartBlock](const std::byte* buffer, size_t size, ReStore::block_id_t blockId) {
                assert(blockId >= myStartBlock);
                auto index = blockId - myStartBlock;
                assert(index < recvData.size());
                // assert(recvData[index].size() == 0);
                recvData[index].clear();
                recvData[index].insert(
                    recvData[index].end(), reinterpret_cast<const ElementType*>(buffer),
                    reinterpret_cast<const ElementType*>(buffer + size));
            });
        benchmark::DoNotOptimize(recvData.data());
        benchmark::ClobberMemory();
        assert(std::all_of(recvData.begin(), recvData.end(), [blockSize](const BlockType& block) {
            return block.size() == blockSize;
        }));
        auto end            = std::chrono::high_resolution_clock::now();
        auto elapsedSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        MPI_Allreduce(MPI_IN_PLACE, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(elapsedSeconds);
        myStartBlock = std::numeric_limits<ReStore::block_id_t>::max();
    }
}

static void BM_pullBlocksRedistribute(benchmark::State& state) {
    // Each rank submits different data. The replication level is set to 3.

    // Parse arguments
    auto blockSize        = throwing_cast<size_t>(state.range(0));
    auto replicationLevel = throwing_cast<uint16_t>(state.range(1));
    auto bytesPerRank     = throwing_cast<size_t>(state.range(2));

    assert(bytesPerRank % blockSize == 0);
    auto blocksPerRank = bytesPerRank / blockSize;

    using ElementType = uint8_t;
    using BlockType   = std::vector<ElementType>;

    // Setup
    uint64_t rankId    = asserting_cast<uint64_t>(myRankId());
    size_t   numBlocks = static_cast<size_t>(numRanks()) * bytesPerRank / blockSize;

    std::vector<BlockType> data;
    for (uint64_t base: range(blocksPerRank * rankId, blocksPerRank * rankId + blocksPerRank)) {
        data.emplace_back();
        data.back().reserve(blockSize);
        for (uint64_t increment: range(0ul, blockSize)) {
            data.back().push_back(static_cast<ElementType>((base - increment) % std::numeric_limits<uint8_t>::max()));
        }
        assert(data.back().size() == blockSize);
    }
    assert(data.size() == blocksPerRank);

    ReStore::ReStore<BlockType> store(
        MPI_COMM_WORLD, replicationLevel, ReStore::OffsetMode::constant, sizeof(uint8_t) * blockSize);

    unsigned counter = 0;

    store.submitBlocks(
        [](const BlockType& range, ReStore::SerializedBlockStoreStream& stream) {
            stream.writeBytes(reinterpret_cast<const std::byte*>(range.data()), range.size() * sizeof(ElementType));
        },
        [&counter, &data]() -> std::optional<ReStore::NextBlock<BlockType>> {
            auto ret = data.size() == counter
                           ? std::nullopt
                           : std::make_optional(ReStore::NextBlock<BlockType>(
                               {counter + static_cast<size_t>(myRankId()) * data.size(), data[counter]}));
            counter++; // We cannot put this in the above line, as we can't assume if the first argument of the pair
                       // is bound before or after the increment.
            return ret;
        },
        numBlocks);
    assert(counter == data.size() + 1);

    std::vector<std::pair<ReStore::block_id_t, size_t>> blockRanges;
    // Get data that was originally on the next rank
    int                 nextRank     = (myRankId() + 1) % numRanks();
    ReStore::block_id_t myStartBlock = static_cast<size_t>(nextRank) * blocksPerRank;
    blockRanges.emplace_back(std::make_pair(myStartBlock, blocksPerRank));

    std::vector<BlockType> recvData(blocksPerRank, BlockType(blockSize));
    // Measurement
    for (auto _: state) {
        UNUSED(_);

        // Ensure, that all ranks start into the times section at about the same time. This prevens faster ranks from
        // having to wait for the slower ranks in the timed section. This ist also a workaround for a bug in the
        // SparseAllToAll implementation which will sometimes allow messages spilling over into the next SparseAllToAll
        // round.
        MPI_Barrier(MPI_COMM_WORLD);

        auto start = std::chrono::high_resolution_clock::now();
        store.pullBlocks(
            blockRanges, [&recvData, myStartBlock](const std::byte* buffer, size_t size, ReStore::block_id_t blockId) {
                assert(blockId >= myStartBlock);
                auto index = blockId - myStartBlock;
                assert(index < recvData.size());
                // assert(recvData[index].size() == 0);
                recvData[index].clear();
                recvData[index].insert(
                    recvData[index].end(), reinterpret_cast<const ElementType*>(buffer),
                    reinterpret_cast<const ElementType*>(buffer + size));
            });
        benchmark::DoNotOptimize(recvData.data());
        benchmark::ClobberMemory();
        assert(std::all_of(recvData.begin(), recvData.end(), [blockSize](const BlockType& block) {
            return block.size() == blockSize;
        }));
        auto end            = std::chrono::high_resolution_clock::now();
        auto elapsedSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        MPI_Allreduce(MPI_IN_PLACE, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(elapsedSeconds);
    }
}

static void BM_pushBlocksSmallRange(benchmark::State& state) {
    // Each rank submits different data. The replication level is set to 3.

    // Parse arguments
    auto blockSize                 = throwing_cast<size_t>(state.range(0));
    auto replicationLevel          = throwing_cast<uint16_t>(state.range(1));
    auto bytesPerRank              = throwing_cast<size_t>(state.range(2));
    auto numRanksDataLoss          = throwing_cast<size_t>(state.range(3));
    auto blocksPerRangePermutation = throwing_cast<size_t>(state.range(4));

    assert(bytesPerRank % blockSize == 0);
    auto blocksPerRank = bytesPerRank / blockSize;

    auto recvBlocksPerRank = blocksPerRank * numRanksDataLoss / asserting_cast<size_t>(numRanks());

    if (recvBlocksPerRank == 0) {
        state.SkipWithError("Parameters set such that no one receives anything!");
        return;
    }

    using ElementType = uint8_t;
    using BlockType   = std::vector<ElementType>;

    // Setup
    uint64_t rankId    = asserting_cast<uint64_t>(myRankId());
    size_t   numBlocks = static_cast<size_t>(numRanks()) * bytesPerRank / blockSize;

    std::vector<BlockType> data;
    for (uint64_t base: range(blocksPerRank * rankId, blocksPerRank * rankId + blocksPerRank)) {
        data.emplace_back();
        data.back().reserve(blockSize);
        for (uint64_t increment: range(0ul, blockSize)) {
            data.back().push_back(static_cast<ElementType>((base - increment) % std::numeric_limits<uint8_t>::max()));
        }
        assert(data.back().size() == blockSize);
    }
    assert(data.size() == blocksPerRank);

    ReStore::ReStore<BlockType> store(
        MPI_COMM_WORLD, replicationLevel, ReStore::OffsetMode::constant, sizeof(uint8_t) * blockSize,
        blocksPerRangePermutation);

    unsigned counter = 0;

    store.submitBlocks(
        [](const BlockType& range, ReStore::SerializedBlockStoreStream& stream) {
            stream.writeBytes(reinterpret_cast<const std::byte*>(range.data()), range.size() * sizeof(ElementType));
        },
        [&counter, &data]() -> std::optional<ReStore::NextBlock<BlockType>> {
            auto ret = data.size() == counter
                           ? std::nullopt
                           : std::make_optional(ReStore::NextBlock<BlockType>(
                               {counter + static_cast<size_t>(myRankId()) * data.size(), data[counter]}));
            counter++; // We cannot put this in the above line, as we can't assume if the first argument of the pair
                       // is bound before or after the increment.
            return ret;
        },
        numBlocks);
    assert(counter == data.size() + 1);

    std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, int>> blockRanges;
    blockRanges.reserve(asserting_cast<size_t>(numRanks()));
    ReStore::block_id_t myStartBlock = std::numeric_limits<ReStore::block_id_t>::max();

    // Use a random Range of numRanksDataLoss PEs whose data we redistribute
    std::mt19937                                             rng(42);
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, asserting_cast<unsigned long>(numRanks()) - numRanksDataLoss);

    std::vector<BlockType> recvData(recvBlocksPerRank, BlockType(blockSize));
    // Measurement
    for (auto _: state) {
        UNUSED(_);

        assert(myStartBlock == std::numeric_limits<ReStore::block_id_t>::max());

        // Build the data structure specifying which block to transfer to which rank.
        blockRanges.clear();
        ReStore::block_id_t startBlockId = dist(rng) * blocksPerRank;
        for (int rank: range(numRanks())) {
            blockRanges.emplace_back(std::make_pair(startBlockId, recvBlocksPerRank), rank);
            if (rank == myRankId()) {
                myStartBlock = startBlockId;
            }
            startBlockId += recvBlocksPerRank;
            assert(startBlockId <= numBlocks);
        }
        assert(myStartBlock != std::numeric_limits<ReStore::block_id_t>::max());

        // Ensure, that all ranks start into the times section at about the same time. This prevens faster ranks from
        // having to wait for the slower ranks in the timed section. This ist also a workaround for a bug in the
        // SparseAllToAll implementation which will sometimes allow messages spilling over into the next SparseAllToAll
        // round.
        MPI_Barrier(MPI_COMM_WORLD);

        auto start = std::chrono::high_resolution_clock::now();
        store.pushBlocksCurrentRankIds(
            blockRanges, [&recvData, myStartBlock](const std::byte* buffer, size_t size, ReStore::block_id_t blockId) {
                assert(blockId >= myStartBlock);
                auto index = blockId - myStartBlock;
                assert(index < recvData.size());
                // assert(recvData[index].size() == 0);
                recvData[index].clear();
                recvData[index].insert(
                    recvData[index].end(), reinterpret_cast<const ElementType*>(buffer),
                    reinterpret_cast<const ElementType*>(buffer + size));
            });
        benchmark::DoNotOptimize(recvData.data());
        benchmark::ClobberMemory();
        assert(std::all_of(recvData.begin(), recvData.end(), [blockSize](const BlockType& block) {
            return block.size() == blockSize;
        }));
        auto end            = std::chrono::high_resolution_clock::now();
        auto elapsedSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        MPI_Allreduce(MPI_IN_PLACE, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(elapsedSeconds);
        myStartBlock = std::numeric_limits<ReStore::block_id_t>::max();
    }
}

template <typename N>
auto constexpr KiB(N n) {
    return n * 1024;
}

template <typename N>
auto constexpr MiB(N n) {
    return n * 1024 * KiB(1);
}

const auto MAX_DATA_LOSS_RANKS   = 8;
const auto MAX_REPLICATION_LEVEL = 4;

BENCHMARK(BM_submitBlocks)          ///
    ->UseManualTime()               ///
    ->Unit(benchmark::kMillisecond) ///
    ->ArgsProduct({
        // {8, 16, 32, 64, 128, 256, 512, KiB(1), MiB(1)}, // block sizes
        // We experimentally determined 64 bytes to be a solid tradeoff between performacne and granularity.
        {64},                               // block sizes
        {1, 2, 3, 4},                       // replication level
        {MiB(1), MiB(16), MiB(32), MiB(64)} //, MiB(128)} // bytes per rank
    });

BENCHMARK(BM_pushBlocksRedistribute) ///
    ->UseManualTime()                ///
    ->Unit(benchmark::kMillisecond)  ///
    ->ArgsProduct({
        // {8, 16, 32, 64, 128, 256, 512, KiB(1), MiB(1)}, // block sizes
        {64},                               // block sizes, see above.
        {2, 3, 4},                          // replication level
        {MiB(1), MiB(16), MiB(32), MiB(64)} //, MiB(128)} // bytes per rank
    });

BENCHMARK(BM_pushBlocksSmallRange)  ///
    ->UseManualTime()               ///
    ->Unit(benchmark::kMillisecond) ///
    ->ArgsProduct({
        // {8, 16, 32, 64, 128, 256, 512, KiB(1), MiB(1)}, // block sizes
        {64},                                // block sizes, see above
        {2, 3, 4},                           // replication level
        {MiB(1), MiB(16), MiB(32), MiB(64)}, //, MiB(128)} // bytes per rank
        {1, 2, 4, 8},                        // Number of ranks from which to get the data
        {1, 8, 128, KiB(1)}                  // Blocks per permutation range
    });

BENCHMARK(BM_pullBlocksRedistribute) ///
    ->UseManualTime()                ///
    ->Unit(benchmark::kMillisecond)  ///
    ->ArgsProduct({
        // {8, 16, 32, 64, 128, 256, 512, KiB(1), MiB(1)}, // block sizes
        {64},                               // block sizes, see above.
        {2, 3, 4},                          // replication level
        {MiB(1), MiB(16), MiB(32), MiB(64)} //, MiB(128)} // bytes per rank
    });

BENCHMARK(BM_pullBlocksSmallRange)  ///
    ->UseManualTime()               ///
    ->Unit(benchmark::kMillisecond) ///
    ->ArgsProduct({
        // {8, 16, 32, 64, 128, 256, 512, KiB(1), MiB(1)}, // block sizes
        {64},                                // block sizes, see above
        {2, 3, 4},                           // replication level
        {MiB(1), MiB(16), MiB(32), MiB(64)}, //, MiB(128)} // bytes per rank
        {1, 2, 4, 8},                        // Number of ranks from which to get the data
        {1, 8, 128, KiB(1)}                  // Blocks per permutation range
    });

// This reporter does nothing.
// We can use it to disable output from all but the root process
class NullReporter : public ::benchmark::BenchmarkReporter {
    public:
    NullReporter() = default;
    bool ReportContext(const Context&) override {
        return true;
    }
    void ReportRuns(const std::vector<Run>&) override {}
    void Finalize() override {}
};

// The main is rewritten to allow for MPI initializing and for selecting a reporter according to the process rank.
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    assert(rank >= 0);

    // Do we have enough MPI ranks?
    if (numRanks() < std::max(MAX_REPLICATION_LEVEL, MAX_DATA_LOSS_RANKS)) {
        std::cout << "Please call this benchmark with at least " << std::max(MAX_REPLICATION_LEVEL, MAX_DATA_LOSS_RANKS)
                  << " ranks." << std::endl;
        return 1;
    }

    if (rank == 0) {
        ::benchmark::Initialize(&argc, argv);

        // Root process will use a reporter from the usual set provided by ::benchmark
        ::benchmark::RunSpecifiedBenchmarks();
    } else {
        // Reporting from other processes is disabled by passing a custom reporter.
        // We have to disable the display AND file reporter.
        NullReporter null;

        // googlebenchmark will check if the benchmark_out parameter is set even when we prove a NullReporter. It does
        // this using the google flags libary. We can therefore specify the benchmark_out parameter on the command line
        // or using an environment variable.
        std::vector<char*> expanded_argv;
        for (int idx = 0; idx < argc; idx++) {
            expanded_argv.push_back(argv[idx]);
        }

        std::string tmpFile = std::filesystem::temp_directory_path();
        tmpFile.append("/restore-microbenchmark-sdfuihK789ahajgdfCVgjhkjFDTSATF.tmp");

        std::string benchmark_out_string = std::string{"--benchmark_out="} + tmpFile;
        auto        benchmark_out =
            std::unique_ptr<char>(reinterpret_cast<char*>(malloc(sizeof(char) * (benchmark_out_string.length() + 1))));
        strcpy(benchmark_out.get(), benchmark_out_string.c_str());

        expanded_argv.push_back(benchmark_out.get());
        argc++;

        // Parse command line parameters
        ::benchmark::Initialize(&argc, expanded_argv.data());

        // Run the benchmarks
        ::benchmark::RunSpecifiedBenchmarks(&null, &null);

        // Clean up the temporary output file
        std::filesystem::remove(tmpFile);
    }

    MPI_Finalize();
    return 0;
}
