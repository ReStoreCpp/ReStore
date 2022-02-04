#include "restore/mpi_context.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cppitertools/range.hpp>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mpi.h>
#include <random>
#include <restore/common.hpp>
#include <string>
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
    auto bytesPerBlock             = throwing_cast<size_t>(state.range(0));
    auto replicationLevel          = throwing_cast<uint16_t>(state.range(1));
    auto bytesPerRank              = throwing_cast<size_t>(state.range(2));
    auto blocksPerPermutationRange = throwing_cast<size_t>(state.range(3));
    auto fractionOfRanksThatFail   = static_cast<double>(state.range(4)) / 1000.;
    UNUSED(fractionOfRanksThatFail);

    assert(bytesPerRank % bytesPerBlock == 0);
    size_t blocksPerRank = bytesPerRank / bytesPerBlock;

    using ElementType = uint8_t;
    using BlockType   = std::vector<ElementType>;

    auto rankId    = asserting_cast<uint64_t>(myRankId());
    auto numBlocks = static_cast<size_t>(numRanks()) * bytesPerRank / bytesPerBlock;

    // Generate the data to be stored in the ReStore.
    std::vector<BlockType> data;
    for (uint64_t base: range(blocksPerRank * rankId, blocksPerRank * rankId + blocksPerRank)) {
        data.emplace_back();
        data.back().reserve(bytesPerBlock);
        for (uint64_t increment: range(0ul, bytesPerBlock)) {
            data.back().push_back(static_cast<ElementType>((base - increment) % std::numeric_limits<uint8_t>::max()));
        }
        assert(data.back().size() == bytesPerBlock);
    }
    assert(data.size() == blocksPerRank);

    // Measurement
    for (auto _: state) {
        UNUSED(_);

        ReStore::ReStore<BlockType> store(
            MPI_COMM_WORLD, replicationLevel, ReStore::OffsetMode::constant, sizeof(uint8_t) * bytesPerBlock,
            blocksPerPermutationRange);

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

// pushBlocks was declassified by pullBlocks in experiments, we therefore no longer benchmark it.
//
// static void BM_pushBlocksRedistribute(benchmark::State& state) {
//     // Each rank submits different data. The replication level is set to 3.
//
//     // Parse arguments
//     auto blockSize        = throwing_cast<size_t>(state.range(0));
//     auto replicationLevel = throwing_cast<uint16_t>(state.range(1));
//     auto bytesPerRank     = throwing_cast<size_t>(state.range(2));
//
//     assert(bytesPerRank % blockSize == 0);
//     auto blocksPerRank = bytesPerRank / blockSize;
//
//     using ElementType = uint8_t;
//     using BlockType   = std::vector<ElementType>;
//
//     // Setup
//     uint64_t rankId    = asserting_cast<uint64_t>(myRankId());
//     size_t   numBlocks = static_cast<size_t>(numRanks()) * bytesPerRank / blockSize;
//
//     std::vector<BlockType> data;
//     for (uint64_t base: range(blocksPerRank * rankId, blocksPerRank * rankId + blocksPerRank)) {
//         data.emplace_back();
//         data.back().reserve(blockSize);
//         for (uint64_t increment: range(0ul, blockSize)) {
//             data.back().push_back(static_cast<ElementType>((base - increment) %
//             std::numeric_limits<uint8_t>::max()));
//         }
//         assert(data.back().size() == blockSize);
//     }
//     assert(data.size() == blocksPerRank);
//
//     ReStore::ReStore<BlockType> store(
//         MPI_COMM_WORLD, replicationLevel, ReStore::OffsetMode::constant, sizeof(uint8_t) * blockSize);
//
//     unsigned counter = 0;
//
//     store.submitBlocks(
//         [](const BlockType& range, ReStore::SerializedBlockStoreStream& stream) {
//             stream.writeBytes(reinterpret_cast<const std::byte*>(range.data()), range.size() * sizeof(ElementType));
//         },
//         [&counter, &data]() -> std::optional<ReStore::NextBlock<BlockType>> {
//             auto ret = data.size() == counter
//                            ? std::nullopt
//                            : std::make_optional(ReStore::NextBlock<BlockType>(
//                                {counter + static_cast<size_t>(myRankId()) * data.size(), data[counter]}));
//             counter++; // We cannot put this in the above line, as we can't assume if the first argument of the pair
//                        // is bound before or after the increment.
//             return ret;
//         },
//         numBlocks);
//     assert(counter == data.size() + 1);
//
//     std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, int>> blockRanges;
//     for (int rank: range(numRanks())) {
//         // Get data that was originally on the next rank
//         int                 rankToLoadFrom   = (rank + 49) % numRanks();
//         ReStore::block_id_t startBlock = static_cast<size_t>(rankToLoadFrom) * blocksPerRank;
//         blockRanges.emplace_back(std::make_pair(startBlock, blocksPerRank), rank);
//     }
//     auto myStartBlock = static_cast<size_t>((myRankId() + 1) % numRanks()) * blocksPerRank;
//
//     std::vector<BlockType> recvData(blocksPerRank, BlockType(blockSize));
//     // Measurement
//     for (auto _: state) {
//         UNUSED(_);
//
//         // Ensure, that all ranks start into the times section at about the same time. This prevens faster ranks from
//         // having to wait for the slower ranks in the timed section. This ist also a workaround for a bug in the
//         // SparseAllToAll implementation which will sometimes allow messages spilling over into the next
//         SparseAllToAll
//         // round.
//         MPI_Barrier(MPI_COMM_WORLD);
//
//         auto start = std::chrono::high_resolution_clock::now();
//         store.pushBlocksCurrentRankIds(
//             blockRanges, [&recvData, myStartBlock](const std::byte* buffer, size_t size, ReStore::block_id_t blockId)
//             {
//                 assert(blockId >= myStartBlock);
//                 auto index = blockId - myStartBlock;
//                 assert(index < recvData.size());
//                 // assert(recvData[index].size() == 0);
//                 recvData[index].clear();
//                 recvData[index].insert(
//                     recvData[index].end(), reinterpret_cast<const ElementType*>(buffer),
//                     reinterpret_cast<const ElementType*>(buffer + size));
//             });
//         benchmark::DoNotOptimize(recvData.data());
//         benchmark::ClobberMemory();
//         assert(std::all_of(recvData.begin(), recvData.end(), [blockSize](const BlockType& block) {
//             return block.size() == blockSize;
//         }));
//         auto end            = std::chrono::high_resolution_clock::now();
//         auto elapsedSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
//         MPI_Allreduce(MPI_IN_PLACE, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//         state.SetIterationTime(elapsedSeconds);
//     }
// }
//
// static void BM_pushBlocksSmallRange(benchmark::State& state) {
//     // Each rank submits different data. The replication level is set to 3.
//
//     // Parse arguments
//     auto blockSize                 = throwing_cast<size_t>(state.range(0));
//     auto replicationLevel          = throwing_cast<uint16_t>(state.range(1));
//     auto bytesPerRank              = throwing_cast<size_t>(state.range(2));
//     auto numRanksDataLoss          = throwing_cast<size_t>(state.range(3));
//     auto blocksPerRangePermutation = throwing_cast<size_t>(state.range(4));
//
//     assert(bytesPerRank % blockSize == 0);
//     auto blocksPerRank = bytesPerRank / blockSize;
//
//     auto recvBlocksPerRank = blocksPerRank * numRanksDataLoss / asserting_cast<size_t>(numRanks());
//
//     if (recvBlocksPerRank == 0) {
//         state.SkipWithError("Parameters set such that no one receives anything!");
//         return;
//     }
//
//     using ElementType = uint8_t;
//     using BlockType   = std::vector<ElementType>;
//
//     // Setup
//     uint64_t rankId    = asserting_cast<uint64_t>(myRankId());
//     size_t   numBlocks = static_cast<size_t>(numRanks()) * bytesPerRank / blockSize;
//
//     std::vector<BlockType> data;
//     for (uint64_t base: range(blocksPerRank * rankId, blocksPerRank * rankId + blocksPerRank)) {
//         data.emplace_back();
//         data.back().reserve(blockSize);
//         for (uint64_t increment: range(0ul, blockSize)) {
//             data.back().push_back(static_cast<ElementType>((base - increment) %
//             std::numeric_limits<uint8_t>::max()));
//         }
//         assert(data.back().size() == blockSize);
//     }
//     assert(data.size() == blocksPerRank);
//
//     ReStore::ReStore<BlockType> store(
//         MPI_COMM_WORLD, replicationLevel, ReStore::OffsetMode::constant, sizeof(uint8_t) * blockSize,
//         blocksPerRangePermutation);
//
//     unsigned counter = 0;
//
//     store.submitBlocks(
//         [](const BlockType& range, ReStore::SerializedBlockStoreStream& stream) {
//             stream.writeBytes(reinterpret_cast<const std::byte*>(range.data()), range.size() * sizeof(ElementType));
//         },
//         [&counter, &data]() -> std::optional<ReStore::NextBlock<BlockType>> {
//             auto ret = data.size() == counter
//                            ? std::nullopt
//                            : std::make_optional(ReStore::NextBlock<BlockType>(
//                                {counter + static_cast<size_t>(myRankId()) * data.size(), data[counter]}));
//             counter++; // We cannot put this in the above line, as we can't assume if the first argument of the pair
//                        // is bound before or after the increment.
//             return ret;
//         },
//         numBlocks);
//     assert(counter == data.size() + 1);
//
//     std::vector<std::pair<std::pair<ReStore::block_id_t, size_t>, int>> blockRanges;
//     blockRanges.reserve(asserting_cast<size_t>(numRanks()));
//     ReStore::block_id_t myStartBlock = std::numeric_limits<ReStore::block_id_t>::max();
//
//     // Use a random Range of numRanksDataLoss PEs whose data we redistribute
//     std::mt19937                                             rng(42);
//     std::uniform_int_distribution<std::mt19937::result_type> dist(
//         0, asserting_cast<unsigned long>(numRanks()) - numRanksDataLoss);
//
//     std::vector<BlockType> recvData(recvBlocksPerRank, BlockType(blockSize));
//     assert(recvData.size() == recvBlocksPerRank);
//     assert(std::all_of(
//         recvData.begin(), recvData.end(), [blockSize](const BlockType& block) { return block.size() == blockSize;
//         }));
//
//     // Measurement
//     for (auto _: state) {
//         UNUSED(_);
//
//         assert(myStartBlock == std::numeric_limits<ReStore::block_id_t>::max());
//
//         // Build the data structure specifying which block to transfer to which rank.
//         blockRanges.clear();
//         ReStore::block_id_t startBlockId = dist(rng) * blocksPerRank;
//         for (int rank: range(numRanks())) {
//             blockRanges.emplace_back(std::make_pair(startBlockId, recvBlocksPerRank), rank);
//             if (rank == myRankId()) {
//                 myStartBlock = startBlockId;
//             }
//             startBlockId += recvBlocksPerRank;
//             assert(startBlockId <= numBlocks);
//         }
//         assert(myStartBlock != std::numeric_limits<ReStore::block_id_t>::max());
//
//         // Ensure, that all ranks start into the times section at about the same time. This prevens faster ranks from
//         // having to wait for the slower ranks in the timed section. This ist also a workaround for a bug in the
//         // SparseAllToAll implementation which will sometimes allow messages spilling over into the next
//         SparseAllToAll
//         // round.
//         MPI_Barrier(MPI_COMM_WORLD);
//
//         auto start = std::chrono::high_resolution_clock::now();
//         store.pushBlocksCurrentRankIds(
//             blockRanges, [&recvData, myStartBlock](const std::byte* buffer, size_t size, ReStore::block_id_t blockId)
//             {
//                 assert(blockId >= myStartBlock);
//                 auto index = blockId - myStartBlock;
//                 assert(index < recvData.size());
//                 // assert(recvData[index].size() == 0);
//                 recvData[index].clear();
//                 recvData[index].insert(
//                     recvData[index].end(), reinterpret_cast<const ElementType*>(buffer),
//                     reinterpret_cast<const ElementType*>(buffer + size));
//             });
//         benchmark::DoNotOptimize(recvData.data());
//         benchmark::ClobberMemory();
//         assert(std::all_of(recvData.begin(), recvData.end(), [blockSize](const BlockType& block) {
//             return block.size() == blockSize;
//         }));
//         auto end            = std::chrono::high_resolution_clock::now();
//         auto elapsedSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
//         MPI_Allreduce(MPI_IN_PLACE, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//         state.SetIterationTime(elapsedSeconds);
//         myStartBlock = std::numeric_limits<ReStore::block_id_t>::max();
//     }
// }

static void BM_pullBlocksSingleRank(benchmark::State& state) {
    // Each rank submits different data. The replication level is set to 3.

    // Parse arguments
    auto bytesPerBlock             = throwing_cast<size_t>(state.range(0));
    auto replicationLevel          = throwing_cast<uint16_t>(state.range(1));
    auto bytesPerRank              = throwing_cast<size_t>(state.range(2));
    auto blocksPerPermutationRange = throwing_cast<size_t>(state.range(3));
    auto fractionOfRanksThatFail   = static_cast<double>(state.range(4)) / 1000.;
    UNUSED(fractionOfRanksThatFail);

    assert(bytesPerRank % bytesPerBlock == 0);
    auto blocksPerRank = bytesPerRank / bytesPerBlock;

    const auto numRankFailures   = 1;
    auto       recvBlocksPerRank = blocksPerRank * numRankFailures / asserting_cast<size_t>(numRanks());

    if (recvBlocksPerRank == 0) {
        state.SkipWithError("Parameters set such that no one receives anything!");
        return;
    }

    using ElementType = uint8_t;
    using BlockType   = std::vector<ElementType>;

    // Setup
    uint64_t rankId    = asserting_cast<uint64_t>(myRankId());
    size_t   numBlocks = static_cast<size_t>(numRanks()) * bytesPerRank / bytesPerBlock;

    std::vector<BlockType> data;
    for (uint64_t base: range(blocksPerRank * rankId, blocksPerRank * rankId + blocksPerRank)) {
        data.emplace_back();
        data.back().reserve(bytesPerBlock);
        for (uint64_t increment: range(0ul, bytesPerBlock)) {
            data.back().push_back(static_cast<ElementType>((base - increment) % std::numeric_limits<uint8_t>::max()));
        }
        assert(data.back().size() == bytesPerBlock);
    }
    assert(data.size() == blocksPerRank);

    ReStore::ReStore<BlockType> store(
        MPI_COMM_WORLD, replicationLevel, ReStore::OffsetMode::constant, sizeof(uint8_t) * bytesPerBlock,
        blocksPerPermutationRange);

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

    // Use a random Range of numRankFailures PEs whose data we redistribute
    std::mt19937                                             rng(42);
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, asserting_cast<unsigned long>(numRanks()) - numRankFailures);

    std::vector<BlockType> recvData(recvBlocksPerRank, BlockType(bytesPerBlock));
    // Measurement
    for (auto _: state) {
        UNUSED(_);

        assert(myStartBlock == std::numeric_limits<ReStore::block_id_t>::max());

        // Build the data structure specifying which block to transfer to which rank.
        blockRanges.clear();
        ReStore::block_id_t startBlockId = dist(rng) * blocksPerRank;
        myStartBlock = startBlockId + recvBlocksPerRank * asserting_cast<ReStore::block_id_t>(myRankId());
        blockRanges.emplace_back(myStartBlock, recvBlocksPerRank);
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
        assert(std::all_of(recvData.begin(), recvData.end(), [bytesPerBlock](const BlockType& block) {
            return block.size() == bytesPerBlock;
        }));
        auto end            = std::chrono::high_resolution_clock::now();
        auto elapsedSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        MPI_Allreduce(MPI_IN_PLACE, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(elapsedSeconds);
        myStartBlock = std::numeric_limits<ReStore::block_id_t>::max();
    }
}

static void BM_pullBlocksSmallRange(benchmark::State& state) {
    // Each rank submits different data. The replication level is set to 3.

    // Parse arguments
    auto bytesPerBlock             = throwing_cast<size_t>(state.range(0));
    auto replicationLevel          = throwing_cast<uint16_t>(state.range(1));
    auto bytesPerRank              = throwing_cast<size_t>(state.range(2));
    auto blocksPerPermutationRange = throwing_cast<size_t>(state.range(3));
    auto fractionOfRanksThatFail   = static_cast<double>(state.range(4)) / 1000.;

    assert(bytesPerRank % bytesPerBlock == 0);
    auto blocksPerRank = bytesPerRank / bytesPerBlock;

    const auto numRankFailures   = static_cast<uint64_t>(std::ceil(fractionOfRanksThatFail * numRanks()));
    auto       recvBlocksPerRank = blocksPerRank * numRankFailures / asserting_cast<size_t>(numRanks());

    if (recvBlocksPerRank == 0) {
        state.SkipWithError("Parameters set such that no one receives anything!");
        return;
    }

    using ElementType = uint8_t;
    using BlockType   = std::vector<ElementType>;

    // Setup
    uint64_t rankId    = asserting_cast<uint64_t>(myRankId());
    size_t   numBlocks = static_cast<size_t>(numRanks()) * bytesPerRank / bytesPerBlock;

    std::vector<BlockType> data;
    for (uint64_t base: range(blocksPerRank * rankId, blocksPerRank * rankId + blocksPerRank)) {
        data.emplace_back();
        data.back().reserve(bytesPerBlock);
        for (uint64_t increment: range(0ul, bytesPerBlock)) {
            data.back().push_back(static_cast<ElementType>((base - increment) % std::numeric_limits<uint8_t>::max()));
        }
        assert(data.back().size() == bytesPerBlock);
    }
    assert(data.size() == blocksPerRank);

    ReStore::ReStore<BlockType> store(
        MPI_COMM_WORLD, replicationLevel, ReStore::OffsetMode::constant, sizeof(uint8_t) * bytesPerBlock,
        blocksPerPermutationRange);

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

    // Use a random Range of numRankFailures PEs whose data we redistribute
    std::mt19937                                             rng(42);
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, asserting_cast<unsigned long>(numRanks()) - numRankFailures);

    std::vector<BlockType> recvData(recvBlocksPerRank, BlockType(bytesPerBlock));
    // Measurement
    for (auto _: state) {
        UNUSED(_);

        assert(myStartBlock == std::numeric_limits<ReStore::block_id_t>::max());

        // Build the data structure specifying which block to transfer to which rank.
        blockRanges.clear();
        ReStore::block_id_t startBlockId = dist(rng) * blocksPerRank;
        myStartBlock = startBlockId + recvBlocksPerRank * asserting_cast<ReStore::block_id_t>(myRankId());
        blockRanges.emplace_back(myStartBlock, recvBlocksPerRank);
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
        assert(std::all_of(recvData.begin(), recvData.end(), [bytesPerBlock](const BlockType& block) {
            return block.size() == bytesPerBlock;
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
    auto bytesPerBlock             = throwing_cast<size_t>(state.range(0));
    auto replicationLevel          = throwing_cast<uint16_t>(state.range(1));
    auto bytesPerRank              = throwing_cast<size_t>(state.range(2));
    auto blocksPerPermutationRange = throwing_cast<size_t>(state.range(3));
    auto fractionOfRanksThatFail   = static_cast<double>(state.range(4)) / 1000.;
    UNUSED(fractionOfRanksThatFail);

    assert(bytesPerRank % bytesPerBlock == 0);
    auto blocksPerRank = bytesPerRank / bytesPerBlock;

    using ElementType = uint8_t;
    using BlockType   = std::vector<ElementType>;

    // Setup
    uint64_t rankId    = asserting_cast<uint64_t>(myRankId());
    size_t   numBlocks = static_cast<size_t>(numRanks()) * bytesPerRank / bytesPerBlock;

    std::vector<BlockType> data;
    for (uint64_t base: range(blocksPerRank * rankId, blocksPerRank * rankId + blocksPerRank)) {
        data.emplace_back();
        data.back().reserve(bytesPerBlock);
        for (uint64_t increment: range(0ul, bytesPerBlock)) {
            data.back().push_back(static_cast<ElementType>((base - increment) % std::numeric_limits<uint8_t>::max()));
        }
        assert(data.back().size() == bytesPerBlock);
    }
    assert(data.size() == blocksPerRank);

    ReStore::ReStore<BlockType> store(
        MPI_COMM_WORLD, replicationLevel, ReStore::OffsetMode::constant, sizeof(uint8_t) * bytesPerBlock,
        blocksPerPermutationRange);

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
    int                 rankToLoadFrom = (myRankId() + 1) % numRanks();
    ReStore::block_id_t myStartBlock   = static_cast<size_t>(rankToLoadFrom) * blocksPerRank;
    blockRanges.emplace_back(std::make_pair(myStartBlock, blocksPerRank));

    std::vector<BlockType> recvData(blocksPerRank, BlockType(bytesPerBlock));
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
        assert(std::all_of(recvData.begin(), recvData.end(), [bytesPerBlock](const BlockType& block) {
            return block.size() == bytesPerBlock;
        }));
        auto end            = std::chrono::high_resolution_clock::now();
        auto elapsedSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        MPI_Allreduce(MPI_IN_PLACE, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(elapsedSeconds);
    }
}


static void BM_DiskRedistribute(benchmark::State& state) {
    // Each rank submits different data. The replication level is set to 3.

    // Parse arguments
    auto bytesPerBlock             = throwing_cast<size_t>(state.range(0));
    auto replicationLevel          = throwing_cast<uint16_t>(state.range(1));
    auto bytesPerRank              = throwing_cast<size_t>(state.range(2));
    auto blocksPerPermutationRange = throwing_cast<size_t>(state.range(3));
    auto fractionOfRanksThatFail   = static_cast<double>(state.range(4)) / 1000.;
    UNUSED(replicationLevel);
    UNUSED(blocksPerPermutationRange);
    UNUSED(fractionOfRanksThatFail);

    assert(bytesPerRank % bytesPerBlock == 0);
    auto blocksPerRank = bytesPerRank / bytesPerBlock;

    using ElementType = uint8_t;
    using BlockType   = std::vector<ElementType>;

    // Setup
    uint64_t rankId = asserting_cast<uint64_t>(myRankId());

    std::string filePrefix      = "checkpoint_redistribute_" + std::to_string(numRanks()) + "_";
    std::string fileNametoWrite = filePrefix + std::to_string(rankId);
    auto        readRank        = (rankId + 49) % asserting_cast<uint64_t>(numRanks());
    std::string fileNameToRead  = filePrefix + std::to_string(readRank);

    std::vector<BlockType> data;
    for (uint64_t base: range(blocksPerRank * rankId, blocksPerRank * rankId + blocksPerRank)) {
        data.emplace_back();
        data.back().reserve(bytesPerBlock);
        for (uint64_t increment: range(0ul, bytesPerBlock)) {
            data.back().push_back(static_cast<ElementType>((base - increment) % std::numeric_limits<uint8_t>::max()));
        }
        assert(data.back().size() == bytesPerBlock);
    }
    assert(data.size() == blocksPerRank);


    std::vector<BlockType> recvData(blocksPerRank, BlockType(bytesPerBlock));
    // Measurement
    for (auto _: state) {
        UNUSED(_);

        // make sure the File doesn't exist already from a previous run
        std::remove(fileNametoWrite.c_str());

        std::ofstream outFileStream(fileNametoWrite, std::ios::binary | std::ios::out | std::ios::app);

        auto writeBlock = blocksPerRank * rankId;
        UNUSED(writeBlock);

        for (const auto& block: data) {
            assert(block.size() * sizeof(ElementType) == bytesPerBlock);
            assert(block[0] == static_cast<ElementType>((writeBlock++) % std::numeric_limits<uint8_t>::max()));
            outFileStream.write((const char*)block.data(), asserting_cast<long>(block.size() * sizeof(ElementType)));
            assert(outFileStream.good());
        }
        outFileStream.close();

        auto readBlock = blocksPerRank * readRank;
        UNUSED(readBlock);
        // std::for_each(
        //     recvData.begin(), recvData.end(), [](auto& block) { std::fill(block.begin(), block.end(), 255); });
        // assert(std::all_of(recvData.begin(), recvData.end(), [](const BlockType& block) {
        //     return std::all_of(
        //         block.begin(), block.end(), [](const ElementType& blockElement) { return blockElement == 255; });
        // }));


        // Ensure, that all ranks start into the times section at about the same time. This prevens faster ranks from
        // having to wait for the slower ranks in the timed section. This ist also a workaround for a bug in the
        // SparseAllToAll implementation which will sometimes allow messages spilling over into the next SparseAllToAll
        // round.
        MPI_Barrier(MPI_COMM_WORLD);
        auto          start = std::chrono::high_resolution_clock::now();
        std::ifstream inFileStream(fileNameToRead, std::ios::binary | std::ios::in);
        for (auto& block: recvData) {
            assert(block.size() * sizeof(ElementType) == bytesPerBlock);
            assert(!inFileStream.eof());
            inFileStream.read((char*)block.data(), asserting_cast<long>(block.size() * sizeof(ElementType)));
            assert(block[0] == static_cast<ElementType>((readBlock++) % std::numeric_limits<uint8_t>::max()));
            assert(!inFileStream.fail());
            assert(inFileStream.good());
        }
        assert(inFileStream.peek() == std::char_traits<char>::eof());
        inFileStream.close();
        benchmark::DoNotOptimize(recvData.data());
        benchmark::ClobberMemory();
        assert(std::all_of(recvData.begin(), recvData.end(), [bytesPerBlock](const BlockType& block) {
            return block.size() == bytesPerBlock;
        }));
        auto end            = std::chrono::high_resolution_clock::now();
        auto elapsedSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        MPI_Allreduce(MPI_IN_PLACE, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(elapsedSeconds);
    }
    std::remove(fileNametoWrite.c_str());
}


static void BM_DiskSmallRange(benchmark::State& state) {
    // Each rank submits different data. The replication level is set to 3.

    // Parse arguments
    auto bytesPerBlock             = throwing_cast<size_t>(state.range(0));
    auto replicationLevel          = throwing_cast<uint16_t>(state.range(1));
    auto bytesPerRank              = throwing_cast<size_t>(state.range(2));
    auto blocksPerPermutationRange = throwing_cast<size_t>(state.range(3));
    auto fractionOfRanksThatFail   = static_cast<double>(state.range(4)) / 1000.;

    assert(bytesPerRank % bytesPerBlock == 0);
    auto blocksPerRank = bytesPerRank / bytesPerBlock;

    const auto numRankFailures   = static_cast<uint64_t>(std::ceil(fractionOfRanksThatFail * numRanks()));
    auto       recvBlocksPerRank = blocksPerRank * numRankFailures / asserting_cast<size_t>(numRanks());

    if (recvBlocksPerRank == 0) {
        state.SkipWithError("Parameters set such that no one receives anything!");
        return;
    }

    using ElementType = uint8_t;
    using BlockType   = std::vector<ElementType>;

    // Setup
    uint64_t rankId    = asserting_cast<uint64_t>(myRankId());
    size_t   numBlocks = static_cast<size_t>(numRanks()) * bytesPerRank / bytesPerBlock;

    std::vector<BlockType> data;
    for (uint64_t base: range(blocksPerRank * rankId, blocksPerRank * rankId + blocksPerRank)) {
        data.emplace_back();
        data.back().reserve(bytesPerBlock);
        for (uint64_t increment: range(0ul, bytesPerBlock)) {
            data.back().push_back(static_cast<ElementType>((base - increment) % std::numeric_limits<uint8_t>::max()));
        }
        assert(data.back().size() == bytesPerBlock);
    }
    assert(data.size() == blocksPerRank);

    ReStore::ReStore<BlockType> store(
        MPI_COMM_WORLD, replicationLevel, ReStore::OffsetMode::constant, sizeof(uint8_t) * bytesPerBlock,
        blocksPerPermutationRange);

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
    ReStore::block_id_t writeStartBlock = std::numeric_limits<ReStore::block_id_t>::max();

    // Use a random Range of numRankFailures PEs whose data we redistribute
    std::mt19937                                             rng(42);
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, asserting_cast<unsigned long>(numRanks()) - numRankFailures);

    std::vector<BlockType> recvData(recvBlocksPerRank, BlockType(bytesPerBlock));

    std::string                filePrefix      = "checkpoint_smallRange_" + std::to_string(numRanks()) + "_";
    ReStoreMPI::current_rank_t rankToWriteFor  = (myRankId() + 49) % numRanks();
    std::string                fileNameToWrite = filePrefix + std::to_string(rankToWriteFor);
    std::string                fileNameToRead  = filePrefix + std::to_string(myRankId());

    // make sure the File doesn't exist already from a previous run
    std::remove(fileNameToWrite.c_str());
    // Measurement
    for (auto _: state) {
        UNUSED(_);

        assert(writeStartBlock == std::numeric_limits<ReStore::block_id_t>::max());

        // Build the data structure specifying which block to transfer to which rank.
        blockRanges.clear();
        ReStore::block_id_t startBlockId = dist(rng) * blocksPerRank;
        writeStartBlock   = startBlockId + recvBlocksPerRank * asserting_cast<ReStore::block_id_t>(rankToWriteFor);
        auto myStartBlock = startBlockId + recvBlocksPerRank * asserting_cast<ReStore::block_id_t>(myRankId());
        UNUSED(myStartBlock);
        blockRanges.emplace_back(writeStartBlock, recvBlocksPerRank);
        assert(writeStartBlock != std::numeric_limits<ReStore::block_id_t>::max());

        store.pullBlocks(
            blockRanges,
            [&recvData, writeStartBlock](const std::byte* buffer, size_t size, ReStore::block_id_t blockId) {
                assert(blockId >= writeStartBlock);
                auto index = blockId - writeStartBlock;
                assert(index < recvData.size());
                // assert(recvData[index].size() == 0);
                recvData[index].clear();
                recvData[index].insert(
                    recvData[index].end(), reinterpret_cast<const ElementType*>(buffer),
                    reinterpret_cast<const ElementType*>(buffer + size));
            });

        std::ofstream outFileStream(fileNameToWrite, std::ios::binary | std::ios::out | std::ios::app);

        for (const auto& block: recvData) {
            assert(block.size() * sizeof(ElementType) == bytesPerBlock);
            outFileStream.write((const char*)block.data(), asserting_cast<long>(block.size() * sizeof(ElementType)));
            assert(block[0] == static_cast<ElementType>((writeStartBlock++) % std::numeric_limits<uint8_t>::max()));
            assert(outFileStream.good());
        }
        outFileStream.close();


        // std::for_each(
        //     recvData.begin(), recvData.end(), [](auto& block) { std::fill(block.begin(), block.end(), 255); });
        // assert(std::all_of(recvData.begin(), recvData.end(), [](const BlockType& block) {
        //     return std::all_of(
        //         block.begin(), block.end(), [](const ElementType& blockElement) { return blockElement == 255; });
        // }));

        // Ensure, that all ranks start into the times section at about the same time. This prevens faster ranks from
        // having to wait for the slower ranks in the timed section. This ist also a workaround for a bug in the
        // SparseAllToAll implementation which will sometimes allow messages spilling over into the next SparseAllToAll
        // round.
        MPI_Barrier(MPI_COMM_WORLD);
        auto          start = std::chrono::high_resolution_clock::now();
        std::ifstream inFileStream(fileNameToRead, std::ios::binary | std::ios::in);
        for (auto& block: recvData) {
            assert(block.size() * sizeof(ElementType) == bytesPerBlock);
            inFileStream.read((char*)block.data(), asserting_cast<long>(block.size() * sizeof(ElementType)));
            assert(block[0] == static_cast<ElementType>((myStartBlock++) % std::numeric_limits<uint8_t>::max()));
            assert(inFileStream.good());
        }
        assert(inFileStream.peek() == std::char_traits<char>::eof());
        inFileStream.close();
        benchmark::DoNotOptimize(recvData.data());
        benchmark::ClobberMemory();
        assert(std::all_of(recvData.begin(), recvData.end(), [bytesPerBlock](const BlockType& block) {
            return block.size() == bytesPerBlock;
        }));
        auto end            = std::chrono::high_resolution_clock::now();
        auto elapsedSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        MPI_Allreduce(MPI_IN_PLACE, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(elapsedSeconds);
        writeStartBlock = std::numeric_limits<ReStore::block_id_t>::max();
        std::remove(fileNameToWrite.c_str());
    }
}

static void BM_DiskSingleRank(benchmark::State& state) {
    // Each rank submits different data. The replication level is set to 3.

    // Parse arguments
    auto bytesPerBlock             = throwing_cast<size_t>(state.range(0));
    auto replicationLevel          = throwing_cast<uint16_t>(state.range(1));
    auto bytesPerRank              = throwing_cast<size_t>(state.range(2));
    auto blocksPerPermutationRange = throwing_cast<size_t>(state.range(3));
    auto fractionOfRanksThatFail   = static_cast<double>(state.range(4)) / 1000.;
    UNUSED(fractionOfRanksThatFail);

    assert(bytesPerRank % bytesPerBlock == 0);
    auto blocksPerRank = bytesPerRank / bytesPerBlock;

    const auto numRankFailures   = 1;
    auto       recvBlocksPerRank = blocksPerRank * numRankFailures / asserting_cast<size_t>(numRanks());

    if (recvBlocksPerRank == 0) {
        state.SkipWithError("Parameters set such that no one receives anything!");
        return;
    }

    using ElementType = uint8_t;
    using BlockType   = std::vector<ElementType>;

    // Setup
    uint64_t rankId    = asserting_cast<uint64_t>(myRankId());
    size_t   numBlocks = static_cast<size_t>(numRanks()) * bytesPerRank / bytesPerBlock;

    std::vector<BlockType> data;
    for (uint64_t base: range(blocksPerRank * rankId, blocksPerRank * rankId + blocksPerRank)) {
        data.emplace_back();
        data.back().reserve(bytesPerBlock);
        for (uint64_t increment: range(0ul, bytesPerBlock)) {
            data.back().push_back(static_cast<ElementType>((base - increment) % std::numeric_limits<uint8_t>::max()));
        }
        assert(data.back().size() == bytesPerBlock);
    }
    assert(data.size() == blocksPerRank);

    ReStore::ReStore<BlockType> store(
        MPI_COMM_WORLD, replicationLevel, ReStore::OffsetMode::constant, sizeof(uint8_t) * bytesPerBlock,
        blocksPerPermutationRange);

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
    ReStore::block_id_t writeStartBlock = std::numeric_limits<ReStore::block_id_t>::max();

    // Use a random Range of numRankFailures PEs whose data we redistribute
    std::mt19937                                             rng(42);
    std::uniform_int_distribution<std::mt19937::result_type> dist(
        0, asserting_cast<unsigned long>(numRanks()) - numRankFailures);

    std::vector<BlockType> recvData(recvBlocksPerRank, BlockType(bytesPerBlock));

    std::string                filePrefix      = "checkpoint_smallRange_" + std::to_string(numRanks()) + "_";
    ReStoreMPI::current_rank_t rankToWriteFor  = (myRankId() + 49) % numRanks();
    std::string                fileNameToWrite = filePrefix + std::to_string(rankToWriteFor);
    std::string                fileNameToRead  = filePrefix + std::to_string(myRankId());

    // make sure the File doesn't exist already from a previous run
    std::remove(fileNameToWrite.c_str());
    // Measurement
    for (auto _: state) {
        UNUSED(_);

        assert(writeStartBlock == std::numeric_limits<ReStore::block_id_t>::max());

        // Build the data structure specifying which block to transfer to which rank.
        blockRanges.clear();
        ReStore::block_id_t startBlockId = dist(rng) * blocksPerRank;
        writeStartBlock   = startBlockId + recvBlocksPerRank * asserting_cast<ReStore::block_id_t>(rankToWriteFor);
        auto myStartBlock = startBlockId + recvBlocksPerRank * asserting_cast<ReStore::block_id_t>(myRankId());
        UNUSED(myStartBlock);
        blockRanges.emplace_back(writeStartBlock, recvBlocksPerRank);
        assert(writeStartBlock != std::numeric_limits<ReStore::block_id_t>::max());

        store.pullBlocks(
            blockRanges,
            [&recvData, writeStartBlock](const std::byte* buffer, size_t size, ReStore::block_id_t blockId) {
                assert(blockId >= writeStartBlock);
                auto index = blockId - writeStartBlock;
                assert(index < recvData.size());
                // assert(recvData[index].size() == 0);
                recvData[index].clear();
                recvData[index].insert(
                    recvData[index].end(), reinterpret_cast<const ElementType*>(buffer),
                    reinterpret_cast<const ElementType*>(buffer + size));
            });

        std::ofstream outFileStream(fileNameToWrite, std::ios::binary | std::ios::out | std::ios::app);

        for (const auto& block: recvData) {
            assert(block.size() * sizeof(ElementType) == bytesPerBlock);
            outFileStream.write((const char*)block.data(), asserting_cast<long>(block.size() * sizeof(ElementType)));
            assert(block[0] == static_cast<ElementType>((writeStartBlock++) % std::numeric_limits<uint8_t>::max()));
            assert(outFileStream.good());
        }
        outFileStream.close();


        // std::for_each(
        //     recvData.begin(), recvData.end(), [](auto& block) { std::fill(block.begin(), block.end(), 255); });
        // assert(std::all_of(recvData.begin(), recvData.end(), [](const BlockType& block) {
        //     return std::all_of(
        //         block.begin(), block.end(), [](const ElementType& blockElement) { return blockElement == 255; });
        // }));

        // Ensure, that all ranks start into the times section at about the same time. This prevens faster ranks from
        // having to wait for the slower ranks in the timed section. This ist also a workaround for a bug in the
        // SparseAllToAll implementation which will sometimes allow messages spilling over into the next SparseAllToAll
        // round.
        MPI_Barrier(MPI_COMM_WORLD);
        auto          start = std::chrono::high_resolution_clock::now();
        std::ifstream inFileStream(fileNameToRead, std::ios::binary | std::ios::in);
        for (auto& block: recvData) {
            assert(block.size() * sizeof(ElementType) == bytesPerBlock);
            inFileStream.read((char*)block.data(), asserting_cast<long>(block.size() * sizeof(ElementType)));
            assert(block[0] == static_cast<ElementType>((myStartBlock++) % std::numeric_limits<uint8_t>::max()));
            assert(inFileStream.good());
        }
        assert(inFileStream.peek() == std::char_traits<char>::eof());
        inFileStream.close();
        benchmark::DoNotOptimize(recvData.data());
        benchmark::ClobberMemory();
        assert(std::all_of(recvData.begin(), recvData.end(), [bytesPerBlock](const BlockType& block) {
            return block.size() == bytesPerBlock;
        }));
        auto end            = std::chrono::high_resolution_clock::now();
        auto elapsedSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        MPI_Allreduce(MPI_IN_PLACE, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        state.SetIterationTime(elapsedSeconds);
        writeStartBlock = std::numeric_limits<ReStore::block_id_t>::max();
        std::remove(fileNameToWrite.c_str());
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

template <
    bool sweepBlocksPerPermutationRange, bool sweepReplicationLevel, bool sweepDataPerRank, bool sweepFailureRateOfPEs>
static void benchmarkArguments(benchmark::internal::Benchmark* benchmark) {
    const int64_t bytesPerBlock             = 64;
    const int64_t replicationLevel          = 4;
    const int64_t bytesPerRank              = MiB(16);
    const int64_t promilleOfRankFailures    = 10;
    const int64_t blocksPerPermutationRange = 4096;

    if (sweepBlocksPerPermutationRange) {
        std::vector<int64_t> blocksPerPermutationRange_values = {
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};

        for (auto b: blocksPerPermutationRange_values) {
            benchmark->Args({bytesPerBlock, replicationLevel, bytesPerRank, b, promilleOfRankFailures});
        }
    }

    // Replication level
    if (sweepReplicationLevel) {
        for (int64_t k: {1, 2, 3, 4, 5, 6}) {
            benchmark->Args({bytesPerBlock, k, bytesPerRank, blocksPerPermutationRange, promilleOfRankFailures});
        }
    }

    // amount of data per rank
    if (sweepDataPerRank) {
        for (int64_t n: {KiB(16), KiB(64), KiB(256), MiB(1), MiB(4), MiB(16), MiB(64)}) {
            benchmark->Args({bytesPerBlock, replicationLevel, n, blocksPerPermutationRange, promilleOfRankFailures});
        }
    }

    // failure rate of PEs
    if (sweepFailureRateOfPEs) {
        for (int64_t f: {5, 10, 20, 30, 40, 50}) {
            benchmark->Args({bytesPerBlock, replicationLevel, bytesPerRank, blocksPerPermutationRange, f});
        }
    }

    // If nothing else is requested, run the default values only.
    if (!sweepBlocksPerPermutationRange && !sweepReplicationLevel && !sweepDataPerRank && !sweepFailureRateOfPEs) {
        benchmark->Args(
            {bytesPerBlock, replicationLevel, bytesPerRank, blocksPerPermutationRange, promilleOfRankFailures});
    }
}

BENCHMARK(BM_submitBlocks)          ///
    ->UseManualTime()               ///
    ->Unit(benchmark::kMillisecond) ///
    ->Apply(benchmarkArguments<false, true, true, true>);

BENCHMARK(BM_pullBlocksRedistribute) ///
    ->UseManualTime()                ///
    ->Unit(benchmark::kMillisecond)  ///
    ->Apply(benchmarkArguments<false, true, true, true>);

BENCHMARK(BM_pullBlocksSmallRange)  ///
    ->UseManualTime()               ///
    ->Unit(benchmark::kMillisecond) ///
    ->Apply(benchmarkArguments<false, true, true, true>);

BENCHMARK(BM_pullBlocksSingleRank)  ///
    ->UseManualTime()               ///
    ->Unit(benchmark::kMillisecond) ///
    ->Apply(benchmarkArguments<false, false, false, false>);

BENCHMARK(BM_DiskRedistribute)      ///
    ->UseManualTime()               ///
    ->Unit(benchmark::kMillisecond) ///
    ->Apply(benchmarkArguments<false, false, true, true>);

BENCHMARK(BM_DiskSmallRange)        ///
    ->UseManualTime()               ///
    ->Unit(benchmark::kMillisecond) ///
    ->Apply(benchmarkArguments<false, false, true, true>);

BENCHMARK(BM_DiskSingleRank)        ///
    ->UseManualTime()               ///
    ->Unit(benchmark::kMillisecond) ///
    ->Apply(benchmarkArguments<false, false, false, false>);

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

    // Warm up the network by sending a message from each rank to each rank.
    std::vector<MPI_Request> sendRequests(asserting_cast<size_t>(numRanks()));
    std::vector<MPI_Status>  recvStatuses(asserting_cast<size_t>(numRanks()));
    int                      buf = 42;
    for (int destRank = 0; destRank < numRanks(); ++destRank) {
        MPI_Isend(&buf, 1, MPI_INT, destRank, 0, MPI_COMM_WORLD, &sendRequests[asserting_cast<size_t>(destRank)]);
    }
    for (int srcRank = 0; srcRank < numRanks(); ++srcRank) {
        MPI_Recv(&buf, 1, MPI_INT, srcRank, 0, MPI_COMM_WORLD, &recvStatuses[asserting_cast<size_t>(srcRank)]);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        ::benchmark::Initialize(&argc, argv);

        // Root process will use a reporter from the usual set provided by ::benchmark
        ::benchmark::RunSpecifiedBenchmarks();
    } else {
        // Reporting from other processes is disabled by passing a custom reporter.
        // We have to disable the display AND file reporter.
        NullReporter null;

        // googlebenchmark will check if the benchmark_out parameter is set even when we prove a NullReporter. It
        // does this using the google flags libary. We can therefore specify the benchmark_out parameter on the
        // command line or using an environment variable.
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
