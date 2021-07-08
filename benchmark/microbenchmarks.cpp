#include <algorithm>
#include <cassert>
#include <chrono>
#include <cppitertools/range.hpp>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <mpi.h>
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
    size_t   blockSize        = throwing_cast<size_t>(state.range(0));
    uint16_t replicationLevel = throwing_cast<uint16_t>(state.range(1));
    size_t   bytesPerRank     = throwing_cast<size_t>(state.range(2));

    assert(bytesPerRank % blockSize == 0);
    size_t blocksPerRank = bytesPerRank / blockSize;

    using ElementType = uint8_t;
    using BlockType   = std::vector<ElementType>;

    // Measurement
    for (auto _: state) {
        // Setup
        uint64_t rankId    = asserting_cast<uint64_t>(myRankId());
        size_t   numBlocks = static_cast<size_t>(numRanks()) * bytesPerRank / blockSize;

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

template <typename N>
auto constexpr KiB(N n) {
    return n * 1024;
}

template <typename N>
auto constexpr MiB(N n) {
    return n * 1024 * KiB(1);
}

BENCHMARK(BM_submitBlocks)          ///
    ->UseManualTime()               ///
    ->Unit(benchmark::kMillisecond) ///
    ->ArgsProduct({
        {8, 16, 32, 64, 128, 256, 512, KiB(1), MiB(1)}, // block sizes
        {1, 2, 3, 4},                                   // replication level
        {MiB(1), MiB(16), MiB(32), MiB(64)}             //, MiB(128)} // bytes per rank
    });

static void BM_pushBlocks(benchmark::State& state) {
    // Each rank submits different data. The replication level is set to 3.

    // Parse arguments
    size_t   blockSize        = throwing_cast<size_t>(state.range(0));
    uint16_t replicationLevel = throwing_cast<uint16_t>(state.range(1));
    size_t   bytesPerRank     = throwing_cast<size_t>(state.range(2));

    assert(bytesPerRank % blockSize == 0);
    size_t blocksPerRank = bytesPerRank / blockSize;

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

    // Measurement
    for (auto _: state) {
        std::vector<BlockType> recvData(blocksPerRank);
        auto                   start = std::chrono::high_resolution_clock::now();
        store.pushBlocksCurrentRankIds(
            blockRanges, [&recvData, myStartBlock](const std::byte* buffer, size_t size, ReStore::block_id_t blockId) {
                assert(blockId >= myStartBlock);
                auto index = blockId - myStartBlock;
                assert(index < recvData.size());
                assert(recvData[index].size() == 0);
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

BENCHMARK(BM_pushBlocks)            ///
    ->UseManualTime()               ///
    ->Unit(benchmark::kMillisecond) ///
    ->ArgsProduct({
        {8, KiB(1), MiB(1)},                // block sizes
        {2, 3, 4},                          // replication level
        {MiB(1), MiB(16), MiB(32), MiB(64)} //, MiB(128)} // bytes per rank
    });
const auto MAX_REPLICATION_LEVEL = 4;

// This reporter does nothing.
// We can use it to disable output from all but the root process
class NullReporter : public ::benchmark::BenchmarkReporter {
    public:
    NullReporter() {}
    virtual bool ReportContext(const Context&) override {
        return true;
    }
    virtual void ReportRuns(const std::vector<Run>&) override {}
    virtual void Finalize() override {}
};

// The main is rewritten to allow for MPI initializing and for selecting a reporter according to the process rank.
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Do we have enough MPI ranks?
    if (numRanks() < MAX_REPLICATION_LEVEL) {
        std::cout << "Please call this benchmark with at least " << MAX_REPLICATION_LEVEL << " ranks." << std::endl;
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

        std::string benchmark_out_string = std::string{"--benchmark_out="} + tmpFile.c_str();
        char*       benchmark_out = reinterpret_cast<char*>(malloc(sizeof(char) * benchmark_out_string.length()));
        strcpy(benchmark_out, benchmark_out_string.c_str());

        expanded_argv.push_back(benchmark_out);
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
