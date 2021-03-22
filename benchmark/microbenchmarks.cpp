#if defined(__GNUC__) && !defined (__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#include <benchmark/benchmark.h>
#pragma GCC diagnostic pop
#else
#include <benchmark/benchmark.h>
#endif

#include <cassert>
#include <chrono>
#include <cppitertools/range.hpp>
#include <mpi.h>

#include <../tests/mpi_helpers.hpp>
#include <restore/core.hpp>
#include <restore/helpers.hpp>

using iter::range;

static void BM_submitBlocks(benchmark::State& state) {
    // Each rank submits different data. The replication level is set to 3.

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

        unsigned counter = 0;

        auto start = std::chrono::high_resolution_clock::now();
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
        auto end            = std::chrono::high_resolution_clock::now();
        auto elapsedSeconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
        MPI_Allreduce(&elapsedSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
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
        {8, KiB(1), MiB(1)},                // block sizes
        {2, 3, 4},                          // replication level
        {MiB(1), MiB(16), MiB(32), MiB(64)} //, MiB(128)} // bytes per rank
    });

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

    ::benchmark::Initialize(&argc, argv);

    if (rank == 0)
        // root process will use a reporter from the usual set provided by ::benchmark
        ::benchmark::RunSpecifiedBenchmarks();
    else {
        // reporting from other processes is disabled by passing a custom reporter
        NullReporter null;
        ::benchmark::RunSpecifiedBenchmarks(&null);
    }

    MPI_Finalize();
    return 0;
}