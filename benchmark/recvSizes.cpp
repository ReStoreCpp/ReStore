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
#include <restore/helpers.hpp>

using iter::range;

static void BM_recvSizes(benchmark::State& state) {
    // Parse arguments
    auto numSendingPEs = throwing_cast<size_t>(state.range(0));
    auto totalSize     = throwing_cast<size_t>(state.range(1));

    assert(totalSize % numSendingPEs == 0);
    size_t dataPerPE = totalSize / numSendingPEs;

    using ElementType = uint8_t;

    // Setup
    auto rankId = asserting_cast<uint64_t>(myRankId());


    std::vector<ElementType> data;
    if (rankId == 0) {
        data.resize(totalSize);
    } else {
        data.resize(dataPerPE);
    }

    // Measurement
    for (auto _: state) {
        UNUSED(_);

        // Start measurement
        auto start = std::chrono::high_resolution_clock::now();

        if (rankId == 0) {
            std::vector<MPI_Request> requests(numSendingPEs);
            for (size_t i = 0; i < numSendingPEs; ++i) {
                int sendingPE = asserting_cast<int>(i) + 1;
                MPI_Irecv(
                    data.data() + asserting_cast<int>(i * dataPerPE), asserting_cast<int>(dataPerPE), MPI_UINT8_T,
                    sendingPE, 42, MPI_COMM_WORLD, &requests[i]);
            }
            MPI_Waitall(asserting_cast<int>(numSendingPEs), requests.data(), MPI_STATUSES_IGNORE);
        } else if (rankId <= numSendingPEs) {
            MPI_Ssend(data.data(), asserting_cast<int>(dataPerPE), MPI_UINT8_T, 0, 42, MPI_COMM_WORLD);
        } else {
            // do nothing
        }

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

BENCHMARK(BM_recvSizes)             ///
    ->UseManualTime()               ///
    ->Unit(benchmark::kMillisecond) ///
    ->ArgsProduct({
        {1, 2, 4, 8, 16, 64, 256, 1024, 2048},                           // numPEs
        {KiB(128), KiB(256), KiB(518), MiB(1), MiB(4), MiB(16), MiB(64)} // Total size to receive
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

// From https://stackoverflow.com/questions/3477525/is-it-possible-to-use-a-c-smart-pointers-together-with-cs-malloc
struct free_delete {
    void operator()(void* x) {
        free(x);
    }
};


// The main is rewritten to allow for MPI initializing and for selecting a reporter according to the process rank.
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    assert(rank >= 0);

    // Do we have enough MPI ranks?
    if (numRanks() < 2049) {
        std::cout << "Please call this benchmark with at least " << 2049 << " ranks." << std::endl;
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
        auto        benchmark_out        = std::unique_ptr<char, free_delete>(
            reinterpret_cast<char*>(malloc(sizeof(char) * benchmark_out_string.length())));
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
