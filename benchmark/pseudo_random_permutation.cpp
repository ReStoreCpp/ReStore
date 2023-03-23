
#if defined(__GNUC__) && !defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wsuggest-override"
    #include <benchmark/benchmark.h>
    #pragma GCC diagnostic pop
#else
    #include <benchmark/benchmark.h>
#endif

#include "restore/helpers.hpp"
#include "restore/pseudo_random_permutation.hpp"

#include <xxhash.h>

static void BM_Feistel(benchmark::State& state) {
    // Setup
    const uint64_t MAX_VALUE  = 100000;
    const uint8_t  NUM_ROUNDS = 4;

    std::random_device        rd;
    std::vector<XXH64_hash_t> keys;
    keys.reserve(NUM_ROUNDS);
    for (uint8_t round = 0; round < NUM_ROUNDS; round++) {
        keys.push_back(rd());
    }
    FeistelPseudoRandomPermutation permutation(MAX_VALUE, keys);

    // Measurement
    for (auto _: state) {
        UNUSED(_);

        for (uint64_t i = 0; i <= MAX_VALUE; i++) {
            benchmark::DoNotOptimize(permutation.f(i));
        }
    }
}
BENCHMARK(BM_Feistel);

static void BM_LCG(benchmark::State& state) {
    // Setup
    const uint64_t MAX_VALUE = 100000;

    LCGPseudoRandomPermutation permutation(MAX_VALUE);

    // Measurement
    for (auto _: state) {
        UNUSED(_);

        for (uint64_t i = 0; i <= MAX_VALUE; i++) {
            benchmark::DoNotOptimize(permutation.f(i));
        }
    }
}
BENCHMARK(BM_LCG);

BENCHMARK_MAIN();
