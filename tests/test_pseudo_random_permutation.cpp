#include <unordered_set>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "restore/helpers.hpp"
#include "restore/pseudo_random_permutation.hpp"

using namespace ::testing;

// TEST(PseudoRandomPermutationTest, LCG) {
//    uint64_t n = 10;
//
//    LCGPseudoRandomPermutation permutation(asserting_cast<int64_t>(n - 1));
//    std::vector<uint64_t> sequence;
//
//    // Build test vector
//    sequence.reserve(n);
//    for (uint64_t i = 0; i < n; ++i) {
//        sequence.push_back(i);
//    }
//
//    // Apply permutation to every element in the test vector.
//    for (size_t idx = 0; idx < n; ++idx) {
//        // Test that the permutation is invertible
//        ASSERT_EQ(sequence[idx], permutation.finv(permutation.f(sequence[idx])));
//
//        // Compute the permutation, to later check that each element only appears once.
//        sequence[idx] = permutation.f(sequence[idx]);
//    }
//
//    // Check that no element appears more than once. Also check, that we did not exceed the given range.
//    EXPECT_THAT(sequence, UnorderedElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
//}

TEST(PseudoRandomPermutationTest, FeistelEvenBitCount) {
    const uint64_t n          = 10;
    const uint8_t  num_rounds = 4;

    std::random_device        rd;
    std::vector<XXH64_hash_t> keys;
    keys.reserve(num_rounds);
    for (uint8_t round = 0; round < num_rounds; round++) {
        keys.push_back(rd());
    }

    FeistelPseudoRandomPermutation permutation(asserting_cast<int64_t>(n - 1), keys);
    std::vector<uint64_t>          sequence;

    // Build test vector
    sequence.reserve(n);
    for (uint64_t i = 0; i < n; ++i) {
        sequence.push_back(i);
    }

    // Apply permutation to every element in the test vector.
    for (size_t idx = 0; idx < n; ++idx) {
        // Test that the permutation is invertible
        ASSERT_EQ(sequence[idx], permutation.finv(permutation.f(sequence[idx])));

        // Compute the permutation, to later check that each element only appears once.
        sequence[idx] = permutation.f(sequence[idx]);
    }

    // Check that no element appears more than once. Also check, that we did not exceed the given range.
    EXPECT_THAT(sequence, UnorderedElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
}

TEST(PseudoRandomPermutationTest, FeistelLargeNumbers) {
    const uint64_t MAX_VALUE  = 100000;
    const uint8_t  NUM_ROUNDS = 4;

    std::random_device        rd;
    std::vector<XXH64_hash_t> keys;
    keys.reserve(NUM_ROUNDS);
    for (uint8_t round = 0; round < NUM_ROUNDS; round++) {
        keys.push_back(rd());
    }
    FeistelPseudoRandomPermutation permutation(MAX_VALUE, keys);

    std::unordered_set<uint64_t> seen;
    for (uint64_t i = 0; i <= MAX_VALUE; i++) {
        auto permuted = permutation.f(i);
        EXPECT_EQ(permutation.finv(permuted), i);
        EXPECT_EQ(seen.count(permuted), 0);
        seen.insert(permuted);
    }
    ASSERT_EQ(seen.size(), MAX_VALUE + 1);
}

TEST(PseudoRandomPermutationTest, FeistelUnevenBitCount) {
    const uint64_t n          = 24;
    const uint8_t  num_rounds = 4;

    std::random_device        rd;
    std::vector<XXH64_hash_t> keys;
    keys.reserve(num_rounds);
    for (uint8_t round = 0; round < num_rounds; round++) {
        keys.push_back(rd());
    }

    FeistelPseudoRandomPermutation permutation(asserting_cast<int64_t>(n - 1), keys);
    std::vector<uint64_t>          sequence;

    // Build test vector
    sequence.reserve(n);
    for (uint64_t i = 0; i < n; ++i) {
        sequence.push_back(i);
    }

    // Apply permutation to every element in the test vector.
    for (size_t idx = 0; idx < n; ++idx) {
        // Test that the permutation is invertible
        ASSERT_EQ(sequence[idx], permutation.finv(permutation.f(sequence[idx])));

        // Compute the permutation, to later check that each element only appears once.
        sequence[idx] = permutation.f(sequence[idx]);
    }

    // Check that no element appears more than once. Also check, that we did not exceed the given range.
    EXPECT_THAT(
        sequence,
        UnorderedElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23));
}