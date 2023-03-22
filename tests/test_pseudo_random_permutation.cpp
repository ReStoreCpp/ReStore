#include <cstdint>
#include <unordered_set>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "restore/helpers.hpp"
#include "restore/pseudo_random_permutation.hpp"

using namespace ::testing;

TEST(PseudoRandomPermutationTest, LCG) {
    uint64_t n = 34;

    LCGPseudoRandomPermutation permutation(asserting_cast<int64_t>(n - 1));
    std::vector<int64_t>       sequence;

    // Build test vector
    sequence.reserve(n);
    for (int64_t i = 0; i < asserting_cast<int64_t>(n); ++i) {
        sequence.push_back(i);
    }

    auto sequence_copy(sequence);

    // Apply permutation to every element in the test vector.
    for (size_t idx = 0; idx < n; ++idx) {
        // Test that the permutation is invertible
        ASSERT_EQ(sequence[idx], permutation.finv(permutation.f(sequence[idx])));

        // Compute the permutation, to later check that each element only appears once.
        sequence[idx] = permutation.f(sequence[idx]);
    }

    // Check that no element appears more than once. Also check, that we did not exceed the given range.
    EXPECT_THAT(sequence, UnorderedElementsAreArray(sequence_copy));
}

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

TEST(PseudoRandomPermutationTest, Range201326592) {
    // microbencharks reported a block id of 201326592 when using exactly 201326592 blocks.

    const uint64_t maxValue       = 201326592 - 1;
    const uint64_t seed           = 0;
    const uint64_t lengthOfRanges = 1000;

    auto permutation = RangePermutation<FeistelPseudoRandomPermutation>(maxValue, lengthOfRanges, seed);

    ASSERT_EQ(permutation.numRanges(), maxValue / lengthOfRanges + 1);
    ASSERT_EQ(permutation.lastIdOfRange(maxValue), maxValue);

    // Apply permutation to every element in the test vector.
    // for (size_t n = 201300000; n < maxValue; ++n) {
    for (size_t n = maxValue - lengthOfRanges * 3; n <= maxValue; ++n) {
        auto permutedN = permutation.f(n);
        auto invertedN = permutation.finv(permutedN);

        // Test that the permutation is invertible
        ASSERT_EQ(n, permutation.finv(permutedN));
        ASSERT_EQ(n, invertedN);

        // And that we do not exceed the given range.
        ASSERT_GE(permutedN, 0);
        ASSERT_LE(permutedN, maxValue);
        ASSERT_GE(invertedN, 0);
        ASSERT_LE(invertedN, maxValue);
    }
}

TEST(PseudoRandomPermutationTest, IdentityPermutation) {
    const uint64_t n = 10;

    IdentityPermutation   permutation;
    std::vector<uint64_t> sequence;

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

    // Check that no element appears more than once. Also check, that we did not exceed the given range and that we
    // really applied the identity permutation.
    EXPECT_THAT(sequence, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
}

TEST(PseudoRandomPermutationTest, RangePermutation) {
    { // Identity permutation
        const uint64_t numElements    = 10;
        const uint64_t lengthOfRanges = 5;
        const uint64_t seed           = 34348128;

        RangePermutation<IdentityPermutation> permutation(numElements - 1, lengthOfRanges, seed);
        ASSERT_EQ(permutation.numRanges(), 2);
        ASSERT_EQ(permutation.maxValue(), numElements - 1);
        ASSERT_EQ(permutation.lengthOfRanges(), lengthOfRanges);

        EXPECT_EQ(permutation.lastIdOfRange(0), 4);
        EXPECT_EQ(permutation.lastIdOfRange(1), 4);
        EXPECT_EQ(permutation.lastIdOfRange(2), 4);
        EXPECT_EQ(permutation.lastIdOfRange(3), 4);
        EXPECT_EQ(permutation.lastIdOfRange(4), 4);
        EXPECT_EQ(permutation.lastIdOfRange(5), 9);
        EXPECT_EQ(permutation.lastIdOfRange(6), 9);
        EXPECT_EQ(permutation.lastIdOfRange(7), 9);
        EXPECT_EQ(permutation.lastIdOfRange(8), 9);
        EXPECT_EQ(permutation.lastIdOfRange(8), 9);

        // Build test vector
        std::vector<uint64_t> sequence;
        sequence.reserve(numElements);
        for (uint64_t i = 0; i < numElements; ++i) {
            sequence.push_back(i);
        }

        // Apply permutation to every element in the test vector.
        for (size_t idx = 0; idx < numElements; ++idx) {
            // Test that the permutation is invertible
            ASSERT_EQ(sequence[idx], permutation.finv(permutation.f(sequence[idx])));

            // Compute the permutation, to later check that each element only appears once.
            sequence[idx] = permutation.f(sequence[idx]);
        }

        // Check that no element appears more than once. Also check, that we did not exceed the given range and that we
        // really applied the identity permutation.
        EXPECT_THAT(sequence, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
    }

    { // Identity permutation numElements is not evenly divisible by numRanges
        const uint64_t                        numElements    = 10;
        const uint64_t                        lengthOfRanges = 3;
        const uint64_t                        seed           = 91244356213;
        RangePermutation<IdentityPermutation> permutation(numElements - 1, lengthOfRanges, seed);
        ASSERT_EQ(permutation.numRanges(), 4);
        ASSERT_EQ(permutation.maxValue(), numElements - 1);
        ASSERT_EQ(permutation.lengthOfRanges(), lengthOfRanges);

        EXPECT_EQ(permutation.lastIdOfRange(0), 2);
        EXPECT_EQ(permutation.lastIdOfRange(1), 2);
        EXPECT_EQ(permutation.lastIdOfRange(2), 2);
        EXPECT_EQ(permutation.lastIdOfRange(3), 5);
        EXPECT_EQ(permutation.lastIdOfRange(4), 5);
        EXPECT_EQ(permutation.lastIdOfRange(5), 5);
        EXPECT_EQ(permutation.lastIdOfRange(6), 8);
        EXPECT_EQ(permutation.lastIdOfRange(7), 8);
        EXPECT_EQ(permutation.lastIdOfRange(8), 8);
        EXPECT_EQ(permutation.lastIdOfRange(9), 9);

        // Build test vector
        std::vector<uint64_t> sequence;
        sequence.reserve(numElements);
        for (uint64_t i = 0; i < numElements; ++i) {
            sequence.push_back(i);
        }

        // Apply permutation to every element in the test vector.
        for (size_t idx = 0; idx < numElements; ++idx) {
            // Test that the permutation is invertible
            ASSERT_EQ(sequence[idx], permutation.finv(permutation.f(sequence[idx])));

            // Compute the permutation, to later check that each element only appears once.
            sequence[idx] = permutation.f(sequence[idx]);
        }

        // Check that no element appears more than once. Also check, that we did not exceed the given range and that we
        // really applied the identity permutation.
        EXPECT_THAT(sequence, ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
    }

    { // Feistel Permutation
        const uint64_t numElements    = 10;
        const uint64_t lengthOfRanges = 2;
        const uint64_t seed           = 91244356213;

        RangePermutation<FeistelPseudoRandomPermutation> permutation(numElements - 1, lengthOfRanges, seed);

        ASSERT_EQ(permutation.numRanges(), 5);
        ASSERT_EQ(permutation.maxValue(), numElements - 1);
        ASSERT_EQ(permutation.lengthOfRanges(), lengthOfRanges);

        EXPECT_EQ(permutation.lastIdOfRange(0), 1);
        EXPECT_EQ(permutation.lastIdOfRange(1), 1);
        EXPECT_EQ(permutation.lastIdOfRange(2), 3);
        EXPECT_EQ(permutation.lastIdOfRange(3), 3);
        EXPECT_EQ(permutation.lastIdOfRange(4), 5);
        EXPECT_EQ(permutation.lastIdOfRange(5), 5);
        EXPECT_EQ(permutation.lastIdOfRange(6), 7);
        EXPECT_EQ(permutation.lastIdOfRange(7), 7);
        EXPECT_EQ(permutation.lastIdOfRange(8), 9);
        EXPECT_EQ(permutation.lastIdOfRange(8), 9);

        // Build test vector
        std::vector<uint64_t> sequence;
        sequence.reserve(numElements);
        for (uint64_t i = 0; i < numElements; ++i) {
            sequence.push_back(i);
        }

        // Apply permutation to every element in the test vector.
        for (size_t idx = 0; idx < numElements; ++idx) {
            // Test that the permutation is invertible.
            ASSERT_EQ(sequence[idx], permutation.finv(permutation.f(sequence[idx])));

            // Compute the permutation, to later check that each element only appears once.
            sequence[idx] = permutation.f(sequence[idx]);
        }

        // Check that no element appears more than once. Also check, that we did not exceed the given range.
        EXPECT_THAT(sequence, UnorderedElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));

        // Check that elements (0, 1), (2, 3), (4, 5), ... have consecutive values.
        for (uint64_t i = 0; i < numElements - 1; i += 2) {
            EXPECT_EQ(sequence[i] + 1, sequence[i + 1]);
        }
    }

    { // Feistel Permutation: bigger example
        const uint64_t numElements    = 100;
        const uint64_t lengthOfRanges = 10;
        const uint64_t seed           = 8234098;

        RangePermutation<FeistelPseudoRandomPermutation> permutation(numElements - 1, lengthOfRanges, seed);

        ASSERT_EQ(permutation.numRanges(), 10);
        ASSERT_EQ(permutation.maxValue(), numElements - 1);
        ASSERT_EQ(permutation.lengthOfRanges(), lengthOfRanges);

        // Build test vector
        std::vector<uint64_t> sequence;
        sequence.reserve(numElements);
        for (uint64_t i = 0; i < numElements; ++i) {
            sequence.push_back(i);
        }

        // Apply permutation to every element in the test vector.
        for (size_t idx = 0; idx < numElements; ++idx) {
            // Test that the permutation is invertible
            ASSERT_EQ(sequence[idx], permutation.finv(permutation.f(sequence[idx])));

            // Compute the permutation, to later check that each element only appears once.
            sequence[idx] = permutation.f(sequence[idx]);
        }

        // Check that elements (0, 1, ..., 10), (11, 12, ..., 19), ... have consecutive values.
        for (uint64_t i = 0; i < numElements - 1; i += permutation.lengthOfRanges()) {
            for (uint64_t j = 0; j < permutation.lengthOfRanges(); ++j) {
                ASSERT_EQ(sequence[i] + j, sequence[i + j]);
            }
        }
    }

    { // lengthOfRanges = 1 should work
        const uint64_t numElements    = 10;
        const uint64_t lengthOfRanges = 1;
        const uint64_t seed           = 145213;

        RangePermutation<FeistelPseudoRandomPermutation> permutation(numElements - 1, lengthOfRanges, seed);

        ASSERT_EQ(permutation.numRanges(), numElements);
        ASSERT_EQ(permutation.maxValue(), numElements - 1);
        ASSERT_EQ(permutation.lengthOfRanges(), 1);

        // Build test vector
        std::vector<uint64_t> sequence;
        sequence.reserve(numElements);
        for (uint64_t i = 0; i < numElements; ++i) {
            sequence.push_back(i);
        }

        // Apply permutation to every element in the test vector.
        for (size_t idx = 0; idx < numElements; ++idx) {
            // Test that the permutation is invertible
            ASSERT_EQ(sequence[idx], permutation.finv(permutation.f(sequence[idx])));

            // Compute the permutation, to later check that each element only appears once.
            sequence[idx] = permutation.f(sequence[idx]);
        }

        // The ids should still be unique and not out of range.
        ASSERT_THAT(sequence, UnorderedElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
    }
}
