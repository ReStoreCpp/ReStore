#include <limits>
#include <memory>
#include <stddef.h>
#include <stdint.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "restore/helpers.hpp"

using namespace ::testing;

TEST(HelpersTest, in_range) {
    uint8_t u8val = 200;
    ASSERT_TRUE(in_range<uint8_t>(u8val));
    ASSERT_TRUE(in_range<uint16_t>(u8val));
    ASSERT_TRUE(in_range<uint32_t>(u8val));
    ASSERT_TRUE(in_range<uint64_t>(u8val));
    ASSERT_FALSE(in_range<int8_t>(u8val));
    ASSERT_TRUE(in_range<int16_t>(u8val));
    ASSERT_TRUE(in_range<int32_t>(u8val));
    ASSERT_TRUE(in_range<int64_t>(u8val));
    u8val = 10;
    ASSERT_TRUE(in_range<int8_t>(u8val));

    auto intMax = std::numeric_limits<int>::max();
    ASSERT_TRUE(in_range<long int>(intMax));
    ASSERT_TRUE(in_range<uintmax_t>(intMax));
    ASSERT_TRUE(in_range<intmax_t>(intMax));

    auto intNeg = -1;
    ASSERT_TRUE(in_range<long int>(intNeg));
    ASSERT_FALSE(in_range<uintmax_t>(intNeg));
    ASSERT_TRUE(in_range<intmax_t>(intNeg));
    ASSERT_FALSE(in_range<size_t>(intNeg));
    ASSERT_TRUE(in_range<short int>(intNeg));

    size_t sizeT = 10000;
    ASSERT_TRUE(in_range<int>(sizeT));
    sizeT = std::numeric_limits<size_t>::max() - 1000;
    ASSERT_FALSE(in_range<int>(sizeT));
    ASSERT_TRUE(in_range<uintmax_t>(sizeT));
}