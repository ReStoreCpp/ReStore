#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "restore/helpers.hpp"

using namespace ::testing;

TEST(HelpersTest, in_range) {
    uint8_t u8val = 200;
    ASSERT_TRUE(in_range<uint8_t>::check(u8val));
    ASSERT_TRUE(in_range<uint16_t>::check(u8val));
    ASSERT_TRUE(in_range<uint32_t>::check(u8val));
    ASSERT_TRUE(in_range<uint64_t>::check(u8val));
    ASSERT_FALSE(in_range<int8_t>::check(u8val));
    ASSERT_TRUE(in_range<int16_t>::check(u8val));
    ASSERT_TRUE(in_range<int32_t>::check(u8val));
    ASSERT_TRUE(in_range<int64_t>::check(u8val));
    u8val = 10;
    ASSERT_TRUE(in_range<int8_t>::check(u8val));

    auto intMax = std::numeric_limits<int>::max();
    ASSERT_TRUE(in_range<long int>::check(intMax));
    ASSERT_TRUE(in_range<uintmax_t>::check(intMax));
    ASSERT_TRUE(in_range<intmax_t>::check(intMax));
    
    auto intNeg = -1;
    ASSERT_TRUE(in_range<long int>::check(intNeg));
    ASSERT_FALSE(in_range<uintmax_t>::check(intNeg));
    ASSERT_TRUE(in_range<intmax_t>::check(intNeg));
    ASSERT_FALSE(in_range<size_t>::check(intNeg));
    ASSERT_TRUE(in_range<short int>::check(intNeg));

    size_t sizeT = 10000;
    ASSERT_TRUE(in_range<int>::check(sizeT));
    sizeT = std::numeric_limits<size_t>::max() - 1000;
    ASSERT_FALSE(in_range<int>::check(sizeT));
    ASSERT_TRUE(in_range<uintmax_t>::check(sizeT));
}