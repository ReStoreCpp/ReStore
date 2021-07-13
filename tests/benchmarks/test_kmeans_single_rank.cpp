#include <cstddef>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "k-means.hpp"

using namespace ::testing;
using namespace kmeans;

TEST(kMeansData, constructor) {
    // 0 dimensions does not make sense
    ASSERT_THROW(kMeansData<float>(0), std::runtime_error);
    ASSERT_THROW(kMeansData<float>(0, 0, 0), std::runtime_error);
    ASSERT_THROW(kMeansData<double>(0), std::runtime_error);
    ASSERT_THROW(kMeansData<double>(0, 0, 0), std::runtime_error);

    // 0 data points is fine, as these can be added later
    ASSERT_NO_THROW(kMeansData(0, 1, 0));
}

TEST(kMeansData, gettersAndSetters) {
    {
        auto data = kMeansData<float>(10, 1, 10);
        ASSERT_EQ(data.numDimensions(), 1);
        ASSERT_EQ(data.numDataPoints(), 10);
        ASSERT_TRUE(data.valid());
    }

    {
        auto data = kMeansData<float>(20, 2, 0);
        ASSERT_EQ(data.numDimensions(), 2);
        ASSERT_EQ(data.numDataPoints(), 20);
        ASSERT_TRUE(data.valid());
    }

    {
        auto data = kMeansData<float>(10, 10, 1);
        ASSERT_EQ(data.numDimensions(), 10);
        ASSERT_EQ(data.numDataPoints(), 10);
        ASSERT_TRUE(data.valid());
    }
}

TEST(kMeansData, appendMoreData) {
    auto data = kMeansData<float>(0, 2, 0);
    ASSERT_EQ(data.numDataPoints(), 0);
    ASSERT_EQ(data.numDimensions(), 2);
    ASSERT_TRUE(data.valid());

    data << 0.5f;
    ASSERT_FALSE(data.valid());
    data << 0.25f;
    ASSERT_FALSE(data.valid());
    data << kMeansData<float>::FinalizeDataPoint();
    ASSERT_TRUE(data.valid());
    ASSERT_EQ(data.numDimensions(), 2);
    ASSERT_EQ(data.numDataPoints(), 1);

    data << 1.f;
    ASSERT_FALSE(data.valid());
    data << 2.f;
    ASSERT_FALSE(data.valid());
    data << kMeansData<float>::FinalizeDataPoint();
    ASSERT_TRUE(data.valid());
    ASSERT_EQ(data.numDimensions(), 2);
    ASSERT_EQ(data.numDataPoints(), 2);

    for (int a = 1; a <= 20; a++) {
        data << static_cast<float>(a) << 2.f * static_cast<float>(a) << kMeansData<float>::FinalizeDataPoint();
        ASSERT_TRUE(data.valid());
        ASSERT_EQ(data.numDimensions(), 2);
        ASSERT_EQ(data.numDataPoints(), 2 + a);
    }
    ASSERT_TRUE(data.valid());
    ASSERT_EQ(data.numDimensions(), 2);
    ASSERT_EQ(data.numDataPoints(), 22);
}

TEST(kMeansData, squareBracketAccess) {
    auto data = kMeansData<double>(4, 2, 1);
    ASSERT_EQ(data[0], 1);
    ASSERT_EQ(data[1], 1);
    ASSERT_EQ(data[2], 1);
    ASSERT_EQ(data[3], 1);
    ASSERT_EQ(data[4], 1);
    ASSERT_EQ(data[5], 1);
    ASSERT_EQ(data[6], 1);
    ASSERT_EQ(data[7], 1);

    for (size_t idx = 1; idx < 5; ++idx) {
        data[idx] = 3;
    }

    ASSERT_EQ(data[0], 1);
    ASSERT_EQ(data[1], 3);
    ASSERT_EQ(data[2], 3);
    ASSERT_EQ(data[3], 3);
    ASSERT_EQ(data[4], 3);
    ASSERT_EQ(data[5], 1);
    ASSERT_EQ(data[6], 1);
    ASSERT_EQ(data[7], 1);
    ASSERT_TRUE(data.valid());
    ASSERT_EQ(data.numDimensions(), 2);
    ASSERT_EQ(data.numDataPoints(), 4);
    ASSERT_EQ(data.dataSize(), 8);
}

TEST(kMeansData, resize) {
    auto data = kMeansData<double>(2, 2, 1);

    ASSERT_TRUE(data.valid());
    ASSERT_EQ(data.numDimensions(), 2);
    ASSERT_EQ(data.numDataPoints(), 2);

    data.resize(4, 42);

    ASSERT_TRUE(data.valid());
    ASSERT_EQ(data.numDimensions(), 2);
    ASSERT_EQ(data.numDataPoints(), 4);
    ASSERT_EQ(data.dataSize(), 8);

    ASSERT_EQ(data[0], 1);
    ASSERT_EQ(data[1], 1);
    ASSERT_EQ(data[2], 1);
    ASSERT_EQ(data[3], 1);
    ASSERT_EQ(data[4], 42);
    ASSERT_EQ(data[5], 42);
    ASSERT_EQ(data[6], 42);
    ASSERT_EQ(data[7], 42);
}

TEST(kMeansHelper, generateRandomData) {
    // Zero dimensional data should throw an error
    ASSERT_THROW(generateRandomData<float>(10, 0), std::runtime_error);

    // Generating a empty data set is fine
    ASSERT_NO_THROW(generateRandomData<float>(0, 1));

    { // Empty data set
        auto data = generateRandomData<float>(0, 1);
        ASSERT_EQ(data.numDataPoints(), 0);
        ASSERT_EQ(data.numDimensions(), 1);
        ASSERT_TRUE(data.valid());
        ASSERT_EQ(data.dataSize(), 0);
    }

    { // Data set with 1 member
        auto data = generateRandomData<float>(1, 1);
        ASSERT_EQ(data.numDataPoints(), 1);
        ASSERT_EQ(data.numDimensions(), 1);
        ASSERT_TRUE(data.valid());
        ASSERT_EQ(data.dataSize(), 1);
    }

    { // Larger example
        auto data = generateRandomData<float>(100, 3);
        ASSERT_EQ(data.numDataPoints(), 100);
        ASSERT_EQ(data.numDimensions(), 3);
        ASSERT_TRUE(data.valid());
        ASSERT_EQ(data.dataSize(), 300);
    }
}