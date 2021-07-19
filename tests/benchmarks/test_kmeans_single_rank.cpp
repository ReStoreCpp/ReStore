#include <cstddef>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <gtest-mpi-listener/include/gtest-mpi-listener.hpp>

#include "k-means.hpp"

using namespace ::testing;
using namespace kmeans;

TEST(kMeansData, constructor) {
    // 0 dimensions does not make sense
    ASSERT_THROW(kMeansData<float>(0), std::invalid_argument);
    ASSERT_THROW(kMeansData<float>(0, 0, 0), std::invalid_argument);
    ASSERT_THROW(kMeansData<double>(0), std::invalid_argument);
    ASSERT_THROW(kMeansData<double>(0, 0, 0), std::invalid_argument);
    ASSERT_THROW(kMeansData<double>(std::vector<double>{0, 1, 2, 3}, 0), std::invalid_argument);
    ASSERT_THROW(kMeansData<double>({0, 1, 2, 3}, 0), std::invalid_argument);

    // Length of provided data not evenly divisible by the number of dimensions
    ASSERT_THROW(kMeansData<double>(std::vector<double>{0, 1, 2, 3}, 3), std::invalid_argument);
    ASSERT_THROW(kMeansData<double>({0, 1, 2, 3}, 3), std::invalid_argument);

    // 0 data points is fine, as these can be added later
    ASSERT_NO_THROW(kMeansData(0, 1, 0));
    ASSERT_NO_THROW(kMeansData<double>(std::vector<double>{}, 1));
    ASSERT_NO_THROW(kMeansData<double>({}, 1));
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
        auto data = kMeansData<float>(11, 10, 1);
        ASSERT_EQ(data.numDimensions(), 10);
        ASSERT_EQ(data.numDataPoints(), 11);
        ASSERT_TRUE(data.valid());
        for (size_t idx = 0; idx < data.numDataPoints(); idx++) {
            ASSERT_EQ(data.getElementDimension(idx, 0), 1);
        }
    }

    {
        auto data = kMeansData<float>(std::vector<float>{1, 2, 3, 4, 5, 6}, 2);
        ASSERT_EQ(data.numDimensions(), 2);
        ASSERT_EQ(data.numDataPoints(), 3);
        ASSERT_TRUE(data.valid());
    }

    {
        auto data = kMeansData<float>({0, 1, 2, 3, 4, 5, 6, 7, 8}, 3);
        ASSERT_EQ(data.numDimensions(), 3);
        ASSERT_EQ(data.numDataPoints(), 3);
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

TEST(kMeansData, getElementDimension) {
    auto data = kMeansData<double>(4, 2, 42);
    ASSERT_EQ(data.getElementDimension(0, 0), 42);
    ASSERT_EQ(data.getElementDimension(1, 0), 42);
    ASSERT_EQ(data.getElementDimension(2, 0), 42);
    ASSERT_EQ(data.getElementDimension(3, 0), 42);
    ASSERT_EQ(data.getElementDimension(0, 1), 42);
    ASSERT_EQ(data.getElementDimension(1, 1), 42);
    ASSERT_EQ(data.getElementDimension(2, 1), 42);
    ASSERT_EQ(data.getElementDimension(3, 1), 42);

    data.getElementDimension(0, 0) = 1337;
    data.getElementDimension(1, 0) = 1337;
    data.getElementDimension(1, 1) = 1337;

    ASSERT_EQ(data.getElementDimension(0, 0), 1337);
    ASSERT_EQ(data.getElementDimension(1, 0), 1337);
    ASSERT_EQ(data.getElementDimension(2, 0), 42);
    ASSERT_EQ(data.getElementDimension(3, 0), 42);
    ASSERT_EQ(data.getElementDimension(0, 1), 42);
    ASSERT_EQ(data.getElementDimension(1, 1), 1337);
    ASSERT_EQ(data.getElementDimension(2, 1), 42);
    ASSERT_EQ(data.getElementDimension(3, 1), 42);
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

    ASSERT_EQ(data.getElementDimension(0, 0), 1);
    ASSERT_EQ(data.getElementDimension(1, 0), 1);
    ASSERT_EQ(data.getElementDimension(2, 0), 42);
    ASSERT_EQ(data.getElementDimension(3, 0), 42);
    ASSERT_EQ(data.getElementDimension(0, 1), 1);
    ASSERT_EQ(data.getElementDimension(1, 1), 1);
    ASSERT_EQ(data.getElementDimension(2, 1), 42);
    ASSERT_EQ(data.getElementDimension(3, 1), 42);
}

TEST(kMeansHelper, generateRandomData) {
    // Zero dimensional data should throw an error
    ASSERT_THROW(generateRandomData<float>(10, 0), std::invalid_argument);

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

TEST(kMeansAlgorithm, constructor) {
    // Constructing with no data points should fail
    ASSERT_THROW(kMeansAlgorithm(generateRandomData<double>(0, 1)), std::invalid_argument);
    ASSERT_THROW(kMeansAlgorithm(generateRandomData<float>(0, 3)), std::invalid_argument);

    { // Constructing with an invalid data object should fail
        auto data = kMeansData<float>(2);
        ASSERT_TRUE(data.valid());
        data << 2.5f;
        ASSERT_FALSE(data.valid());
        ASSERT_THROW(kMeansAlgorithm<float>(std::move(data)), std::invalid_argument);
    }

    { // Constructing with a piecewise constructed kMeansData object should work
        auto data = kMeansData<float>(2);
        ASSERT_TRUE(data.valid());
        data << 2.5f;
        ASSERT_FALSE(data.valid());
        data << 2.5f;
        data << typename kMeansData<float>::FinalizeDataPoint();
        ASSERT_TRUE(data.valid());
        ASSERT_NO_THROW(kMeansAlgorithm<float>(std::move(data)));
    }

    { // Constructing with a valid, non-empty data object should work
        auto data1 = generateRandomData<double>(100, 30);
        ASSERT_NO_THROW(kMeansAlgorithm<double>(std::move(data1)));

        auto data2 = kMeansData<double>(2, 2, 2);
        ASSERT_NO_THROW(kMeansAlgorithm<double>(std::move(data2)));
    }

    { // Length of input vector not evenly dividible by the number of dimensions should fail
        ASSERT_THROW(kMeansAlgorithm<float>(std::vector<float>{1, 2, 3}, 2), std::invalid_argument);
        ASSERT_THROW(kMeansAlgorithm<float>({1, 2, 3}, 2), std::invalid_argument);
    }

    { // Is fine
        ASSERT_NO_THROW(kMeansAlgorithm<float>(std::vector<float>{1, 2, 3, 4}, 2));
        ASSERT_NO_THROW(kMeansAlgorithm<float>({1, 2, 3, 4}, 2));
    }
}

TEST(kMeansAlgorithm, pickCentersRandomly) {
    auto kmeansInstance = kMeansAlgorithm<float>(generateRandomData<float>(100, 3));
    ASSERT_EQ(kmeansInstance.numCenters(), 0);
    ASSERT_EQ(kmeansInstance.numDataPoints(), 100);
    ASSERT_EQ(kmeansInstance.numDimensions(), 3);

    // Picking random centers should result in this amount of centers and not affect the dimensionality or the number of
    // data points.
    kmeansInstance.pickCentersRandomly(10);
    ASSERT_EQ(kmeansInstance.numCenters(), 10);
    ASSERT_EQ(kmeansInstance.centers().numDataPoints(), 10);
    ASSERT_EQ(kmeansInstance.centers().numDimensions(), 3);
    ASSERT_TRUE(kmeansInstance.centers().valid());
    ASSERT_EQ(kmeansInstance.numDataPoints(), 100);
    ASSERT_EQ(kmeansInstance.numDimensions(), 3);

    // Picking the initial centers twice should be no problem
    kmeansInstance.pickCentersRandomly(20);
    ASSERT_EQ(kmeansInstance.numCenters(), 20);
    ASSERT_EQ(kmeansInstance.centers().numDataPoints(), 20);
    ASSERT_EQ(kmeansInstance.centers().numDimensions(), 3);
    ASSERT_TRUE(kmeansInstance.centers().valid());
    ASSERT_EQ(kmeansInstance.numDataPoints(), 100);
    ASSERT_EQ(kmeansInstance.numDimensions(), 3);

    // Picking as many centers as there are data points is fine
    kmeansInstance.pickCentersRandomly(100);
    ASSERT_EQ(kmeansInstance.numCenters(), 100);
    ASSERT_EQ(kmeansInstance.centers().numDataPoints(), 100);
    ASSERT_EQ(kmeansInstance.centers().numDimensions(), 3);
    ASSERT_TRUE(kmeansInstance.centers().valid());
    ASSERT_EQ(kmeansInstance.numDataPoints(), 100);
    ASSERT_EQ(kmeansInstance.numDimensions(), 3);

    // Picking more centers than there are data points does not really make sense and should result in an error.
    ASSERT_THROW(kmeansInstance.pickCentersRandomly(101), std::invalid_argument);
}

TEST(kMeansAlgorithm, setCenters) {
    auto kmeansInstance = kMeansAlgorithm<float>(generateRandomData<float>(100, 3));
    ASSERT_EQ(kmeansInstance.numCenters(), 0);
    ASSERT_EQ(kmeansInstance.numDataPoints(), 100);
    ASSERT_EQ(kmeansInstance.numDimensions(), 3);

    kmeansInstance.setCenters(kMeansData<float>(3, 3, 0));
    ASSERT_EQ(kmeansInstance.numCenters(), 3);
    ASSERT_EQ(kmeansInstance.numDataPoints(), 100);
    ASSERT_EQ(kmeansInstance.numDimensions(), 3);
    ASSERT_THAT(kmeansInstance.centers(), ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0));

    kmeansInstance.setCenters({0.5, 1.0, 2.0, 3.5, 3.5, 3.5});
    ASSERT_EQ(kmeansInstance.numCenters(), 2);
    ASSERT_EQ(kmeansInstance.numDataPoints(), 100);
    ASSERT_EQ(kmeansInstance.numDimensions(), 3);
    ASSERT_THAT(kmeansInstance.centers(), ElementsAre(0.5, 1.0, 2.0, 3.5, 3.5, 3.5));

    // Length of initializer list or vector not evenly divisible by number of dimensions
    ASSERT_THROW(kmeansInstance.setCenters(std::vector<float>{1, 3, 2, 4}), std::invalid_argument);
    ASSERT_THROW(kmeansInstance.setCenters({1, 3, 2, 4}), std::invalid_argument);
}

TEST(kMeansAlgorithm, assignPointsToCenters) {
    {
        auto data           = kMeansData<float>(5, 2, 0);
        auto kmeansInstance = kMeansAlgorithm<float>(std::move(data));

        kmeansInstance.setCenters(std::vector<float>{0, 0});
        kmeansInstance.assignPointsToCenters();
        ASSERT_THAT(kmeansInstance.pointToCenterAssignment().assignedCenter, ElementsAre(0, 0, 0, 0, 0));

        kmeansInstance.setCenters(std::vector<float>{1, 2});
        kmeansInstance.assignPointsToCenters();
        ASSERT_THAT(kmeansInstance.pointToCenterAssignment().assignedCenter, ElementsAre(0, 0, 0, 0, 0));
    }
    {
        auto kmeansInstance = kMeansAlgorithm<float>({1, 2, 3, 7, 8, 9}, 1);
        kmeansInstance.setCenters({2, 8});
        kmeansInstance.assignPointsToCenters();
        ASSERT_THAT(kmeansInstance.pointToCenterAssignment().assignedCenter, ElementsAre(0, 0, 0, 1, 1, 1));
        ASSERT_THAT(kmeansInstance.pointToCenterAssignment().numPointsAssignedToCenter, ElementsAre(3, 3));

        kmeansInstance.setCenters({9.1f, -0.1f});
        kmeansInstance.assignPointsToCenters();
        ASSERT_THAT(kmeansInstance.pointToCenterAssignment().assignedCenter, ElementsAre(1, 1, 1, 0, 0, 0));
        ASSERT_THAT(kmeansInstance.pointToCenterAssignment().numPointsAssignedToCenter, ElementsAre(3, 3));
    }

    {
        //auto kmeansInstance = kMeansAlgorithm<float>({0, 0.5f, 1, 1.1f, 2, 2, 6, 6, 7, 7, 9, 10}, 2);
        auto kmeansInstance = kMeansAlgorithm<float>({0, 0, 1, 1, 2, 2, 6, 6, 7, 7, 9, 10}, 2);
        kmeansInstance.setCenters({1, 1, 8, 8});
        kmeansInstance.assignPointsToCenters();
        ASSERT_THAT(kmeansInstance.pointToCenterAssignment().assignedCenter, ElementsAre(0, 0, 0, 1, 1, 1));
        ASSERT_THAT(kmeansInstance.pointToCenterAssignment().numPointsAssignedToCenter, ElementsAre(3, 3));
    }
}

TEST(kMeansAlgorithm, updateCenters) {
    { // If the centers already match the data, nothing should change
    
    }
    
    { // If the centers are not already matching the data, they should be updated

    }
}

TEST(kMeansAlgorithm, performIterations) {
    // 0 iterations does nothing

    // single iteration

    // two iterations
}

TEST(kMeansAlgorithm, smallExample) {}

TEST(kMeansAlgorithm, slightlyLargerExample) {}

int main(int argc, char** argv) {
    // Filter out Google Test arguments
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Set errorhandler to return so we have a chance to mitigate failures
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    // Add object that will finalize MPI on exit; Google Test owns this pointer
    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);

    // Get the event listener list.
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

    // Remove default listener: the default printer and the default XML printer
    ::testing::TestEventListener* l = listeners.Release(listeners.default_result_printer());

    // Adds MPI listener; Google Test owns this pointer
    listeners.Append(new GTestMPIListener::MPIWrapperPrinter(l, MPI_COMM_WORLD));

    int result = RUN_ALL_TESTS();

    return result;
}
