#include <cstddef>
#include <gmock/gmock.h>
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>

#include "k-means.hpp"
#include "restore/mpi_context.hpp"

using namespace ::testing;
using namespace kmeans;

// These tests are run on a single rank, we therefore cannot fulfil a replication level > 1.
const uint16_t REPLICATION_LEVEL = 1;
const uint64_t BLOCKS_PER_PERMUTATION_RANGE = 1;

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
    // Initialize the MPI_Context
    using MPIContext = ReStoreMPI::MPIContext;
    MPIContext mpiContext(MPI_COMM_WORLD);

    // Constructing with no data points should fail
    ASSERT_THROW(
        kMeansAlgorithm(generateRandomData<double>(0, 1), mpiContext, true, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE), std::invalid_argument);
    ASSERT_THROW(
        kMeansAlgorithm(generateRandomData<float>(0, 3), mpiContext, true, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE), std::invalid_argument);

    { // Constructing with an invalid data object should fail
        auto data = kMeansData<float>(2);
        ASSERT_TRUE(data.valid());
        data << 2.5f;
        ASSERT_FALSE(data.valid());
        ASSERT_THROW(
            (kMeansAlgorithm<float, MPIContext>(std::move(data), mpiContext, true, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE)),
            std::invalid_argument);
    }

    { // Constructing with a piecewise constructed kMeansData object should work
        auto data = kMeansData<float>(2);
        ASSERT_TRUE(data.valid());
        data << 2.5f;
        ASSERT_FALSE(data.valid());
        data << 2.5f;
        data << typename kMeansData<float>::FinalizeDataPoint();
        ASSERT_TRUE(data.valid());
        data << 1.3f << 1.7f << typename kMeansData<float>::FinalizeDataPoint();
        ASSERT_TRUE(data.valid());
        ASSERT_GT(data.dataSize(), 0);
        ASSERT_NO_THROW((kMeansAlgorithm<float, MPIContext>(std::move(data), mpiContext, true, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE)));
    }

    { // Constructing with a valid, non-empty data object should work
        auto data1 = generateRandomData<double>(100, 30);
        ASSERT_NO_THROW((kMeansAlgorithm<double, MPIContext>(std::move(data1), mpiContext, true, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE)));

        auto data2 = kMeansData<double>(2, 2, 2);
        ASSERT_NO_THROW((kMeansAlgorithm<double, MPIContext>(std::move(data2), mpiContext, true, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE)));
    }

    { // Length of input vector not evenly dividible by the number of dimensions should fail
        ASSERT_THROW(
            (kMeansAlgorithm<float, MPIContext>(std::vector<float>{1, 2, 3}, 2, mpiContext, true, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE)),
            std::invalid_argument);
        ASSERT_THROW(
            (kMeansAlgorithm<float, MPIContext>({1, 2, 3}, 2, mpiContext, true, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE)),
            std::invalid_argument);
    }

    { // When using fault tolerance, the replication level must be greater than 0
        ASSERT_THROW((kMeansAlgorithm(std::vector<float>{1, 2, 3, 4}, 2, mpiContext, true, 0, BLOCKS_PER_PERMUTATION_RANGE)), std::invalid_argument);
    }

    { // Is fine
        ASSERT_NO_THROW((kMeansAlgorithm(std::vector<float>{1, 2, 3, 4}, 2, mpiContext, true, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE)));
        ASSERT_NO_THROW((kMeansAlgorithm<float, MPIContext>({1, 2, 3, 4}, 2, mpiContext, true, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE)));
        ASSERT_NO_THROW((kMeansAlgorithm<float, MPIContext>({1, 2, 3, 4}, 2, mpiContext, false, 0, BLOCKS_PER_PERMUTATION_RANGE)));
    }
}

TEST(kMeansAlgorithm, pickCentersRandomly) {
    // Initialize the MPIContext
    using MPIContext = ReStoreMPI::MPIContext;
    MPIContext mpiContext(MPI_COMM_WORLD);

    auto kmeansInstance =
        kMeansAlgorithm<float, MPIContext>(generateRandomData<float>(100, 3), mpiContext, true, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE);
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
    // Initialize the MPIContext
    using MPIContext = ReStoreMPI::MPIContext;
    MPIContext mpiContext(MPI_COMM_WORLD);

    auto kmeansInstance =
        kMeansAlgorithm<float, MPIContext>(generateRandomData<float>(100, 3), mpiContext, true, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE);
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
    // Initialize the MPIContext
    using MPIContext = ReStoreMPI::MPIContext;
    MPIContext mpiContext(MPI_COMM_WORLD);

    for (bool useFaultTolerance: {false, true}) {
        {
            auto data = kMeansData<float>(5, 2, 0);
            auto kmeansInstance =
                kMeansAlgorithm<float, MPIContext>(std::move(data), mpiContext, useFaultTolerance, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE);

            kmeansInstance.setCenters(std::vector<float>{0, 0});
            kmeansInstance.assignPointsToCenters();
            ASSERT_THAT(kmeansInstance.pointToCenterAssignment().assignedCenter, ElementsAre(0, 0, 0, 0, 0));

            kmeansInstance.setCenters(std::vector<float>{1, 2});
            kmeansInstance.assignPointsToCenters();
            ASSERT_THAT(kmeansInstance.pointToCenterAssignment().assignedCenter, ElementsAre(0, 0, 0, 0, 0));
        }
        {
            auto kmeansInstance = kMeansAlgorithm<float, MPIContext>(
                {1, 2, 3, 7, 8, 9}, 1, mpiContext, useFaultTolerance, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE);
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
            // auto kmeansInstance = kMeansAlgorithm<float, MPIContext>({0, 0.5f, 1, 1.1f, 2, 2, 6, 6, 7, 7, 9, 10}, 2);
            auto kmeansInstance = kMeansAlgorithm<float, MPIContext>(
                {0, 0, 1, 1, 2, 2, 6, 6, 7, 7, 9, 10}, 2, mpiContext, useFaultTolerance, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE);
            kmeansInstance.setCenters({1, 1, 8, 8});
            kmeansInstance.assignPointsToCenters();
            ASSERT_THAT(kmeansInstance.pointToCenterAssignment().assignedCenter, ElementsAre(0, 0, 0, 1, 1, 1));
            ASSERT_THAT(kmeansInstance.pointToCenterAssignment().numPointsAssignedToCenter, ElementsAre(3, 3));
        }

        { // A center not getting assigned to any points should not be a problem
            auto kmeansInstance =
                kMeansAlgorithm<float, MPIContext>({0, 0, 1, 1}, 1, mpiContext, useFaultTolerance, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE);
            kmeansInstance.setCenters({0, 20});
            ASSERT_THAT(kmeansInstance.centers(), ElementsAre(0, 20));
            ASSERT_NO_THROW(kmeansInstance.assignPointsToCenters());
            ASSERT_THAT(kmeansInstance.pointToCenterAssignment().assignedCenter, ElementsAre(0, 0, 0, 0));
            ASSERT_THAT(kmeansInstance.pointToCenterAssignment().numPointsAssignedToCenter, ElementsAre(4, 0));
        }
    }
}

TEST(kMeansAlgorithm, updateCenters) {
    // Initialize the MPIContext
    using MPIContext = ReStoreMPI::MPIContext;
    MPIContext mpiContext(MPI_COMM_WORLD);

    for (bool useFaultTolerance: {false, true}) {
        { // If the centers already match the data, nothing should change
            auto kmeansInstance =
                kMeansAlgorithm<float, MPIContext>({0, 0, 1, 1}, 1, mpiContext, useFaultTolerance, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE);
            kmeansInstance.setCenters({0, 1});
            ASSERT_THAT(kmeansInstance.centers(), ElementsAre(0, 1));
            kmeansInstance.assignPointsToCenters();
            ASSERT_THAT(kmeansInstance.pointToCenterAssignment().assignedCenter, ElementsAre(0, 0, 1, 1));
            ASSERT_THAT(kmeansInstance.pointToCenterAssignment().numPointsAssignedToCenter, ElementsAre(2, 2));
            kmeansInstance.updateCenters();
            kmeansInstance.assignPointsToCenters();
            ASSERT_THAT(kmeansInstance.pointToCenterAssignment().numPointsAssignedToCenter, ElementsAre(2, 2));
            ASSERT_THAT(kmeansInstance.centers(), ElementsAre(0, 1));
            ASSERT_THAT(kmeansInstance.pointToCenterAssignment().assignedCenter, ElementsAre(0, 0, 1, 1));
        }

        { // If the centers already match the data, nothing should change
            auto kmeansInstance = kMeansAlgorithm<float, MPIContext>(
                {0, 0, 1, 1, 1, 1, 0, 0}, 2, mpiContext, useFaultTolerance, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE);
            kmeansInstance.setCenters({0, 0, 1, 1});
            ASSERT_THAT(kmeansInstance.centers(), ElementsAre(0, 0, 1, 1));
            kmeansInstance.assignPointsToCenters();
            kmeansInstance.updateCenters();
            ASSERT_THAT(kmeansInstance.centers(), ElementsAre(0, 0, 1, 1));
        }

        { // If the centers are already at their optimal position, nothing should change
            auto kmeansInstance =
                kMeansAlgorithm<float, MPIContext>({0, 0, 1, 1}, 1, mpiContext, useFaultTolerance, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE);
            kmeansInstance.setCenters({0.5});
            ASSERT_THAT(kmeansInstance.centers(), ElementsAre(0.5));
            kmeansInstance.assignPointsToCenters();
            kmeansInstance.updateCenters();
            ASSERT_THAT(kmeansInstance.centers(), ElementsAre(0.5));
        }

        { // If the centers are already at their optimal position, nothing should change
            auto kmeansInstance =
                kMeansAlgorithm<float, MPIContext>({0, 0, 1, 1}, 2, mpiContext, useFaultTolerance, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE);
            kmeansInstance.setCenters({0.5, 0.5});
            ASSERT_THAT(kmeansInstance.centers(), ElementsAre(0.5, 0.5));
            kmeansInstance.assignPointsToCenters();
            kmeansInstance.updateCenters();
            ASSERT_THAT(kmeansInstance.centers(), ElementsAre(0.5, 0.5));
        }

        { // Else, the centers should be updated
            auto kmeansInstance =
                kMeansAlgorithm<float, MPIContext>({0, 0, 1, 1}, 2, mpiContext, useFaultTolerance, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE);
            kmeansInstance.setCenters({0, 0});
            ASSERT_THAT(kmeansInstance.centers(), ElementsAre(0, 0));
            kmeansInstance.assignPointsToCenters();
            kmeansInstance.updateCenters();
            ASSERT_THAT(kmeansInstance.centers(), ElementsAre(0.5, 0.5));
        }

        { // A center not getting assigned to any points should not be a problem
            auto kmeansInstance =
                kMeansAlgorithm<float, MPIContext>({0, 0, 1, 1}, 1, mpiContext, useFaultTolerance, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE);
            kmeansInstance.setCenters({0, 20});
            ASSERT_THAT(kmeansInstance.centers(), ElementsAre(0, 20));
            kmeansInstance.assignPointsToCenters();
            ASSERT_THAT(kmeansInstance.pointToCenterAssignment().assignedCenter, ElementsAre(0, 0, 0, 0));
            ASSERT_THAT(kmeansInstance.pointToCenterAssignment().numPointsAssignedToCenter, ElementsAre(4, 0));
            kmeansInstance.updateCenters();
            kmeansInstance.assignPointsToCenters();
            ASSERT_THAT(kmeansInstance.centers(), ElementsAre(0.5, 20));
            ASSERT_THAT(kmeansInstance.pointToCenterAssignment().assignedCenter, ElementsAre(0, 0, 0, 0));
            ASSERT_THAT(kmeansInstance.pointToCenterAssignment().numPointsAssignedToCenter, ElementsAre(4, 0));
        }
    }
}

TEST(kMeansAlgorithm, performIterations) {
    // Initialize the MPIContext
    using MPIContext = ReStoreMPI::MPIContext;
    MPIContext mpiContext(MPI_COMM_WORLD);

    for (bool useFaultTolerance: {false, true}) {
        { // 0 iterations does nothing
            auto kmeansInstance = kMeansAlgorithm<float, MPIContext>(
                {0, 0, 1, 1, 2, 2, 6, 6, 7, 7, 8, 8}, 2, mpiContext, useFaultTolerance, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE);
            kmeansInstance.setCenters({5, 5, 8, 8});
            ASSERT_THAT(kmeansInstance.centers(), ElementsAre(5, 5, 8, 8));
            kmeansInstance.performIterations(0);
            ASSERT_THAT(kmeansInstance.centers(), ElementsAre(5, 5, 8, 8));
        }

        { // single iteration
            auto kmeansInstance = kMeansAlgorithm<float, MPIContext>(
                {0, 0, 1, 1, 2, 2, 6, 6, 7, 7, 8, 8}, 2, mpiContext, useFaultTolerance, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE);
            kmeansInstance.setCenters({5, 5, 8, 8});
            ASSERT_THAT(kmeansInstance.centers(), ElementsAre(5, 5, 8, 8));
            kmeansInstance.performIterations(1);
            ASSERT_THAT(kmeansInstance.centers(), ElementsAre(2.25, 2.25, 7.5, 7.5));
        }

        { // two iterations
            auto kmeansInstance = kMeansAlgorithm<float, MPIContext>(
                {0, 0, 1, 1, 2, 2, 6, 6, 7, 7, 8, 8}, 2, mpiContext, useFaultTolerance, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE);
            kmeansInstance.setCenters({5, 5, 8, 8});
            ASSERT_THAT(kmeansInstance.centers(), ElementsAre(5, 5, 8, 8));
            kmeansInstance.performIterations(2);
            ASSERT_THAT(kmeansInstance.centers(), ElementsAre(1, 1, 7, 7));
        }
    }
}

TEST(kMeansAlgorithm, smallExample) {
    // Initialize the MPIContext
    using MPIContext = ReStoreMPI::MPIContext;
    MPIContext mpiContext(MPI_COMM_WORLD);

    for (bool useFaultTolerance: {false, true}) {
        auto kmeansInstance = kMeansAlgorithm<double, MPIContext>(
            {1, 1, 2, 10, 13, 18, 20, 21}, 1, mpiContext, useFaultTolerance, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE);
        kmeansInstance.setCenters({5, 17});
        ASSERT_THAT(kmeansInstance.centers(), ElementsAre(5, 17));

        kmeansInstance.performIterations(1);
        kmeansInstance.assignPointsToCenters(); // We want to test the local assignment.
        ASSERT_THAT(kmeansInstance.pointToCenterAssignment().assignedCenter, ElementsAre(0, 0, 0, 0, 1, 1, 1, 1));
        ASSERT_THAT(kmeansInstance.pointToCenterAssignment().numPointsAssignedToCenter, ElementsAre(4, 4));
        ASSERT_THAT(kmeansInstance.centers(), ElementsAre(3.5, 18));

        kmeansInstance.performIterations(1);
        kmeansInstance.assignPointsToCenters(); // We want to test the local assignment.
        ASSERT_THAT(kmeansInstance.pointToCenterAssignment().assignedCenter, ElementsAre(0, 0, 0, 0, 1, 1, 1, 1));
        ASSERT_THAT(kmeansInstance.pointToCenterAssignment().numPointsAssignedToCenter, ElementsAre(4, 4));
        ASSERT_THAT(kmeansInstance.centers(), ElementsAre(3.5, 18));
    }
}

TEST(kMeansAlgorithm, slightlyLargerExample) {
    // Initialize the MPIContext
    using MPIContext = ReStoreMPI::MPIContext;
    MPIContext mpiContext(MPI_COMM_WORLD);

    for (bool useFaultTolerance: {false, true}) {
        // Data taken from https://scistatcalc.blogspot.com/2014/01/k-means-clustering-calculator.html
        auto kMeansInstance = kMeansAlgorithm<double, MPIContext>(
            {1.870,  1.991,  2.007,  2.186,  2.137,  1.982,  1.883,  2.103,  1.917,  2.019,  1.935,  2.083,  1.849,
             2.087,  0.122,  -0.223, 2.003,  2.031,  0.055,  0.029,  0.054,  0.107,  0.827,  0.981,  0.980,  0.887,
             1.222,  0.883,  1.056,  0.921,  2.080,  1.996,  1.021,  1.066,  -0.055, 0.016,  0.970,  1.027,  1.125,
             0.860,  1.023,  0.978,  -0.036, -0.019, 0.200,  0.109,  -0.163, -0.131, -0.176, -0.122, -0.023, -0.044,
             -0.029, 0.028,  1.920,  1.991,  -0.030, -0.031, 1.940,  1.967,  1.021,  1.141,  1.025,  1.097,  1.997,
             2.081,  1.921,  2.026,  -0.107, -0.042, 0.911,  1.093,  1.027,  1.151,  1.992,  2.079,  1.088,  0.979,
             -0.031, 0.019,  0.082,  -0.033, -0.012, -0.034, 2.110,  1.828,  1.960,  1.887,  0.892,  0.854,  1.900,
             2.054,  0.073,  0.103,  2.142,  1.770,  0.034,  0.069,  0.109,  0.037,  1.011,  0.808,  1.976,  1.800,
             2.050,  2.045,  -0.150, -0.015, 2.145,  2.127,  2.085,  2.116,  2.090,  1.974,  0.106,  -0.044, 0.044,
             0.109,  0.934,  1.016,  1.897,  1.986,  1.105,  0.926,  -0.056, 0.092,  0.993,  1.006,  -0.026, -0.031,
             0.984,  1.079,  1.027,  1.033,  0.225,  -0.012, 1.903,  1.710,  0.058,  0.004,  1.042,  1.208,  1.824,
             2.033,  -0.082, -0.052, 0.910,  0.996,  2.004,  2.134,  -0.082, 0.153,  -0.075, 0.136,  1.078,  1.035,
             2.013,  1.968,  0.174,  -0.006, -0.072, 0.002,  1.147,  0.961,  1.952,  1.947,  0.051,  0.048,  -0.087,
             -0.127, 2.076,  2.166,  1.932,  1.960,  -0.015, -0.056, 0.997,  0.833,  1.049,  0.886,  2.059,  2.220,
             1.081,  0.999,  1.032,  1.001,  0.892,  1.078,  1.929,  1.969,  0.810,  1.012,  1.287,  0.992,  2.045,
             2.056,  -0.009, -0.049, -0.136, 0.046,  0.934,  1.020,  0.973,  0.963,  0.118,  -0.025, 1.982,  1.971,
             1.959,  1.964,  -0.014, 0.017,  0.017,  0.122,  0.916,  1.063,  -0.020, -0.087, 0.020,  -0.145, 0.992,
             0.957,  1.990,  1.909,  0.959,  1.013,  -0.008, 0.038,  0.985,  0.986,  0.042,  -0.013, 0.096,  -0.013,
             0.069,  -0.137, 0.115,  0.119,  2.118,  2.105,  0.952,  1.005,  2.065,  1.884,  -0.056, -0.111, 0.104,
             -0.026, 1.901,  1.851,  -0.128, 0.003,  1.961,  1.941,  1.086,  0.951,  1.933,  1.960,  2.026,  2.178,
             0.889,  0.845,  1.153,  0.894,  0.106,  0.039,  0.993,  1.106,  1.016,  1.036,  -0.248, -0.030, 1.865,
             1.974,  1.139,  1.005,  1.077,  1.001,  0.005,  0.074,  2.019,  2.069,  -0.032, 0.007,  0.692,  1.134,
             1.909,  1.964,  2.194,  1.919,  2.095,  1.964,  1.114,  1.074,  0.936,  1.066,  0.996,  0.968,  -0.107,
             0.010,  0.135,  0.099,  0.046,  -0.128, 1.076,  0.962,  -0.164, 0.044,  0.787,  1.078,  0.112,  -0.110,
             0.046,  0.053,  0.747,  0.848,  0.021,  0.099,  0.146,  0.042,  1.844,  1.819,  1.066,  0.987,  0.855,
             0.922,  -0.158, -0.033, 2.008,  1.980,  1.029,  0.987,  -0.017, 0.057,  2.045,  1.996,  2.129,  1.986,
             -0.046, 0.076,  1.030,  0.959,  -0.002, -0.208, 1.028,  1.076,  -0.021, 0.059,  2.039,  2.056,  0.991,
             1.002,  2.109,  2.080,  1.120,  0.834,  1.956,  2.128,  1.913,  1.988,  1.932,  1.974,  1.041,  0.975,
             1.210,  0.872,  -0.003, -0.159, 1.926,  2.076,  2.127,  2.007,  -0.010, 0.234,  2.094,  1.938,  -0.183,
             -0.136, 1.989,  2.211,  -0.008, 0.033,  1.946,  1.911,  2.215,  1.822,  1.917,  1.968,  2.077,  2.110,
             2.190,  1.934,  1.009,  0.995,  0.042,  0.028,  1.977,  2.179,  1.018,  1.014},
            2, mpiContext, useFaultTolerance, REPLICATION_LEVEL, BLOCKS_PER_PERMUTATION_RANGE);

        kMeansInstance.setCenters({2.045, 1.987, 1.589, 0.764, -0.12, 0.43});
        kMeansInstance.performIterations(10);
        ASSERT_THAT(
            kMeansInstance.pointToCenterAssignment().assignedCenter,
            ElementsAre(
                0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 0, 2, 0, 1, 1, 0, 0, 2,
                1, 1, 0, 1, 2, 2, 2, 0, 0, 1, 0, 2, 0, 2, 2, 1, 0, 0, 2, 0, 0, 0, 2, 2, 1, 0, 1, 2, 1, 2, 1, 1, 2, 0, 2,
                1, 0, 2, 1, 0, 2, 2, 1, 0, 2, 2, 1, 0, 2, 2, 0, 0, 2, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 2, 2, 1, 1, 2, 0, 0,
                2, 2, 1, 2, 2, 1, 0, 1, 2, 1, 2, 2, 2, 2, 0, 1, 0, 2, 2, 0, 2, 0, 1, 0, 0, 1, 1, 2, 1, 1, 2, 0, 1, 1, 2,
                0, 2, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, 2, 2, 0, 1, 1, 2, 0, 1, 2, 0, 0, 2, 1, 2, 1, 2, 0,
                1, 0, 1, 0, 0, 0, 1, 1, 2, 0, 0, 2, 0, 2, 0, 2, 0, 0, 0, 0, 0, 1, 2, 0, 1));
    }
}

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
