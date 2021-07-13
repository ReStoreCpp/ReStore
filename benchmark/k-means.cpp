#include <array>
#include <cstdint>
#include <mpi.h>
#include <random>
#include <vector>

#include <dbg.h>

#include "restore/helpers.hpp"
#include "k-means.hpp"

int main(int argc, char** argv) {
    using namespace kmeans;

    UNUSED(argc);
    UNUSED(argv);

    MPI_Init(&argc, &argv);

    const size_t numDataPoints = 10000;
    const size_t numCenters    = 10;
    const size_t numIterations = 10;
    const size_t numDimensions = 2;

    auto kmeansInstance = kmeans::kMeansAlgorithm<float>(
        kmeans::generateRandomData<float>(numDataPoints, numDimensions), numCenters, numIterations);
    kmeansInstance();

    dbg("done");
    MPI_Finalize();
    return 0;
}