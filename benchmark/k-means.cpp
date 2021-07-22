#include <array>
#include <cstdint>
#include <mpi.h>
#include <random>
#include <vector>

#include "k-means.hpp"
#include "restore/helpers.hpp"
#include "restore/mpi_context.hpp"

int main(int argc, char** argv) {
    using namespace kmeans;

    UNUSED(argc);
    UNUSED(argv);

    MPI_Init(&argc, &argv);

    ReStoreMPI::MPIContext mpiContext(MPI_COMM_WORLD);

    const size_t numDataPoints = 10000;
    const size_t numCenters    = 10;
    const size_t numIterations = 10;
    const size_t numDimensions = 2;

    auto kmeansInstance = kmeans::kMeansAlgorithm<float, ReStoreMPI::MPIContext>(
        kmeans::generateRandomData<float>(numDataPoints, numDimensions), mpiContext);

    kmeansInstance.pickCentersRandomly(numCenters);
    kmeansInstance.performIterations(numIterations);

    MPI_Finalize();
    return 0;
}