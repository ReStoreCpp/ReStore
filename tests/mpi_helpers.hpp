#include <cstdlib>
#include <mpi.h>
#include <signal.h>

constexpr int EXIT_SIMULATED_FAILURE = 42;

int myRankId() {
    int rankId;
    MPI_Comm_rank(MPI_COMM_WORLD, &rankId);
    return rankId;
}

int numRanks() {
    int numRanks;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    return numRanks;
}

void failRank(int rankId) {
    if (myRankId() == rankId) {
        //raise(SIGKILL);
        exit(EXIT_SIMULATED_FAILURE);
    }
}
