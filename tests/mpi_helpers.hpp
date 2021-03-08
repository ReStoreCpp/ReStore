#include <mpi.h>
#include <signal.h>

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
        raise(SIGKILL);
    }
}
