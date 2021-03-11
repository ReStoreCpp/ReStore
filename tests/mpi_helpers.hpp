#include <cstdlib>
#include <gtest/gtest.h>
#include <mpi.h>
#include <signal.h>

#include <mpi-ext.h>

constexpr int EXIT_SIMULATED_FAILURE = 42;

int myRankId(MPI_Comm _comm = MPI_COMM_WORLD) {
    int rankId;
    MPI_Comm_rank(_comm, &rankId);
    return rankId;
}

int numRanks(MPI_Comm _comm = MPI_COMM_WORLD) {
    int numRanks;
    MPI_Comm_size(_comm, &numRanks);
    return numRanks;
}

void failRank(int rankId) {
    if (myRankId() == rankId) {
        // raise(SIGKILL);
        exit(EXIT_SIMULATED_FAILURE);
    }
}

inline MPI_Comm getFixedCommunicator(MPI_Comm _comm = MPI_COMM_WORLD) {
    int rc, ec;
    rc = MPI_Barrier(_comm);
    MPI_Error_class(rc, &ec);

    EXPECT_TRUE((ec == MPI_ERR_PROC_FAILED || ec == MPI_ERR_REVOKED));
    if (ec == MPI_ERR_PROC_FAILED) {
        MPIX_Comm_revoke(_comm);
    }

    // Build a new communicator without the failed ranks
    MPI_Comm newComm;
    rc = MPIX_Comm_shrink(_comm, &newComm);
    EXPECT_EQ(MPI_SUCCESS, rc);
    // As for the ULFM documentation, freeing the communicator is recommended but will probably
    // not succeed. This is why we do not check for an error here.
    // I checked that --mca mpi_show_handle_leaks 1 does not show a leaked handle
    MPI_Comm_free(&_comm);
    return newComm;
}
