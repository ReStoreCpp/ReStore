#pragma once

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <signal.h>
#include <vector>

#include "restore/mpi_context.hpp"

#ifdef USE_FTMPI
    #include <mpi-ext.h>
#elif !defined(SIMULATE_FAILURES)
    #error "If not using a fault-tolerant MPI implementation, you can use only simulated failures."
#endif

constexpr int EXIT_SIMULATED_FAILURE = 42;

// Returns a descriptive string associated with the given MPI error code.
std::string mpi_error_code_to_string(int errorCode) {
    int  errorClass;
    int  errorStringLength;
    int  errorClassStringLength;
    char errorString[BUFSIZ];
    char errorClassString[BUFSIZ];

    // Get the error string for this error.
    MPI_Error_string(errorCode, errorString, &errorStringLength);

    // Get the error class and error string for the error class.
    MPI_Error_class(errorCode, &errorClass);
    MPI_Error_string(errorClass, errorClassString, &errorClassStringLength);

    // Assemble the error message and return.
    return std::string(errorString) + " (" + std::string(errorClassString) + ")";
}

int myRankId(MPI_Comm comm = MPI_COMM_WORLD) {
    assert(comm != MPI_COMM_NULL);
    int rankId;
    if (MPI_Comm_rank(comm, &rankId) != MPI_SUCCESS) {
        std::cerr << "Error in MPI_Comm_rank" << std::endl;
        MPI_Abort(comm, EXIT_FAILURE);
    }
    return rankId;
}

int numRanks(MPI_Comm comm = MPI_COMM_WORLD) {
    assert(comm != MPI_COMM_NULL);
    int numRanks;
    if (MPI_Comm_size(comm, &numRanks) != MPI_SUCCESS) {
        std::cerr << "Error in MPI_Comm_size" << std::endl;
        MPI_Abort(comm, EXIT_FAILURE);
    }
    return numRanks;
}

#define EXIT_IF_FAILED(FAILED) \
    if (FAILED) {              \
        return;                \
    }

class RankFailureManager {
    public:
    explicit RankFailureManager(MPI_Comm comm) noexcept : _comm(comm), _iFailed(false), _noMoreCollectives(false) {
        assert(comm != MPI_COMM_NULL);
    }

    // failRanks()
    //
    // Depending on SIMULATE_FAILURES, either simulates the failure of the ranks given in failedRanks or calls exit() on
    // them.
    MPI_Comm failRanks(std::vector<ReStoreMPI::original_rank_t> failedRanks) {
        assert(!iFailed());
        _iFailed = std::find(failedRanks.begin(), failedRanks.end(), myRankId(_comm)) != failedRanks.end();

#ifdef SIMULATE_FAILURES
        return simulateFailure(_iFailed);
#else
        if (_iFailed) {
            // raise(SIGKILL);
            exit(EXIT_SIMULATED_FAILURE);
        }
        return repairCommunicator();
#endif
    }

    // everyoneStillRunning()
    //
    // Reports if this rank is still running (answering to collectives) and checks if everyone else is, too. If someone
    // reports to not answer collectives from now on, we should not rely on them to do so. Depending on where we are in
    // our testcase (in the middle? at the end?) this can mean that an ASSERT_* failed on a remote rank or simply that
    // we are done with this testcase.
    // We must call this before each collective operation and if it returns false, we should quit the testcase. We can
    // use the EXIT_IF_FAILED() macro for this (don't forget to invert the return value of this function).
    // This function will always return true if SIMULATE_FAILURES is set to false.
    bool everyoneStillRunning(bool running = true) {
#ifdef SIMULATE_FAILURES
        // If we detect that a remote rank is no longer running and therefore exit our testcase, the test fixture's
        // TearDown() will call endOfTestcase() which will then call us. We therefore have to respect
        // _noMoreCollectives.
        if (!_noMoreCollectives) {
            bool everyoneStillRunning;
            MPI_Barrier(_comm);
            MPI_Reduce(&running, &everyoneStillRunning, 1, MPI_C_BOOL, MPI_LAND, 0, _comm);
            MPI_Bcast(&everyoneStillRunning, 1, MPI_C_BOOL, 0, _comm);
            // MPI_Allreduce(MPI_IN_PLACE, &running, 1, MPI_C_BOOL, MPI_LAND, _comm);
            if (!everyoneStillRunning) {
                // Someone stopped running, we should not expect them to participate in collective operations.
                _noMoreCollectives = true;
            }
            return running;
        } else {
            return false;
        }
#else
        UNUSED(running);
        return true;
#endif
    }

    // Signals that this ranks test case ended. Either because it ran until the end or because some ASSERT_* failed.
    // Either way, this rank is no longer answering collectives.
    // If SIMULATE_FAILURES is set to false, this does nothing.
    void endOfTestcase() {
        // In the case that this rank simulated a failure, it has already been split off into another communicator. As a
        // result, nobody expect this rank to participate in any collective operation and it can therefore silently
        // exit.
#ifdef SIMULATE_FAILURES
        if (!iFailed()) {
            everyoneStillRunning(false);
        }
#endif
    }

    // iFailed()
    //
    // Return true if this rank simulated its own failure.
    bool iFailed() const noexcept {
        return _iFailed;
    }

    // resetCommunicator()
    //
    // Reset the communicator to the one given as comm. We can use this if there was a failure somewhere else in the
    // program.
    void resetCommunicator(MPI_Comm comm) noexcept {
        assert(comm != MPI_COMM_NULL);
        _comm = comm;
    }

    private:
#ifdef SIMULATE_FAILURES
    // Simulates the failure of this rank if iFailed is set. This is done by splitting the current communictor into
    // alive and failed ranks. The failed ranks must then exit their test case.
    MPI_Comm simulateFailure(bool iFailed) {
        assert(!_noMoreCollectives);
        assert(_comm != MPI_COMM_NULL);
        MPI_Comm newComm = MPI_COMM_NULL;

        // Create new communicators for the alive and failed ranks.
        auto result = MPI_Comm_split(
            _comm,
            iFailed,    // color
            myRankId(), // key
            &newComm);
        if (result != MPI_SUCCESS) {
            std::cerr << "MPI_Comm_split failed: " << mpi_error_code_to_string(result) << std::endl;
            MPI_Abort(_comm, EXIT_FAILURE);
        } else if (newComm == MPI_COMM_NULL) {
            std::cerr << "MPI_Comm_split returned MPI_COMM_NULL" << std::endl;
            MPI_Abort(_comm, EXIT_FAILURE);
        }

        // Free the previous communicator.
        // result = MPI_Comm_free(&_comm);
        // if (result != MPI_SUCCESS) {
        //     std::cerr << "MPI_Comm_free failed: " << mpi_error_code_to_string(result) << std::endl;
        //     MPI_Abort(_comm, EXIT_FAILURE);
        // }
        // assert(_comm == MPI_COMM_NULL);

        // Switch to the new communicator.
        _comm = newComm;
        assert(_comm != MPI_COMM_NULL);

        return newComm;
    }
#endif

#ifndef SIMULATE_FAILURES
    // repairCommunicator()
    //
    // If there was a (non simulated) rank failure we can use this to repair the MPI communicator. It will return
    // the new communicator. Do not use this when simulating failures.
    MPI_Comm repairCommunicator() {
        int rc, ec;
        rc = MPI_Barrier(_comm);
        MPI_Error_class(rc, &ec);

        assert((ec == MPI_ERR_PROC_FAILED || ec == MPI_ERR_REVOKED));
        if (ec == MPI_ERR_PROC_FAILED) {
            MPIX_Comm_revoke(_comm);
        }

        // Build a new communicator without the failed ranks
        MPI_Comm newComm;
        rc = MPIX_Comm_shrink(_comm, &newComm);
        assert(MPI_SUCCESS == rc);
        // As for the ULFM documentation, freeing the communicator is recommended but will probably
        // not succeed. This is why we do not check for an error here.
        // I checked that --mca mpi_show_handle_leaks 1 does not show a leaked handle
        MPI_Comm_free(&_comm);
        _comm = newComm;
        return newComm;
    }
#endif

    MPI_Comm _comm;
    bool     _iFailed;
    bool     _noMoreCollectives;
};
