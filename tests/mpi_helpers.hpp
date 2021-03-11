#include <cstdlib>
#include <mpi.h>
#include <restore/mpi_context.hpp>
#include <signal.h>

#include <gtest/gtest.h>
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

#define EXIT_IF_FAILED(FAILED) \
    if (FAILED) {              \
        return;                \
    }

class RankFailureManager {
    public:
    RankFailureManager(MPI_Comm comm) noexcept : _comm(comm), _iFailed(false), _noMoreCollectives(false) {}

    // failRanks()
    //
    // Depending on SIMULATE_FAILURES, either simulates the failure of the ranks given in failedRanks or calls exit() on
    // them.
    MPI_Comm failRanks(std::vector<ReStoreMPI::original_rank_t> failedRanks) {
        assert(!iFailed());
        _iFailed = std::find(failedRanks.begin(), failedRanks.end(), myRankId(_comm)) != failedRanks.end();
        std::cout << myRankId() << ": " << _iFailed << std::endl;

        if constexpr (SIMULATE_FAILURES) {
            return simulateFailure(_iFailed);
        } else {
            if (_iFailed) {
                // raise(SIGKILL);
                exit(EXIT_SIMULATED_FAILURE);
            }
            return repairCommunicator();
        }
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
        if (!SIMULATE_FAILURES) {
            return true;
        }
        // If we detect that a remote rank is no longer running and therefore exit our testcase, the test fixture's
        // TearDown() will call endOfTestcase() which will then call us. We therefore have to respect
        // _noMoreCollectives.
        if (!_noMoreCollectives) {
            MPI_Allreduce(&running, &running, 1, MPI_BYTE, MPI_LAND, _comm);
            if (!running) {
                // Someone stopped running, we should not expect them to participate in collective operations.
                _noMoreCollectives = true;
            }
            return running;
        } else {
            return false;
        }
    }

    // Signals that this ranks test case ended. Either because it ran until the end or because some ASSERT_* failed.
    // Either way, this rank is no longer answering collectives.
    // If SIMULATE_FAILURES is set to false, this does nothing.
    void endOfTestcase() {
        // In the case that this rank simulated a failure, it has already been split off into another communicator. As a
        // result, nobody expect this rank to participate in any collective operation and it can therefore silently
        // exit.
        if (SIMULATE_FAILURES && !iFailed()) {
            everyoneStillRunning(false);
        }
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
    // progam.
    void resetCommunicator(MPI_Comm comm) noexcept {
        _comm = comm;
    }

    private:
    // Simulates the failure of this rank if iFailed is set. This is done by splitting the current communictor into
    // alive and failed ranks. The failed ranks must then exit their test case.
    MPI_Comm simulateFailure(bool iFailed) {
        assert(!_noMoreCollectives);
        MPI_Comm newComm;
        MPI_Comm_split(
            _comm,
            iFailed,    // color 1
            myRankId(), // key
            &newComm);
        MPI_Comm_free(&_comm);

        _comm = newComm;
        return newComm;
    }

    // repairCommunicator()
    //
    // If there was a (non simulated) rank failure we can use this to repair the MPI communicator. It will return
    // the new communicator. Do not use this when simulating failures.
    MPI_Comm repairCommunicator() {
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
        _comm = newComm;
        return newComm;
    }

    MPI_Comm _comm;
    bool     _iFailed;
    bool     _noMoreCollectives;
};

class ReStoreTestWithFailures : public ::testing::Test {
    protected:
    RankFailureManager _rankFailureManager;

    ReStoreTestWithFailures() : _rankFailureManager(MPI_COMM_WORLD) {}

    virtual ~ReStoreTestWithFailures() override {}

    virtual void SetUp() override {}

    virtual void TearDown() override {
        _rankFailureManager.endOfTestcase();
    }
};
