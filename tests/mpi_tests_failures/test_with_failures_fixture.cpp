#include <gtest/gtest.h>
#include "../mpi_helpers.hpp"

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