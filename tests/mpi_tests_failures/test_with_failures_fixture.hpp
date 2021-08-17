#include "../mpi_helpers.hpp"
#include <gtest/gtest.h>

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

class ReStoreVectorTest : public ::testing::Test {
    protected:
    RankFailureManager _rankFailureManager;

    ReStoreVectorTest () : _rankFailureManager(MPI_COMM_WORLD) {}

    virtual ~ReStoreVectorTest() override {}

    virtual void SetUp() override {}

    virtual void TearDown() override {
        _rankFailureManager.endOfTestcase();
    }
};

class kMeansTestWithFailures : public ::testing::Test {
    protected:
    RankFailureManager _rankFailureManager;

    kMeansTestWithFailures () : _rankFailureManager(MPI_COMM_WORLD) {}

    virtual ~kMeansTestWithFailures() override {}

    virtual void SetUp() override {}

    virtual void TearDown() override {
        _rankFailureManager.endOfTestcase();
    }
};