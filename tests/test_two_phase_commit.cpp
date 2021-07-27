#include <cassert>
#include <cstddef>
#include <cstdint>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <utility>

#include "restore/two_phase_commit.hpp"

using namespace ::testing;

TEST(TwoPhaseCommit, Basic) {
    TwoPhaseCommit<int> intObj(0);

    // No changes yet
    ASSERT_EQ(*intObj, 0);
    ASSERT_EQ(intObj.activeCopy(), 0);
    ASSERT_EQ(intObj.backupCopy(), 0);

    // Start commiting a change
    *intObj = 1;
    ASSERT_EQ(*intObj, 1);
    ASSERT_EQ(intObj.activeCopy(), 1);
    ASSERT_EQ(intObj.backupCopy(), 0);

    // Commit the change
    intObj.commit();
    ASSERT_EQ(*intObj, 1);
    ASSERT_EQ(intObj.activeCopy(), 1);
    ASSERT_EQ(intObj.backupCopy(), 1);

    // Commit another change
    *intObj = 2;
    ASSERT_EQ(*intObj, 2);
    ASSERT_EQ(intObj.activeCopy(), 2);
    ASSERT_EQ(intObj.backupCopy(), 1);

    intObj.commit();
    ASSERT_EQ(*intObj, 2);
    ASSERT_EQ(intObj.activeCopy(), 2);
    ASSERT_EQ(intObj.backupCopy(), 2);

    // Abort commiting a change
    *intObj = 3;
    ASSERT_EQ(*intObj, 3);
    ASSERT_EQ(intObj.activeCopy(), 3);
    ASSERT_EQ(intObj.backupCopy(), 2);

    intObj.rollback();
    ASSERT_EQ(*intObj, 2);
    ASSERT_EQ(intObj.activeCopy(), 2);
    ASSERT_EQ(intObj.backupCopy(), 2);
}

TEST(TwoPhaseCommit, ComplexDataType) {
    TwoPhaseCommit<std::vector<int>> vec (std::initializer_list<int>{0, 1, 2});

    // No changes yet
    ASSERT_THAT(*vec, ElementsAre(0, 1, 2));
    ASSERT_THAT(vec.activeCopy(), ElementsAre(0, 1, 2));
    ASSERT_THAT(vec.backupCopy(), ElementsAre(0, 1, 2));

    // Start commiting a change
    vec->push_back(3);
    ASSERT_THAT(*vec, ElementsAre(0, 1, 2, 3));
    ASSERT_THAT(vec.activeCopy(), ElementsAre(0, 1, 2, 3));
    ASSERT_THAT(vec.backupCopy(), ElementsAre(0, 1, 2));

    vec.commit();
    ASSERT_THAT(*vec, ElementsAre(0, 1, 2, 3));
    ASSERT_THAT(vec.activeCopy(), ElementsAre(0, 1, 2, 3));
    ASSERT_THAT(vec.backupCopy(), ElementsAre(0, 1, 2, 3));

    // Abort a commit
    vec->pop_back();
    vec->pop_back();

    ASSERT_THAT(*vec, ElementsAre(0, 1));
    ASSERT_THAT(vec.activeCopy(), ElementsAre(0, 1));
    ASSERT_THAT(vec.backupCopy(), ElementsAre(0, 1, 2, 3));

    vec.rollback();
    ASSERT_THAT(*vec, ElementsAre(0, 1, 2, 3));
    ASSERT_THAT(vec.activeCopy(), ElementsAre(0, 1, 2, 3));
    ASSERT_THAT(vec.backupCopy(), ElementsAre(0, 1, 2, 3));
}