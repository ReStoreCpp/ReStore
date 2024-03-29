
# Keeps CACHE cleaner
mark_as_advanced(
    BUILD_GMOCK BUILD_GTEST BUILD_SHARED_LIBS
    gmock_build_tests gtest_build_samples gtest_build_tests
    gtest_disable_pthreads gtest_force_shared_crt gtest_hide_internal_symbols
)

# Keep IDE clean
set_target_properties(gtest PROPERTIES FOLDER extern)
set_target_properties(gtest_main PROPERTIES FOLDER extern)
set_target_properties(gmock PROPERTIES FOLDER extern)
set_target_properties(gmock_main PROPERTIES FOLDER extern)

# Get the testname from the name of the .cpp file
macro(restore_get_test_name CPPFILE OUTVAR)
    get_filename_component(${OUTVAR} "${CPPFILE}" NAME_WE)
endmacro()

# create an exectuable in which the tests will be stored
function(restore_add_test_executable CPPFILE TESTNAME_OUTVAR)
    restore_get_test_name(${CPPFILE} ${TESTNAME_OUTVAR})

    set(TESTNAME ${${TESTNAME_OUTVAR}})
    if(${PROJECT_NAME}_BACKWARD_ENABLED)
        add_executable(${TESTNAME} ${CPPFILE} "${${PROJECT_NAME}_BACKWARDCPP_DIR}/backward.cpp")
        target_link_libraries(${TESTNAME} Backward::Backward)
    else()
        add_executable(${TESTNAME} ${CPPFILE})
    endif()

    set(${TESTNAME_OUTVAR} ${TESTNAME} PARENT_SCOPE)
endfunction()

# Link in the google test and google mock libararies
macro(restore_enable_gtest TARGETNAME DISCOVER_TESTS)
    # link the Google test infrastructure, mocking library, and a default main fuction to
    # the test executable.  Remove gtest_main if writing your own main function.
    target_link_libraries(${TARGETNAME} gtest gmock gtest_main)

    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    if (${DISCOVER_TESTS})
        gtest_discover_tests(${TARGETNAME}
            # set a working directory so your project root so that you can find test data via paths relative to the project root
            WORKING_DIRECTORY ${PROJECT_DIR}
            PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
        )
    endif()
endmacro()

function(restore_add_test CPPFILE LIBRARIES)
    restore_add_test_executable(${CPPFILE} TESTNAME)

    restore_enable_gtest(${TESTNAME} ON)
    target_link_libraries(${TESTNAME} ${LIBRARIES})

    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
    target_compile_options(${TESTNAME} PRIVATE -g -O0 ${WARNING_FLAGS})
endfunction()

function(restore_add_mpi_test CPPFILE DISPLAY_NAME NO_MPI_PROC LIBRARIES)
    restore_add_test_executable(${CPPFILE} TESTNAME)

    restore_enable_gtest(${TESTNAME} OFF)
    target_link_libraries(${TESTNAME} ${MPI_LIBRARIES} ${LIBRARIES})

    target_include_directories(${TESTNAME} SYSTEM PRIVATE ../extern/gtest-mpi-listener/include/)
    target_include_directories(${TESTNAME} PRIVATE .)
    # target_include_directories(${NAME} SYSTEM PRIVATE ../include/)

    # ULFM does not pass along the return code, so we need to use a wrapper here.
    add_custom_command( # copy over the wrapper
        TARGET ${TESTNAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${PROJECT_SOURCE_DIR}/tests/mpi_test_wrapper.sh
                ${CMAKE_CURRENT_BINARY_DIR}/mpi_test_wrapper.sh
    )
    add_test( # Register the custom test
        NAME ${DISPLAY_NAME}
        COMMAND "./mpi_test_wrapper.sh" ${MPIEXEC_EXECUTABLE} ${NO_MPI_PROC} "./${TESTNAME}"
    )

    set_target_properties(${TESTNAME} PROPERTIES FOLDER mpi_tests)
    target_compile_options(${TESTNAME} PRIVATE -g -O0 ${WARNING_FLAGS})
endfunction()

# Load the backward-cpp library
set(Backward_DIR "${PROJECT_SOURCE_DIR}/extern/backward-cpp")
find_package(Backward)

# Register our tests
## Without MPI
restore_add_test(test_block_distribution.cpp "ReStore::ReStore;cppitertools")
restore_add_test(test_serialization.cpp "ReStore::ReStore;cppitertools")
restore_add_test(test_block_submission.cpp "ReStore::ReStore;cppitertools")
restore_add_test(test_block_retrieval.cpp "ReStore::ReStore")
restore_add_test(test_helpers.cpp "ReStore::ReStore;cppitertools")
restore_add_test(test_equal_load_balancer.cpp "ReStore::ReStore")
restore_add_test(test_two_phase_commit.cpp "ReStore::ReStore")
restore_add_test(test_pseudo_random_permutation.cpp "ReStore::ReStore;cppitertools")

## Single-rank MPI tests
restore_add_mpi_test(
    mpi_tests_failures/test_restore_constructor.cpp TestRestoreConstructor
    4 "ReStore::ReStore;cppitertools"
)

## Multi-rank MPI tests
restore_add_mpi_test(
    mpi_tests_failures/test_restore_end_to_end_simple1_push_blocks.cpp TestRestore_EndToEnd_Simple1_PushBlocks
    4 "ReStore::ReStore;cppitertools"
)

restore_add_mpi_test(
  mpi_tests_failures/test_restore_end_to_end_simple1_pull_blocks.cpp TestRestore_EndToEnd_Simple1_PullBlocks
  4 "ReStore::ReStore;cppitertools"
  )

restore_add_mpi_test(
    mpi_tests_failures/test_restore_end_to_end_simple2_push_blocks.cpp TestRestore_EndToEnd_Simple2_PushBlocks
    4 "ReStore::ReStore;cppitertools"
)

restore_add_mpi_test(
    mpi_tests_failures/test_restore_end_to_end_asynchronous_submit.cpp TestRestore_EndToEnd_Asynchronous_Submit
    4 "ReStore::ReStore;cppitertools"
)

restore_add_mpi_test(
  mpi_tests_failures/test_restore_end_to_end_simple2_pull_blocks.cpp TestRestore_EndToEnd_Simple2_PullBlocks
  4 "ReStore::ReStore;cppitertools"
  )

restore_add_mpi_test(
    mpi_tests_failures/test_restore_end_to_end_single_failure.cpp TestRestore_EndToEnd_SingleFailure
    4 "ReStore::ReStore;cppitertools"
)

restore_add_mpi_test(
    mpi_tests_failures/test_restore_end_to_end_two_failures_push_blocks.cpp TestRestore_EndToEnd_TwoFailures_PushBlocks
    4 "ReStore::ReStore;cppitertools"
)

restore_add_mpi_test(
  mpi_tests_failures/test_restore_end_to_end_two_failures_pull_blocks.cpp TestRestore_EndToEnd_TwoFailures_PullBlocks
  4 "ReStore::ReStore;cppitertools"
  )

restore_add_mpi_test(
    mpi_tests_failures/test_restore_end_to_end_complex_data_type.cpp TestRestore_EndToEnd_ComplextDataType
    4 "ReStore::ReStore;cppitertools"
)

restore_add_mpi_test(
    mpi_tests_failures/test_restore_end_to_end_irrecoverable_data_loss.cpp TestRestore_EndToEnd_IrrecoverableDataLoss
    4 "ReStore::ReStore;cppitertools"
)

restore_add_mpi_test(
    mpi_tests_failures/test_restore_end_to_end_proxy_blocktype.cpp TestRestore_EndToEnd_ProxyBlockType
    4 "ReStore::ReStore"
)

if (NOT ${PROJECT_NAME}_ID_RANDOMIZATION)
restore_add_mpi_test(
    test_restore_end_to_end_already_serialized_data.cpp TestRestore_EndToEnd_AlreadySerializedData
    4 "ReStore::ReStore"
)
endif()

restore_add_mpi_test(
    test_mpi_context.cpp MPIContext
    4 "ReStore::ReStore"
)

restore_add_mpi_test(
    mpi_tests_failures/test_restore_vector.cpp TestRestore_ReStoreVector
    4 "ReStore::ReStore"
)

## Test suite of the benchmarks
restore_add_mpi_test(
    test_kmeans_single_rank.cpp "kmeansSingleRank"
    1 "ReStore::ReStore;ReStore::kmeans"
)

restore_add_mpi_test(
    test_kmeans_multi_rank.cpp "kmeansMultiRank"
    4 "ReStore::ReStore;ReStore::kmeans"
)

if (${PROJECT_NAME}_SIMULATED_FAILURES)
    restore_add_mpi_test(
        mpi_tests_failures/test_kmeans_update_centers_throws_on_failure.cpp "TestKMeans_updateCenters_throwsOnDFailure"
        4 "ReStore::ReStore;ReStore::kmeans"
    )

    restore_add_mpi_test(
        mpi_tests_failures/test_kmeans_end_to_end_single_failure.cpp "TestKMeans_EndToEnd_SingleFailure"
        4 "ReStore::ReStore;ReStore::kmeans"
    )

    restore_add_mpi_test(
        mpi_tests_failures/test_kmeans_end_to_end_single_failure_2.cpp "TestKMeans_EndToEnd_SingleFailure_2"
        4 "ReStore::ReStore;ReStore::kmeans"
    )

    restore_add_mpi_test(
        mpi_tests_failures/test_kmeans_end_to_end_multi_failure.cpp "TestKMeans_EndToEnd_MultiFailure"
        4 "ReStore::ReStore;ReStore::kmeans"
    )
endif()

# PageRank Tests
set(PAGERANK "pageRankForTest")
add_executable(${PAGERANK} ../benchmark/pageRank.cpp)
target_include_directories(${PAGERANK} SYSTEM PRIVATE ../extern/)
target_include_directories(${PAGERANK} PRIVATE ../include/)
target_link_libraries(${PAGERANK} gtest gmock "ReStore::ReStore;cppitertools;cxxopts")

target_compile_options(${PAGERANK} PRIVATE -g -O0 ${WARNING_FLAGS})

configure_file( ${PROJECT_SOURCE_DIR}/tests/pageRankTestGraph/pageRankTestGraph_0 ${CMAKE_CURRENT_BINARY_DIR}/pageRankTestGraph/pageRankTestGraph_0 COPYONLY)
configure_file( ${PROJECT_SOURCE_DIR}/tests/pageRankTestGraph/pageRankTestGraph_1 ${CMAKE_CURRENT_BINARY_DIR}/pageRankTestGraph/pageRankTestGraph_1 COPYONLY)
configure_file( ${PROJECT_SOURCE_DIR}/tests/pageRankTestGraph/pageRankTestGraph_2 ${CMAKE_CURRENT_BINARY_DIR}/pageRankTestGraph/pageRankTestGraph_2 COPYONLY)
configure_file( ${PROJECT_SOURCE_DIR}/tests/pageRankTestGraph/pageRankTestGraph_3 ${CMAKE_CURRENT_BINARY_DIR}/pageRankTestGraph/pageRankTestGraph_3 COPYONLY)

configure_file(${PROJECT_SOURCE_DIR}/benchmark/pageRankNetworKit.py ${CMAKE_CURRENT_BINARY_DIR}/pageRankNetworKit.py COPYONLY)

add_test( # Register the custom test
  NAME TestPageRank_Full
  COMMAND python3 pageRankNetworKit.py pageRankTestGraph --test -e ./pageRankForTest -m ${MPIEXEC_EXECUTABLE}
  )

# Make ctest play nicely with Demian's spacemacs
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --force-new-ctest-process --output-on-failure)
