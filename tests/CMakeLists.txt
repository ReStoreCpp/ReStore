
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

macro(restore_add_test TESTNAME LIBRARIES)
    # create an exectuable in which the tests will be stored
    if(${PROJECT_NAME}_BACKWARD_ENABLED)
        add_executable(${TESTNAME} ${ARGN} "${${PROJECT_NAME}_BACKWARDCPP_DIR}/backward.cpp")
    else()
        add_executable(${TESTNAME} ${ARGN})
    endif()
    # link the Google test infrastructure, mocking library, and a default main fuction to
    # the test executable.  Remove g_test_main if writing your own main function.
    target_link_libraries(${TESTNAME} gtest gmock gtest_main Backward::Backward ${MPI_LIBRARIES} ${LIBRARIES})
    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TESTNAME}
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
    )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
    target_compile_options(${TESTNAME} PRIVATE -g -O0 ${WARNING_FLAGS})
endmacro()

function(add_mpi_test FILE NAME DISPLAY_NAME NO_MPI_PROC LIBRARIES)
    add_executable(${NAME} ${FILE})
    target_include_directories(${NAME} SYSTEM PRIVATE ../extern/)
    target_include_directories(${NAME} PRIVATE ../include/)
    target_include_directories(${NAME} PRIVATE .)
    target_link_libraries(${NAME} gtest gmock Backward::Backward ${MPI_LIBRARIES} ${LIBRARIES})

    add_custom_command(
        TARGET ${NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_SOURCE_DIR}/tests/mpi_test_wrapper.sh
                ${CMAKE_CURRENT_BINARY_DIR}/mpi_test_wrapper.sh
    )

    add_test(NAME ${DISPLAY_NAME} COMMAND "./mpi_test_wrapper.sh" ${MPIEXEC_EXECUTABLE} ${NO_MPI_PROC} "./${NAME}")
    target_compile_options(${NAME} PRIVATE -g -Og ${WARNING_FLAGS})
    #set_tests_properties(${DISPLAY_NAME} PROPERTIES
    #    FAIL_REGULAR_EXPRESSION " FAILED "
    #    PASS_REGULAR_EXPRESSION " PASSED "
    #)
endfunction(add_mpi_test)

# Register our tests
## Without MPI
restore_add_test(test_block_distribution "ReStore::ReStore;cppitertools" test_block_distribution.cpp)
restore_add_test(test_serialization "ReStore::ReStore;cppitertools" test_serialization.cpp)
restore_add_test(test_block_submission "ReStore::ReStore;cppitertools" test_block_submission.cpp)
restore_add_test(test_block_retrieval "ReStore::ReStore;" test_block_retrieval.cpp)
restore_add_test(test_helpers "ReStore::ReStore;cppitertools" test_helpers.cpp)
restore_add_test(test_equal_load_balancer "ReStore::ReStore" test_equal_load_balancer.cpp)
add_mpi_test(mpi_tests_failures/test_restore_constructor.cpp test_restore_constructor TestRestoreConstructor 4 "ReStore::ReStore;cppitertools")

## With MPI
add_compile_definitions(SIMULATE_FAILURES=$<BOOL:${SIMULATE_FAILURES}>)
add_compile_definitions(USE_FTMPI=$<BOOL:${USE_FTMPI}>)
add_mpi_test(mpi_tests_failures/test_restore_end_to_end_simple1.cpp test_restore_end_to_end_simple1 TestRestore_EndToEnd_Simple1 4 "ReStore::ReStore;cppitertools")
add_mpi_test(mpi_tests_failures/test_restore_end_to_end_simple2.cpp test_restore_end_to_end_simple2 TestRestore_EndToEnd_Simple2 4 "ReStore::ReStore;cppitertools")
add_mpi_test(mpi_tests_failures/test_restore_end_to_end_single_failure.cpp test_restore_end_to_end_single_failure TestRestore_EndToEnd_SingleFailure 4 "ReStore::ReStore;cppitertools")
add_mpi_test(mpi_tests_failures/test_restore_end_to_end_two_failures.cpp test_restore_end_to_end_two_failures TestRestore_EndToEnd_TwoFailures 4 "ReStore::ReStore;cppitertools")
add_mpi_test(mpi_tests_failures/test_restore_end_to_end_complex_data_type.cpp test_restore_end_to_end_complex_data_type TestRestore_EndToEnd_ComplextDataType 4 "ReStore::ReStore;cppitertools")
add_mpi_test(mpi_tests_failures/test_restore_end_to_end_irrecoverable_data_loss.cpp test_restore_end_to_end_irrecoverable_data_loss TestRestore_EndToEnd_IrrecoverableDataLoss 4 "ReStore::ReStore;cppitertools")
add_mpi_test(test_mpi_context.cpp test_mpi_context MPIContext 4 "")

# Make ctest play nicely with Demian's spacemacs
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --force-new-ctest-process --output-on-failure)
