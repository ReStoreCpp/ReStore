
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
    add_executable(${TESTNAME} ${ARGN})
    # link the Google test infrastructure, mocking library, and a default main fuction to
    # the test executable.  Remove g_test_main if writing your own main function.
    target_link_libraries(${TESTNAME} gtest gmock gtest_main ${MPI_LIBRARIES} ${LIBRARIES})
    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TESTNAME}
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
    )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
    target_compile_options(${TESTNAME} PRIVATE -g -O0 -Wall -Wextra -Wcast-align -Wconversion -Wnoexcept -Wnon-virtual-dtor -Wnull-dereference -Woverloaded-virtual -Wpedantic -Wsuggest-attribute=const -Wsuggest-attribute=noreturn -Wshadow -Wsign-conversion -Wsuggest-override -Wundef -Wunreachable-code -Wunused)
endmacro()

set(SIM_TILL_DATA_LOSS_TARGET "simulate-failures-until-data-loss")
add_executable(${SIM_TILL_DATA_LOSS_TARGET} simulate_failures_until_data_loss.cpp)
target_include_directories(${SIM_TILL_DATA_LOSS_TARGET} SYSTEM PRIVATE ../extern/)
target_include_directories(${SIM_TILL_DATA_LOSS_TARGET} SYSTEM PRIVATE ../include/)
target_link_libraries(${SIM_TILL_DATA_LOSS_TARGET} gtest gmock "ReStore::ReStore;cppitertools;cxxopts")
target_compile_options(${SIM_TILL_DATA_LOSS_TARGET} PRIVATE -O2 -Wall -Wextra -Wcast-align -Wconversion -Wnoexcept -Wnon-virtual-dtor -Wnull-dereference -Woverloaded-virtual -Wpedantic -Wsuggest-attribute=const -Wsuggest-attribute=noreturn -Wshadow -Wsign-conversion -Wsuggest-override -Wundef -Wunreachable-code -Wunused)
