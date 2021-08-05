cmake_minimum_required(VERSION 3.16)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
    set(IS_SUBPROJECT OFF)
else()
    set(IS_SUBPROJECT ON)
endif()

# Use ccache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    message("Found ccache; using it to speed up compilation.")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

# Project settings
project(ReStore VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
if(MPI_CXX_FOUND)
  include_directories(BEFORE SYSTEM ${MPI_CXX_INCLUDE_PATH})
  link_libraries(${MPI_LIBRARIES})
endif()

# Make additinal cmake modules finadble
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/extern/sanitizers-cmake/cmake")

# Require out-of-source builds
include(require_out_of_source_builds)
require_out_of_source_builds()

include(CMakeDependentOption)
#include(CMakePackageConfigHelpers)
include(CTest)
include(GNUInstallDirs)

# Load (memory, adress, ...) sanitizer support
find_package(Sanitizers)

# Enable sorting targets (of external libraries) into folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# Are we building in DEBUG mode?
if(CMAKE_BUILD_TYPE MATCHES Debug)
  message("!!!!! Building in DEBUG mode !!!!!")
  add_definitions(-D_GLIBCXX_DEBUG)
  add_definitions(-D_GLIBCXX_DEBUG_PEDANTIC)
endif(CMAKE_BUILD_TYPE MATCHES Debug)

option(${PROJECT_NAME}_INSTALL "Add ${PROJECT_NAME} to the install list" ON)

# Options to disable building the tests, examples, benchmarks, ...
option(${PROJECT_NAME}_BUILD_TESTS "Build unit tests" ON)
option(${PROJECT_NAME}_BUILD_BENCHMARKS "Build benchmarks" ON)
option(${PROJECT_NAME}_BUILD_EXAMPLES "Build examples" OFF)

# Other build options
option("SIMULATE_FAILURES" "Simulate node failues (for example when running unit test)." OFF)
option("USE_FTMPI" "Use a fault-tolerant MPI implementation" ON)
option("WARNINGS_ARE_ERRORS" "Treat warnings as errors." OFF)

# Output config
if(NOT SIMULATE_FAILURES AND NOT USE_FTMPI)
    message(FATAL_ERROR "You have to either use a fault-tolerant MPI implementation or simulated failures (or both).")
endif()

message(" ## Using simulated failures: ${SIMULATE_FAILURES}")
message(" ## Using a fault-tolerant MPI implementation: ${USE_FTMPI}")

# Populate some of the options to the source code
add_compile_definitions(SIMULATE_FAILURES=$<BOOL:${SIMULATE_FAILURES}>)
add_compile_definitions(USE_FTMPI=$<BOOL:${USE_FTMPI}>)

# Extra options when building tests
cmake_dependent_option("${PROJECT_NAME}_SYSTEM_GTEST" "Use googletest version installed on the system" OFF "${PROJECT_NAME}_BUILD_TESTS" OFF)

# Library specific settings
set("NEEDS_GOOGLETEST" ${PROJECT_NAME}_BUILD_TESTS OR ${PROJECT_NAME}_BUILD_BENCHMARKS)
set("NEEDS_CPPITERTOOLS" ${PROJECT_NAME}_BUILD_TESTS OR ${PROJECT_NAME}_BUILD_BENCHMARKS)
set("NEEDS_CXXOPTS" ${PROJECT_NAME}_BUILD_BENCHMARKS)
set("NEEDS_GOOGLEBENCHMARK" ${PROJECT_NAME}_BUILD_BENCHMARKS)
set("NEEDS_BACKWARDCPP" ${PROJECT_NAME}_BUILD_TESTS)

if(NEEDS_GOOGLETEST)
    message("Using googletest")
    set("${PROJECT_NAME}_GTEST_DIR" "${CMAKE_CURRENT_LIST_DIR}/extern/googletest" CACHE PATH "Path to the googletest source directory")
endif()

if(NEEDS_GOOGLEBENCHMARK)
    message("Using googlebenchmark")
    set("${PROJECT_NAME}_GBENCHMARK_DIR" "${CMAKE_CURRENT_LIST_DIR}/extern/googlebenchmark" CACHE PATH "Path to the googlebenchmark source directory")
endif()

if(NEEDS_CPPITERTOOLS)
    message("Using cppitertools")
    set("${PROJECT_NAME}_CPPITERTOOLS_DIR" "${CMAKE_CURRENT_LIST_DIR}/extern/cppitertools" CACHE PATH "Path to the cppitertools source directory")
endif()

if(NEEDS_CXXOPTS)
    message("Using cxxopts")
    set("${PROJECT_NAME}_CXXOPTS_DIR" "${CMAKE_CURRENT_LIST_DIR}/extern/cxxopts" CACHE PATH "Path to the cxxopts source directory")
endif()

if(NEEDS_BACKWARDCPP)
    message("Using backwardcpp")
    set("${PROJECT_NAME}_BACKWARDCPP_DIR" "${CMAKE_CURRENT_LIST_DIR}/extern/backward-cpp" CACHE PATH "Path to the backward-cpp source directory")
    option(${PROJECT_NAME}_BACKWARD_ENABLED "Enable pretty printing of stack traces when a test case fails." ON)
else()
    option(${PROJECT_NAME}_BACKWARD_ENABLED "Enable pretty printing of stack traces when a test case fails." OFF)
endif()

# Organize targets into folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# The target to be linked against by other targets. This library is an
# interface target and as such does not generate any artefacts. It rather sets
# include directories and required compiler flags.
add_library("${PROJECT_NAME}" INTERFACE)
target_sources("${PROJECT_NAME}" INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(
    "${PROJECT_NAME}" INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

### Compiler and linker settings ###
# This interface reflects the requirements to the compiler for building targets
# linking against.
add_library("${PROJECT_NAME}_compile_requirements" INTERFACE)
# We just want c++17 or above
target_compile_features(
    "${PROJECT_NAME}_compile_requirements" INTERFACE
    cxx_std_17
)
target_link_libraries("${PROJECT_NAME}" INTERFACE "${PROJECT_NAME}_compile_requirements")

# The namespace alias can be used as link target if this project is a
# subproject.
add_library("${PROJECT_NAME}::${PROJECT_NAME}" ALIAS "${PROJECT_NAME}")

# Not added as link target to avoid propagation of warning
# flags. Only to be used by internal targets that compile the library.
add_library("${PROJECT_NAME}_warnings" INTERFACE)

# TODO: Use CheckCXXCompilerFlag for this?
list(
    APPEND WARNING_FLAGS
    "-Wall"
    "-Wextra"
    "-Wconversion"
    "-Wnon-virtual-dtor"
    "-Woverloaded-virtual"
    "-Wshadow"
    "-Wsign-conversion"
    "-Wundef"
    "-Wunreachable-code"
    "-Wunused"
)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    list(
        APPEND WARNING_FLAGS
        "-Wcast-align"
        "-Wnull-dereference"
        "-Wpedantic"
    )
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    list(
        APPEND WARNING_FLAGS
        "-Wcast-align"
        "-Wnull-dereference"
        "-Wpedantic"
        "-Wnoexcept"
        "-Wsuggest-attribute=const"
        "-Wsuggest-attribute=noreturn"
        "-Wsuggest-override"
    )
endif()

if(WARNINGS_ARE_ERRORS)
  list(
    APPEND WARNING_FLAGS
    "-Werror"
    )
endif()

target_compile_options(
    "${PROJECT_NAME}_warnings" INTERFACE
    $<BUILD_INTERFACE:${WARNING_FLAGS}>
)

### Libraries ###
# Load and include cppitertools       
if(NEEDS_CPPITERTOOLS)
    if(NOT EXISTS "${${PROJECT_NAME}_CPPITERTOOLS_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "Could not find cppitertools in ${${PROJECT_NAME}_CPPITERTOOLS_DIR}")
    endif()
    set("ENV{cppitertools_INSTALL_CMAKE_DIR}" "share") # default value; supress the warning message
    add_subdirectory("${${PROJECT_NAME}_CPPITERTOOLS_DIR}")
endif()

# Load and include cxxopts       
if(NEEDS_CXXOPTS)
    if(NOT EXISTS "${${PROJECT_NAME}_CXXOPTS_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "Could not find cxxopts in ${${PROJECT_NAME}_CXXOPTS_DIR}")
    endif()
    set("ENV{cxxopts_INSTALL_CMAKE_DIR}" "share") # default value; supress the warning message
    add_subdirectory("${${PROJECT_NAME}_CXXOPTS_DIR}")
endif()

# Load and include googletest
if(NEEDS_GOOGLETEST)
    if(${PROJECT_NAME}_SYSTEM_GTEST)
        find_package(GTest REQUIRED)
    else()
        if(NOT EXISTS "${${PROJECT_NAME}_GTEST_DIR}/CMakeLists.txt")
            message(FATAL_ERROR "Could not find googletest in ${${PROJECT_NAME}_GTEST_DIR}")
        endif()
        
        add_subdirectory("${${PROJECT_NAME}_GTEST_DIR}" EXCLUDE_FROM_ALL)
        list(APPEND CMAKE_MODULE_PATH "${${PROJECT_NAME}_GTEST_DIR}/contrib")
    endif()

    include(GoogleTest)
    set_target_properties(gtest gmock PROPERTIES FOLDER googletest)
endif()

# Load and include googlebenchmark
if(NEEDS_GOOGLEBENCHMARK)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
    add_subdirectory("${${PROJECT_NAME}_GBENCHMARK_DIR}" EXCLUDE_FROM_ALL)
    set_target_properties(benchmark PROPERTIES FOLDER googlebenchmark)
endif()


### Subdirectories ###
# Build the unit tests
if(${PROJECT_NAME}_BUILD_TESTS)
    message("Building unit tests: YES")
    set(TEST_RUNNER_PARAMS "" CACHE STRING "Options added to the test runner")
    enable_testing()
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/tests")
else()
    message("Building unit tests: NO")
endif()

# Build the benchmarks
if(${PROJECT_NAME}_BUILD_BENCHMARKS)
    message("Building benchmarks: YES")
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/benchmark")
else()
    message("Building benchmarks: NO")
endif()

# Build the examples
if(${PROJECT_NAME}_BUILD_EXAMPLES)
    message("Building examples: YES")
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/examples")
else()
    message("Building examples: NO")
endif()