cmake_minimum_required(VERSION 3.16)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
    set(IS_SUBPROJECT OFF)
    message(STATUS "Building ReStore as a stand-alone project.")
else()
    set(IS_SUBPROJECT ON)
    message(STATUS "Building ReStore as a subbroject.")
endif()

# Use ccache if available
find_program(CCACHE_PROGRAM ccache)
if((NOT ${IS_SUBPROBJECT}) AND ${CCACHE_PROGRAM})
    message(STATUS "Found ccache; using it to speed up compilation.")
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

# Make additinal cmake modules findable
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
  # Enable extra checks in the STL
  add_definitions(-D_GLIBCXX_DEBUG)
  add_definitions(-D_GLIBCXX_DEBUG_PEDANTIC)
endif(CMAKE_BUILD_TYPE MATCHES Debug)

option(${PROJECT_NAME}_INSTALL "Add ${PROJECT_NAME} to the install list" ON)

# Options to disable building the tests, examples, benchmarks, ...
option(${PROJECT_NAME}_BUILD_TESTS "Build unit tests" ON)
option(${PROJECT_NAME}_BUILD_BENCHMARKS "Build benchmarks" ON)
option(${PROJECT_NAME}_BUILD_EXAMPLES "Build examples" OFF)

# Other build options
option(${PROJECT_NAME}_SIMULATE_FAILURES "Simulate node failues (for example when running unit test)." OFF)
option(${PROJECT_NAME}_USE_FTMPI "Use a fault-tolerant MPI implementation" ON)
option(${PROJECT_NAME}_WARNINGS_ARE_ERRORS "Treat warnings as errors." OFF)

# Output config
if(NOT ${PROJECT_NAME}_SIMULATE_FAILURES AND NOT ${PROJECT_NAME}_USE_FTMPI)
    message(FATAL_ERROR "You have to either use a fault-tolerant MPI implementation or simulated failures (or both).")
endif()

message(STATUS "Using simulated failures: ${SIMULATE_FAILURES}")
message(STATUS "Using a fault-tolerant MPI implementation: ${USE_FTMPI}")

# Extra options when building tests
cmake_dependent_option("${PROJECT_NAME}_SYSTEM_GTEST" "Use googletest version installed on the system" OFF "${PROJECT_NAME}_BUILD_TESTS" OFF)

# Library specific settings
set("NEEDS_GOOGLETEST" ${PROJECT_NAME}_BUILD_TESTS OR ${PROJECT_NAME}_BUILD_BENCHMARKS)
set("NEEDS_CPPITERTOOLS" ${PROJECT_NAME}_BUILD_TESTS OR ${PROJECT_NAME}_BUILD_BENCHMARKS)
set("NEEDS_CXXOPTS" ${PROJECT_NAME}_BUILD_BENCHMARKS)
set("NEEDS_GOOGLEBENCHMARK" ${PROJECT_NAME}_BUILD_BENCHMARKS)
set("NEEDS_BACKWARDCPP" ${PROJECT_NAME}_BUILD_TESTS OR ${PROJECT_NAME}_BUILD_BENCHMARKS)
set("NEEDS_XXHASH" ON)

if(NEEDS_GOOGLETEST)
    message(STATUS "Using googletest library.")
    set("${PROJECT_NAME}_GTEST_DIR" "${CMAKE_CURRENT_LIST_DIR}/extern/googletest" CACHE PATH "Path to the googletest source directory")
endif()

if(NEEDS_GOOGLEBENCHMARK)
    message(STATUS "Using googlebenchmark library.")
    set("${PROJECT_NAME}_GBENCHMARK_DIR" "${CMAKE_CURRENT_LIST_DIR}/extern/googlebenchmark" CACHE PATH "Path to the googlebenchmark source directory")
endif()

if(NEEDS_CPPITERTOOLS)
    message(STATUS "Using cppitertools library.")
    set("${PROJECT_NAME}_CPPITERTOOLS_DIR" "${CMAKE_CURRENT_LIST_DIR}/extern/cppitertools" CACHE PATH "Path to the cppitertools source directory")
endif()

if(NEEDS_CXXOPTS)
    message(STATUS "Using cxxopts library.")
    set("${PROJECT_NAME}_CXXOPTS_DIR" "${CMAKE_CURRENT_LIST_DIR}/extern/cxxopts" CACHE PATH "Path to the cxxopts source directory")
endif()

if(NEEDS_BACKWARDCPP)
    message(STATUS "Using backwardcpp library.")
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
target_include_directories("${PROJECT_NAME}" INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

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

if(${PROJECT_NAME}_WARNINGS_ARE_ERRORS)
  list(
    APPEND WARNING_FLAGS
    "-Werror"
    )
endif()

target_compile_options(
    "${PROJECT_NAME}_warnings" INTERFACE
    $<BUILD_INTERFACE:${WARNING_FLAGS}>
)

# Populate some of the options to the source code
if (${PROJECT_NAME}_SIMULATE_FAILURES) 
    target_compile_definitions(${PROJECT_NAME}_compile_requirements INTERFACE SIMULATE_FAILURES)
endif()
if (${PROJECT_NAME}_USE_FTMPI)
    target_compile_definitions(${PROJECT_NAME}_compile_requirements INTERFACE USE_FTMPI)
endif()

### Libraries ###
# Load and include cppitertools       
if(NEEDS_CPPITERTOOLS)
    if(NOT EXISTS "${${PROJECT_NAME}_CPPITERTOOLS_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "Could not find cppitertools in ${${PROJECT_NAME}_CPPITERTOOLS_DIR}")
    endif()
    message(STATUS "Configuring cppitertools...")
    set("ENV{cppitertools_INSTALL_CMAKE_DIR}" "share") # default value; supress the warning message
    add_subdirectory("${${PROJECT_NAME}_CPPITERTOOLS_DIR}")
    message(STATUS "cppitertools configured.")
endif()

# Load and include cxxopts       
if(NEEDS_CXXOPTS)
    if(NOT EXISTS "${${PROJECT_NAME}_CXXOPTS_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "Could not find cxxopts in ${${PROJECT_NAME}_CXXOPTS_DIR}")
    endif()
    message(STATUS "Configuring cxxopts...")
    set("ENV{cxxopts_INSTALL_CMAKE_DIR}" "share") # default value; supress the warning message
    add_subdirectory("${${PROJECT_NAME}_CXXOPTS_DIR}")
    message(STATUS "cxxopts configured.")
endif()

# Load and include googletest
if(NEEDS_GOOGLETEST)
    if(${PROJECT_NAME}_SYSTEM_GTEST)
        find_package(GTest REQUIRED)
    else()
        if(NOT EXISTS "${${PROJECT_NAME}_GTEST_DIR}/CMakeLists.txt")
            message(FATAL_ERROR "Could not find googletest in ${${PROJECT_NAME}_GTEST_DIR}")
        endif()
        
        message(STATUS "Configuring googletest...")
        add_subdirectory("${${PROJECT_NAME}_GTEST_DIR}" EXCLUDE_FROM_ALL)
        list(APPEND CMAKE_MODULE_PATH "${${PROJECT_NAME}_GTEST_DIR}/contrib")
        message(STATUS "googletest configured.")
    endif()

    include(GoogleTest)
    set_target_properties(gtest gmock PROPERTIES FOLDER googletest)
endif()

# Load and include googlebenchmark
if(NEEDS_GOOGLEBENCHMARK)
    message(STATUS "Configuring googlebenchmark...")
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
    add_subdirectory("${${PROJECT_NAME}_GBENCHMARK_DIR}" EXCLUDE_FROM_ALL)
    set_target_properties(benchmark PROPERTIES FOLDER googlebenchmark)
    message(STATUS "googlebenchmark configured.")
endif()

# Load and include backward-cpp
if (NEEDS_BACKWARDCPP)
    message(STATUS "Configuring backward-cpp...")
    set(Backward_DIR "${PROJECT_SOURCE_DIR}/extern/backward-cpp")
    find_package(Backward)
    message(STATUS "backward-cpp configured.")
endif()

# Load and include xxHash
message(STATUS "Configuring xxHash...")
set(${PROJECT_NAME}_XXHASH_DIR "${CMAKE_CURRENT_LIST_DIR}/extern/xxHash" CACHE PATH "Path to the xxHash source directory")
set(BUILD_SHARED_LIBS OFF)
set(XXHASH_BUILD_ENABLE_INLINE_API ON)
set(XXHASH_BUILD_XXHSUM OFF)
add_subdirectory("${${PROJECT_NAME}_XXHASH_DIR}/cmake_unofficial" "${${PROJECT_NAME}_XXHASH_DIR/build}" EXCLUDE_FROM_ALL)

# Link xxHash with ReStore
target_link_libraries("${PROJECT_NAME}" INTERFACE xxhash)

### Subdirectories ###
# Build the unit tests
if(${PROJECT_NAME}_BUILD_TESTS)
    message(STATUS "Building unit tests: YES")
    set(TEST_RUNNER_PARAMS "" CACHE STRING "Options added to the test runner")
    enable_testing()
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/tests")
else()
    message(STATUS "Building unit tests: NO")
endif()

# Build the benchmarks
if(${PROJECT_NAME}_BUILD_BENCHMARKS)
    message(STATUS "Building benchmarks: YES")
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/benchmark")
else()
    message(STATUS "Building benchmarks: NO")
endif()

# Build the examples
if(${PROJECT_NAME}_BUILD_EXAMPLES)
    message(STATUS "Building examples: YES")
    add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/examples")
else()
    message(STATUS "Building examples: NO")
endif()
