cmake_minimum_required(VERSION 3.16)

if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
    set(IS_SUBPROJECT OFF)
else()
    set(IS_SUBPROJECT ON)
endif()

# Project settings
project(ReStore VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
if(MPI_CXX_FOUND)
  include_directories(BEFORE SYSTEM ${MPI_CXX_INCLUDE_PATH})
  link_libraries(${MPI_LIBRARIES})
endif()

# Look in the /cmake dir for CMake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# Require out-of-source builds
include(require_out_of_source_builds)
require_out_of_source_builds()

include(CMakeDependentOption)
#include(CMakePackageConfigHelpers)
include(CTest)
include(GNUInstallDirs)

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
option("WARNINGS_ARE_ERRORS" "Treat warnings as errors." OFF)

# Extra options when building tests
cmake_dependent_option("${PROJECT_NAME}_SYSTEM_GTEST" "Use googletest version installed on the system" OFF "${PROJECT_NAME}_BUILD_TESTS" OFF)

# Library specific settings
set("NEEDS_GOOGLETEST" ${PROJECT_NAME}_BUILD_TESTS OR ${PROJECT_NAME}_BUILD_BENCHMARKS)
set("NEEDS_CPPITERTOOLS" ${PROJECT_NAME}_BUILD_TESTS OR ${PROJECT_NAME}_BUILD_BENCHMARKS)
set("NEEDS_CXXOPTS" ${PROJECT_NAME}_BUILD_BENCHMARKS)
set("NEEDS_GOOGLEBENCHMARK" ${PROJECT_NAME}_BUILD_BENCHMARKS)
set("NEEDS_BACKWARDCPP" OFF)

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
endif()

# Organize targets into folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#configure_file(
#    "version.hpp.in"
#    "version.hpp"
#)
#configure_file(
#    "system_config.hpp.in"
#    "system_config.hpp"
#)

# The target to be linked against by other targets. This library is an
# interface target and as such does not generate any artefacts. It rather sets
# include directories and required compiler flags.
add_library("${PROJECT_NAME}" INTERFACE)
target_include_directories(
    "${PROJECT_NAME}" INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Include sources generated at build time. When installing the headers/library,
# these sources will be copied to appropriate locations such that no additional
# directories have to be included.
#target_include_directories(
#    "${PROJECT_NAME}" INTERFACE
#    $<BUILD_INTERFACE:${PROJECT_CURRENT_BINARY_DIR}>
#)

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
# flags.    Only to be used by internal targets that compile the library.
add_library("${PROJECT_NAME}_warnings" INTERFACE)

list(
    APPEND WARNING_FLAGS
    "-Wall"
    "-Wextra"
    "-Wcast-align"
    "-Wconversion"
    "-Wnon-virtual-dtor"
    "-Wnull-dereference"
    "-Woverloaded-virtual"
    "-Wpedantic"
    "-Wshadow"
    "-Wsign-conversion"
    "-Wundef"
    "-Wunreachable-code"
    "-Wunused"
)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    list(
        APPEND WARNING_FLAGS
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
endif()

# Load and include googlebenchmark
if(NEEDS_GOOGLEBENCHMARK)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
    add_subdirectory("${${PROJECT_NAME}_GBENCHMARK_DIR}" EXCLUDE_FROM_ALL)
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

#if(${PROJECT_NAME}_INSTALL)
#    include(InstallRequiredSystemLibraries)
#    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
#    set(CPACK_RESOURCE_FILE_README ${CMAKE_CURRENT_LIST_DIR}/README.md)
#    include(CPack)
#    set(INSTALL_MODULEDIR "${CMAKE_INSTALL_LIBDIR}/cmake")
#    set(INSTALL_CMAKEDIR "${INSTALL_MODULEDIR}/${PROJECT_NAME}")
#    configure_package_config_file(
#        "${CMAKE_CURRENT_LIST_DIR}/cmake/${PROJECT_NAME}Config.cmake.in"
#        "${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}Config.cmake"
#        INSTALL_DESTINATION "${INSTALL_CMAKEDIR}"
#        PATH_VARS INSTALL_MODULEDIR
#    )
#    write_basic_package_version_file(
#        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
#        VERSION ${PROJECT_VERSION}
#        COMPATIBILITY
#        SameMajorVersion
#        ARCH_INDEPENDENT
#    )
#    # On install, copy the header files to the appropriate location.
#    install(
#        DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}/"
#        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}"
#        FILES_MATCHING PATTERN "*.hpp"
#    )
#    # Also copy the artefacts generated at build time into the include directory.
#    install(
#        FILES "${CMAKE_CURRENT_BINARY_DIR}/version.hpp"
#        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}"
#    )
#    install(TARGETS "${PROJECT_NAME}" "${PROJECT_NAME}_compile_requirements" EXPORT "${PROJECT_NAME}Export")
#    install(
#        FILES
#        "${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}Config.cmake"
#        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
#        DESTINATION
#        "${INSTALL_CMAKEDIR}"
#    )
#    install(
#        EXPORT "${PROJECT_NAME}Export"
#        DESTINATION "${INSTALL_MODULEDIR}"
#        NAMESPACE "${PROJECT_NAME}::"
#        FILE "${PROJECT_NAME}Targets.cmake"
#    )
#    if(${PROJECT_NAME}_INSTALL_DOCS)
#        install(
#            DIRECTORY docs/
#            DESTINATION "${CMAKE_INSTALL_DOCDIR}"
#        )
#    endif()
#endif()
