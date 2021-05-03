
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

# TODO Get rid of the googletest and googlemock dependencies

# simulate-failures-until-data-loss
set(SIM_TILL_DATA_LOSS_TARGET "simulate-failures-until-data-loss")
add_executable(${SIM_TILL_DATA_LOSS_TARGET} simulate_failures_until_data_loss.cpp)
target_include_directories(${SIM_TILL_DATA_LOSS_TARGET} SYSTEM PRIVATE ../extern/)
target_include_directories(${SIM_TILL_DATA_LOSS_TARGET} SYSTEM PRIVATE ../include/)
target_link_libraries(${SIM_TILL_DATA_LOSS_TARGET} gtest gmock "ReStore::ReStore;cppitertools;cxxopts")
target_compile_options(${SIM_TILL_DATA_LOSS_TARGET} PRIVATE -O3 "${WARNING_FLAGS}")

set(PAGERANK "pageRank")
add_executable(${PAGERANK} pageRank.cpp)
target_include_directories(${PAGERANK} SYSTEM PRIVATE ../extern/)
target_include_directories(${PAGERANK} SYSTEM PRIVATE ../include/)
target_link_libraries(${PAGERANK} gtest gmock "ReStore::ReStore;cppitertools;cxxopts")
target_compile_options(${PAGERANK} PRIVATE -O3 "${WARNING_FLAGS}")

# microbenchmarks
add_compile_definitions(SIMULATE_FAILURES=$<BOOL:${SIMULATE_FAILURES}>)
set(MICROBENCHMARKS_TARGET "microbenchmarks")
add_executable(${MICROBENCHMARKS_TARGET} microbenchmarks.cpp)
target_include_directories(${MICROBENCHMARKS_TARGET} SYSTEM PRIVATE ../extern/)
target_include_directories(${MICROBENCHMARKS_TARGET} SYSTEM PRIVATE ../include/)
set(BENCHMARK_ENABLE_TESTING OFF)
add_subdirectory(../extern/googlebenchmark ${CMAKE_CURRENT_BINARY_DIR}/../googlebenchmark)
target_link_libraries(${MICROBENCHMARKS_TARGET} "ReStore::ReStore;benchmark::benchmark")
target_compile_options(${MICROBENCHMARKS_TARGET} PRIVATE -O3 -g "${WARNING_FLAGS}")
