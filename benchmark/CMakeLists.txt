
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

# simulate-failures-until-data-loss
set(SIM_TILL_DATA_LOSS_TARGET "simulate-failures-until-data-loss")
add_executable(${SIM_TILL_DATA_LOSS_TARGET} simulate_failures_until_data_loss.cpp)
target_include_directories(${SIM_TILL_DATA_LOSS_TARGET} SYSTEM PRIVATE ../extern/)
target_include_directories(${SIM_TILL_DATA_LOSS_TARGET} PRIVATE ../include/)
target_link_libraries(${SIM_TILL_DATA_LOSS_TARGET} gtest gmock "ReStore::ReStore;cppitertools;cxxopts")
target_compile_options(${SIM_TILL_DATA_LOSS_TARGET} PRIVATE -O3 "${WARNING_FLAGS}")

# page rank
set(PAGERANK "pageRank")
add_executable(${PAGERANK} pageRank.cpp)
target_include_directories(${PAGERANK} SYSTEM PRIVATE ../extern/)
target_include_directories(${PAGERANK} PRIVATE ../include/)
target_link_libraries(${PAGERANK} gtest gmock "ReStore::ReStore;cppitertools;cxxopts")
target_compile_options(${PAGERANK} PRIVATE -O3 "${WARNING_FLAGS}")
target_compile_definitions(${PAGERANK} PRIVATE "ENABLE_TIMERS")

# TODO Shall we put this in the root CMakeLists.txt?
# Load the backward-cpp library
set(Backward_DIR "${PROJECT_SOURCE_DIR}/extern/backward-cpp")
find_package(Backward)

# k-means - library
add_library("${PROJECT_NAME}_kmeans" INTERFACE)
target_sources("${PROJECT_NAME}_kmeans" INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(
    "${PROJECT_NAME}_kmeans" INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries("${PROJECT_NAME}_kmeans" INTERFACE "${PROJECT_NAME}_compile_requirements")
add_library("${PROJECT_NAME}::kmeans" ALIAS "${PROJECT_NAME}_kmeans")

# k-means - benchmark
set(KMEANS "k-means")
if(${PROJECT_NAME}_BACKWARD_ENABLED)
    add_executable(${KMEANS} k-means.cpp "${${PROJECT_NAME}_BACKWARDCPP_DIR}/backward.cpp")
else()
    add_executable(${KMEANS} k-means.cpp)
endif()
# target_include_directories(${KMEANS} SYSTEM PRIVATE ../extern/)
# target_include_directories(${KMEANS} PRIVATE ../include/)
# target_include_directories(${KMEANS} PRIVATE ./include/)
target_link_libraries(${KMEANS} "ReStore::ReStore;Backward::Backward;ReStore::kmeans;cxxopts")
target_compile_options(${KMEANS} PRIVATE -O3 "${WARNING_FLAGS}")
target_compile_definitions(${PROJECT_NAME}_kmeans INTERFACE "ENABLE_TIMERS")

# microbenchmarks
set(MICROBENCHMARKS_TARGET "microbenchmarks")
add_executable(${MICROBENCHMARKS_TARGET} microbenchmarks.cpp)
target_include_directories(${MICROBENCHMARKS_TARGET} SYSTEM PRIVATE ../extern/)
target_include_directories(${MICROBENCHMARKS_TARGET} PRIVATE ../include/)
target_link_libraries(${MICROBENCHMARKS_TARGET} "ReStore::ReStore;benchmark::benchmark")
target_compile_options(${MICROBENCHMARKS_TARGET} PRIVATE -O3 "${WARNING_FLAGS}")
