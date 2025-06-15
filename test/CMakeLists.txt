cmake_minimum_required(VERSION 3.10)
project(MixedRadixFFT_Tests C CXX)

# Manually control whether GoogleTest is enabled
# Change to ON to enable GoogleTest-based tests
set(ENABLE_GTEST OFF CACHE BOOL "Enable GoogleTest-based tests")

# Add the executable and configuration based on ENABLE_GTEST
if(ENABLE_GTEST)
    message(STATUS "Building with GoogleTest support")

    add_executable(test_mixedRadixFFT
        test_mixedRadixFFT.cpp
    )

    target_include_directories(test_mixedRadixFFT PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${fff_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(test_mixedRadixFFT
        highspeedFFT
        gtest_main
        gmock_main
    )

    include(GoogleTest)
    gtest_discover_tests(test_mixedRadixFFT)

else()
    message(STATUS "Building without GoogleTest (using main.c)")

    add_executable(test_mixedRadixFFT
        main.c
    )

    target_include_directories(test_mixedRadixFFT PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(test_mixedRadixFFT
        highspeedFFT
        m
    )
endif()