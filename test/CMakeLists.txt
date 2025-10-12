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
    )
    
    # Link math library for non-MSVC
    if(NOT MSVC)
        target_link_libraries(test_mixedRadixFFT m)
    endif()
    
    #=======================================================================
    # AVX-512 SUPPORT FOR TESTS
    #=======================================================================
    # Inherit AVX-512 flags from parent if enabled
    if(ENABLE_AVX512)
        message(STATUS "Test executable: AVX-512 enabled")
        target_compile_definitions(test_mixedRadixFFT PRIVATE HAS_AVX512)
        
        if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(test_mixedRadixFFT PRIVATE -mavx512f -mavx512dq)
        elseif(MSVC)
            target_compile_options(test_mixedRadixFFT PRIVATE /arch:AVX512)
        endif()
    else()
        message(STATUS "Test executable: AVX-512 disabled (use -DENABLE_AVX512=ON to enable)")
    endif()
endif()