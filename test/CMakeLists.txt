cmake_minimum_required(VERSION 3.10)
project(MixedRadixFFT_Tests C CXX)

#==============================================================================
# TEST EXECUTABLE CONFIGURATION
#==============================================================================
if(ENABLE_GTEST)
    #==========================================================================
    # GoogleTest-based tests (DISABLED for now)
    #==========================================================================
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
    #==========================================================================
    # Standalone tests using main.c (DEFAULT)
    #==========================================================================
    message(STATUS "Building standalone tests (main.c)")

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
    
    # Link math library (non-MSVC)
    if(NOT MSVC)
        target_link_libraries(test_mixedRadixFFT m pthread)
    endif()
    
    #==========================================================================
    # INHERIT FLAGS FROM LIBRARY
    #==========================================================================
    
    # Prefetch system
    if(ENABLE_PREFETCH)
        message(STATUS "Test executable: Prefetch system ENABLED")
        target_compile_definitions(test_mixedRadixFFT PRIVATE FFT_ENABLE_PREFETCH)
        
        if(DEBUG_PREFETCH)
            message(STATUS "Test executable: Prefetch debugging ENABLED")
            target_compile_definitions(test_mixedRadixFFT PRIVATE FFT_DEBUG_PREFETCH)
        endif()
    else()
        message(STATUS "Test executable: Prefetch system DISABLED")
    endif()
    
    # AVX-512 support
    if(ENABLE_AVX512)
        message(STATUS "Test executable: AVX-512 ENABLED")
        target_compile_definitions(test_mixedRadixFFT PRIVATE HAS_AVX512)
        
        if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(test_mixedRadixFFT PRIVATE -mavx512f -mavx512dq)
        elseif(MSVC)
            target_compile_options(test_mixedRadixFFT PRIVATE /arch:AVX512)
        endif()
    else()
        message(STATUS "Test executable: AVX-512 DISABLED")
    endif()
    
    #==========================================================================
    # OPTIMIZATION FLAGS
    #==========================================================================
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_mixedRadixFFT PRIVATE 
            -O3 
            -march=native
            -Wall
            -Wextra
        )
    elseif(MSVC)
        target_compile_options(test_mixedRadixFFT PRIVATE 
            /O2
            /W4
        )
    endif()
    
    #==========================================================================
    # REGISTER TEST WITH CTEST
    #==========================================================================
    add_test(NAME standalone_fft_tests COMMAND test_mixedRadixFFT)
endif()

#==============================================================================
# TEST CONFIGURATION SUMMARY
#==============================================================================
message(STATUS "============================================")
message(STATUS "Test Configuration:")
if(ENABLE_GTEST)
    message(STATUS "  Framework:      GoogleTest (C++)")
    message(STATUS "  Source:         test_mixedRadixFFT.cpp")
else()
    message(STATUS "  Framework:      Standalone (C)")
    message(STATUS "  Source:         main.c")
endif()

if(ENABLE_PREFETCH)
    message(STATUS "  Prefetch:       ENABLED")
    if(DEBUG_PREFETCH)
        message(STATUS "    → Debug:      YES")
    endif()
else()
    message(STATUS "  Prefetch:       DISABLED")
endif()

if(ENABLE_AVX512)
    message(STATUS "  AVX-512:        ENABLED")
else()
    message(STATUS "  AVX-512:        DISABLED")
endif()
message(STATUS "============================================")