cmake_minimum_required(VERSION 3.10)
project(MixedRadixFFT_Tests C CXX)

#==============================================================================
# CHOOSE TEST FRAMEWORK
#==============================================================================
if(ENABLE_GTEST)
    #==========================================================================
    # GoogleTest-based tests
    #==========================================================================
    message(STATUS "Building with GoogleTest support")

    add_executable(test_mixedRadixFFT_gtest
        test_mixedRadixFFT.cpp
    )

    target_include_directories(test_mixedRadixFFT_gtest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${fff_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(test_mixedRadixFFT_gtest PRIVATE
        highspeedFFT
        gtest_main
        gmock_main
    )
    
    #==========================================================================
    # INHERIT FLAGS FROM LIBRARY (GoogleTest)
    #==========================================================================
    if(ENABLE_PREFETCH)
        target_compile_definitions(test_mixedRadixFFT_gtest PRIVATE FFT_ENABLE_PREFETCH)
        if(DEBUG_PREFETCH)
            target_compile_definitions(test_mixedRadixFFT_gtest PRIVATE FFT_DEBUG_PREFETCH)
        endif()
    endif()
    
    if(ENABLE_AVX512)
        target_compile_definitions(test_mixedRadixFFT_gtest PRIVATE HAS_AVX512)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(test_mixedRadixFFT_gtest PRIVATE -mavx512f -mavx512dq)
        elseif(MSVC)
            target_compile_options(test_mixedRadixFFT_gtest PRIVATE /arch:AVX512)
        endif()
    endif()
    
    #==========================================================================
    # OPTIMIZATION FLAGS (GoogleTest)
    #==========================================================================
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_mixedRadixFFT_gtest PRIVATE 
            -O3 
            -march=native
            -mavx2
            -mfma
            -Wall
            -Wextra
            -Wpedantic
        )
    elseif(MSVC)
        target_compile_options(test_mixedRadixFFT_gtest PRIVATE 
            /O2
            /arch:AVX2
            /W4
        )
    endif()

    include(GoogleTest)
    gtest_discover_tests(test_mixedRadixFFT_gtest)
    
    message(STATUS "  Test executable: test_mixedRadixFFT_gtest")

else()
    #==========================================================================
    # Standalone tests using main.c (DEFAULT)
    #==========================================================================
    message(STATUS "Building standalone tests (main.c)")

    add_executable(test_mixedRadixFFT_standalone
        main.c
    )

    target_include_directories(test_mixedRadixFFT_standalone PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
    )

    target_link_libraries(test_mixedRadixFFT_standalone PRIVATE
        highspeedFFT
    )
    
    # Link math library (non-MSVC)
    if(NOT MSVC)
        target_link_libraries(test_mixedRadixFFT_standalone PRIVATE m pthread)
    endif()
    
    #==========================================================================
    # INHERIT FLAGS FROM LIBRARY (main.c)
    #==========================================================================
    if(ENABLE_PREFETCH)
        message(STATUS "Test executable: Prefetch system ENABLED")
        target_compile_definitions(test_mixedRadixFFT_standalone PRIVATE FFT_ENABLE_PREFETCH)
        
        if(DEBUG_PREFETCH)
            message(STATUS "Test executable: Prefetch debugging ENABLED")
            target_compile_definitions(test_mixedRadixFFT_standalone PRIVATE FFT_DEBUG_PREFETCH)
        endif()
    else()
        message(STATUS "Test executable: Prefetch system DISABLED")
    endif()
    
    if(ENABLE_AVX512)
        message(STATUS "Test executable: AVX-512 ENABLED")
        target_compile_definitions(test_mixedRadixFFT_standalone PRIVATE HAS_AVX512)
        
        if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(test_mixedRadixFFT_standalone PRIVATE -mavx512f -mavx512dq)
        elseif(MSVC)
            target_compile_options(test_mixedRadixFFT_standalone PRIVATE /arch:AVX512)
        endif()
    else()
        message(STATUS "Test executable: AVX-512 DISABLED")
    endif()
    
    #==========================================================================
    # OPTIMIZATION FLAGS (main.c)
    #==========================================================================
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_mixedRadixFFT_standalone PRIVATE 
            -O3 
            -march=native
            -mavx2
            -mfma
            -Wall
            -Wextra
        )
    elseif(MSVC)
        target_compile_options(test_mixedRadixFFT_standalone PRIVATE 
            /O2
            /arch:AVX2
            /W4
        )
    endif()
    
    #==========================================================================
    # REGISTER TEST WITH CTEST
    #==========================================================================
    add_test(NAME standalone_fft_tests COMMAND test_mixedRadixFFT_standalone)
    
    message(STATUS "  Test executable: test_mixedRadixFFT_standalone")
endif()

#==============================================================================
# TEST CONFIGURATION SUMMARY
#==============================================================================
message(STATUS "============================================")
message(STATUS "Test Configuration:")
if(ENABLE_GTEST)
    message(STATUS "  Framework:      GoogleTest (C++)")
    message(STATUS "  Source:         test_mixedRadixFFT.cpp")
    message(STATUS "  Executable:     test_mixedRadixFFT_gtest")
else()
    message(STATUS "  Framework:      Standalone (C)")
    message(STATUS "  Source:         main.c")
    message(STATUS "  Executable:     test_mixedRadixFFT_standalone")
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