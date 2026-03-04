cmake_minimum_required(VERSION 3.16)

# ── Find FFTW3 ──
# Try find_package first (works with vcpkg, system installs, etc.)
# Fall back to manual paths via FFTW3_DIR.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3 QUIET fftw3)
endif()

if(NOT FFTW3_FOUND)
    find_path(FFTW3_INCLUDE_DIRS fftw3.h
        HINTS
            ${FFTW3_DIR}/include
            $ENV{FFTW3_DIR}/include
            "C:/fftw"
            "C:/Program Files/fftw"
    )
    find_library(FFTW3_LIB
        NAMES fftw3 libfftw3-3
        HINTS
            ${FFTW3_DIR}/lib
            $ENV{FFTW3_DIR}/lib
            "C:/fftw"
            "C:/Program Files/fftw"
    )
    if(FFTW3_INCLUDE_DIRS AND FFTW3_LIB)
        set(FFTW3_FOUND TRUE)
        set(FFTW3_LIBRARIES ${FFTW3_LIB})
        message(STATUS "FFTW3 found: ${FFTW3_LIB}")
    else()
        message(FATAL_ERROR
            "FFTW3 not found. Set FFTW3_DIR to your FFTW installation root, e.g.:\n"
            "  cmake -DFFTW3_DIR=C:/fftw ..\n"
            "  cmake -DFFTW3_DIR=/usr/local ..")
    endif()
endif()

# ── Common test setup ──
# All tests link against FFTW and the radix128 header-only lib.
# On Linux, also need libm.

function(add_radix128_test target source simd_level)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE radix128)
    target_compile_options(${target} PRIVATE ${RADIX128_COMMON_FLAGS})

    # SIMD flags
    if(simd_level STREQUAL "avx512")
        target_compile_options(${target} PRIVATE ${RADIX128_AVX512_FLAGS})
    elseif(simd_level STREQUAL "avx2")
        target_compile_options(${target} PRIVATE ${RADIX128_AVX2_FLAGS})
    endif()

    # FFTW
    if(FFTW3_INCLUDE_DIRS)
        target_include_directories(${target} PRIVATE ${FFTW3_INCLUDE_DIRS})
    endif()
    if(FFTW3_LIBRARIES)
        target_link_libraries(${target} PRIVATE ${FFTW3_LIBRARIES})
    elseif(FFTW3_LINK_LIBRARIES)
        target_link_libraries(${target} PRIVATE ${FFTW3_LINK_LIBRARIES})
    endif()

    # Math library (Linux/macOS)
    if(NOT WIN32)
        target_link_libraries(${target} PRIVATE m)
    endif()

    # Suppress harmless warnings from generated code
    if(CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM")
        if(CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
            target_compile_options(${target} PRIVATE /Qdiag-disable:177,869)
        else()
            target_compile_options(${target} PRIVATE
                -Wno-unused-function -Wno-unused-parameter)
        endif()
    elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${target} PRIVATE
            -Wno-unused-function -Wno-unused-parameter)
    endif()

    add_test(NAME ${target} COMMAND ${target})
endfunction()

# ── Test executables ──

# AVX-512 benchmark (gen kernel only)
add_radix128_test(test_bench_r128_gen
    test_bench_r128_gen.c
    avx512)

# Scalar + AVX2 benchmark
add_radix128_test(test_bench_r128_scalar_avx2
    test_bench_r128_scalar_avx2.c
    avx2)

# Unified header test (all ISAs)
add_radix128_test(test_radix128_unified
    test_radix128_unified.c
    avx512)
