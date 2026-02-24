cmake_minimum_required(VERSION 3.16)
project(fft_radix5 LANGUAGES C)

# ──────────────────────────────────────────────────────────────────────
#  Radix-5 WFTA butterfly kernels + stage visitors
#
#  Targets:
#    fft_radix5  — static library (visitors + butterfly symbols)
#
#  Usage from parent CMake:
#    add_subdirectory(radix5)
#    target_link_libraries(my_planner PRIVATE fft_radix5)
# ──────────────────────────────────────────────────────────────────────

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ── ISA detection ────────────────────────────────────────────────────

set(R5_MAX_ISA "avx2" CACHE STRING "Max ISA: auto, avx512, avx2, scalar (avx512 multi-stage has known bug)")

include(CheckCCompilerFlag)
check_c_compiler_flag("-mavx2 -mfma" HAS_AVX2)
check_c_compiler_flag("-mavx512f"    HAS_AVX512F)

# Allow capping the ISA level (useful if AVX-512 kernels have issues)
if(R5_MAX_ISA STREQUAL "scalar")
    set(HAS_AVX2    FALSE)
    set(HAS_AVX512F FALSE)
elseif(R5_MAX_ISA STREQUAL "avx2")
    set(HAS_AVX512F FALSE)
endif()

if(HAS_AVX512F)
    set(R5_SIMD_FLAGS "-mavx512f;-mavx2;-mfma")
    message(STATUS "radix5: AVX-512F + AVX2 + FMA")
elseif(HAS_AVX2)
    set(R5_SIMD_FLAGS "-mavx2;-mfma")
    message(STATUS "radix5: AVX2 + FMA")
else()
    set(R5_SIMD_FLAGS "")
    message(STATUS "radix5: Scalar only")
endif()

# ── Static library ───────────────────────────────────────────────────

add_library(fft_radix5 STATIC
    fft_radix5_fv.c
    fft_radix5_bv.c
)

target_include_directories(fft_radix5
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/scalar
            ${CMAKE_CURRENT_SOURCE_DIR}/avx2
            ${CMAKE_CURRENT_SOURCE_DIR}/avx512
)

target_compile_options(fft_radix5 PRIVATE ${R5_SIMD_FLAGS})

# Suppress -Wrestrict for in-place visitor calls (input == output is safe)
include(CheckCCompilerFlag)
check_c_compiler_flag("-Wno-restrict" HAS_WNO_RESTRICT)
if(HAS_WNO_RESTRICT)
    target_compile_options(fft_radix5 PRIVATE -Wno-restrict)
endif()

# Math library: built into CRT on Windows, need -lm on Unix
if(NOT WIN32)
    target_link_libraries(fft_radix5 PUBLIC m)
    set(R5_MATH_LIB "m" CACHE INTERNAL "math library")
else()
    set(R5_MATH_LIB ""  CACHE INTERNAL "math library (none on Windows)")
endif()

# Export to parent scope / cache so tests and other subdirs can use them
set(R5_SIMD_FLAGS  "${R5_SIMD_FLAGS}"  CACHE INTERNAL "radix5 SIMD flags")
set(R5_HAS_AVX512F "${HAS_AVX512F}"    CACHE INTERNAL "radix5 AVX-512F available")

# ── Tests (optional) ────────────────────────────────────────────────

option(R5_BUILD_TESTS "Build radix-5 tests and benchmarks" ON)

if(R5_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
