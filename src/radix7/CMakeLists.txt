cmake_minimum_required(VERSION 3.16)
project(fft_radix7 LANGUAGES C)

# ──────────────────────────────────────────────────────────────────────
#  Radix-7 Rader butterfly kernels + stage visitors
#
#  Targets:
#    fft_radix7  — static library (visitors + butterfly symbols)
#
#  Usage from parent CMake:
#    add_subdirectory(radix7)
#    target_link_libraries(my_planner PRIVATE fft_radix7)
# ──────────────────────────────────────────────────────────────────────

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ── ISA detection ────────────────────────────────────────────────────

include(CheckCCompilerFlag)
check_c_compiler_flag("-mavx2 -mfma" HAS_AVX2)
check_c_compiler_flag("-mavx512f"    HAS_AVX512F)

if(HAS_AVX512F)
    set(R7_SIMD_FLAGS "-mavx512f;-mavx2;-mfma")
    message(STATUS "radix7: AVX-512F + AVX2 + FMA")
elseif(HAS_AVX2)
    set(R7_SIMD_FLAGS "-mavx2;-mfma")
    message(STATUS "radix7: AVX2 + FMA")
else()
    set(R7_SIMD_FLAGS "")
    message(STATUS "radix7: Scalar only")
endif()

# ── Static library ───────────────────────────────────────────────────

add_library(fft_radix7 STATIC
    fft_radix7_fv.c
    fft_radix7_bv.c
)

target_include_directories(fft_radix7 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(fft_radix7 PRIVATE ${R7_SIMD_FLAGS})
target_link_libraries(fft_radix7 PUBLIC m)

# Export to parent scope / cache so tests and other subdirs can use them
set(R7_SIMD_FLAGS  "${R7_SIMD_FLAGS}"  CACHE INTERNAL "radix7 SIMD flags")
set(R7_HAS_AVX512F "${HAS_AVX512F}"    CACHE INTERNAL "radix7 AVX-512F available")

# ── Tests (optional) ────────────────────────────────────────────────

option(R7_BUILD_TESTS "Build radix-7 tests and benchmarks" ON)

if(R7_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
