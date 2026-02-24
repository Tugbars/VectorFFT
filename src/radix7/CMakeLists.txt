cmake_minimum_required(VERSION 3.16)
project(fft_radix7 LANGUAGES C)

# ──────────────────────────────────────────────────────────────────────
#  Radix-7 Rader butterfly kernels + stage visitors
#
#  Targets:
#    fft_radix7              — static library (visitors + butterfly headers)
#    test_butterfly          — butterfly-level correctness (scalar + AVX2)
#    test_butterfly_avx512   — butterfly-level correctness (AVX-512)
#    test_visitors           — end-to-end multi-stage test (simulates planner)
#    bench_butterfly         — butterfly-level benchmark (all ISAs)
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
    set(SIMD_FLAGS "-mavx512f;-mavx2;-mfma")
    message(STATUS "radix7: AVX-512F + AVX2 + FMA")
elseif(HAS_AVX2)
    set(SIMD_FLAGS "-mavx2;-mfma")
    message(STATUS "radix7: AVX2 + FMA")
else()
    set(SIMD_FLAGS "")
    message(STATUS "radix7: Scalar only")
endif()

# ── Static library: visitors ─────────────────────────────────────────

add_library(fft_radix7 STATIC
    fft_radix7_fv.c
    fft_radix7_bv.c
)

target_include_directories(fft_radix7 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(fft_radix7 PRIVATE ${SIMD_FLAGS})
target_link_libraries(fft_radix7 PUBLIC m)

# ── Tests ────────────────────────────────────────────────────────────

add_executable(test_butterfly tests/test_radix7_rader.c)
target_include_directories(test_butterfly PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(test_butterfly PRIVATE ${SIMD_FLAGS})
target_link_libraries(test_butterfly PRIVATE m)

if(HAS_AVX512F)
    add_executable(test_butterfly_avx512 tests/test_radix7_avx512.c)
    target_include_directories(test_butterfly_avx512 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_options(test_butterfly_avx512 PRIVATE ${SIMD_FLAGS})
    target_link_libraries(test_butterfly_avx512 PRIVATE m)
endif()

add_executable(test_visitors tests/test_visitors.c)
target_link_libraries(test_visitors PRIVATE fft_radix7)
target_compile_options(test_visitors PRIVATE ${SIMD_FLAGS})

add_executable(bench_butterfly tests/bench_radix7.c)
target_include_directories(bench_butterfly PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(bench_butterfly PRIVATE ${SIMD_FLAGS})
target_link_libraries(bench_butterfly PRIVATE m)

# ── CTest ────────────────────────────────────────────────────────────

enable_testing()

add_test(NAME r7_butterfly_scalar_avx2 COMMAND test_butterfly)
add_test(NAME r7_visitors              COMMAND test_visitors)
add_test(NAME r7_bench                 COMMAND bench_butterfly)
set_tests_properties(r7_butterfly_scalar_avx2 PROPERTIES TIMEOUT 30)
set_tests_properties(r7_visitors              PROPERTIES TIMEOUT 120)
set_tests_properties(r7_bench                 PROPERTIES TIMEOUT 120)

if(HAS_AVX512F)
    add_test(NAME r7_butterfly_avx512 COMMAND test_butterfly_avx512)
    set_tests_properties(r7_butterfly_avx512 PROPERTIES TIMEOUT 30)
endif()
