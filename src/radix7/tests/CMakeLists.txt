cmake_minimum_required(VERSION 3.16)

# ──────────────────────────────────────────────────────────────────────
#  Radix-7 tests — each independently toggleable
#
#  Options (all ON by default when R7_BUILD_TESTS=ON):
#    R7_TEST_BUTTERFLY        — scalar + AVX2 butterfly unit tests
#    R7_TEST_BUTTERFLY_AVX512 — AVX-512 butterfly unit tests
#    R7_TEST_VISITORS         — end-to-end multi-stage visitor tests
#    R7_TEST_BENCH            — butterfly-level ISA benchmarks
# ──────────────────────────────────────────────────────────────────────

enable_testing()

option(R7_TEST_BUTTERFLY        "Butterfly unit tests (scalar + AVX2)"  ON)
option(R7_TEST_BUTTERFLY_AVX512 "Butterfly unit tests (AVX-512)"        ON)
option(R7_TEST_VISITORS         "End-to-end multi-stage visitor tests"   ON)
option(R7_TEST_BENCH            "Butterfly-level benchmark (all ISAs)"   ON)

# Parent dir is the radix7 root (include path for headers)
set(R7_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

# ── Butterfly scalar + AVX2 ─────────────────────────────────────────

if(R7_TEST_BUTTERFLY)
    add_executable(test_butterfly test_radix7_rader.c)
    target_include_directories(test_butterfly PRIVATE ${R7_ROOT})
    target_compile_options(test_butterfly PRIVATE ${R7_SIMD_FLAGS})
    target_link_libraries(test_butterfly PRIVATE ${R7_MATH_LIB})

    add_test(NAME r7_butterfly COMMAND test_butterfly)
    set_tests_properties(r7_butterfly PROPERTIES TIMEOUT 30)
endif()

# ── Butterfly AVX-512 ───────────────────────────────────────────────

if(R7_TEST_BUTTERFLY_AVX512 AND R7_HAS_AVX512F)
    add_executable(test_butterfly_avx512 test_radix7_avx512.c)
    target_include_directories(test_butterfly_avx512 PRIVATE ${R7_ROOT})
    target_compile_options(test_butterfly_avx512 PRIVATE ${R7_SIMD_FLAGS})
    target_link_libraries(test_butterfly_avx512 PRIVATE ${R7_MATH_LIB})

    add_test(NAME r7_butterfly_avx512 COMMAND test_butterfly_avx512)
    set_tests_properties(r7_butterfly_avx512 PROPERTIES TIMEOUT 30)
endif()

# ── End-to-end visitor tests ────────────────────────────────────────

if(R7_TEST_VISITORS)
    add_executable(test_visitors test_visitors.c)
    target_link_libraries(test_visitors PRIVATE fft_radix7)
    target_compile_options(test_visitors PRIVATE ${R7_SIMD_FLAGS})

    add_test(NAME r7_visitors COMMAND test_visitors)
    set_tests_properties(r7_visitors PROPERTIES TIMEOUT 120)
endif()

# ── Benchmark ────────────────────────────────────────────────────────

if(R7_TEST_BENCH)
    add_executable(bench_butterfly bench_radix7.c)
    target_include_directories(bench_butterfly PRIVATE ${R7_ROOT})
    target_compile_options(bench_butterfly PRIVATE ${R7_SIMD_FLAGS})
    target_link_libraries(bench_butterfly PRIVATE ${R7_MATH_LIB})

    add_test(NAME r7_bench COMMAND bench_butterfly)
    set_tests_properties(r7_bench PROPERTIES TIMEOUT 120)
endif()