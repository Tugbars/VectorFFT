cmake_minimum_required(VERSION 3.20)
# ============================================================================
# Common compile options for test targets
# ============================================================================
set(RADIX32_TEST_FLAGS
    -O2
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
    -march=native
)
# ============================================================================
# Unit tests
# ============================================================================
add_executable(test_radix32_avx2 test_radix32_avx2.c)
target_link_libraries(test_radix32_avx2 PRIVATE radix32_avx2 m)
target_compile_options(test_radix32_avx2 PRIVATE ${RADIX32_TEST_FLAGS})
add_test(
    NAME    radix32_avx2_unit
    COMMAND test_radix32_avx2
)
# ============================================================================
# Latency benchmark
# ============================================================================
add_executable(bench_radix32_avx2 bench_radix32_avx2.c)
target_link_libraries(bench_radix32_avx2 PRIVATE radix32_avx2 m)
target_compile_options(bench_radix32_avx2 PRIVATE ${RADIX32_TEST_FLAGS})
add_test(
    NAME    radix32_avx2_bench
    COMMAND bench_radix32_avx2
)
# ============================================================================
# GTest: n1 codelet roundtrip
# ============================================================================
add_executable(test_radix32_n1_roundtrip test_radix32_n1_roundtrip.cpp)
target_link_libraries(test_radix32_n1_roundtrip PRIVATE
    radix32_avx2
    GTest::gtest_main
    m
)
target_compile_options(test_radix32_n1_roundtrip PRIVATE
    ${RADIX32_TEST_FLAGS}
)
include(GoogleTest)
gtest_discover_tests(test_radix32_n1_roundtrip
    TEST_PREFIX "n1_roundtrip/"
)
# ============================================================================
# GTest: twiddle codelet roundtrip
# ============================================================================
add_executable(test_radix32_twiddle_roundtrip test_radix32_twiddle_roundtrip.cpp)
target_link_libraries(test_radix32_twiddle_roundtrip PRIVATE
    radix32_avx2
    GTest::gtest_main
    m
)
target_compile_options(test_radix32_twiddle_roundtrip PRIVATE
    ${RADIX32_TEST_FLAGS}
    -fpermissive     # C header uses int for _mm_prefetch hint (valid C, pedantic C++)
    -Wno-pedantic
)
gtest_discover_tests(test_radix32_twiddle_roundtrip
    TEST_PREFIX "twiddle_roundtrip/"
)
