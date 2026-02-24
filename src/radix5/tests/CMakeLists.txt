cmake_minimum_required(VERSION 3.16)

# ──────────────────────────────────────────────────────────────────────
#  Radix-5 tests — each independently toggleable
#
#  Options (all ON by default when R5_BUILD_TESTS=ON):
#    R5_TEST_BUTTERFLY_AVX2   — AVX2 butterfly unit tests
#    R5_TEST_BUTTERFLY_AVX512 — AVX-512 butterfly unit tests
#    R5_TEST_VISITORS         — end-to-end multi-stage tests + benchmark
# ──────────────────────────────────────────────────────────────────────

enable_testing()

option(R5_TEST_BUTTERFLY_AVX2   "Butterfly unit tests (AVX2)"           ON)
option(R5_TEST_BUTTERFLY_AVX512 "Butterfly unit tests (AVX-512)"        ON)
option(R5_TEST_VISITORS         "End-to-end visitor tests + benchmark"  ON)

set(R5_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

# ── Butterfly AVX2 ──────────────────────────────────────────────────

if(R5_TEST_BUTTERFLY_AVX2 AND HAS_AVX2)
    add_executable(test_r5_avx2 test_radix5_avx2.c)
    target_include_directories(test_r5_avx2 PRIVATE
        ${R5_ROOT}
        ${R5_ROOT}/scalar
        ${R5_ROOT}/avx2
    )
    target_compile_options(test_r5_avx2 PRIVATE ${R5_SIMD_FLAGS})
    target_link_libraries(test_r5_avx2 PRIVATE ${R5_MATH_LIB})

    add_test(NAME r5_butterfly_avx2 COMMAND test_r5_avx2)
    set_tests_properties(r5_butterfly_avx2 PROPERTIES TIMEOUT 30)
endif()

# ── Butterfly AVX-512 ──────────────────────────────────────────────

if(R5_TEST_BUTTERFLY_AVX512 AND R5_HAS_AVX512F)
    add_executable(test_r5_avx512 test_radix5_avx512.c)
    target_include_directories(test_r5_avx512 PRIVATE
        ${R5_ROOT}
        ${R5_ROOT}/scalar
        ${R5_ROOT}/avx512
    )
    target_compile_options(test_r5_avx512 PRIVATE ${R5_SIMD_FLAGS})
    target_link_libraries(test_r5_avx512 PRIVATE ${R5_MATH_LIB})

    add_test(NAME r5_butterfly_avx512 COMMAND test_r5_avx512)
    set_tests_properties(r5_butterfly_avx512 PROPERTIES TIMEOUT 30)
endif()

# ── End-to-end visitor tests + benchmark ────────────────────────────

if(R5_TEST_VISITORS)
    add_executable(test_r5_visitors test_radix5_visitors.c)
    target_link_libraries(test_r5_visitors PRIVATE fft_radix5)
    target_compile_options(test_r5_visitors PRIVATE ${R5_SIMD_FLAGS})

    add_test(NAME r5_visitors COMMAND test_r5_visitors)
    set_tests_properties(r5_visitors PROPERTIES TIMEOUT 120)
endif()
