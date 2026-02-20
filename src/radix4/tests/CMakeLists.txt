# ============================================================================
# src/radix4/tests/CMakeLists.txt
# Tests and benchmarks for radix-4
# ============================================================================

set(RADIX4_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)

set(RADIX4_ISAS "AVX512;AVX2")

set(RADIX4_INCLUDES
    ${RADIX4_DIR}
    ${RADIX4_DIR}/avx2
    ${RADIX4_DIR}/avx512
    ${RADIX4_DIR}/scalar
)

# ── Correctness tests ────────────────────────────────────────────────────

vfft_add_isa_tests(test_radix4_bv "${RADIX4_ISAS}"
    test_radix4_bv.c
    ${RADIX4_DIR}/fft_radix4_bv.c
    LABELS radix4
    EXTRA_INCLUDES ${RADIX4_INCLUDES}
)

vfft_add_isa_tests(test_radix4_fv "${RADIX4_ISAS}"
    test_radix4_fv.c
    ${RADIX4_DIR}/fft_radix4_fv.c
    LABELS radix4
    EXTRA_INCLUDES ${RADIX4_INCLUDES}
)

vfft_add_isa_tests(test_roundtrip "${RADIX4_ISAS}"
    test_roundtrip.c
    ${RADIX4_DIR}/fft_radix4_fv.c
    ${RADIX4_DIR}/fft_radix4_bv.c
    LABELS radix4
    EXTRA_INCLUDES ${RADIX4_INCLUDES}
)

# ── Benchmarks ───────────────────────────────────────────────────────────

vfft_add_isa_benchmarks(bench_radix4 "${RADIX4_ISAS}"
    bench_radix4.c
    ${RADIX4_DIR}/fft_radix4_fv.c
    ${RADIX4_DIR}/fft_radix4_bv.c
    LABELS radix4
    EXTRA_INCLUDES ${RADIX4_INCLUDES}
)