# ============================================================================
# tests/CMakeLists.txt
# Tests and benchmarks for radix-4
#
# One test source → N executables (one per ISA) via vfft_add_isa_tests.
# Benchmarks built similarly but not registered with CTest.
#
# NOTE: vfft_add_isa_tests sets include paths relative to CMAKE_CURRENT_SOURCE_DIR
# (which is tests/). Since radix4 headers live in src/radix4/, we add that path
# via RADIX4_INCLUDE_HINT which the helper picks up from the parent scope.
# Alternatively, the dispatch .c sources are compiled directly into each test
# executable, so the compiler needs -I src/radix4/ to resolve the internal
# #include "avx2/..." paths.
# ============================================================================

set(RADIX4_DIR ${PROJECT_SOURCE_DIR}/src/radix4)

set(RADIX4_TEST_ISAS "AVX512;AVX2")
# Append SSE2 when kernels are ready:
# set(RADIX4_TEST_ISAS "AVX512;AVX2;SSE2")

# ── Helper: add radix4 include paths to test/bench targets ───────────────
# The vfft_add_isa_* helpers create targets named <base>_<isa_lower>.
# We post-fix them with the include dirs they need.
macro(radix4_fixup_includes BASE_NAME ISA_LIST)
    foreach(ISA IN LISTS ${ISA_LIST})
        string(TOLOWER ${ISA} _isa_lower)
        set(_tgt "${BASE_NAME}_${_isa_lower}")
        if(TARGET ${_tgt})
            target_include_directories(${_tgt} PRIVATE
                ${RADIX4_DIR}
                ${RADIX4_DIR}/avx2
                ${RADIX4_DIR}/avx512
                ${RADIX4_DIR}/scalar
            )
        endif()
    endforeach()
endmacro()

# ── Correctness tests ────────────────────────────────────────────────────

vfft_add_isa_tests(test_radix4_bv "${RADIX4_TEST_ISAS}"
    test_radix4_bv.c
    ${RADIX4_DIR}/fft_radix4_bv.c
)
radix4_fixup_includes(test_radix4_bv RADIX4_TEST_ISAS)

vfft_add_isa_tests(test_radix4_fv "${RADIX4_TEST_ISAS}"
    test_radix4_fv.c
    ${RADIX4_DIR}/fft_radix4_fv.c
)
radix4_fixup_includes(test_radix4_fv RADIX4_TEST_ISAS)

vfft_add_isa_tests(test_roundtrip "${RADIX4_TEST_ISAS}"
    test_roundtrip.c
    ${RADIX4_DIR}/fft_radix4_fv.c
    ${RADIX4_DIR}/fft_radix4_bv.c
)
radix4_fixup_includes(test_roundtrip RADIX4_TEST_ISAS)

# ── Benchmarks ───────────────────────────────────────────────────────────

vfft_add_isa_benchmarks(bench_radix4 "${RADIX4_TEST_ISAS}"
    bench_radix4.c
    ${RADIX4_DIR}/fft_radix4_fv.c
    ${RADIX4_DIR}/fft_radix4_bv.c
)
radix4_fixup_includes(bench_radix4 RADIX4_TEST_ISAS)
