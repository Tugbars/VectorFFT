# ===========================================================================
# src/radix2/CMakeLists.txt — Radix-2 butterfly kernels
#
# Builds fft_radix2_bv.c three times (one per ISA tier).
# Test and bench sources live in tests/ and are built per-ISA.
# ===========================================================================

# ── Source files ─────────────────────────────────────────────────────────
set(RADIX2_SOURCES
    fft_radix2_bv.c
)

# fft_radix2_bv.c includes ISA headers via #ifdef, so each build
# picks up the right avx512/*.h, avx2/*.h, sse2/*.h, scalar/*.h
# automatically. Include dirs are set by vfft_add_isa_library().

# ── Object libraries (one per ISA) ──────────────────────────────────────
vfft_add_isa_library(radix2_avx512 AVX512 ${RADIX2_SOURCES})
vfft_add_isa_library(radix2_avx2   AVX2   ${RADIX2_SOURCES})
vfft_add_isa_library(radix2_sse2   SSE2   ${RADIX2_SOURCES})

# ── Tests ────────────────────────────────────────────────────────────────
if(VFFT_BUILD_TESTS)
    # Correctness: single source → 3 executables
    # test_radix2_bv_avx512, test_radix2_bv_avx2, test_radix2_bv_sse2
    vfft_add_isa_tests(test_radix2_bv "AVX512;AVX2;SSE2"
        tests/test_radix2_bv.c
        ${RADIX2_SOURCES}
    )
endif()

# ── Benchmarks ───────────────────────────────────────────────────────────
if(VFFT_BUILD_BENCHMARKS)
    # Self-benchmark: radix2 standalone perf
    vfft_add_isa_benchmarks(bench_radix2_bv "AVX512;AVX2;SSE2"
        tests/bench_radix2_bv.c
        ${RADIX2_SOURCES}
    )

    # FFTW comparison (only built if FFTW3 found)
    if(TARGET FFTW3::fftw3)
        vfft_add_isa_benchmarks(bench_radix2_vs_fftw "AVX512;AVX2;SSE2"
            tests/bench_vs_fftw.c
            ${RADIX2_SOURCES}
        )
    endif()
endif()
