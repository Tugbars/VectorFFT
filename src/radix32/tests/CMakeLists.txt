# ============================================================================
# VectorFFT — radix32/tests
#
# Correctness tests and performance benchmarks for radix-32 codelets.
#
# Tests:
#   test_radix32_dispatch_{isa}   — Full dispatch: all ISA paths, fwd+bwd+roundtrip
#   test_radix32_scalar           — Scalar-only: tw + notw correctness
#
# Benchmarks (require FFTW3):
#   bench_radix32_avx512_tw_{isa} — Twiddled AVX-512 (flat+ladder) vs FFTW
#   bench_radix32_avx2_tw_{isa}   — Twiddled AVX2 flat vs FFTW
#   bench_radix32_notw_{isa}      — Twiddle-less AVX-512+AVX2 vs FFTW
# ============================================================================

# ── Include paths for radix32 headers ────────────────────────────────────
# Tests live in radix32/tests/ but headers are in radix32/{avx512,avx2,scalar}/
set(R32_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(R32_INCLUDE_DIRS
    ${R32_ROOT}             # fft_radix32_dispatch.h
    ${R32_ROOT}/avx512      # fft_radix32_avx512_*.h
    ${R32_ROOT}/avx2        # fft_radix32_avx2_*.h
    ${R32_ROOT}/scalar      # fft_radix32_scalar_*.h
    ${CMAKE_CURRENT_SOURCE_DIR}  # radix32_test_utils.h
)

# ══════════════════════════════════════════════════════════════════════════
# CORRECTNESS TESTS
# ══════════════════════════════════════════════════════════════════════════

# ── test_dispatch: exercises all dispatch paths (needs highest ISA) ──────
#    Compiled with AVX-512 flags: enables __AVX512F__ and __AVX2__ both,
#    so all #ifdef branches in dispatch.h are active.
#    On AVX2-only systems, falls back to AVX2 + scalar paths only.

if(VFFT_HAS_AVX512)
    vfft_add_isa_tests(test_radix32_dispatch "AVX512"
        test_dispatch.c
        LABELS radix32 correctness
        EXTRA_INCLUDES ${R32_INCLUDE_DIRS}
    )
elseif(VFFT_HAS_AVX2)
    vfft_add_isa_tests(test_radix32_dispatch "AVX2"
        test_dispatch.c
        LABELS radix32 correctness
        EXTRA_INCLUDES ${R32_INCLUDE_DIRS}
    )
endif()

# ── test_scalar: no SIMD required, portable baseline ────────────────────
add_executable(test_radix32_scalar test_scalar.c)
target_compile_options(test_radix32_scalar PRIVATE ${VFFT_BASE_FLAGS})
set_target_properties(test_radix32_scalar PROPERTIES C_STANDARD ${VFFT_C_STANDARD})
target_include_directories(test_radix32_scalar PRIVATE ${R32_INCLUDE_DIRS})
if(VFFT_BASE_DEFINITIONS)
    target_compile_definitions(test_radix32_scalar PRIVATE ${VFFT_BASE_DEFINITIONS})
endif()
if(NOT WIN32)
    target_link_libraries(test_radix32_scalar PRIVATE m)
endif()

add_test(NAME test_radix32_scalar COMMAND test_radix32_scalar)
set_tests_properties(test_radix32_scalar PROPERTIES LABELS "scalar;radix32;correctness")
message(STATUS "[VectorFFT] Test: test_radix32_scalar [scalar;radix32;correctness]")

# ══════════════════════════════════════════════════════════════════════════
# BENCHMARKS (require FFTW3)
# ══════════════════════════════════════════════════════════════════════════

if(VFFT_BUILD_BENCHMARKS)

    # ── AVX-512 twiddled vs FFTW ─────────────────────────────────────────
    if(VFFT_HAS_AVX512)
        vfft_add_isa_benchmarks(bench_radix32_avx512_tw "AVX512"
            bench_avx512_tw_vs_fftw.c
            LABELS radix32 benchmark twiddled
            EXTRA_INCLUDES ${R32_INCLUDE_DIRS}
        )
    endif()

    # ── AVX2 twiddled vs FFTW ────────────────────────────────────────────
    #    Compiled with -mavx2 -mfma -mno-avx512f so FFTW comparison is
    #    AVX2 VecFFT vs FFTW's best (which may use AVX-512 internally).
    if(VFFT_HAS_AVX2)
        vfft_add_isa_benchmarks(bench_radix32_avx2_tw "AVX2"
            bench_avx2_tw_vs_fftw.c
            LABELS radix32 benchmark twiddled
            EXTRA_INCLUDES ${R32_INCLUDE_DIRS}
        )
    endif()

    # ── Twiddle-less (AVX-512 + AVX2 + FFTW, all in one binary) ─────────
    #    Compiled with AVX-512 flags to enable both ISA codepaths.
    if(VFFT_HAS_AVX512)
        vfft_add_isa_benchmarks(bench_radix32_notw "AVX512"
            bench_notw_vs_fftw.c
            LABELS radix32 benchmark notw
            EXTRA_INCLUDES ${R32_INCLUDE_DIRS}
        )
    elseif(VFFT_HAS_AVX2)
        vfft_add_isa_benchmarks(bench_radix32_notw "AVX2"
            bench_notw_vs_fftw.c
            LABELS radix32 benchmark notw
            EXTRA_INCLUDES ${R32_INCLUDE_DIRS}
        )
    endif()

endif()