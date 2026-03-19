# ============================================================================
# VectorFFT — radix64 codelet module
#
# Header-only: all codelets are generated .h files included by dispatch.
# This CMakeLists creates an INTERFACE library for include paths and
# adds the test/benchmark subdirectory.
#
# Codelet inventory (v2):
#   dispatch    — fft_radix64_dispatch.h (N1 + tw DIT/DIF + IL tw + N1 mono IL)
#   avx2/       — n1_gen, tw, il_tw, n1_mono_il
#   avx512/     — tw, il_tw, n1_mono_il
#
# v2 optimizations:
#   - 8x8 CT + log3 twiddles (depth-3 tree, sequential cross tw,
#     static W64 arrays for {1to8} broadcast, XOR negation)
#   - Monolithic genfft N1 native IL (translated from FFTW DAG, zero spills)
# ============================================================================

# ── Interface library (header-only, carries include paths) ───────────────
add_library(vfft_radix64 INTERFACE)

target_include_directories(vfft_radix64 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}            # fft_radix64_dispatch.h
    ${CMAKE_CURRENT_SOURCE_DIR}/avx512     # fft_radix64_avx512_*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/avx2       # fft_radix64_avx2_*.h
)

message(STATUS "[VectorFFT] Module: radix64 (header-only, v2 — 8x8 CT + log3 + mono N1 IL)")

# ── Tests + benchmarks ───────────────────────────────────────────────────
if(VFFT_BUILD_TESTS OR VFFT_BUILD_BENCHMARKS)
    add_subdirectory(tests)
endif()
