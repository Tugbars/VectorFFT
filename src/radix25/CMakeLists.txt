# ============================================================================
# VectorFFT — radix25 codelet module
#
# Header-only: all codelets are generated .h files included by dispatch.
# This CMakeLists creates an INTERFACE library for include paths and
# adds the test/benchmark subdirectory.
#
# Codelet inventory (v2):
#   dispatch    — fft_radix25_dispatch.h (N1 + tw DIT/DIF + IL tw + N1 mono IL)
#   scalar/     — n1, tw, dif_tw, il_tw, il_dif_tw
#   avx2/       — n1, tw, dif_tw, il_tw, il_dif_tw, n1_mono_il
#   avx512/     — n1, tw, dif_tw, il_tw, il_dif_tw, n1_mono_il
#
# v2 optimizations:
#   - 5×5 CT + log3 twiddles (depth-2 tree, sequential cross tw, inline broadcasts)
#   - Monolithic genfft N1 native IL (translated from FFTW DAG, zero spills)
# ============================================================================

# ── Interface library (header-only, carries include paths) ───────────────
add_library(vfft_radix25 INTERFACE)

target_include_directories(vfft_radix25 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}            # fft_radix25_dispatch.h
    ${CMAKE_CURRENT_SOURCE_DIR}/avx512     # fft_radix25_avx512_*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/avx2       # fft_radix25_avx2_*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/scalar     # fft_radix25_scalar_*.h
)

message(STATUS "[VectorFFT] Module: radix25 (header-only, v2 — 5x5 CT + log3 + mono N1 IL)")

# ── Tests + benchmarks ───────────────────────────────────────────────────
if(VFFT_BUILD_TESTS OR VFFT_BUILD_BENCHMARKS)
    add_subdirectory(tests)
endif()
