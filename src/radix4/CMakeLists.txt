# ============================================================================
# src/radix4/CMakeLists.txt
# Radix-4 DIF butterfly — SoA native, forward + backward
#
# Builds one OBJECT library per ISA (radix4_avx512, radix4_avx2, radix4_sse2).
# The dispatch .c files use #ifdef to select the active code path, so the
# same source is compiled 3× with different -m flags.
# ============================================================================

set(RADIX4_SOURCES
    fft_radix4_bv.c
    fft_radix4_fv.c
)

# ── Per-ISA object libraries ─────────────────────────────────────────────
# vfft_add_isa_library() comes from cmake/VectorFFTCompiler.cmake.
# It sets flags, include paths, and skips gracefully if ISA unsupported.

vfft_add_isa_library(radix4_avx512  AVX512  ${RADIX4_SOURCES})
vfft_add_isa_library(radix4_avx2    AVX2    ${RADIX4_SOURCES})
# vfft_add_isa_library(radix4_sse2  SSE2    ${RADIX4_SOURCES})  # TODO: SSE2 kernels
