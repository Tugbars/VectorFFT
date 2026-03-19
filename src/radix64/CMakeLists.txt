# ============================================================================
# VectorFFT — radix64 codelet module
#
# Codelet inventory (v2):
#   dispatch    — fft_radix64_dispatch.h
#   avx2/       — n1_gen, tw, il_tw, n1_mono_il
#   avx512/     — n1_gen, tw, il_tw, n1_mono_il
# ============================================================================

add_library(vfft_radix64 INTERFACE)

target_include_directories(vfft_radix64 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/avx512
    ${CMAKE_CURRENT_SOURCE_DIR}/avx2
)

message(STATUS "[VectorFFT] Module: radix64 (header-only, v2 — 8x8 CT + log3 + mono N1 IL)")

if(VFFT_BUILD_TESTS OR VFFT_BUILD_BENCHMARKS)
    add_subdirectory(tests)
endif()
