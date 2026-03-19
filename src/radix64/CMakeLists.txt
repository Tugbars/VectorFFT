# ============================================================================
# VectorFFT — radix64 codelet module
#
# Codelet inventory:
#   dispatch    — fft_radix64_dispatch.h (DAG K≤12, 8×8CT K>12)
#   avx2/       — n1_gen, tw, tw_dag, il_tw, n1_mono_il
#   avx512/     — n1_gen, tw, tw_dag, il_tw, n1_mono_il
#
# Architecture:
#   DIT tw: genfft DAG (Frigo/Johnson) at low K, 8×8 CT + log3 at high K
#   DIF tw: 8×8 CT + log3 (all K)
#   N1: 8×8 CT (split) + genfft DAG (mono IL)
#   AVX2: log3 + split butterfly (peak 12 YMM)
#   AVX-512: log3 + monolithic butterfly (32 ZMM)
# ============================================================================

add_library(vfft_radix64 INTERFACE)

target_include_directories(vfft_radix64 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/avx512
    ${CMAKE_CURRENT_SOURCE_DIR}/avx2
)

message(STATUS "[VectorFFT] Module: radix64 (DAG + 8x8CT, ISA-specific)")

if(VFFT_BUILD_TESTS OR VFFT_BUILD_BENCHMARKS)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        add_subdirectory(tests)
    endif()
endif()
