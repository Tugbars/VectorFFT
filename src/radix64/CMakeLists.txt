# ============================================================================
# VectorFFT — radix64 codelet module
#
# Architecture:
#   DIT tw: genfft DAG (K≤12), 8×8 CT + log3 (K>12), scalar DAG fallback
#   DIF tw: 8×8 CT + log3 (SIMD), scalar N1+tw fallback
#   N1: 8×8 CT (split) + genfft DAG (mono IL), scalar fallback
# ============================================================================

add_library(vfft_radix64 INTERFACE)

target_include_directories(vfft_radix64 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/avx512
    ${CMAKE_CURRENT_SOURCE_DIR}/avx2
    ${CMAKE_CURRENT_SOURCE_DIR}/scalar
)

message(STATUS "[VectorFFT] Module: radix64 (DAG + 8x8CT + scalar fallback)")

if(VFFT_BUILD_TESTS OR VFFT_BUILD_BENCHMARKS)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
        add_subdirectory(tests)
    endif()
endif()
