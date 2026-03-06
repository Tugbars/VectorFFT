# ============================================================================
# VectorFFT — radix3 codelet module
#
# Header-only: all codelets are generated .h files included by dispatch.
# This CMakeLists creates an INTERFACE library for include paths and
# adds the test/benchmark subdirectory.
# ============================================================================

# ── Interface library (header-only, carries include paths) ───────────────
add_library(vfft_radix3 INTERFACE)

target_include_directories(vfft_radix3 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}            # fft_radix3_dispatch.h, fft_radix3_tw_packed.h
    ${CMAKE_CURRENT_SOURCE_DIR}/avx512     # fft_radix3_avx512.h
    ${CMAKE_CURRENT_SOURCE_DIR}/avx2       # fft_radix3_avx2.h
    ${CMAKE_CURRENT_SOURCE_DIR}/scalar     # fft_radix3_scalar.h
)

message(STATUS "[VectorFFT] Module: radix3 (header-only)")

# ── Tests + benchmarks ───────────────────────────────────────────────────
if(VFFT_BUILD_TESTS OR VFFT_BUILD_BENCHMARKS)
    add_subdirectory(tests)
endif()
