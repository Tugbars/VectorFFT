# ============================================================================
# VectorFFT — radix25 codelet module
#
# Header-only: all codelets are generated .h files included by dispatch.
# This CMakeLists creates an INTERFACE library for include paths and
# adds the test/benchmark subdirectory.
# ============================================================================

# ── Interface library (header-only, carries include paths) ───────────────
add_library(vfft_radix25 INTERFACE)

target_include_directories(vfft_radix25 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}            # fft_radix25_dispatch.h, fft_radix25_dif_dispatch.h
    ${CMAKE_CURRENT_SOURCE_DIR}/avx512     # fft_radix25_avx512_*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/avx2       # fft_radix25_avx2_*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/scalar     # fft_radix25_scalar_*.h
)

message(STATUS "[VectorFFT] Module: radix25 (header-only)")

# ── Tests + benchmarks ───────────────────────────────────────────────────
if(VFFT_BUILD_TESTS OR VFFT_BUILD_BENCHMARKS)
    add_subdirectory(tests)
endif()
