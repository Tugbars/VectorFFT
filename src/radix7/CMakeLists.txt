# ============================================================================
# VectorFFT — radix7 codelet module
#
# Header-only: all codelets are generated .h files included by dispatch.
# This CMakeLists creates an INTERFACE library for include paths and
# adds the test/benchmark subdirectory.
# ============================================================================

add_library(vfft_radix7 INTERFACE)

target_include_directories(vfft_radix7 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}            # fft_radix7_dispatch.h, fft_radix7_tw_packed.h
    ${CMAKE_CURRENT_SOURCE_DIR}/avx512     # fft_radix7_avx512.h
    ${CMAKE_CURRENT_SOURCE_DIR}/avx2       # fft_radix7_avx2.h
    ${CMAKE_CURRENT_SOURCE_DIR}/scalar     # fft_radix7_scalar.h
)

message(STATUS "[VectorFFT] Module: radix7 (header-only)")

if(VFFT_BUILD_TESTS OR VFFT_BUILD_BENCHMARKS)
    add_subdirectory(tests)
endif()
