cmake_minimum_required(VERSION 3.16)

# ═══════════════════════════════════════════════════════════════
# VectorFFT examples
#
# Builds the quickstart demo that exercises every public API entry
# point in <vfft.h>. Links against the vfft target defined by
# src/CMakeLists.txt. No internal headers used.
# ═══════════════════════════════════════════════════════════════

add_executable(vfft_quickstart quickstart.c)

target_link_libraries(vfft_quickstart PRIVATE vfft)

target_compile_definitions(vfft_quickstart PRIVATE
    _CRT_SECURE_NO_WARNINGS
    $<$<PLATFORM_ID:Windows>:_USE_MATH_DEFINES>
)

# Match the lib's ISA flags so the example uses the same SIMD path.
# vfft target inherits these flags via target_compile_options; the
# example doesn't need them for its own code (no intrinsics), but
# /arch:AVX2 is harmless.
if(VFFT_STRIDE_ISA STREQUAL "avx512")
    if(MSVC OR CMAKE_C_COMPILER_ID MATCHES "Intel")
        target_compile_options(vfft_quickstart PRIVATE /arch:AVX512)
    else()
        target_compile_options(vfft_quickstart PRIVATE -mavx512f -mavx512dq -mfma)
    endif()
elseif(VFFT_STRIDE_ISA STREQUAL "avx2")
    if(MSVC OR CMAKE_C_COMPILER_ID MATCHES "Intel")
        target_compile_options(vfft_quickstart PRIVATE /arch:AVX2)
    else()
        target_compile_options(vfft_quickstart PRIVATE -mavx2 -mfma)
    endif()
endif()

set_target_properties(vfft_quickstart PROPERTIES
    C_STANDARD 11
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
