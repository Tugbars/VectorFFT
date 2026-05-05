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

# Math library is needed under GCC/Clang on Linux and MinGW; harmless on MSVC.
if(NOT MSVC)
    target_link_libraries(vfft_quickstart PRIVATE m)
endif()

# Bake an absolute path to the calibrated wisdom file shipped with the
# repo. The example tries this path first; if missing, falls back to
# VFFT_ESTIMATE. Users who run the binary out-of-tree can override at
# runtime with the VFFT_WISDOM environment variable.
target_compile_definitions(vfft_quickstart PRIVATE
    _CRT_SECURE_NO_WARNINGS
    $<$<PLATFORM_ID:Windows>:_USE_MATH_DEFINES>
    VFFT_WISDOM_DEFAULT_PATH="${CMAKE_SOURCE_DIR}/build_tuned/vfft_wisdom_tuned.txt"
)

# GCC on MinGW has a known issue at -O3 where compile-then-link of this
# specific source segfaults (works fine at -O2 or with -flto). Override
# the default Release flags down to -O2 for this target only.
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(vfft_quickstart PRIVATE -O2)
endif()

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
