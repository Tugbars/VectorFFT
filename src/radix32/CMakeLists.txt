cmake_minimum_required(VERSION 3.20)
project(radix32 LANGUAGES C)

# ============================================================================
# Compiler detection
# ============================================================================
set(R32_IS_ICX   FALSE)
set(R32_IS_GCC   FALSE)
set(R32_IS_CLANG FALSE)
set(R32_IS_MSVC  FALSE)

if(CMAKE_C_COMPILER_ID MATCHES "IntelLLVM")
    set(R32_IS_ICX TRUE)
    message(STATUS "radix32: ICX detected (${CMAKE_C_COMPILER_VERSION})")
elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(R32_IS_GCC TRUE)
    message(STATUS "radix32: GCC detected (${CMAKE_C_COMPILER_VERSION})")
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(R32_IS_CLANG TRUE)
    message(STATUS "radix32: Clang detected (${CMAKE_C_COMPILER_VERSION})")
elseif(MSVC)
    set(R32_IS_MSVC TRUE)
    message(STATUS "radix32: MSVC detected (${CMAKE_C_COMPILER_VERSION})")
endif()

# ============================================================================
# Options
# ============================================================================
option(R32_BUILD_TESTS    "Build benchmarks + correctness tests"            ON)
option(R32_BUILD_AVX512   "Enable AVX-512 backend (requires F+DQ)"         ON)
option(R32_BUILD_SSE2     "Enable SSE2 backend (placeholder)"              OFF)

set(R32_ICX_AVX2_ARCH   "CORE-AVX2"   CACHE STRING "ICX -x target for AVX2")
set(R32_ICX_AVX512_ARCH "CORE-AVX512" CACHE STRING "ICX -x target for AVX-512")

# ============================================================================
# Global settings
# ============================================================================
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ============================================================================
# Warning flags
# ============================================================================
if(R32_IS_MSVC)
    set(R32_WARN /W3)
else()
    set(R32_WARN -Wall -Wextra -Wpedantic -Wno-unused-parameter -Wno-unused-function)
endif()

# ============================================================================
# INTERFACE LIBRARY: radix32_scalar
#
# Headers: scalar/fft_radix32_scalar_gen.h
# No ISA flags — pure C.
# ============================================================================
add_library(radix32_scalar INTERFACE)
target_include_directories(radix32_scalar INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/scalar>
    $<INSTALL_INTERFACE:include/radix32/scalar>
)

# ============================================================================
# INTERFACE LIBRARY: radix32_avx2
#
# Headers: avx2/fft_radix32_avx2_n1.h
#          avx2/fft_radix32_avx2_tw.h
#          avx2/fft_radix32_avx2_tw_packed.h
#          avx2/fft_radix32_avx2_tw_unified.h
# ============================================================================
add_library(radix32_avx2 INTERFACE)
target_include_directories(radix32_avx2 INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/avx2>
    $<INSTALL_INTERFACE:include/radix32/avx2>
)
if(R32_IS_ICX)
    target_compile_options(radix32_avx2 INTERFACE -x${R32_ICX_AVX2_ARCH})
elseif(R32_IS_MSVC)
    target_compile_options(radix32_avx2 INTERFACE /arch:AVX2)
else()
    target_compile_options(radix32_avx2 INTERFACE -mavx2 -mfma)
endif()

# ============================================================================
# INTERFACE LIBRARY: radix32_avx512  (conditional)
#
# Headers: avx512/fft_radix32_avx512_n1_u2.h
#          avx512/fft_radix32_avx512_tw_ladder.h
#          avx512/fft_radix32_avx512_tw_packed.h
#          avx512/fft_radix32_avx512_tw_unified.h
# ============================================================================
if(R32_BUILD_AVX512)
    add_library(radix32_avx512 INTERFACE)
    target_include_directories(radix32_avx512 INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/avx512>
        $<INSTALL_INTERFACE:include/radix32/avx512>
    )
    if(R32_IS_ICX)
        target_compile_options(radix32_avx512 INTERFACE -x${R32_ICX_AVX512_ARCH})
    elseif(R32_IS_MSVC)
        target_compile_options(radix32_avx512 INTERFACE /arch:AVX512)
    else()
        target_compile_options(radix32_avx512 INTERFACE -mavx512f -mavx512dq -mfma)
    endif()
    message(STATUS "radix32: AVX-512 enabled")
endif()

# ============================================================================
# TOP-LEVEL INTERFACE: radix32
#
# Header: fft_radix32_dispatch.h
#
# This is what users link against. It includes all backends and does
# runtime ISA dispatch. The top-level include directory is radix32/
# itself, so #include "fft_radix32_dispatch.h" works.
#
# Compile with the HIGHEST ISA flags so all backend headers can be
# parsed. Runtime CPUID gates which path actually executes.
# ============================================================================
add_library(radix32 INTERFACE)

target_include_directories(radix32 INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/radix32>
)

target_link_libraries(radix32 INTERFACE radix32_scalar radix32_avx2)
if(R32_BUILD_AVX512)
    target_link_libraries(radix32 INTERFACE radix32_avx512)
endif()

# ============================================================================
# Tests
# ============================================================================
if(R32_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "  radix32 configuration")
message(STATUS "  ---------------------")
message(STATUS "  Compiler:   ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Scalar:     ON")
message(STATUS "  AVX2:       ON")
message(STATUS "  AVX-512:    ${R32_BUILD_AVX512}")
message(STATUS "  Tests:      ${R32_BUILD_TESTS}")
if(R32_IS_ICX)
    message(STATUS "  ICX AVX2:   -x${R32_ICX_AVX2_ARCH}")
    message(STATUS "  ICX AVX512: -x${R32_ICX_AVX512_ARCH}")
endif()
message(STATUS "")
