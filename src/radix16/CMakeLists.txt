cmake_minimum_required(VERSION 3.20)
project(radix16 LANGUAGES C)

# ============================================================================
# Compiler detection
# ============================================================================
set(R16_IS_ICX   FALSE)
set(R16_IS_GCC   FALSE)
set(R16_IS_CLANG FALSE)
set(R16_IS_MSVC  FALSE)

if(CMAKE_C_COMPILER_ID MATCHES "IntelLLVM")
    set(R16_IS_ICX TRUE)
    message(STATUS "radix16: ICX detected (${CMAKE_C_COMPILER_VERSION})")
elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(R16_IS_GCC TRUE)
    message(STATUS "radix16: GCC detected (${CMAKE_C_COMPILER_VERSION})")
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(R16_IS_CLANG TRUE)
    message(STATUS "radix16: Clang detected (${CMAKE_C_COMPILER_VERSION})")
elseif(MSVC)
    set(R16_IS_MSVC TRUE)
    message(STATUS "radix16: MSVC detected (${CMAKE_C_COMPILER_VERSION})")
endif()

# ============================================================================
# Options
# ============================================================================
option(R16_BUILD_TESTS    "Build benchmarks + correctness tests"            ON)
option(R16_BUILD_AVX512   "Enable AVX-512 backend (requires F+DQ)"         ON)

set(R16_ICX_AVX2_ARCH   "CORE-AVX2"   CACHE STRING "ICX -x target for AVX2")
set(R16_ICX_AVX512_ARCH "CORE-AVX512" CACHE STRING "ICX -x target for AVX-512")

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
if(R16_IS_MSVC)
    set(R16_WARN /W3)
else()
    set(R16_WARN -Wall -Wextra -Wpedantic -Wno-unused-parameter -Wno-unused-function)
endif()

# ============================================================================
# INTERFACE LIBRARY: radix16_scalar
# ============================================================================
add_library(radix16_scalar INTERFACE)
target_include_directories(radix16_scalar INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/scalar>
    $<INSTALL_INTERFACE:include/radix16/scalar>
)

# ============================================================================
# INTERFACE LIBRARY: radix16_avx2
# ============================================================================
add_library(radix16_avx2 INTERFACE)
target_include_directories(radix16_avx2 INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/avx2>
    $<INSTALL_INTERFACE:include/radix16/avx2>
)
if(R16_IS_ICX)
    target_compile_options(radix16_avx2 INTERFACE -x${R16_ICX_AVX2_ARCH})
elseif(R16_IS_MSVC)
    target_compile_options(radix16_avx2 INTERFACE /arch:AVX2)
else()
    target_compile_options(radix16_avx2 INTERFACE -mavx2 -mfma)
endif()

# ============================================================================
# INTERFACE LIBRARY: radix16_avx512 (conditional)
# ============================================================================
if(R16_BUILD_AVX512)
    add_library(radix16_avx512 INTERFACE)
    target_include_directories(radix16_avx512 INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/avx512>
        $<INSTALL_INTERFACE:include/radix16/avx512>
    )
    if(R16_IS_ICX)
        target_compile_options(radix16_avx512 INTERFACE -x${R16_ICX_AVX512_ARCH})
    elseif(R16_IS_MSVC)
        target_compile_options(radix16_avx512 INTERFACE /arch:AVX512)
    else()
        target_compile_options(radix16_avx512 INTERFACE -mavx512f -mavx512dq -mfma)
    endif()
    message(STATUS "radix16: AVX-512 enabled")
endif()

# ============================================================================
# TOP-LEVEL INTERFACE: radix16
#
# Header: fft_radix16_dispatch.h + fft_radix16_tw_packed.h
# ============================================================================
add_library(radix16 INTERFACE)
target_include_directories(radix16 INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/radix16>
)
target_link_libraries(radix16 INTERFACE radix16_scalar radix16_avx2)
if(R16_BUILD_AVX512)
    target_link_libraries(radix16 INTERFACE radix16_avx512)
endif()

# ============================================================================
# Tests
# ============================================================================
if(R16_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "  radix16 configuration")
message(STATUS "  ---------------------")
message(STATUS "  Compiler:   ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Scalar:     ON")
message(STATUS "  AVX2:       ON")
message(STATUS "  AVX-512:    ${R16_BUILD_AVX512}")
message(STATUS "  Tests:      ${R16_BUILD_TESTS}")
if(R16_IS_ICX)
    message(STATUS "  ICX AVX2:   -x${R16_ICX_AVX2_ARCH}")
    message(STATUS "  ICX AVX512: -x${R16_ICX_AVX512_ARCH}")
endif()
message(STATUS "")
