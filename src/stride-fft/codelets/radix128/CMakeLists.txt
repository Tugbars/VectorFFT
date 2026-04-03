cmake_minimum_required(VERSION 3.16)
project(radix128 LANGUAGES C)

# ── Header-only interface library ──
add_library(radix128 INTERFACE)
target_include_directories(radix128 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/scalar
    ${CMAKE_CURRENT_SOURCE_DIR}/avx2
    ${CMAKE_CURRENT_SOURCE_DIR}/avx512
)

# ── Compiler-specific SIMD flags ──
# Detect IntelLLVM (icx) vs GCC/Clang vs MSVC
# ICX on Windows uses MSVC frontend by default; on Linux uses GNU frontend.

if(CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM")
    if(CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        # ICX Windows MSVC-like driver
        set(RADIX128_AVX2_FLAGS   /arch:AVX2 /QxCORE-AVX2)
        set(RADIX128_AVX512_FLAGS /arch:AVX512 /QxCORE-AVX512)
        set(RADIX128_COMMON_FLAGS /O2 /Qfma)
    else()
        # ICX Linux / ICX Windows with GNU-like driver
        set(RADIX128_AVX2_FLAGS   -mavx2 -mfma)
        set(RADIX128_AVX512_FLAGS -mavx512f -mavx512dq -mfma)
        set(RADIX128_COMMON_FLAGS -O2)
    endif()
elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
    set(RADIX128_AVX2_FLAGS   -mavx2 -mfma)
    set(RADIX128_AVX512_FLAGS -mavx512f -mavx512dq -mfma)
    set(RADIX128_COMMON_FLAGS -O2)
elseif(MSVC)
    set(RADIX128_AVX2_FLAGS   /arch:AVX2)
    set(RADIX128_AVX512_FLAGS /arch:AVX512)
    set(RADIX128_COMMON_FLAGS /O2)
else()
    message(WARNING "Unknown compiler ${CMAKE_C_COMPILER_ID} — SIMD flags may need manual setup")
    set(RADIX128_AVX2_FLAGS "")
    set(RADIX128_AVX512_FLAGS "")
    set(RADIX128_COMMON_FLAGS "")
endif()

# Export flags for test subdirectory
set(RADIX128_AVX2_FLAGS   ${RADIX128_AVX2_FLAGS}   CACHE INTERNAL "")
set(RADIX128_AVX512_FLAGS ${RADIX128_AVX512_FLAGS} CACHE INTERNAL "")
set(RADIX128_COMMON_FLAGS ${RADIX128_COMMON_FLAGS} CACHE INTERNAL "")

# ── Tests ──
option(RADIX128_BUILD_TESTS "Build radix128 test executables" ON)
if(RADIX128_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
