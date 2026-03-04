cmake_minimum_required(VERSION 3.16)
project(radix64 LANGUAGES C)

# ── Header-only interface library ──
add_library(radix64 INTERFACE)
target_include_directories(radix64 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/scalar
    ${CMAKE_CURRENT_SOURCE_DIR}/avx2
    ${CMAKE_CURRENT_SOURCE_DIR}/avx512
)

# ── Compiler-specific SIMD flags ──
if(CMAKE_C_COMPILER_ID STREQUAL "IntelLLVM")
    if(CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
        set(RADIX64_AVX2_FLAGS   /arch:AVX2 /QxCORE-AVX2)
        set(RADIX64_AVX512_FLAGS /arch:AVX512 /QxCORE-AVX512)
        set(RADIX64_COMMON_FLAGS /O2 /Qfma)
    else()
        set(RADIX64_AVX2_FLAGS   -mavx2 -mfma)
        set(RADIX64_AVX512_FLAGS -mavx512f -mavx512dq -mfma)
        set(RADIX64_COMMON_FLAGS -O2)
    endif()
elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
    set(RADIX64_AVX2_FLAGS   -mavx2 -mfma)
    set(RADIX64_AVX512_FLAGS -mavx512f -mavx512dq -mfma)
    set(RADIX64_COMMON_FLAGS -O2)
elseif(MSVC)
    set(RADIX64_AVX2_FLAGS   /arch:AVX2)
    set(RADIX64_AVX512_FLAGS /arch:AVX512)
    set(RADIX64_COMMON_FLAGS /O2)
else()
    message(WARNING "Unknown compiler ${CMAKE_C_COMPILER_ID} — SIMD flags may need manual setup")
    set(RADIX64_AVX2_FLAGS "")
    set(RADIX64_AVX512_FLAGS "")
    set(RADIX64_COMMON_FLAGS "")
endif()

set(RADIX64_AVX2_FLAGS   ${RADIX64_AVX2_FLAGS}   CACHE INTERNAL "")
set(RADIX64_AVX512_FLAGS ${RADIX64_AVX512_FLAGS} CACHE INTERNAL "")
set(RADIX64_COMMON_FLAGS ${RADIX64_COMMON_FLAGS} CACHE INTERNAL "")

# ── Tests ──
option(RADIX64_BUILD_TESTS "Build radix64 test executables" ON)
if(RADIX64_BUILD_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()
