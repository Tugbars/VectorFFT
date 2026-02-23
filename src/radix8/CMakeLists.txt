##############################################################################
# radix8/ CMakeLists.txt
#
# Radix-8 FFT stage library — SoA split-complex, hybrid blocked twiddles.
# Dispatch chain: AVX-512 → AVX2 → scalar (FMA).
#
# Usage:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake -B build -DCMAKE_BUILD_TYPE=Release -DRADIX8_BUILD_TESTS=ON
#   cmake -B build -DCMAKE_BUILD_TYPE=Release -DRADIX8_ISA=AVX2     # force ISA
#   cmake --build build
#   ctest --test-dir build --output-on-failure
#
# Options:
#   RADIX8_BUILD_TESTS   ON/OFF  Build test executables         (default: OFF)
#   RADIX8_ISA           AUTO | AVX512 | AVX2 | SCALAR          (default: AUTO)
#
##############################################################################
cmake_minimum_required(VERSION 3.16)
project(radix8 LANGUAGES C)

# ── Options ─────────────────────────────────────────────────────────────────
option(RADIX8_BUILD_TESTS "Build radix-8 test executables" OFF)

set(RADIX8_ISA "AUTO" CACHE STRING "Target ISA: AUTO, AVX512, AVX2, SCALAR")
set_property(CACHE RADIX8_ISA PROPERTY STRINGS AUTO AVX512 AVX2 SCALAR)

# ── Compiler baseline ───────────────────────────────────────────────────────
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W3)
else()
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

# ── ISA detection / flag setup ──────────────────────────────────────────────
#
# We probe the compiler to see what it can emit, then set:
#   RADIX8_RESOLVED_ISA   = AVX512 | AVX2 | SCALAR
#   RADIX8_C_FLAGS        = list of compiler flags for the driver TUs
#
include(CheckCCompilerFlag)

# Helper: test if a flag compiles
macro(radix8_check_flag flag result_var)
    check_c_compiler_flag("${flag}" ${result_var})
endmacro()

if(MSVC)
    # MSVC: /arch:AVX512, /arch:AVX2, or nothing (scalar+FMA via AVX2)
    radix8_check_flag("/arch:AVX512" HAVE_AVX512)
    radix8_check_flag("/arch:AVX2"   HAVE_AVX2)
    set(FLAGS_AVX512 "/arch:AVX512")
    set(FLAGS_AVX2   "/arch:AVX2")
    set(FLAGS_SCALAR "")   # MSVC scalar still gets FMA via /arch:AVX2
else()
    # GCC / Clang
    radix8_check_flag("-mavx512f"    HAVE_AVX512)
    radix8_check_flag("-mavx2"       HAVE_AVX2)
    set(FLAGS_AVX512 "-mavx512f;-mavx512dq;-mfma")
    set(FLAGS_AVX2   "-mavx2;-mfma")
    set(FLAGS_SCALAR "-mfma")
endif()

# Resolve ISA
if(RADIX8_ISA STREQUAL "AUTO")
    if(HAVE_AVX512)
        set(RADIX8_RESOLVED_ISA "AVX512")
    elseif(HAVE_AVX2)
        set(RADIX8_RESOLVED_ISA "AVX2")
    else()
        set(RADIX8_RESOLVED_ISA "SCALAR")
    endif()
elseif(RADIX8_ISA STREQUAL "AVX512")
    if(NOT HAVE_AVX512)
        message(FATAL_ERROR "RADIX8_ISA=AVX512 requested but compiler lacks AVX-512 support")
    endif()
    set(RADIX8_RESOLVED_ISA "AVX512")
elseif(RADIX8_ISA STREQUAL "AVX2")
    if(NOT HAVE_AVX2)
        message(FATAL_ERROR "RADIX8_ISA=AVX2 requested but compiler lacks AVX2 support")
    endif()
    set(RADIX8_RESOLVED_ISA "AVX2")
elseif(RADIX8_ISA STREQUAL "SCALAR")
    set(RADIX8_RESOLVED_ISA "SCALAR")
else()
    message(FATAL_ERROR "Unknown RADIX8_ISA: ${RADIX8_ISA}  (expected AUTO|AVX512|AVX2|SCALAR)")
endif()

# Map resolved ISA → compiler flags
if(RADIX8_RESOLVED_ISA STREQUAL "AVX512")
    set(RADIX8_C_FLAGS ${FLAGS_AVX512})
elseif(RADIX8_RESOLVED_ISA STREQUAL "AVX2")
    set(RADIX8_C_FLAGS ${FLAGS_AVX2})
else()
    set(RADIX8_C_FLAGS ${FLAGS_SCALAR})
endif()

message(STATUS "Radix-8 ISA: ${RADIX8_RESOLVED_ISA}  (flags: ${RADIX8_C_FLAGS})")

# ── Library target ──────────────────────────────────────────────────────────
#
# The two driver TUs (fv.c, bv.c) include ISA-specific headers via #ifdef
# and do runtime K-based dispatch.  Compile them with the resolved ISA flags
# so the highest-available backend is compiled in, with scalar fallback
# always present for small K values.
#
add_library(radix8 STATIC
    fft_radix8_fv.c
    fft_radix8_bv.c
)

target_include_directories(radix8
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/avx512
        ${CMAKE_CURRENT_SOURCE_DIR}/avx2
        ${CMAKE_CURRENT_SOURCE_DIR}/scalar
        ${CMAKE_CURRENT_SOURCE_DIR}/sse2
)

target_compile_options(radix8 PRIVATE ${RADIX8_C_FLAGS})

# Expose include dirs to consumers
# Users just: target_link_libraries(myapp PRIVATE radix8)
#             #include "fft_radix8.h"

# ── Math library (Unix) ────────────────────────────────────────────────────
if(NOT MSVC)
    find_library(MATH_LIB m)
    if(MATH_LIB)
        target_link_libraries(radix8 PRIVATE ${MATH_LIB})
    endif()
endif()

# ── Tests ───────────────────────────────────────────────────────────────────
if(RADIX8_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
