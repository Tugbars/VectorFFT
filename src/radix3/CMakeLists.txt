##############################################################################
# radix3/ CMakeLists.txt
#
# Radix-3 FFT stage library — SoA split-complex.
# Dispatch chain: AVX-512 → AVX2 → scalar (FMA).
#
# Usage:
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake -B build -DCMAKE_BUILD_TYPE=Release -DRADIX3_BUILD_TESTS=ON
#   cmake -B build -DCMAKE_BUILD_TYPE=Release -DRADIX3_ISA=AVX2
#   cmake --build build
#   ctest --test-dir build --output-on-failure
#
# Options:
#   RADIX3_BUILD_TESTS   ON/OFF  Build test executables         (default: OFF)
#   RADIX3_ISA           AUTO | AVX512 | AVX2 | SCALAR          (default: AUTO)
#
##############################################################################
cmake_minimum_required(VERSION 3.16)
project(radix3 LANGUAGES C)

# ── Options ─────────────────────────────────────────────────────────────────
option(RADIX3_BUILD_TESTS "Build radix-3 test executables" OFF)

set(RADIX3_ISA "AUTO" CACHE STRING "Target ISA: AUTO, AVX512, AVX2, SCALAR")
set_property(CACHE RADIX3_ISA PROPERTY STRINGS AUTO AVX512 AVX2 SCALAR)

# ── Compiler baseline ───────────────────────────────────────────────────────
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/W3)
else()
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
endif()

# ── ISA detection / flag setup ──────────────────────────────────────────────
include(CheckCCompilerFlag)

macro(radix3_check_flag flag result_var)
    check_c_compiler_flag("${flag}" ${result_var})
endmacro()

if(MSVC)
    radix3_check_flag("/arch:AVX512" R3_HAVE_AVX512)
    radix3_check_flag("/arch:AVX2"   R3_HAVE_AVX2)
    set(R3_FLAGS_AVX512 "/arch:AVX512")
    set(R3_FLAGS_AVX2   "/arch:AVX2")
    set(R3_FLAGS_SCALAR "")
else()
    radix3_check_flag("-mavx512f"    R3_HAVE_AVX512)
    radix3_check_flag("-mavx2"       R3_HAVE_AVX2)
    set(R3_FLAGS_AVX512 "-mavx512f;-mavx512dq;-mfma")
    set(R3_FLAGS_AVX2   "-mavx2;-mfma")
    set(R3_FLAGS_SCALAR "-mfma")
endif()

# Resolve ISA
if(RADIX3_ISA STREQUAL "AUTO")
    if(R3_HAVE_AVX512)
        set(R3_RESOLVED_ISA "AVX512")
    elseif(R3_HAVE_AVX2)
        set(R3_RESOLVED_ISA "AVX2")
    else()
        set(R3_RESOLVED_ISA "SCALAR")
    endif()
elseif(RADIX3_ISA STREQUAL "AVX512")
    if(NOT R3_HAVE_AVX512)
        message(FATAL_ERROR "RADIX3_ISA=AVX512 requested but compiler lacks AVX-512")
    endif()
    set(R3_RESOLVED_ISA "AVX512")
elseif(RADIX3_ISA STREQUAL "AVX2")
    if(NOT R3_HAVE_AVX2)
        message(FATAL_ERROR "RADIX3_ISA=AVX2 requested but compiler lacks AVX2")
    endif()
    set(R3_RESOLVED_ISA "AVX2")
elseif(RADIX3_ISA STREQUAL "SCALAR")
    set(R3_RESOLVED_ISA "SCALAR")
else()
    message(FATAL_ERROR "Unknown RADIX3_ISA: ${RADIX3_ISA}")
endif()

if(R3_RESOLVED_ISA STREQUAL "AVX512")
    set(R3_C_FLAGS ${R3_FLAGS_AVX512})
elseif(R3_RESOLVED_ISA STREQUAL "AVX2")
    set(R3_C_FLAGS ${R3_FLAGS_AVX2})
else()
    set(R3_C_FLAGS ${R3_FLAGS_SCALAR})
endif()

message(STATUS "Radix-3 ISA: ${R3_RESOLVED_ISA}  (flags: ${R3_C_FLAGS})")

# ── Library target ──────────────────────────────────────────────────────────
add_library(radix3 STATIC
    fft_radix3_fv.c
    fft_radix3_bv.c
)

target_include_directories(radix3
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_compile_options(radix3 PRIVATE ${R3_C_FLAGS})

if(NOT MSVC)
    find_library(R3_MATH_LIB m)
    if(R3_MATH_LIB)
        target_link_libraries(radix3 PRIVATE ${R3_MATH_LIB})
    endif()
endif()

# ── Tests ───────────────────────────────────────────────────────────────────
if(RADIX3_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
