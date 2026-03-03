cmake_minimum_required(VERSION 3.20)
project(radix32 LANGUAGES C CXX)

# ============================================================================
# Compiler detection — ICX preferred, GCC/Clang supported
# ============================================================================
set(RADIX32_IS_ICX   FALSE)
set(RADIX32_IS_GCC   FALSE)
set(RADIX32_IS_CLANG FALSE)

if(CMAKE_C_COMPILER_ID MATCHES "IntelLLVM")
    set(RADIX32_IS_ICX TRUE)
    message(STATUS "radix32: Intel ICX detected (${CMAKE_C_COMPILER_VERSION})")
elseif(CMAKE_C_COMPILER_ID STREQUAL "GNU")
    set(RADIX32_IS_GCC TRUE)
    message(STATUS "radix32: GCC detected (${CMAKE_C_COMPILER_VERSION})")
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(RADIX32_IS_CLANG TRUE)
    message(STATUS "radix32: Clang detected (${CMAKE_C_COMPILER_VERSION})")
else()
    message(WARNING
        "radix32: Untested compiler '${CMAKE_C_COMPILER_ID}'. "
        "ICX, GCC, or Clang recommended.")
endif()

# ============================================================================
# Options
# ============================================================================
option(RADIX32_BUILD_TESTS    "Build unit tests and benchmarks"            ON)
option(RADIX32_BUILD_AVX512   "Build AVX-512 codelet (requires AVX-512)"   OFF)
option(RADIX32_BUILD_SSE2     "Build SSE2 codelet"                         OFF)
option(RADIX32_BUILD_DRIVERS  "Build forward/backward driver objects"       OFF)
option(RADIX32_OPT_REPORT     "Enable ICX optimization reports"            OFF)

# ============================================================================
# Global settings
# ============================================================================
set(CMAKE_C_STANDARD   11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ============================================================================
# ISA detection helpers
#
# ICX: -x flag selects exact µarch (e.g. -xCORE-AVX2, -xCORE-AVX512)
#      -ax adds runtime dispatch path alongside baseline
#      -march=native works but -x gives tighter codegen on known targets
# GCC: -march=native or explicit -mavx2 -mfma
# ============================================================================

# Base warning flags shared across all compilers
set(RADIX32_WARN_FLAGS
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
    -Wno-unused-function
)

# ICX opt-report (generates .optrpt files alongside objects)
set(RADIX32_ICX_OPT_REPORT_FLAGS "")
if(RADIX32_IS_ICX AND RADIX32_OPT_REPORT)
    set(RADIX32_ICX_OPT_REPORT_FLAGS -qopt-report=5 -qopt-report-phase=vec,loop)
    message(STATUS "radix32: ICX optimization reports enabled")
endif()

# ============================================================================
# INTERFACE LIBRARY: radix32_avx2  (header-only)
#
# Consumers get: -I avx2/  + ISA flags for AVX2+FMA
# ============================================================================
add_library(radix32_avx2 INTERFACE)

target_include_directories(radix32_avx2 INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/avx2>
    $<INSTALL_INTERFACE:include/radix32/avx2>
)

if(RADIX32_IS_ICX)
    # -xCORE-AVX2 implies FMA, BMI2, POPCNT — tighter than -mavx2 -mfma
    target_compile_options(radix32_avx2 INTERFACE -xCORE-AVX2)
else()
    target_compile_options(radix32_avx2 INTERFACE -mavx2 -mfma)
endif()

# ============================================================================
# INTERFACE LIBRARY: radix32_scalar  (header-only)
#
# Depends on avx2 for shared twiddle structs (tw_stage8_t, tw_mode_t, etc.)
# Only needs FMA for scalar vfmadd/vfmsub instructions — inherited from avx2.
# ============================================================================
add_library(radix32_scalar INTERFACE)

target_include_directories(radix32_scalar INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/scalar>
    $<INSTALL_INTERFACE:include/radix32/scalar>
)

target_link_libraries(radix32_scalar INTERFACE radix32_avx2)

# ============================================================================
# INTERFACE LIBRARY: radix32_avx512  (header-only, optional)
# ============================================================================
if(RADIX32_BUILD_AVX512)
    add_library(radix32_avx512 INTERFACE)

    target_include_directories(radix32_avx512 INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/avx512>
        $<INSTALL_INTERFACE:include/radix32/avx512>
    )

    if(RADIX32_IS_ICX)
        # -xCORE-AVX512 = SKX baseline (512-bit FMA, no VNNI/BF16)
        # For ICX/SPR with AVX512-IFMA etc: -xICELAKE-SERVER or -xSAPPHIRERAPIDS
        target_compile_options(radix32_avx512 INTERFACE -xCORE-AVX512)
    else()
        target_compile_options(radix32_avx512 INTERFACE
            -mavx512f -mavx512dq -mavx512vl -mfma)
    endif()

    target_link_libraries(radix32_avx512 INTERFACE radix32_avx2)
    message(STATUS "radix32: AVX-512 codelet enabled")
endif()

# ============================================================================
# INTERFACE LIBRARY: radix32_sse2  (header-only, optional)
# ============================================================================
if(RADIX32_BUILD_SSE2)
    add_library(radix32_sse2 INTERFACE)

    target_include_directories(radix32_sse2 INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/sse2>
        $<INSTALL_INTERFACE:include/radix32/sse2>
    )

    if(RADIX32_IS_ICX)
        target_compile_options(radix32_sse2 INTERFACE -xSSE4.2)
    else()
        target_compile_options(radix32_sse2 INTERFACE -msse2 -msse4.2)
    endif()

    message(STATUS "radix32: SSE2 codelet enabled")
endif()

# ============================================================================
# STATIC LIBRARY: radix32  (compiled drivers, optional)
#
# Root-level fft_radix32_fv.c / fft_radix32_bv.c / fft_radix32_uniform.h
# These contain the dispatch logic that selects AVX2/AVX512/scalar.
# Enable with -DRADIX32_BUILD_DRIVERS=ON once drivers are implemented.
# ============================================================================
if(RADIX32_BUILD_DRIVERS)
    set(RADIX32_DRIVER_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/fft_radix32_fv.c
        ${CMAKE_CURRENT_SOURCE_DIR}/fft_radix32_bv.c
    )

    foreach(_src IN LISTS RADIX32_DRIVER_SOURCES)
        if(NOT EXISTS "${_src}")
            message(FATAL_ERROR
                "radix32: Driver source '${_src}' not found.\n"
                "  Disable with -DRADIX32_BUILD_DRIVERS=OFF")
        endif()
    endforeach()

    add_library(radix32 STATIC ${RADIX32_DRIVER_SOURCES})

    target_include_directories(radix32 PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include/radix32>
    )

    target_link_libraries(radix32 PUBLIC radix32_avx2 radix32_scalar)

    if(RADIX32_BUILD_AVX512)
        target_link_libraries(radix32 PUBLIC radix32_avx512)
    endif()

    target_compile_options(radix32 PRIVATE
        ${RADIX32_WARN_FLAGS}
        ${RADIX32_ICX_OPT_REPORT_FLAGS}
        $<$<CONFIG:Release>:-O2>
        $<$<CONFIG:RelWithDebInfo>:-O2 -g>
    )

    if(RADIX32_IS_ICX)
        target_compile_options(radix32 PRIVATE -ipo)
    endif()

    message(STATUS "radix32: Driver library enabled")
endif()

# ============================================================================
# Tests
# ============================================================================
if(RADIX32_BUILD_TESTS)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_subdirectory(tests)
endif()

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "  radix32 configuration")
message(STATUS "  ---------------------")
message(STATUS "  Compiler:     ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "  Build type:   ${CMAKE_BUILD_TYPE}")
message(STATUS "  AVX2 codelet: ON  (always)")
message(STATUS "  Scalar:       ON  (always)")
message(STATUS "  AVX-512:      ${RADIX32_BUILD_AVX512}")
message(STATUS "  SSE2:         ${RADIX32_BUILD_SSE2}")
message(STATUS "  Drivers:      ${RADIX32_BUILD_DRIVERS}")
message(STATUS "  Tests:        ${RADIX32_BUILD_TESTS}")
if(RADIX32_IS_ICX)
    message(STATUS "  Opt reports:  ${RADIX32_OPT_REPORT}")
endif()
message(STATUS "")
