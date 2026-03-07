# ============================================================================
# VectorFFT — High-Performance Mixed-Radix FFT Library
# Root CMakeLists.txt
# ============================================================================

cmake_minimum_required(VERSION 3.16)

# Auto-detect vcpkg if present in project root
if(EXISTS "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
        CACHE STRING "Vcpkg toolchain file")
    message(STATUS "vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
else()
    message(WARNING "vcpkg not found at ${CMAKE_SOURCE_DIR}/vcpkg/")
endif()

project(VectorFFT
    VERSION   1.0.0
    LANGUAGES C
)

# ── Build options ────────────────────────────────────────────────────────
option(VFFT_BUILD_TESTS      "Build correctness tests"       ON)
option(VFFT_BUILD_BENCHMARKS "Build performance benchmarks"   ON)

# ── Compiler helpers (ISA detection, per-target flags) ───────────────────
# Move VectorFFTCompiler.cmake into cmake/ at project root:
#   VectorFFT/cmake/VectorFFTCompiler.cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(VectorFFTCompiler)

# After include(VectorFFTCompiler), add:
if(VFFT_COMPILER STREQUAL "ICX" AND WIN32)
    get_filename_component(_icx_bin "${CMAKE_C_COMPILER}" DIRECTORY)
    get_filename_component(_icx_root "${_icx_bin}/.." ABSOLUTE)
    link_directories("${_icx_root}/lib")
    message(STATUS "[VectorFFT] Intel lib path: ${_icx_root}/lib")
endif()

# ── Source tree ──────────────────────────────────────────────────────────
add_subdirectory(src)

# ── Benchmarks ───────────────────────────────────────────────────────────
if(VFFT_BUILD_BENCHMARKS)
    add_subdirectory(test)
endif()

# ── CTest ────────────────────────────────────────────────────────────────
if(VFFT_BUILD_TESTS)
    enable_testing()
endif()

# ── Summary ──────────────────────────────────────────────────────────────
message(STATUS "")
message(STATUS "╔══════════════════════════════════════════╗")
message(STATUS "║  VectorFFT ${PROJECT_VERSION}                        ║")
message(STATUS "╠══════════════════════════════════════════╣")
message(STATUS "║  Compiler:    ${VFFT_COMPILER} (${CMAKE_C_COMPILER_VERSION})")
message(STATUS "║  AVX-512:     ${VFFT_HAS_AVX512}")
message(STATUS "║  AVX2:        ${VFFT_HAS_AVX2}")
message(STATUS "║  SSE2:        ${VFFT_HAS_SSE2}")
message(STATUS "║  Tests:       ${VFFT_BUILD_TESTS}")
message(STATUS "║  Benchmarks:  ${VFFT_BUILD_BENCHMARKS}")
message(STATUS "╚══════════════════════════════════════════╝")
message(STATUS "")