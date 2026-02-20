# ============================================================================
# VectorFFT — High-Performance Mixed-Radix FFT Library
# Root CMakeLists.txt
# ============================================================================

cmake_minimum_required(VERSION 3.16)

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

# ── Source tree ──────────────────────────────────────────────────────────
add_subdirectory(src)

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