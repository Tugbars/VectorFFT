cmake_minimum_required(VERSION 3.16)

# ═══════════════════════════════════════════════════════════════
# VectorFFT Stride-FFT Module
#
# Header-only library with generated SIMD codelets.
#
# Usage (from parent CMakeLists.txt):
#   add_subdirectory(src/stride-fft)
#   target_link_libraries(myapp vfft_stride)
#
# Options:
#   VFFT_STRIDE_BUILD_BENCH  - Build benchmark executable (default: OFF)
#   VFFT_STRIDE_ISA          - Force ISA: "avx512", "avx2", "scalar", "auto" (default: auto)
# ═══════════════════════════════════════════════════════════════

option(VFFT_STRIDE_BUILD_BENCH "Build stride-fft benchmark" OFF)
set(VFFT_STRIDE_ISA "auto" CACHE STRING "ISA for stride-fft codelets (auto/avx512/avx2/scalar)")

# ── ISA Detection ──
if(VFFT_STRIDE_ISA STREQUAL "auto")
    include(CheckCSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "")

    # Check AVX-512
    if(MSVC OR CMAKE_C_COMPILER_ID MATCHES "Intel")
        set(_avx512_flag "/arch:AVX512")
    else()
        set(_avx512_flag "-mavx512f -mavx512dq -mfma")
    endif()
    set(CMAKE_REQUIRED_FLAGS "${_avx512_flag}")
    check_c_source_compiles("
        #include <immintrin.h>
        int main() { __m512d a = _mm512_setzero_pd(); (void)a; return 0; }
    " VFFT_HAS_AVX512)

    # Check AVX2
    if(MSVC OR CMAKE_C_COMPILER_ID MATCHES "Intel")
        set(_avx2_flag "/arch:AVX2")
    else()
        set(_avx2_flag "-mavx2 -mfma")
    endif()
    set(CMAKE_REQUIRED_FLAGS "${_avx2_flag}")
    check_c_source_compiles("
        #include <immintrin.h>
        int main() { __m256d a = _mm256_setzero_pd(); (void)a; return 0; }
    " VFFT_HAS_AVX2)

    set(CMAKE_REQUIRED_FLAGS "")

    if(VFFT_HAS_AVX512)
        set(_isa "avx512")
    elseif(VFFT_HAS_AVX2)
        set(_isa "avx2")
    else()
        set(_isa "scalar")
    endif()
else()
    set(_isa "${VFFT_STRIDE_ISA}")
endif()

message(STATUS "VectorFFT stride-fft ISA: ${_isa}")

# ── ISA compile flags ──
if(_isa STREQUAL "avx512")
    if(MSVC OR CMAKE_C_COMPILER_ID MATCHES "Intel")
        set(_isa_flags "/arch:AVX512")
    else()
        set(_isa_flags "-mavx512f;-mavx512dq;-mfma")
    endif()
elseif(_isa STREQUAL "avx2")
    if(MSVC OR CMAKE_C_COMPILER_ID MATCHES "Intel")
        set(_isa_flags "/arch:AVX2")
    else()
        set(_isa_flags "-mavx2;-mfma")
    endif()
else()
    set(_isa_flags "")
endif()

# ── Codelet directory ──
set(VFFT_CODELET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/codelets/${_isa}")

if(NOT EXISTS "${VFFT_CODELET_DIR}/fft_radix16_${_isa}_ct_n1.h")
    message(WARNING "Codelets not found in ${VFFT_CODELET_DIR}. "
                    "Run generators/generate_all.bat first.")
endif()

# ── Interface library (header-only) ──
add_library(vfft_stride INTERFACE)

target_include_directories(vfft_stride INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/core
    ${VFFT_CODELET_DIR}
)

target_compile_options(vfft_stride INTERFACE ${_isa_flags})

target_compile_definitions(vfft_stride INTERFACE
    $<$<PLATFORM_ID:Windows>:_USE_MATH_DEFINES>
)

# ── Benchmark target (optional) ──
if(VFFT_STRIDE_BUILD_BENCH)
    find_package(PkgConfig QUIET)
    find_library(FFTW3_LIB fftw3 HINTS
        "${CMAKE_SOURCE_DIR}/build/vcpkg_installed/x64-windows/lib"
    )
    find_path(FFTW3_INC fftw3.h HINTS
        "${CMAKE_SOURCE_DIR}/build/vcpkg_installed/x64-windows/include"
    )

    if(FFTW3_LIB AND FFTW3_INC)
        add_executable(vfft_stride_bench bench/bench_planner.c)
        target_link_libraries(vfft_stride_bench PRIVATE vfft_stride ${FFTW3_LIB})
        target_include_directories(vfft_stride_bench PRIVATE ${FFTW3_INC})
        target_compile_definitions(vfft_stride_bench PRIVATE
            _CRT_SECURE_NO_WARNINGS
        )
        message(STATUS "VectorFFT stride-fft bench: enabled")
    else()
        message(WARNING "FFTW3 not found — stride-fft bench disabled")
    endif()
endif()
