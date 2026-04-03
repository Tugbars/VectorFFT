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

set(VFFT_STRIDE_ISA "auto" CACHE STRING "ISA for stride-fft codelets (auto/avx512/avx2/scalar)")

# ── ISA Detection ──
if(VFFT_STRIDE_ISA STREQUAL "auto")
    include(CheckCSourceCompiles)
    set(CMAKE_REQUIRED_FLAGS "")

    # Check AVX-512 (compile + run: must actually execute on this CPU)
    if(MSVC OR CMAKE_C_COMPILER_ID MATCHES "Intel")
        set(_avx512_flag "/arch:AVX512")
    else()
        set(_avx512_flag "-mavx512f -mavx512dq -mfma")
    endif()
    set(CMAKE_REQUIRED_FLAGS "${_avx512_flag}")
    set(VFFT_HAS_AVX512 FALSE)
    if(NOT CMAKE_CROSSCOMPILING)
        try_run(_avx512_run _avx512_compile
            "${CMAKE_BINARY_DIR}/avx512_check"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_avx512.c"
            CMAKE_FLAGS "-DCMAKE_C_FLAGS=${_avx512_flag}"
        )
        if(_avx512_compile AND _avx512_run EQUAL 0)
            set(VFFT_HAS_AVX512 TRUE)
        endif()
    endif()

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
                    "Run: cmake --build . --target generate_codelets")
endif()

# ── Codelet generation target ──
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
    if(WIN32)
        set(_gen_script "${CMAKE_CURRENT_SOURCE_DIR}/generators/generate_all.bat")
    else()
        set(_gen_script bash "${CMAKE_CURRENT_SOURCE_DIR}/generators/generate_all.sh")
    endif()
    add_custom_target(generate_codelets
        COMMAND ${_gen_script}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/generators"
        COMMENT "Generating stride-fft codelets (AVX2 + AVX-512 + scalar)..."
    )
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

# ── FFTW3 discovery (shared by bench targets) ──
find_library(FFTW3_LIB fftw3 HINTS
    "${CMAKE_SOURCE_DIR}/build/vcpkg_installed/x64-windows/lib"
)
find_path(FFTW3_INC fftw3.h HINTS
    "${CMAKE_SOURCE_DIR}/build/vcpkg_installed/x64-windows/include"
)

# ── MKL discovery (optional) ──
set(VFFT_HAS_MKL OFF)
find_path(MKL_INC mkl_dfti.h HINTS
    "$ENV{MKLROOT}/include"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/include"
)
find_library(MKL_CORE mkl_core HINTS
    "$ENV{MKLROOT}/lib"
    "$ENV{MKLROOT}/lib/intel64"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/intel64"
)
find_library(MKL_SEQ mkl_sequential HINTS
    "$ENV{MKLROOT}/lib"
    "$ENV{MKLROOT}/lib/intel64"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/intel64"
)
find_library(MKL_ILP64 mkl_intel_ilp64 HINTS
    "$ENV{MKLROOT}/lib"
    "$ENV{MKLROOT}/lib/intel64"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib"
    "C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/intel64"
)
if(MKL_INC AND MKL_CORE AND MKL_SEQ AND MKL_ILP64)
    set(VFFT_HAS_MKL ON)
    set(MKL_LIBS ${MKL_ILP64} ${MKL_SEQ} ${MKL_CORE})
    message(STATUS "Intel MKL found: ${MKL_INC}")
else()
    message(STATUS "Intel MKL not found — MKL benchmark disabled")
endif()

# ── Benchmark target — always built, shows as VS Code target ──
if(FFTW3_LIB AND FFTW3_INC)
    add_executable(vfft_bench bench/bench_planner.c)
    target_link_libraries(vfft_bench PRIVATE vfft_stride ${FFTW3_LIB})
    target_include_directories(vfft_bench PRIVATE ${FFTW3_INC})
    target_compile_definitions(vfft_bench PRIVATE _CRT_SECURE_NO_WARNINGS)
    set_target_properties(vfft_bench PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )

    if(VFFT_HAS_MKL)
        target_link_libraries(vfft_bench PRIVATE ${MKL_LIBS})
        target_include_directories(vfft_bench PRIVATE ${MKL_INC})
        target_compile_definitions(vfft_bench PRIVATE VFFT_HAS_MKL MKL_ILP64)
        message(STATUS "VectorFFT bench: vfft_bench (FFTW + MKL)")
    else()
        message(STATUS "VectorFFT bench: vfft_bench (FFTW only)")
    endif()
else()
    message(WARNING "FFTW3 not found — vfft_bench disabled. Install via: vcpkg install fftw3")
endif()
