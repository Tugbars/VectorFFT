# ===========================================================================
# VectorFFTCompiler.cmake
# Compiler detection, ISA flags, and target helpers for VectorFFT
#
# Primary target:   Intel ICX (oneAPI DPC++/C++)
# Fallback:         GCC 12+, Clang 15+
# ===========================================================================

include(CheckCCompilerFlag)

# ── Detect compiler ──────────────────────────────────────────────────────
set(VFFT_COMPILER "UNKNOWN")

if(CMAKE_C_COMPILER_ID MATCHES "IntelLLVM")
    set(VFFT_COMPILER "ICX")
    message(STATUS "[VectorFFT] Compiler: Intel ICX (${CMAKE_C_COMPILER_VERSION})")
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(VFFT_COMPILER "CLANG")
    message(STATUS "[VectorFFT] Compiler: Clang (${CMAKE_C_COMPILER_VERSION})")
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU")
    set(VFFT_COMPILER "GCC")
    message(STATUS "[VectorFFT] Compiler: GCC (${CMAKE_C_COMPILER_VERSION})")
    if(CMAKE_C_COMPILER_VERSION VERSION_LESS "12.0")
        message(WARNING "[VectorFFT] GCC < 12 may produce suboptimal AVX-512 codegen")
    endif()
else()
    message(WARNING "[VectorFFT] Unknown compiler: ${CMAKE_C_COMPILER_ID}")
endif()

# ── Base flags (all compilers) ───────────────────────────────────────────
set(VFFT_C_STANDARD 11)

set(VFFT_BASE_FLAGS
    -O2
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wshadow
    -Wstrict-prototypes
    -fstrict-aliasing
)

# ── ICX-specific flags ──────────────────────────────────────────────────
if(VFFT_COMPILER STREQUAL "ICX")
    list(APPEND VFFT_BASE_FLAGS
        -qopt-zmm-usage=high          # Don't downgrade 512-bit to 256-bit
        -qopt-report=3                 # Optimization reports (level 3)
        -qopt-report-phase=vec,loop    # Focus on vectorization + loop opts
        -fno-omit-frame-pointer        # Better profiling with VTune
    )
endif()

# ── GCC-specific flags ──────────────────────────────────────────────────
if(VFFT_COMPILER STREQUAL "GCC")
    list(APPEND VFFT_BASE_FLAGS
        -fopt-info-vec-missed          # Report failed vectorizations
        -fno-semantic-interposition     # Better inlining for shared libs
    )
endif()

# ── ISA flag sets ────────────────────────────────────────────────────────
# These are applied per-target, not globally.

set(VFFT_FLAGS_AVX512  -mavx512f -mfma)
set(VFFT_FLAGS_AVX2    -mavx2 -mfma -mno-avx512f)
set(VFFT_FLAGS_SSE2    -msse2 -mno-sse3 -mno-avx -mno-avx2 -mno-fma)

# ICX: add explicit arch tuning
if(VFFT_COMPILER STREQUAL "ICX")
    # Default to Skylake-AVX512 unless user overrides
    if(NOT DEFINED VFFT_ARCH)
        set(VFFT_ARCH "skylake-avx512")
    endif()
    list(APPEND VFFT_FLAGS_AVX512 -march=${VFFT_ARCH})
    message(STATUS "[VectorFFT] AVX-512 target arch: ${VFFT_ARCH}")
endif()

# ── ISA detection (what the build machine supports) ──────────────────────
set(VFFT_HAS_AVX512 OFF)
set(VFFT_HAS_AVX2   OFF)
set(VFFT_HAS_SSE2   OFF)

check_c_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
check_c_compiler_flag("-mavx2"    COMPILER_SUPPORTS_AVX2)
check_c_compiler_flag("-msse2"    COMPILER_SUPPORTS_SSE2)

if(COMPILER_SUPPORTS_AVX512)
    set(VFFT_HAS_AVX512 ON)
endif()
if(COMPILER_SUPPORTS_AVX2)
    set(VFFT_HAS_AVX2 ON)
endif()
if(COMPILER_SUPPORTS_SSE2)
    set(VFFT_HAS_SSE2 ON)
endif()

message(STATUS "[VectorFFT] ISA support: AVX-512=${VFFT_HAS_AVX512} AVX2=${VFFT_HAS_AVX2} SSE2=${VFFT_HAS_SSE2}")

# ── Helper: create a library target with specific ISA flags ──────────────
#
# vfft_add_isa_library(<name> <isa> <sources...>)
#   isa: AVX512 | AVX2 | SSE2
#
# Example:
#   vfft_add_isa_library(radix2_avx512 AVX512 fft_radix2_bv.c)
#
function(vfft_add_isa_library NAME ISA)
    set(SOURCES ${ARGN})

    if(ISA STREQUAL "AVX512" AND NOT VFFT_HAS_AVX512)
        message(STATUS "[VectorFFT] Skipping ${NAME} — AVX-512 not supported")
        return()
    elseif(ISA STREQUAL "AVX2" AND NOT VFFT_HAS_AVX2)
        message(STATUS "[VectorFFT] Skipping ${NAME} — AVX2 not supported")
        return()
    elseif(ISA STREQUAL "SSE2" AND NOT VFFT_HAS_SSE2)
        message(STATUS "[VectorFFT] Skipping ${NAME} — SSE2 not supported")
        return()
    endif()

    add_library(${NAME} OBJECT ${SOURCES})
    target_compile_options(${NAME} PRIVATE ${VFFT_BASE_FLAGS} ${VFFT_FLAGS_${ISA}})
    set_target_properties(${NAME} PROPERTIES C_STANDARD ${VFFT_C_STANDARD})

    # Include paths: public API + kernel root + ISA subdirs
    target_include_directories(${NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/include/vectorfft
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/avx2
        ${CMAKE_CURRENT_SOURCE_DIR}/avx512
        ${CMAKE_CURRENT_SOURCE_DIR}/sse2
        ${CMAKE_CURRENT_SOURCE_DIR}/scalar
        ${PROJECT_SOURCE_DIR}/src/common
    )
endfunction()

# ── Helper: create test executables for all supported ISAs ───────────────
#
# vfft_add_isa_tests(<base_name> <isa_list> <sources...>)
#
# Creates: <base_name>_avx512, <base_name>_avx2, <base_name>_sse2
# Each links against the corresponding ISA object library.
#
# Example:
#   vfft_add_isa_tests(test_radix2_bv "AVX512;AVX2;SSE2"
#       tests/test_radix2_bv.c fft_radix2_bv.c)
#
function(vfft_add_isa_tests BASE_NAME ISA_LIST)
    set(SOURCES ${ARGN})

    foreach(ISA IN LISTS ISA_LIST)
        string(TOLOWER ${ISA} isa_lower)
        set(TARGET_NAME "${BASE_NAME}_${isa_lower}")

        # Check ISA is available
        if(ISA STREQUAL "AVX512" AND NOT VFFT_HAS_AVX512)
            continue()
        elseif(ISA STREQUAL "AVX2" AND NOT VFFT_HAS_AVX2)
            continue()
        elseif(ISA STREQUAL "SSE2" AND NOT VFFT_HAS_SSE2)
            continue()
        endif()

        add_executable(${TARGET_NAME} ${SOURCES})
        target_compile_options(${TARGET_NAME} PRIVATE ${VFFT_BASE_FLAGS} ${VFFT_FLAGS_${ISA}})
        set_target_properties(${TARGET_NAME} PROPERTIES C_STANDARD ${VFFT_C_STANDARD})

        target_include_directories(${TARGET_NAME} PRIVATE
            ${PROJECT_SOURCE_DIR}/include/vectorfft
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/avx2
            ${CMAKE_CURRENT_SOURCE_DIR}/avx512
            ${CMAKE_CURRENT_SOURCE_DIR}/sse2
            ${CMAKE_CURRENT_SOURCE_DIR}/scalar
            ${PROJECT_SOURCE_DIR}/src/common
        )

        target_link_libraries(${TARGET_NAME} PRIVATE m)

        # Register with CTest
        add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
        set_tests_properties(${TARGET_NAME} PROPERTIES
            LABELS "${isa_lower};radix2"
        )

        message(STATUS "[VectorFFT] Test: ${TARGET_NAME}")
    endforeach()
endfunction()

# ── Helper: create benchmark executables ─────────────────────────────────
#
# Same as vfft_add_isa_tests but links FFTW3 if available, and does NOT
# register with CTest (benchmarks are run manually).
#
function(vfft_add_isa_benchmarks BASE_NAME ISA_LIST)
    set(SOURCES ${ARGN})

    foreach(ISA IN LISTS ISA_LIST)
        string(TOLOWER ${ISA} isa_lower)
        set(TARGET_NAME "${BASE_NAME}_${isa_lower}")

        if(ISA STREQUAL "AVX512" AND NOT VFFT_HAS_AVX512)
            continue()
        elseif(ISA STREQUAL "AVX2" AND NOT VFFT_HAS_AVX2)
            continue()
        elseif(ISA STREQUAL "SSE2" AND NOT VFFT_HAS_SSE2)
            continue()
        endif()

        add_executable(${TARGET_NAME} ${SOURCES})
        target_compile_options(${TARGET_NAME} PRIVATE ${VFFT_BASE_FLAGS} ${VFFT_FLAGS_${ISA}})
        set_target_properties(${TARGET_NAME} PROPERTIES C_STANDARD ${VFFT_C_STANDARD})

        target_include_directories(${TARGET_NAME} PRIVATE
            ${PROJECT_SOURCE_DIR}/include/vectorfft
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/avx2
            ${CMAKE_CURRENT_SOURCE_DIR}/avx512
            ${CMAKE_CURRENT_SOURCE_DIR}/sse2
            ${CMAKE_CURRENT_SOURCE_DIR}/scalar
            ${PROJECT_SOURCE_DIR}/src/common
        )

        target_link_libraries(${TARGET_NAME} PRIVATE m)

        # Link FFTW3 if available (for comparison benchmarks)
        if(TARGET FFTW3::fftw3)
            target_link_libraries(${TARGET_NAME} PRIVATE FFTW3::fftw3)
            target_compile_definitions(${TARGET_NAME} PRIVATE VFFT_HAS_FFTW3=1)
        endif()

        message(STATUS "[VectorFFT] Bench: ${TARGET_NAME}")
    endforeach()
endfunction()