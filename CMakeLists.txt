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
# Applied per-target, not globally.

set(VFFT_FLAGS_AVX512  -mavx512f -mfma)
set(VFFT_FLAGS_AVX2    -mavx2 -mfma -mno-avx512f)
set(VFFT_FLAGS_SSE2    -msse2 -mno-sse3 -mno-avx -mno-avx2 -mno-fma)

# ICX: explicit arch tuning for AVX-512
if(VFFT_COMPILER STREQUAL "ICX")
    if(NOT DEFINED VFFT_ARCH)
        set(VFFT_ARCH "skylake-avx512")
    endif()
    list(APPEND VFFT_FLAGS_AVX512 -march=${VFFT_ARCH})
    message(STATUS "[VectorFFT] AVX-512 target arch: ${VFFT_ARCH}")
endif()

# ── ISA detection (what the compiler supports) ───────────────────────────
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

# ── Internal: common target setup ────────────────────────────────────────
#
# _vfft_setup_target(<target> <isa> [EXTRA_INCLUDES dir1 dir2 ...])
#
# Applies flags, C standard, and include paths. Used by all public helpers.
#
function(_vfft_setup_target TARGET ISA)
    cmake_parse_arguments(ARG "" "" "EXTRA_INCLUDES" ${ARGN})

    target_compile_options(${TARGET} PRIVATE ${VFFT_BASE_FLAGS} ${VFFT_FLAGS_${ISA}})
    set_target_properties(${TARGET} PROPERTIES C_STANDARD ${VFFT_C_STANDARD})

    target_include_directories(${TARGET} PRIVATE
        ${PROJECT_SOURCE_DIR}/include/vectorfft
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/avx2
        ${CMAKE_CURRENT_SOURCE_DIR}/avx512
        ${CMAKE_CURRENT_SOURCE_DIR}/sse2
        ${CMAKE_CURRENT_SOURCE_DIR}/scalar
        ${PROJECT_SOURCE_DIR}/src/common
    )

    if(ARG_EXTRA_INCLUDES)
        target_include_directories(${TARGET} PRIVATE ${ARG_EXTRA_INCLUDES})
    endif()
endfunction()

# ── Internal: check ISA availability ─────────────────────────────────────
#
# _vfft_check_isa(<isa> <result_var>)
#
function(_vfft_check_isa ISA RESULT_VAR)
    set(${RESULT_VAR} ON PARENT_SCOPE)
    if(ISA STREQUAL "AVX512" AND NOT VFFT_HAS_AVX512)
        set(${RESULT_VAR} OFF PARENT_SCOPE)
    elseif(ISA STREQUAL "AVX2" AND NOT VFFT_HAS_AVX2)
        set(${RESULT_VAR} OFF PARENT_SCOPE)
    elseif(ISA STREQUAL "SSE2" AND NOT VFFT_HAS_SSE2)
        set(${RESULT_VAR} OFF PARENT_SCOPE)
    endif()
endfunction()

# ===========================================================================
# PUBLIC API
# ===========================================================================

# ── vfft_add_isa_library ─────────────────────────────────────────────────
#
# vfft_add_isa_library(<name> <isa> <sources...>)
#
# Creates an OBJECT library compiled with the given ISA flags.
# Skips gracefully if the compiler doesn't support the ISA.
#
# Example:
#   vfft_add_isa_library(radix4_avx512 AVX512 fft_radix4_bv.c fft_radix4_fv.c)
#
function(vfft_add_isa_library NAME ISA)
    _vfft_check_isa(${ISA} _available)
    if(NOT _available)
        message(STATUS "[VectorFFT] Skipping ${NAME} — ${ISA} not supported")
        return()
    endif()

    add_library(${NAME} OBJECT ${ARGN})
    _vfft_setup_target(${NAME} ${ISA})
endfunction()

# ── vfft_add_isa_tests ───────────────────────────────────────────────────
#
# vfft_add_isa_tests(<base_name> <isa_list> <sources...>
#                    [LABELS label1 label2 ...]
#                    [EXTRA_INCLUDES dir1 dir2 ...])
#
# Creates one test executable per ISA: <base_name>_avx512, <base_name>_avx2, ...
# Registers each with CTest.
#
# LABELS:         Additional CTest labels (ISA label is always added automatically).
#                 If omitted, only the ISA label is applied.
# EXTRA_INCLUDES: Additional include directories (e.g. for cross-module headers).
#
# Examples:
#   vfft_add_isa_tests(test_radix2_bv "AVX512;AVX2"
#       test_radix2_bv.c ../fft_radix2_bv.c
#       LABELS radix2)
#
#   vfft_add_isa_tests(test_radix4_bv "AVX512;AVX2"
#       test_radix4_bv.c ${RADIX4_DIR}/fft_radix4_bv.c
#       LABELS radix4
#       EXTRA_INCLUDES ${RADIX4_DIR} ${RADIX4_DIR}/avx2 ${RADIX4_DIR}/avx512 ${RADIX4_DIR}/scalar)
#
function(vfft_add_isa_tests BASE_NAME ISA_LIST)
    cmake_parse_arguments(ARG "" "" "LABELS;EXTRA_INCLUDES" ${ARGN})
    set(SOURCES ${ARG_UNPARSED_ARGUMENTS})

    foreach(ISA IN LISTS ISA_LIST)
        _vfft_check_isa(${ISA} _available)
        if(NOT _available)
            continue()
        endif()

        string(TOLOWER ${ISA} isa_lower)
        set(TARGET_NAME "${BASE_NAME}_${isa_lower}")

        add_executable(${TARGET_NAME} ${SOURCES})
        _vfft_setup_target(${TARGET_NAME} ${ISA} EXTRA_INCLUDES ${ARG_EXTRA_INCLUDES})
        target_link_libraries(${TARGET_NAME} PRIVATE m)

        # CTest registration: ISA label always present, user labels appended
        set(_labels ${isa_lower})
        if(ARG_LABELS)
            list(APPEND _labels ${ARG_LABELS})
        endif()

        add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
        set_tests_properties(${TARGET_NAME} PROPERTIES LABELS "${_labels}")

        message(STATUS "[VectorFFT] Test: ${TARGET_NAME} [${_labels}]")
    endforeach()
endfunction()

# ── vfft_add_isa_benchmarks ──────────────────────────────────────────────
#
# vfft_add_isa_benchmarks(<base_name> <isa_list> <sources...>
#                         [LABELS label1 label2 ...]
#                         [EXTRA_INCLUDES dir1 dir2 ...])
#
# Same as vfft_add_isa_tests but:
#   - NOT registered with CTest (run manually)
#   - Links FFTW3 if available (for comparison benchmarks)
#
# Example:
#   vfft_add_isa_benchmarks(bench_radix4 "AVX512;AVX2"
#       bench_radix4.c ${RADIX4_DIR}/fft_radix4_fv.c ${RADIX4_DIR}/fft_radix4_bv.c
#       LABELS radix4
#       EXTRA_INCLUDES ${RADIX4_DIR})
#
function(vfft_add_isa_benchmarks BASE_NAME ISA_LIST)
    cmake_parse_arguments(ARG "" "" "LABELS;EXTRA_INCLUDES" ${ARGN})
    set(SOURCES ${ARG_UNPARSED_ARGUMENTS})

    foreach(ISA IN LISTS ISA_LIST)
        _vfft_check_isa(${ISA} _available)
        if(NOT _available)
            continue()
        endif()

        string(TOLOWER ${ISA} isa_lower)
        set(TARGET_NAME "${BASE_NAME}_${isa_lower}")

        add_executable(${TARGET_NAME} ${SOURCES})
        _vfft_setup_target(${TARGET_NAME} ${ISA} EXTRA_INCLUDES ${ARG_EXTRA_INCLUDES})
        target_link_libraries(${TARGET_NAME} PRIVATE m)

        # FFTW3 for comparison benchmarks
        if(TARGET FFTW3::fftw3)
            target_link_libraries(${TARGET_NAME} PRIVATE FFTW3::fftw3)
            target_compile_definitions(${TARGET_NAME} PRIVATE VFFT_HAS_FFTW3=1)
        endif()

        message(STATUS "[VectorFFT] Bench: ${TARGET_NAME}")
    endforeach()
endfunction()
