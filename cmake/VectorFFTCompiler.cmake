# ===========================================================================
# VectorFFTCompiler.cmake
# Compiler detection, ISA flags, and target helpers for VectorFFT
#
# Primary target:   Intel ICX (oneAPI DPC++/C++)
# Fallback:         GCC 12+, Clang 15+
# Platform:         Linux + Windows (clang-cl mode)
# ===========================================================================

include(CheckCCompilerFlag)

# ── Detect compiler + platform ───────────────────────────────────────────
set(VFFT_COMPILER "UNKNOWN")
set(VFFT_WINDOWS_CLANGCL OFF)

if(CMAKE_C_COMPILER_ID MATCHES "IntelLLVM")
    set(VFFT_COMPILER "ICX")
    message(STATUS "[VectorFFT] Compiler: Intel ICX (${CMAKE_C_COMPILER_VERSION})")
    # ICX on Windows uses clang-cl driver (MSVC-style flags mixed with clang)
    if(WIN32)
        set(VFFT_WINDOWS_CLANGCL ON)
        message(STATUS "[VectorFFT] Platform: Windows (clang-cl mode)")
    endif()
elseif(CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(VFFT_COMPILER "CLANG")
    message(STATUS "[VectorFFT] Compiler: Clang (${CMAKE_C_COMPILER_VERSION})")
    if(MSVC)
        set(VFFT_WINDOWS_CLANGCL ON)
        message(STATUS "[VectorFFT] Platform: Windows (clang-cl mode)")
    endif()
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU")
    set(VFFT_COMPILER "GCC")
    message(STATUS "[VectorFFT] Compiler: GCC (${CMAKE_C_COMPILER_VERSION})")
    if(CMAKE_C_COMPILER_VERSION VERSION_LESS "12.0")
        message(WARNING "[VectorFFT] GCC < 12 may produce suboptimal AVX-512 codegen")
    endif()
else()
    message(WARNING "[VectorFFT] Unknown compiler: ${CMAKE_C_COMPILER_ID}")
endif()

# ── Base flags ───────────────────────────────────────────────────────────
set(VFFT_C_STANDARD 11)

if(VFFT_WINDOWS_CLANGCL)
    # clang-cl mode: use /flags for MSVC-style, -flags for clang extensions
    set(VFFT_BASE_FLAGS
        /W3
        -Wconversion
        -Wshadow
        -Wstrict-prototypes
        -fstrict-aliasing
    )

    # ICX Windows-specific
    if(VFFT_COMPILER STREQUAL "ICX")
        list(APPEND VFFT_BASE_FLAGS
            -Qopt-zmm-usage=high
            -Qopt-report=3
            -Qopt-report-phase=vec,loop
            /Oy-                           # Keep frame pointers (VTune)
        )
    endif()

    # _USE_MATH_DEFINES for M_PI, etc.
    set(VFFT_BASE_DEFINITIONS _USE_MATH_DEFINES)
else()
    # Linux / native GCC / Clang
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

    set(VFFT_BASE_DEFINITIONS "")

    if(VFFT_COMPILER STREQUAL "ICX")
        list(APPEND VFFT_BASE_FLAGS
            -qopt-zmm-usage=high
            -qopt-report=3
            -qopt-report-phase=vec,loop
            -fno-omit-frame-pointer
        )
    endif()

    if(VFFT_COMPILER STREQUAL "GCC")
        list(APPEND VFFT_BASE_FLAGS
            -fopt-info-vec-missed
            -fno-semantic-interposition
        )
    endif()
endif()

# ── ISA flag sets ────────────────────────────────────────────────────────
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

# ── ISA detection ────────────────────────────────────────────────────────
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
function(_vfft_setup_target TARGET ISA)
    cmake_parse_arguments(ARG "" "" "EXTRA_INCLUDES" ${ARGN})

    target_compile_options(${TARGET} PRIVATE ${VFFT_BASE_FLAGS} ${VFFT_FLAGS_${ISA}})
    set_target_properties(${TARGET} PROPERTIES C_STANDARD ${VFFT_C_STANDARD})

    # Base definitions (M_PI on Windows, etc.)
    if(VFFT_BASE_DEFINITIONS)
        target_compile_definitions(${TARGET} PRIVATE ${VFFT_BASE_DEFINITIONS})
    endif()

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

        # Platform-specific link libraries
        if(WIN32)
            # No -lm needed on Windows
        else()
            target_link_libraries(${TARGET_NAME} PRIVATE m)
        endif()

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

        if(WIN32)
            # No -lm needed
        else()
            target_link_libraries(${TARGET_NAME} PRIVATE m)
        endif()

        if(TARGET FFTW3::fftw3)
            target_link_libraries(${TARGET_NAME} PRIVATE FFTW3::fftw3)
            target_compile_definitions(${TARGET_NAME} PRIVATE VFFT_HAS_FFTW3=1)
        endif()

        message(STATUS "[VectorFFT] Bench: ${TARGET_NAME}")
    endforeach()
endfunction()