# ============================================================================
# radix3/tests/CMakeLists.txt
#
# Correctness tests and performance benchmarks for radix-3 codelets.
#
# Correctness (no FFTW needed):
#   test_radix3_dispatch    — All dispatch paths: scalar + AVX2 + AVX-512
#   test_radix3_scalar      — Scalar-only: tw + notw
#
# Benchmarks (no FFTW needed):
#   bench_radix3_three_way  — Old hand-written vs New generated vs Packed
# ============================================================================

# ── Include paths ────────────────────────────────────────────────────────
set(R3_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(R3_INCLUDE_DIRS
    ${R3_ROOT}                        # fft_radix3_dispatch.h, fft_radix3_tw_packed.h
    ${R3_ROOT}/avx512
    ${R3_ROOT}/avx2
    ${R3_ROOT}/scalar
    ${CMAKE_CURRENT_SOURCE_DIR}       # test utils
)

# ── Highest available ISA flags ──────────────────────────────────────────
set(R3_TEST_ISA "")
set(R3_TEST_ISA_FLAGS "")

if(VFFT_HAS_AVX512)
    set(R3_TEST_ISA "AVX512")
    if(VFFT_WINDOWS_CLANGCL)
        set(R3_TEST_ISA_FLAGS -mavx512f -mavx512dq -mavx2 -mfma)
    else()
        set(R3_TEST_ISA_FLAGS -mavx512f -mavx512dq -mavx2 -mfma)
    endif()
elseif(VFFT_HAS_AVX2)
    set(R3_TEST_ISA "AVX2")
    if(VFFT_WINDOWS_CLANGCL)
        set(R3_TEST_ISA_FLAGS -mavx2 -mfma)
    else()
        set(R3_TEST_ISA_FLAGS -mavx2 -mfma)
    endif()
endif()

# ── Find FFTW3 ───────────────────────────────────────────────────────────
set(R3_HAS_FFTW OFF)

find_package(FFTW3 CONFIG QUIET)
if(FFTW3_FOUND OR TARGET FFTW3::fftw3)
    set(R3_HAS_FFTW ON)
    message(STATUS "[radix3/tests] FFTW3 found via CMake config")
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFTW3 QUIET fftw3)
        if(FFTW3_FOUND)
            set(R3_HAS_FFTW ON)
            message(STATUS "[radix3/tests] FFTW3 found via pkg-config")
        endif()
    endif()
endif()

if(NOT R3_HAS_FFTW)
    find_library(FFTW3_LIB NAMES fftw3 libfftw3-3)
    find_path(FFTW3_INC NAMES fftw3.h)
    if(FFTW3_LIB AND FFTW3_INC)
        set(R3_HAS_FFTW ON)
        set(FFTW3_LIBRARIES ${FFTW3_LIB})
        set(FFTW3_INCLUDE_DIRS ${FFTW3_INC})
        message(STATUS "[radix3/tests] FFTW3 found via manual search: ${FFTW3_LIB}")
    endif()
endif()

if(NOT R3_HAS_FFTW)
    message(STATUS "[radix3/tests] FFTW3 not found — FFTW benchmarks will be skipped.\n"
                   "  Install: apt install libfftw3-dev  (Linux)\n"
                   "           brew install fftw          (macOS)\n"
                   "           vcpkg install fftw3        (Windows)")
endif()

# ── Helper: link FFTW to a target ────────────────────────────────────────
function(_r3_link_fftw target)
    if(TARGET FFTW3::fftw3)
        target_link_libraries(${target} PRIVATE FFTW3::fftw3)
    else()
        if(FFTW3_INCLUDE_DIRS)
            target_include_directories(${target} PRIVATE ${FFTW3_INCLUDE_DIRS})
        endif()
        if(FFTW3_LIBRARIES)
            target_link_libraries(${target} PRIVATE ${FFTW3_LIBRARIES})
        endif()
    endif()
endfunction()

# ── Helper: add a radix3 test/bench ──────────────────────────────────────
function(_r3_add_target name source)
    cmake_parse_arguments(ARG "NEEDS_FFTW" "" "ISA_FLAGS" ${ARGN})

    add_executable(${name} ${source})

    target_include_directories(${name} PRIVATE ${R3_INCLUDE_DIRS})

    # ISA flags
    if(ARG_ISA_FLAGS)
        target_compile_options(${name} PRIVATE ${ARG_ISA_FLAGS})
    endif()

    # Base flags from VectorFFTCompiler
    target_compile_options(${name} PRIVATE ${VFFT_BASE_FLAGS})
    set_target_properties(${name} PROPERTIES C_STANDARD ${VFFT_C_STANDARD})

    # Platform definitions (_USE_MATH_DEFINES, etc.)
    if(VFFT_BASE_DEFINITIONS)
        target_compile_definitions(${name} PRIVATE ${VFFT_BASE_DEFINITIONS})
    endif()

    # Link math library (Linux/macOS)
    if(NOT WIN32)
        target_link_libraries(${name} PRIVATE m)
    endif()

    # Link FFTW if needed
    if(ARG_NEEDS_FFTW)
        _r3_link_fftw(${name})
        target_compile_definitions(${name} PRIVATE VFFT_HAS_FFTW3=1)
    endif()

    add_test(NAME ${name} COMMAND ${name})
endfunction()

# ══════════════════════════════════════════════════════════════════════════
# BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════

if(VFFT_BUILD_BENCHMARKS)

    # ── Three-way: old vs new vs packed (no FFTW needed) ─────────────────
    if(VFFT_HAS_AVX512)
        _r3_add_target(bench_radix3_three_way bench_r3_three_way.c
            ISA_FLAGS ${VFFT_FLAGS_AVX512})
        set_tests_properties(bench_radix3_three_way
            PROPERTIES LABELS "radix3;benchmark;avx512")
        message(STATUS "[radix3/tests] Bench: bench_radix3_three_way [AVX-512]")
    endif()

elseif(NOT VFFT_BUILD_BENCHMARKS)
    message(STATUS "[radix3/tests] Benchmarks disabled (VFFT_BUILD_BENCHMARKS=OFF)")
endif()

# ── Summary ──────────────────────────────────────────────────────────────
get_property(_tests DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY TESTS)
list(LENGTH _tests _n)
message(STATUS "[radix3/tests] ${_n} target(s) registered")