# ============================================================================
# radix32/tests/CMakeLists.txt
#
# Correctness tests and performance benchmarks for radix-32 codelets.
#
# Correctness (no FFTW needed):
#   test_radix32_dispatch   — All dispatch paths: scalar + AVX2 + AVX-512
#   test_radix32_scalar     — Scalar-only: tw + notw
#
# Benchmarks (require FFTW3):
#   bench_radix32_avx512_tw — Twiddled AVX-512 (flat+ladder) vs FFTW
#   bench_radix32_avx2_tw   — Twiddled AVX2 flat vs FFTW
#   bench_radix32_notw      — Twiddle-less AVX-512+AVX2 vs FFTW
# ============================================================================

# ── Include paths ────────────────────────────────────────────────────────
set(R32_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(R32_INCLUDE_DIRS
    ${R32_ROOT}                       # fft_radix32_dispatch.h
    ${R32_ROOT}/avx512
    ${R32_ROOT}/avx2
    ${R32_ROOT}/scalar
    ${CMAKE_CURRENT_SOURCE_DIR}       # radix32_test_utils.h
)

# ── Highest available ISA flags ──────────────────────────────────────────
# dispatch.h conditionally includes all backends; the TU needs the highest
# ISA so every #ifdef branch parses.
set(R32_TEST_ISA "")
set(R32_TEST_ISA_FLAGS "")

if(VFFT_HAS_AVX512)
    set(R32_TEST_ISA "AVX512")
    if(VFFT_WINDOWS_CLANGCL)
        set(R32_TEST_ISA_FLAGS -mavx512f -mavx512dq -mavx2 -mfma)
    else()
        set(R32_TEST_ISA_FLAGS -mavx512f -mavx512dq -mavx2 -mfma)
    endif()
elseif(VFFT_HAS_AVX2)
    set(R32_TEST_ISA "AVX2")
    if(VFFT_WINDOWS_CLANGCL)
        set(R32_TEST_ISA_FLAGS -mavx2 -mfma)
    else()
        set(R32_TEST_ISA_FLAGS -mavx2 -mfma)
    endif()
endif()

# ── Find FFTW3 ───────────────────────────────────────────────────────────
# 1. CMake config mode (vcpkg, manual installs with CMake configs)
# 2. pkg-config (Linux system packages)
# 3. Manual search (fallback)

set(R32_HAS_FFTW OFF)

find_package(FFTW3 CONFIG QUIET)
if(FFTW3_FOUND OR TARGET FFTW3::fftw3)
    set(R32_HAS_FFTW ON)
    message(STATUS "[radix32/tests] FFTW3 found via CMake config")
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFTW3 QUIET fftw3)
        if(FFTW3_FOUND)
            set(R32_HAS_FFTW ON)
            message(STATUS "[radix32/tests] FFTW3 found via pkg-config")
        endif()
    endif()
endif()

if(NOT R32_HAS_FFTW)
    find_library(FFTW3_LIB NAMES fftw3 libfftw3-3)
    find_path(FFTW3_INC NAMES fftw3.h)
    if(FFTW3_LIB AND FFTW3_INC)
        set(R32_HAS_FFTW ON)
        set(FFTW3_LIBRARIES ${FFTW3_LIB})
        set(FFTW3_INCLUDE_DIRS ${FFTW3_INC})
        message(STATUS "[radix32/tests] FFTW3 found via manual search: ${FFTW3_LIB}")
    endif()
endif()

if(NOT R32_HAS_FFTW)
    message(STATUS "[radix32/tests] FFTW3 not found — benchmarks will be skipped.\n"
                   "  Install: apt install libfftw3-dev  (Linux)\n"
                   "           brew install fftw          (macOS)\n"
                   "           vcpkg install fftw3        (Windows)")
endif()

# ── Helper: link FFTW to a target ────────────────────────────────────────
function(_r32_link_fftw target)
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

# ── Helper: add a radix32 test/bench ─────────────────────────────────────
function(_r32_add_target name source)
    cmake_parse_arguments(ARG "NEEDS_FFTW" "" "ISA_FLAGS" ${ARGN})

    add_executable(${name} ${source})

    target_include_directories(${name} PRIVATE ${R32_INCLUDE_DIRS})

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
        _r32_link_fftw(${name})
        target_compile_definitions(${name} PRIVATE VFFT_HAS_FFTW3=1)
    endif()

    add_test(NAME ${name} COMMAND ${name})
endfunction()

# ══════════════════════════════════════════════════════════════════════════
# CORRECTNESS TESTS (no FFTW dependency)
# ══════════════════════════════════════════════════════════════════════════

# ── test_dispatch: all ISA paths (compiled with highest available) ────────
if(R32_TEST_ISA)
    _r32_add_target(test_radix32_dispatch test_dispatch.c
        ISA_FLAGS ${R32_TEST_ISA_FLAGS})
    set_tests_properties(test_radix32_dispatch
        PROPERTIES LABELS "radix32;correctness;${R32_TEST_ISA}")
    message(STATUS "[radix32/tests] Test: test_radix32_dispatch [${R32_TEST_ISA}]")
endif()

# ── test_scalar: portable, no SIMD flags ─────────────────────────────────
_r32_add_target(test_radix32_scalar test_scalar.c)
set_tests_properties(test_radix32_scalar
    PROPERTIES LABELS "radix32;correctness;scalar")
message(STATUS "[radix32/tests] Test: test_radix32_scalar [scalar]")

# ══════════════════════════════════════════════════════════════════════════
# BENCHMARKS (require FFTW3)
# ══════════════════════════════════════════════════════════════════════════

if(VFFT_BUILD_BENCHMARKS AND R32_HAS_FFTW)

    # ── AVX-512 twiddled vs FFTW ─────────────────────────────────────────
    if(VFFT_HAS_AVX512)
        _r32_add_target(bench_radix32_avx512_tw bench_avx512_tw_vs_fftw.c
            NEEDS_FFTW
            ISA_FLAGS ${VFFT_FLAGS_AVX512})
        set_tests_properties(bench_radix32_avx512_tw
            PROPERTIES LABELS "radix32;benchmark;avx512")
        message(STATUS "[radix32/tests] Bench: bench_radix32_avx512_tw [AVX-512]")
    endif()

    # ── AVX2 twiddled vs FFTW ────────────────────────────────────────────
    if(VFFT_HAS_AVX2)
        _r32_add_target(bench_radix32_avx2_tw bench_avx2_tw_vs_fftw.c
            NEEDS_FFTW
            ISA_FLAGS ${VFFT_FLAGS_AVX2})
        set_tests_properties(bench_radix32_avx2_tw
            PROPERTIES LABELS "radix32;benchmark;avx2")
        message(STATUS "[radix32/tests] Bench: bench_radix32_avx2_tw [AVX2]")
    endif()

    # ── Twiddle-less (AVX-512 + AVX2 in one binary) vs FFTW ─────────────
    if(VFFT_HAS_AVX512)
        _r32_add_target(bench_radix32_notw bench_notw_vs_fftw.c
            NEEDS_FFTW
            ISA_FLAGS ${VFFT_FLAGS_AVX512})
        set_tests_properties(bench_radix32_notw
            PROPERTIES LABELS "radix32;benchmark;notw;avx512")
        message(STATUS "[radix32/tests] Bench: bench_radix32_notw [AVX-512]")
    elseif(VFFT_HAS_AVX2)
        _r32_add_target(bench_radix32_notw bench_notw_vs_fftw.c
            NEEDS_FFTW
            ISA_FLAGS ${VFFT_FLAGS_AVX2})
        set_tests_properties(bench_radix32_notw
            PROPERTIES LABELS "radix32;benchmark;notw;avx2")
        message(STATUS "[radix32/tests] Bench: bench_radix32_notw [AVX2]")
    endif()

elseif(VFFT_BUILD_BENCHMARKS AND NOT R32_HAS_FFTW)
    message(STATUS "[radix32/tests] Benchmarks skipped (FFTW3 not found)")
endif()

# ── Summary ──────────────────────────────────────────────────────────────
get_property(_tests DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY TESTS)
list(LENGTH _tests _n)
message(STATUS "[radix32/tests] ${_n} target(s) registered")