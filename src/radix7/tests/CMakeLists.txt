# ============================================================================
# radix7/tests/CMakeLists.txt
#
# Correctness tests and performance benchmarks for radix-7 codelets.
#
# Correctness (no FFTW needed):
#   test_radix7_dispatch    — All dispatch paths: scalar + AVX2 + AVX-512
#
# Benchmarks:
#   bench_radix7_vs_fftw    — Strided vs Packed vs FFTW (requires FFTW3)
# ============================================================================

set(R7_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(R7_INCLUDE_DIRS
    ${R7_ROOT}
    ${R7_ROOT}/avx512
    ${R7_ROOT}/avx2
    ${R7_ROOT}/scalar
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ── ISA flags ────────────────────────────────────────────────────────────
set(R7_TEST_ISA "")
set(R7_TEST_ISA_FLAGS "")

if(VFFT_HAS_AVX512)
    set(R7_TEST_ISA "AVX512")
    set(R7_TEST_ISA_FLAGS -mavx512f -mavx512dq -mavx2 -mfma)
elseif(VFFT_HAS_AVX2)
    set(R7_TEST_ISA "AVX2")
    set(R7_TEST_ISA_FLAGS -mavx2 -mfma)
endif()

# ── Find FFTW3 ───────────────────────────────────────────────────────────
set(R7_HAS_FFTW OFF)

find_package(FFTW3 CONFIG QUIET)
if(FFTW3_FOUND OR TARGET FFTW3::fftw3)
    set(R7_HAS_FFTW ON)
    message(STATUS "[radix7/tests] FFTW3 found via CMake config")
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFTW3 QUIET fftw3)
        if(FFTW3_FOUND)
            set(R7_HAS_FFTW ON)
            message(STATUS "[radix7/tests] FFTW3 found via pkg-config")
        endif()
    endif()
endif()

if(NOT R7_HAS_FFTW)
    find_library(FFTW3_LIB NAMES fftw3 libfftw3-3)
    find_path(FFTW3_INC NAMES fftw3.h)
    if(FFTW3_LIB AND FFTW3_INC)
        set(R7_HAS_FFTW ON)
        set(FFTW3_LIBRARIES ${FFTW3_LIB})
        set(FFTW3_INCLUDE_DIRS ${FFTW3_INC})
        message(STATUS "[radix7/tests] FFTW3 found via manual search: ${FFTW3_LIB}")
    endif()
endif()

if(NOT R7_HAS_FFTW)
    message(STATUS "[radix7/tests] FFTW3 not found — FFTW benchmarks will be skipped.\n"
                   "  Install: apt install libfftw3-dev  (Linux)\n"
                   "           brew install fftw          (macOS)\n"
                   "           vcpkg install fftw3        (Windows)")
endif()

# ── Helpers ──────────────────────────────────────────────────────────────
function(_r7_link_fftw target)
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

function(_r7_add_target name source)
    cmake_parse_arguments(ARG "NEEDS_FFTW" "" "ISA_FLAGS" ${ARGN})

    add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE ${R7_INCLUDE_DIRS})

    if(ARG_ISA_FLAGS)
        target_compile_options(${name} PRIVATE ${ARG_ISA_FLAGS})
    endif()

    target_compile_options(${name} PRIVATE ${VFFT_BASE_FLAGS})
    set_target_properties(${name} PROPERTIES C_STANDARD ${VFFT_C_STANDARD})

    if(VFFT_BASE_DEFINITIONS)
        target_compile_definitions(${name} PRIVATE ${VFFT_BASE_DEFINITIONS})
    endif()

    if(NOT WIN32)
        target_link_libraries(${name} PRIVATE m)
    endif()

    if(ARG_NEEDS_FFTW)
        _r7_link_fftw(${name})
        target_compile_definitions(${name} PRIVATE VFFT_HAS_FFTW3=1)
    endif()

    add_test(NAME ${name} COMMAND ${name})
endfunction()

# ══════════════════════════════════════════════════════════════════════════
# CORRECTNESS TESTS
# ══════════════════════════════════════════════════════════════════════════

if(R7_TEST_ISA)
    _r7_add_target(test_radix7_dispatch test_radix7_dispatch.c
        ISA_FLAGS ${R7_TEST_ISA_FLAGS})
    set_tests_properties(test_radix7_dispatch
        PROPERTIES LABELS "radix7;correctness;${R7_TEST_ISA}")
    message(STATUS "[radix7/tests] Test: test_radix7_dispatch [${R7_TEST_ISA}]")
endif()

# ══════════════════════════════════════════════════════════════════════════
# BENCHMARKS
# ══════════════════════════════════════════════════════════════════════════

if(VFFT_BUILD_BENCHMARKS)

    if(R7_HAS_FFTW AND VFFT_HAS_AVX512)
        _r7_add_target(bench_radix7_vs_fftw bench_r7_vs_fftw.c
            NEEDS_FFTW
            ISA_FLAGS ${VFFT_FLAGS_AVX512})
        set_tests_properties(bench_radix7_vs_fftw
            PROPERTIES LABELS "radix7;benchmark;fftw;avx512")
        message(STATUS "[radix7/tests] Bench: bench_radix7_vs_fftw [AVX-512]")
    endif()

endif()

# ── Summary ──────────────────────────────────────────────────────────────
get_property(_tests DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY TESTS)
list(LENGTH _tests _n)
message(STATUS "[radix7/tests] ${_n} target(s) registered")
