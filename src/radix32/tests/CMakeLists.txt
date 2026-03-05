# ============================================================================
# radix32/tests/CMakeLists.txt
#
# Each test is a standalone C executable that benchmarks against FFTW.
# All tests include fft_radix32_dispatch.h (top-level cross-ISA header).
#
# Tests:
#   bench_radix32_scalar_gen    — scalar N1 + tw vs naive + FFTW
#   bench_radix32_avx2_n1      — AVX2 N1 vs naive + FFTW
#   bench_radix32_tw_unified   — AVX-512 twiddled unified dispatch vs FFTW
#   bench_radix32_u2           — AVX-512 N1 U1+U2 vs naive + FFTW
# ============================================================================

# ── Find FFTW ──────────────────────────────────────────────────────────────
# 1. CMake config mode (vcpkg, manual installs with CMake configs)
# 2. pkg-config (Linux system packages)
# 3. Manual search (fallback)

find_package(FFTW3 CONFIG QUIET)
if(FFTW3_FOUND OR TARGET FFTW3::fftw3)
    set(FFTW3_FOUND TRUE)
    if(TARGET FFTW3::fftw3)
        set(FFTW3_LIBRARIES FFTW3::fftw3)
    endif()
    message(STATUS "radix32/tests: FFTW3 found via CMake config")
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFTW3 QUIET fftw3)
        if(FFTW3_FOUND)
            message(STATUS "radix32/tests: FFTW3 found via pkg-config")
        endif()
    endif()
endif()

if(NOT FFTW3_FOUND)
    find_library(FFTW3_LIB NAMES fftw3 libfftw3-3)
    find_path(FFTW3_INC NAMES fftw3.h)
    if(FFTW3_LIB AND FFTW3_INC)
        set(FFTW3_FOUND TRUE)
        set(FFTW3_LIBRARIES ${FFTW3_LIB})
        set(FFTW3_INCLUDE_DIRS ${FFTW3_INC})
        message(STATUS "radix32/tests: FFTW3 found via manual search")
    endif()
endif()

if(NOT FFTW3_FOUND)
    message(WARNING "radix32/tests: FFTW3 not found — tests will not be built.\n"
                    "  Install: apt install libfftw3-dev  (Linux)\n"
                    "           brew install fftw          (macOS)\n"
                    "           vcpkg install fftw3        (Windows)")
    return()
endif()

message(STATUS "radix32/tests: FFTW3 found")

# ── Highest ISA flags for test TUs ────────────────────────────────────────
# Tests include dispatch.h which conditionally includes all backends.
# The TU must compile with the highest ISA so all headers parse.
set(R32_TEST_ISA_FLAGS "")
if(R32_BUILD_AVX512)
    if(R32_IS_ICX)
        set(R32_TEST_ISA_FLAGS -x${R32_ICX_AVX512_ARCH})
    elseif(R32_IS_MSVC)
        set(R32_TEST_ISA_FLAGS /arch:AVX512)
    else()
        set(R32_TEST_ISA_FLAGS -mavx512f -mavx512dq -mavx2 -mfma)
    endif()
else()
    if(R32_IS_ICX)
        set(R32_TEST_ISA_FLAGS -x${R32_ICX_AVX2_ARCH})
    elseif(R32_IS_MSVC)
        set(R32_TEST_ISA_FLAGS /arch:AVX2)
    else()
        set(R32_TEST_ISA_FLAGS -mavx2 -mfma)
    endif()
endif()

# ── Helper: add a radix32 test ────────────────────────────────────────────
function(r32_add_test name source)
    add_executable(${name} ${source})

    # Include paths: tests/ dir + all backend dirs via radix32 interface
    target_link_libraries(${name} PRIVATE radix32)

    # FFTW
    if(TARGET FFTW3::fftw3)
        target_link_libraries(${name} PRIVATE FFTW3::fftw3)
    else()
        target_include_directories(${name} PRIVATE ${FFTW3_INCLUDE_DIRS})
        target_link_libraries(${name} PRIVATE ${FFTW3_LIBRARIES})
    endif()
    if(NOT R32_IS_MSVC)
        target_link_libraries(${name} PRIVATE m)
    endif()

    # ISA + warning flags
    target_compile_options(${name} PRIVATE
        ${R32_TEST_ISA_FLAGS}
        ${R32_WARN}
        $<$<NOT:$<BOOL:${R32_IS_MSVC}>>:-O2>
        $<$<BOOL:${R32_IS_MSVC}>:/O2>
    )

    add_test(NAME ${name} COMMAND ${name})
endfunction()

# ── Tests ─────────────────────────────────────────────────────────────────

r32_add_test(bench_radix32_scalar_gen
    ${CMAKE_CURRENT_SOURCE_DIR}/bench_radix32_scalar_gen.c)

r32_add_test(bench_radix32_avx2_n1
    ${CMAKE_CURRENT_SOURCE_DIR}/bench_radix32_avx2_n1.c)

if(R32_BUILD_AVX512)
    r32_add_test(bench_radix32_tw_unified
        ${CMAKE_CURRENT_SOURCE_DIR}/bench_radix32_tw_unified.c)

    r32_add_test(bench_radix32_u2
        ${CMAKE_CURRENT_SOURCE_DIR}/bench_radix32_u2.c)
endif()

# ── Summary ───────────────────────────────────────────────────────────────
get_property(_tests DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY TESTS)
list(LENGTH _tests _n)
message(STATUS "radix32/tests: ${_n} test(s) registered")