# ============================================================================
# radix16/tests/CMakeLists.txt
#
# Tests:
#   bench_radix16_avx2_n1          — N1 AVX2 vs FFTW
#   bench_radix16_avx2_n1_packed   — N1 strided vs packed vs FFTW
#   bench_radix16_avx2_tw_packed   — TW strided vs packed vs FFTW
#   bench_radix16_avx512_tw        — TW AVX-512 vs FFTW (skips if no HW)
# ============================================================================

# ── Find FFTW ──────────────────────────────────────────────────────────────
find_package(FFTW3 CONFIG QUIET)
if(FFTW3_FOUND OR TARGET FFTW3::fftw3)
    set(FFTW3_FOUND TRUE)
    if(TARGET FFTW3::fftw3)
        set(FFTW3_LIBRARIES FFTW3::fftw3)
    endif()
    message(STATUS "radix16/tests: FFTW3 found via CMake config")
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFTW3 QUIET fftw3)
        if(FFTW3_FOUND)
            message(STATUS "radix16/tests: FFTW3 found via pkg-config")
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
        message(STATUS "radix16/tests: FFTW3 found via manual search")
    endif()
endif()

if(NOT FFTW3_FOUND)
    message(WARNING "radix16/tests: FFTW3 not found — tests will not be built.\n"
                    "  Install: apt install libfftw3-dev  (Linux)\n"
                    "           brew install fftw          (macOS)\n"
                    "           vcpkg install fftw3        (Windows)")
    return()
endif()

# ── Highest ISA flags for test TUs ────────────────────────────────────────
set(R16_TEST_AVX2_FLAGS "")
set(R16_TEST_AVX512_FLAGS "")

if(R16_IS_ICX)
    set(R16_TEST_AVX2_FLAGS   -x${R16_ICX_AVX2_ARCH})
    set(R16_TEST_AVX512_FLAGS -x${R16_ICX_AVX512_ARCH})
elseif(R16_IS_MSVC)
    set(R16_TEST_AVX2_FLAGS   /arch:AVX2)
    set(R16_TEST_AVX512_FLAGS /arch:AVX512)
else()
    set(R16_TEST_AVX2_FLAGS   -mavx2 -mfma)
    set(R16_TEST_AVX512_FLAGS -mavx512f -mavx512dq -mavx2 -mfma)
endif()

# ── Helper: add a radix16 test ────────────────────────────────────────────
function(r16_add_test name source isa_flags isa_level)
    add_executable(${name} ${source})

    target_include_directories(${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
    )

    # Link only the ISA targets needed — avoids inheriting AVX-512 flags on AVX2 tests
    target_link_libraries(${name} PRIVATE radix16_scalar)
    if(isa_level STREQUAL "avx2" OR isa_level STREQUAL "avx512")
        target_link_libraries(${name} PRIVATE radix16_avx2)
    endif()
    if(isa_level STREQUAL "avx512" AND R16_BUILD_AVX512)
        target_link_libraries(${name} PRIVATE radix16_avx512)
    endif()

    # FFTW
    if(TARGET FFTW3::fftw3)
        target_link_libraries(${name} PRIVATE FFTW3::fftw3)
    else()
        target_include_directories(${name} PRIVATE ${FFTW3_INCLUDE_DIRS})
        target_link_libraries(${name} PRIVATE ${FFTW3_LIBRARIES})
    endif()
    if(NOT WIN32)
        target_link_libraries(${name} PRIVATE m)
    endif()

    # ISA + warning flags
    target_compile_options(${name} PRIVATE
        ${isa_flags}
        ${R16_WARN}
        $<$<NOT:$<BOOL:${R16_IS_MSVC}>>:-O2>
        $<$<BOOL:${R16_IS_MSVC}>:/O2>
    )

    add_test(NAME ${name} COMMAND ${name})
endfunction()

# ── AVX2 tests (always built) ────────────────────────────────────────────

r16_add_test(bench_radix16_avx2_n1
    ${CMAKE_CURRENT_SOURCE_DIR}/bench_radix16_avx2_n1.c
    "${R16_TEST_AVX2_FLAGS}" "avx2")

r16_add_test(bench_radix16_avx2_n1_packed
    ${CMAKE_CURRENT_SOURCE_DIR}/bench_radix16_avx2_n1_packed.c
    "${R16_TEST_AVX2_FLAGS}" "avx2")

r16_add_test(bench_radix16_avx2_tw_packed
    ${CMAKE_CURRENT_SOURCE_DIR}/bench_radix16_avx2_tw_packed.c
    "${R16_TEST_AVX2_FLAGS}" "avx2")

# ── AVX-512 tests (conditional) ──────────────────────────────────────────

if(R16_BUILD_AVX512)
    r16_add_test(bench_radix16_avx512_tw
        ${CMAKE_CURRENT_SOURCE_DIR}/bench_radix16_avx512_tw.c
        "${R16_TEST_AVX512_FLAGS}" "avx512")
endif()

# ── Summary ───────────────────────────────────────────────────────────────
get_property(_tests DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY TESTS)
list(LENGTH _tests _n)
message(STATUS "radix16/tests: ${_n} test(s) registered")