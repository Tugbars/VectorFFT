# ============================================================================
# radix11/CMakeLists.txt
#
# DFT-11 genfft codelet — scalar + AVX2 + AVX-512
# Headers: fft_radix11_genfft.h, fft_radix11_dispatch.h
#          avx512/fft_radix11_avx512_tw_pack_walk.h
#          avx2/fft_radix11_avx2_tw_pack_walk.h
#
# Targets:
#   bench_final_r11    — correctness + benchmark vs FFTW
#   test_radix11_walk  — pack+walk correctness + packed-table vs walk benchmark
# ============================================================================

add_library(vfft_radix11 INTERFACE)
target_include_directories(vfft_radix11 INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/avx512
    ${CMAKE_CURRENT_SOURCE_DIR}/avx2
)
message(STATUS "[VectorFFT] Module: radix11 (header-only)")

# ── Find FFTW ──────────────────────────────────────────────────────────────
find_package(FFTW3 CONFIG QUIET)
if(FFTW3_FOUND OR TARGET FFTW3::fftw3)
    set(FFTW3_FOUND TRUE)
    if(TARGET FFTW3::fftw3)
        set(FFTW3_LIBRARIES FFTW3::fftw3)
    endif()
    message(STATUS "radix11: FFTW3 found via CMake config")
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFTW3 QUIET fftw3)
        if(FFTW3_FOUND)
            message(STATUS "radix11: FFTW3 found via pkg-config")
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
        message(STATUS "radix11: FFTW3 found via manual search")
    endif()
endif()

# ── Highest ISA flags for test TU ─────────────────────────────────────────
set(R11_TEST_ISA_FLAGS "")
set(R11_TEST_ISA "scalar")
if(VFFT_HAS_AVX512)
    set(R11_TEST_ISA_FLAGS ${VFFT_FLAGS_AVX512})
    set(R11_TEST_ISA "AVX512")
elseif(VFFT_HAS_AVX2)
    set(R11_TEST_ISA_FLAGS ${VFFT_FLAGS_AVX2})
    set(R11_TEST_ISA "AVX2")
elseif(VFFT_HAS_SSE2)
    set(R11_TEST_ISA_FLAGS ${VFFT_FLAGS_SSE2})
    set(R11_TEST_ISA "SSE2")
endif()

# ── Helper: create a radix-11 target ──────────────────────────────────────
set(R11_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/avx512
    ${CMAKE_CURRENT_SOURCE_DIR}/avx2
    ${PROJECT_SOURCE_DIR}/src/common
)

function(_r11_add_target name source)
    cmake_parse_arguments(ARG "" "" "ISA_FLAGS;LIBS" ${ARGN})
    add_executable(${name} ${source})
    target_include_directories(${name} PRIVATE ${R11_INCLUDE_DIRS})
    if(ARG_ISA_FLAGS)
        target_compile_options(${name} PRIVATE ${ARG_ISA_FLAGS})
    endif()
    target_compile_options(${name} PRIVATE ${VFFT_BASE_FLAGS})
    set_target_properties(${name} PROPERTIES C_STANDARD ${VFFT_C_STANDARD})
    if(VFFT_BASE_DEFINITIONS)
        target_compile_definitions(${name} PRIVATE ${VFFT_BASE_DEFINITIONS})
    endif()
    if(ARG_LIBS)
        target_link_libraries(${name} PRIVATE ${ARG_LIBS})
    endif()
    if(NOT WIN32)
        target_link_libraries(${name} PRIVATE m)
    endif()
endfunction()

# ── test_radix11_walk — pack+walk correctness + benchmark ─────────────────
if(R11_TEST_ISA)
    _r11_add_target(test_radix11_walk tests/test_radix11_walk.c
        ISA_FLAGS ${R11_TEST_ISA_FLAGS})
    add_test(NAME radix11_walk COMMAND test_radix11_walk)
    set_tests_properties(radix11_walk PROPERTIES LABELS "radix11;correctness;${R11_TEST_ISA}")
    message(STATUS "  radix11: test_radix11_walk [${R11_TEST_ISA}]")
endif()

# ── bench_final_r11 — correctness + benchmark vs FFTW ─────────────────────
if(FFTW3_FOUND)
    set(_fftw_libs "")
    if(TARGET FFTW3::fftw3)
        set(_fftw_libs FFTW3::fftw3)
    else()
        set(_fftw_libs ${FFTW3_LIBRARIES})
    endif()

    _r11_add_target(bench_final_r11 tests/bench_final_r11.c
        ISA_FLAGS ${R11_TEST_ISA_FLAGS}
        LIBS ${_fftw_libs})

    if(FFTW3_INCLUDE_DIRS)
        target_include_directories(bench_final_r11 PRIVATE ${FFTW3_INCLUDE_DIRS})
    endif()

    add_test(NAME radix11_correctness COMMAND bench_final_r11)
    set_tests_properties(radix11_correctness PROPERTIES LABELS "radix11;correctness;bench")
    message(STATUS "  radix11: bench_final_r11 [${R11_TEST_ISA}] (vs FFTW)")
else()
    message(WARNING "radix11: FFTW3 not found — bench_final_r11 will not be built.\n"
                    "  Install: apt install libfftw3-dev  (Linux)\n"
                    "           brew install fftw          (macOS)\n"
                    "           vcpkg install fftw3        (Windows)")
endif()

# ── Summary ───────────────────────────────────────────────────────────────
get_property(_tests DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY TESTS)
list(LENGTH _tests _n)
message(STATUS "[radix11] ${_n} target(s) registered")