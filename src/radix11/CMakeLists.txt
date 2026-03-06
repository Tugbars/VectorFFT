# ============================================================================
# radix11/CMakeLists.txt
#
# DFT-11 genfft codelet — scalar + AVX2 + AVX-512
# Headers: fft_radix11_genfft.h, fft_radix11_dispatch.h
#
# Tests:
#   bench_final_r11 — correctness (41 tests) + benchmark vs FFTW
# ============================================================================

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

if(NOT FFTW3_FOUND)
    message(WARNING "radix11: FFTW3 not found — bench will not be built.\n"
                    "  Install: apt install libfftw3-dev  (Linux)\n"
                    "           brew install fftw          (macOS)\n"
                    "           vcpkg install fftw3        (Windows)")
    return()
endif()

# ── Highest ISA flags for test TU ─────────────────────────────────────────
# The bench includes dispatch.h which conditionally includes all backends.
# The TU must compile with the highest ISA so all headers parse.
set(R11_TEST_ISA_FLAGS "")
if(VFFT_HAS_AVX512)
    set(R11_TEST_ISA_FLAGS ${VFFT_FLAGS_AVX512})
elseif(VFFT_HAS_AVX2)
    set(R11_TEST_ISA_FLAGS ${VFFT_FLAGS_AVX2})
elseif(VFFT_HAS_SSE2)
    set(R11_TEST_ISA_FLAGS ${VFFT_FLAGS_SSE2})
endif()

# ── Bench target ──────────────────────────────────────────────────────────

add_executable(bench_final_r11 bench_final_r11.c)

target_compile_options(bench_final_r11 PRIVATE
    ${VFFT_BASE_FLAGS}
    ${R11_TEST_ISA_FLAGS}
)

set_target_properties(bench_final_r11 PROPERTIES C_STANDARD ${VFFT_C_STANDARD})

if(VFFT_BASE_DEFINITIONS)
    target_compile_definitions(bench_final_r11 PRIVATE ${VFFT_BASE_DEFINITIONS})
endif()

target_include_directories(bench_final_r11 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src/common
)

# FFTW linkage
if(TARGET FFTW3::fftw3)
    target_link_libraries(bench_final_r11 PRIVATE FFTW3::fftw3)
else()
    target_include_directories(bench_final_r11 PRIVATE ${FFTW3_INCLUDE_DIRS})
    target_link_libraries(bench_final_r11 PRIVATE ${FFTW3_LIBRARIES})
endif()

if(NOT WIN32)
    target_link_libraries(bench_final_r11 PRIVATE m)
endif()

add_test(NAME radix11_correctness COMMAND bench_final_r11)
set_tests_properties(radix11_correctness PROPERTIES LABELS "radix11")

# ── Summary ───────────────────────────────────────────────────────────────
message(STATUS "radix11: bench_final_r11 [ISA: ${R11_TEST_ISA_FLAGS}]")