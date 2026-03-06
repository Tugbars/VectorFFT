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
# 1. CMake config mode (vcpkg, manual installs with CMake configs)
# 2. pkg-config (Linux system packages)
# 3. Manual search (fallback)

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
endif()

# ── Header-only library ───────────────────────────────────────────────────

add_library(radix11 INTERFACE)
target_include_directories(radix11 INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/vectorfft>
)

# ── ISA flags ─────────────────────────────────────────────────────────────
# Inherit from parent if available, otherwise detect locally

if(NOT DEFINED R11_TEST_ISA_FLAGS)
    set(R11_TEST_ISA_FLAGS "")
    include(CheckCCompilerFlag)

    # Check compiler identity — ICX BEFORE MSVC (ICX sets MSVC=TRUE on Windows)
    set(R11_IS_MSVC FALSE)
    set(R11_IS_ICX FALSE)
    if(CMAKE_C_COMPILER_ID MATCHES "IntelLLVM")
        set(R11_IS_ICX TRUE)
    elseif(MSVC)
        set(R11_IS_MSVC TRUE)
    endif()

    if(R11_IS_ICX)
        # ICX supports GCC-style flags on all platforms
        check_c_compiler_flag("-mavx512f" R11_HAS_AVX512)
        check_c_compiler_flag("-mavx2" R11_HAS_AVX2)
        check_c_compiler_flag("-mfma" R11_HAS_FMA)
        if(R11_HAS_AVX512)
            set(R11_TEST_ISA_FLAGS -mavx512f -mavx512dq -mavx2 -mfma)
        elseif(R11_HAS_AVX2 AND R11_HAS_FMA)
            set(R11_TEST_ISA_FLAGS -mavx2 -mfma)
        endif()
    elseif(R11_IS_MSVC)
        check_c_compiler_flag("/arch:AVX512" R11_HAS_AVX512)
        check_c_compiler_flag("/arch:AVX2" R11_HAS_AVX2)
        if(R11_HAS_AVX512)
            set(R11_TEST_ISA_FLAGS /arch:AVX512)
        elseif(R11_HAS_AVX2)
            set(R11_TEST_ISA_FLAGS /arch:AVX2)
        endif()
    else()
        check_c_compiler_flag("-mavx512f" R11_HAS_AVX512)
        check_c_compiler_flag("-mavx2" R11_HAS_AVX2)
        check_c_compiler_flag("-mfma" R11_HAS_FMA)
        if(R11_HAS_AVX512)
            list(APPEND R11_TEST_ISA_FLAGS -mavx512f -mavx512dq)
        endif()
        if(R11_HAS_AVX2)
            list(APPEND R11_TEST_ISA_FLAGS -mavx2)
        endif()
        if(R11_HAS_FMA)
            list(APPEND R11_TEST_ISA_FLAGS -mfma)
        endif()
    endif()

    message(STATUS "radix11: compiler=${CMAKE_C_COMPILER_ID} ISA flags=${R11_TEST_ISA_FLAGS}")
endif()

# ── Tests ─────────────────────────────────────────────────────────────────

if(FFTW3_FOUND)
    enable_testing()

    add_executable(bench_final_r11 bench_final_r11.c)
    target_include_directories(bench_final_r11 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(bench_final_r11 PRIVATE radix11)

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

    target_compile_options(bench_final_r11 PRIVATE
        ${R11_TEST_ISA_FLAGS}
        $<$<BOOL:${R11_IS_MSVC}>:/O2>
        $<$<NOT:$<BOOL:${R11_IS_MSVC}>>:-O2 -Wno-unused-function>
    )

    add_test(NAME radix11_correctness COMMAND bench_final_r11)
    message(STATUS "radix11: bench_final_r11 registered")
endif()

# ── Install ───────────────────────────────────────────────────────────────

install(FILES
    fft_radix11_genfft.h
    fft_radix11_dispatch.h
    DESTINATION include/vectorfft
)