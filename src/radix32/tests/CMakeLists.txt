# ============================================================================
# radix32/tests — Unit tests + benchmarks
#
# Build (GCC):
#   cd radix32 && mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   cmake --build . -j$(nproc)
#   ctest --output-on-failure
#
# Build (ICX):
#   cmake .. -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
#            -DCMAKE_BUILD_TYPE=Release
#   cmake --build . -j$(nproc)
#   ctest --output-on-failure
#
# Build (ICX, SPR-tuned AVX-512):
#   cmake .. -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
#            -DRADIX32_ICX_AVX512_ARCH=SAPPHIRERAPIDS \
#            -DCMAKE_BUILD_TYPE=Release
#
# Run a single test:
#   ctest -R test_radix32_n1 --output-on-failure
# ============================================================================

cmake_minimum_required(VERSION 3.20)

# ============================================================================
# Common compile options
# ============================================================================
if(RADIX32_IS_MSVC)
    set(RADIX32_TEST_WARN /W3)
    set(RADIX32_TEST_OPT
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/O2 /Zi>
        $<$<CONFIG:Debug>:/Od /Zi>
    )
else()
    set(RADIX32_TEST_WARN
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -Wno-unused-function
    )
    set(RADIX32_TEST_OPT
        $<$<CONFIG:Release>:-O2>
        $<$<CONFIG:RelWithDebInfo>:-O2 -g>
        $<$<CONFIG:Debug>:-O0 -g>
    )
endif()

set(RADIX32_TEST_ICX_FLAGS "")
if(RADIX32_IS_ICX)
    set(RADIX32_TEST_ICX_FLAGS ${RADIX32_ICX_OPT_REPORT_FLAGS})
endif()

set(RADIX32_TEST_FLAGS
    ${RADIX32_TEST_WARN}
    ${RADIX32_TEST_OPT}
    ${RADIX32_TEST_ICX_FLAGS}
)

# ============================================================================
# Highest ISA flags for multi-ISA tests (dispatch, N=1, etc.)
#
# Tests that exercise multiple backends in the same TU need the highest
# ISA flags so all inline functions can be instantiated. Runtime CPUID
# in the test ensures only supported paths execute.
# ============================================================================
if(RADIX32_BUILD_AVX512)
    if(RADIX32_IS_ICX)
        set(RADIX32_ALL_ISA_FLAGS -x${RADIX32_ICX_AVX512_ARCH})
    elseif(RADIX32_IS_MSVC)
        set(RADIX32_ALL_ISA_FLAGS /arch:AVX512)
    else()
        set(RADIX32_ALL_ISA_FLAGS -mavx512f -mavx512dq -mavx512vl -mavx2 -mfma)
    endif()
else()
    if(RADIX32_IS_ICX)
        set(RADIX32_ALL_ISA_FLAGS -x${RADIX32_ICX_AVX2_ARCH})
    elseif(RADIX32_IS_MSVC)
        set(RADIX32_ALL_ISA_FLAGS /arch:AVX2)
    else()
        set(RADIX32_ALL_ISA_FLAGS -mavx2 -mfma)
    endif()
endif()

# Math library: -lm on Unix, not needed on MSVC
if(RADIX32_IS_MSVC)
    set(RADIX32_MATH_LIB "")
else()
    set(RADIX32_MATH_LIB m)
endif()

# ============================================================================
# Helper: add a C test executable
#
# Usage: radix32_add_c_test(name source [LIBS lib1 lib2] [FLAGS -f1 -f2])
# ============================================================================
function(radix32_add_c_test name source)
    cmake_parse_arguments(ARG "" "" "LIBS;FLAGS" ${ARGN})

    add_executable(${name} ${source})

    target_link_libraries(${name} PRIVATE
        radix32_avx2
        radix32_scalar
        ${RADIX32_MATH_LIB}
        ${ARG_LIBS}
    )

    target_compile_options(${name} PRIVATE
        ${RADIX32_TEST_FLAGS}
        ${ARG_FLAGS}
    )

    add_test(NAME ${name} COMMAND ${name})
endfunction()

# ============================================================================
# Helper: add a C++ GTest executable
# ============================================================================
function(radix32_add_gtest name source)
    cmake_parse_arguments(ARG "" "" "LIBS;FLAGS" ${ARGN})

    add_executable(${name} ${source})

    target_link_libraries(${name} PRIVATE
        radix32_avx2
        radix32_scalar
        GTest::gtest_main
        ${RADIX32_MATH_LIB}
        ${ARG_LIBS}
    )

    target_compile_options(${name} PRIVATE
        ${RADIX32_TEST_FLAGS}
        ${ARG_FLAGS}
    )

    include(GoogleTest)
    gtest_discover_tests(${name} TEST_PREFIX "${name}/")
endfunction()

# ============================================================================
# Helper: add a benchmark (built but not registered with ctest)
# ============================================================================
function(radix32_add_bench name source)
    cmake_parse_arguments(ARG "" "" "LIBS;FLAGS" ${ARGN})

    add_executable(${name} ${source})

    target_link_libraries(${name} PRIVATE
        radix32_avx2
        radix32_scalar
        ${RADIX32_MATH_LIB}
        ${ARG_LIBS}
    )

    target_compile_options(${name} PRIVATE
        ${RADIX32_TEST_FLAGS}
        ${ARG_FLAGS}
    )
endfunction()

# ============================================================================
# C unit tests — always built
# ============================================================================

# --- AVX2 codelet tests ---
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_radix32_avx2.c")
    radix32_add_c_test(test_radix32_avx2 test_radix32_avx2.c)
endif()

# --- Scalar codelet tests (BLOCKED8 + BLOCKED4 + RECURRENCE) ---
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_radix32_scalar.c")
    radix32_add_c_test(test_radix32_scalar test_radix32_scalar.c)
endif()

# ============================================================================
# AVX-512 tests (conditional on RADIX32_BUILD_AVX512)
# ============================================================================
if(RADIX32_BUILD_AVX512)

    # --- AVX-512 core butterfly tests (14 tests) ---
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_radix32_avx512_core.c")
        radix32_add_c_test(test_radix32_avx512_core
            test_radix32_avx512_core.c
            LIBS radix32_avx512
            FLAGS ${RADIX32_ALL_ISA_FLAGS}
        )
    endif()

    # --- AVX-512 driver tests (32 tests: U=1/U=2 pass2, fwd/bwd) ---
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_radix32_avx512_driver.c")
        radix32_add_c_test(test_radix32_avx512_driver
            test_radix32_avx512_driver.c
            LIBS radix32_avx512
            FLAGS ${RADIX32_ALL_ISA_FLAGS}
        )
    endif()

endif()

# ============================================================================
# N=1 (twiddle-less) codelet tests
#
# Exercises scalar N=1 and AVX-512 N=1 (if enabled):
#   - N=1 ≡ twiddled(identity)  (bit-exact for scalar, ~0 ULP for AVX-512)
#   - Cross-ISA agreement
#   - Roundtrip: backward(forward(x)) = 32·x
#   - Parseval energy conservation
# ============================================================================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_radix32_n1.c")
    set(_N1_LIBS radix32_scalar)
    set(_N1_FLAGS "")

    if(RADIX32_BUILD_AVX512)
        list(APPEND _N1_LIBS radix32_avx512)
        set(_N1_FLAGS ${RADIX32_ALL_ISA_FLAGS})
    endif()

    radix32_add_c_test(test_radix32_n1
        test_radix32_n1.c
        LIBS ${_N1_LIBS}
        FLAGS ${_N1_FLAGS}
    )
endif()

# ============================================================================
# Dispatch driver tests (links against radix32 static lib)
#
# Verifies:
#   - Three-way agreement: AVX-512 ≡ AVX2 ≡ scalar
#   - Auto-dispatch ISA selection (CPUID-based)
#   - Constraint downgrades (BLOCKED4 → AVX2, K=8 → AVX2)
#   - Auto ≡ forced paths (bit-exact)
# ============================================================================
if(RADIX32_BUILD_DRIVERS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_radix32_dispatch.c")
    add_executable(test_radix32_dispatch test_radix32_dispatch.c)

    target_include_directories(test_radix32_dispatch PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
    )

    # Link against the compiled driver library (fv.o + bv.o)
    target_link_libraries(test_radix32_dispatch PRIVATE
        radix32
        ${RADIX32_MATH_LIB}
    )

    if(RADIX32_BUILD_AVX512)
        target_link_libraries(test_radix32_dispatch PRIVATE radix32_avx512)
    endif()

    target_compile_options(test_radix32_dispatch PRIVATE
        ${RADIX32_TEST_FLAGS}
        ${RADIX32_ALL_ISA_FLAGS}
    )

    add_test(NAME test_radix32_dispatch COMMAND test_radix32_dispatch)
endif()

# ============================================================================
# C++ GTest suites
# ============================================================================

# Suppress C-header pedantic warnings in C++ (e.g. int literal in _mm_prefetch)
set(RADIX32_CXX_RELAX_FLAGS
    $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fpermissive -Wno-pedantic>
    $<$<COMPILE_LANG_AND_ID:CXX,IntelLLVM>:-Wno-pedantic>
    $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wno-pedantic>
)

# --- N=1 codelet roundtrip (GTest) ---
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_radix32_n1_roundtrip.cpp")
    radix32_add_gtest(test_radix32_n1_roundtrip
        test_radix32_n1_roundtrip.cpp
        FLAGS ${RADIX32_CXX_RELAX_FLAGS}
    )
endif()

# --- Twiddle codelet roundtrip (GTest, 130+ parameterized tests) ---
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_radix32_twiddle_roundtrip.cpp")
    radix32_add_gtest(test_radix32_twiddle_roundtrip
        test_radix32_twiddle_roundtrip.cpp
        FLAGS ${RADIX32_CXX_RELAX_FLAGS}
    )
endif()

# ============================================================================
# Benchmarks (built but not registered with ctest)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_radix32_avx2.c")
    radix32_add_bench(bench_radix32_avx2 bench_radix32_avx2.c)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_radix32_pass2.c")
    radix32_add_bench(bench_radix32_pass2 bench_radix32_pass2.c)
endif()

# ============================================================================
# FFTW comparison benchmark (optional — requires libfftw3-dev)
# ============================================================================
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3 QUIET fftw3)
endif()

# Fallback: try find_library if pkg-config didn't find it
if(NOT FFTW3_FOUND)
    find_library(FFTW3_LIB fftw3)
    find_path(FFTW3_INC fftw3.h)
    if(FFTW3_LIB AND FFTW3_INC)
        set(FFTW3_FOUND TRUE)
        set(FFTW3_LIBRARIES ${FFTW3_LIB})
        set(FFTW3_INCLUDE_DIRS ${FFTW3_INC})
    endif()
endif()

if(FFTW3_FOUND AND RADIX32_BUILD_DRIVERS
   AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_radix32_vs_fftw.c")

    add_executable(bench_radix32_vs_fftw bench_radix32_vs_fftw.c)

    target_include_directories(bench_radix32_vs_fftw PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${FFTW3_INCLUDE_DIRS}
    )

    target_link_libraries(bench_radix32_vs_fftw PRIVATE
        radix32                     # dispatch drivers (fv.o + bv.o)
        radix32_scalar
        ${RADIX32_MATH_LIB}
        ${FFTW3_LIBRARIES}
    )

    if(RADIX32_BUILD_AVX512)
        target_link_libraries(bench_radix32_vs_fftw PRIVATE radix32_avx512)
    endif()

    target_compile_options(bench_radix32_vs_fftw PRIVATE
        ${RADIX32_TEST_FLAGS}
        ${RADIX32_ALL_ISA_FLAGS}
    )

    message(STATUS "radix32: FFTW benchmark enabled (fftw3 found)")
else()
    if(NOT FFTW3_FOUND)
        message(STATUS "radix32: FFTW benchmark skipped (libfftw3-dev not found)")
    endif()
endif()