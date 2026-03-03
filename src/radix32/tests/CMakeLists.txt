# ============================================================================
# radix32/tests — Unit tests + benchmarks
#
# Build:
#   cd radix32 && mkdir build && cd build
#
#   # GCC
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#
#   # ICX
#   cmake .. -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
#            -DCMAKE_BUILD_TYPE=Release
#
#   cmake --build . -j$(nproc)
#   ctest --output-on-failure
# ============================================================================

cmake_minimum_required(VERSION 3.20)

# ============================================================================
# Common compile options
# ============================================================================
set(RADIX32_TEST_WARN
    -Wall -Wextra -Wpedantic
    -Wno-unused-parameter
    -Wno-unused-function
)

# Per-config optimization (works with both single-config and multi-config)
set(RADIX32_TEST_OPT
    $<$<CONFIG:Release>:-O2>
    $<$<CONFIG:RelWithDebInfo>:-O2 -g>
    $<$<CONFIG:Debug>:-O0 -g>
)

# ICX-specific flags
set(RADIX32_TEST_ICX_FLAGS "")
if(RADIX32_IS_ICX)
    set(RADIX32_TEST_ICX_FLAGS
        ${RADIX32_ICX_OPT_REPORT_FLAGS}
    )
endif()

# Aggregate
set(RADIX32_TEST_FLAGS
    ${RADIX32_TEST_WARN}
    ${RADIX32_TEST_OPT}
    ${RADIX32_TEST_ICX_FLAGS}
)

# ============================================================================
# Helper: add a C test executable
# ============================================================================
function(radix32_add_c_test name source)
    # Parse optional arguments: LIBS (extra link libs), FLAGS (extra flags)
    cmake_parse_arguments(ARG "" "" "LIBS;FLAGS" ${ARGN})

    add_executable(${name} ${source})

    target_link_libraries(${name} PRIVATE
        radix32_avx2
        radix32_scalar
        m
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
        m
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
# Helper: add a benchmark (no ctest registration by default)
# ============================================================================
function(radix32_add_bench name source)
    cmake_parse_arguments(ARG "" "" "LIBS;FLAGS" ${ARGN})

    add_executable(${name} ${source})

    target_link_libraries(${name} PRIVATE
        radix32_avx2
        radix32_scalar
        m
        ${ARG_LIBS}
    )

    target_compile_options(${name} PRIVATE
        ${RADIX32_TEST_FLAGS}
        ${ARG_FLAGS}
    )
endfunction()

# ============================================================================
# C unit tests
# ============================================================================

# AVX2 codelet tests
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_radix32_avx2.c")
    radix32_add_c_test(test_radix32_avx2 test_radix32_avx2.c)
endif()

# Scalar codelet tests (BLOCKED8 + BLOCKED4 + RECURRENCE, fwd/bwd, roundtrip)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_radix32_scalar.c")
    radix32_add_c_test(test_radix32_scalar test_radix32_scalar.c)
endif()

# ============================================================================
# C++ GTest suites
# ============================================================================

# N=1 codelet roundtrip
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_radix32_n1_roundtrip.cpp")
    radix32_add_gtest(test_radix32_n1_roundtrip
        test_radix32_n1_roundtrip.cpp)
endif()

# Twiddle codelet roundtrip (130 parameterized tests)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_radix32_twiddle_roundtrip.cpp")
    radix32_add_gtest(test_radix32_twiddle_roundtrip
        test_radix32_twiddle_roundtrip.cpp
        FLAGS
            # C header passes int literal for _mm_prefetch hint (valid C, pedantic C++)
            $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fpermissive -Wno-pedantic>
            $<$<COMPILE_LANG_AND_ID:CXX,IntelLLVM>:-Wno-pedantic>
            $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-Wno-pedantic>
    )
endif()

# ============================================================================
# Benchmarks (built but not run by ctest)
# ============================================================================

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_radix32_avx2.c")
    radix32_add_bench(bench_radix32_avx2 bench_radix32_avx2.c)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench_radix32_pass2.c")
    radix32_add_bench(bench_radix32_pass2 bench_radix32_pass2.c)
endif()

# ============================================================================
# AVX-512 tests (conditional)
# ============================================================================
if(RADIX32_BUILD_AVX512)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test_radix32_avx512.c")
        radix32_add_c_test(test_radix32_avx512 test_radix32_avx512.c
            LIBS radix32_avx512)
    endif()
endif()