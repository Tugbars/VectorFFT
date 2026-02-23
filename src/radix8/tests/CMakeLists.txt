##############################################################################
# radix8/tests/ CMakeLists.txt
#
# Test executables for radix-8 FFT.
#
# Two categories:
#   1. STANDALONE header tests — compile a single .c that #includes the
#      ISA-specific header directly.  Only built when the ISA is available.
#   2. DRIVER integration test — links against the radix8 library and
#      exercises the public API at the resolved ISA level.
#
# All tests register with CTest:
#   ctest --test-dir build --output-on-failure
#
##############################################################################
enable_testing()

# Inherit from parent: RADIX8_RESOLVED_ISA, RADIX8_C_FLAGS, FLAGS_*

# ── Math library ────────────────────────────────────────────────────────────
if(NOT MSVC)
    find_library(TEST_MATH_LIB m)
endif()

# ── Helper: add a standalone header test ────────────────────────────────────
#
# standalone_test(
#   NAME   test_radix8_avx2         # target & CTest name
#   SOURCE test_radix8_avx2.c       # source file
#   FLAGS  -mavx2 -mfma             # compile flags
#   DIRS   avx2 scalar              # subdirs under radix8/ to add as includes
# )
#
function(standalone_test)
    cmake_parse_arguments(ST "" "NAME;SOURCE" "FLAGS;DIRS" ${ARGN})

    add_executable(${ST_NAME} ${ST_SOURCE})
    target_compile_options(${ST_NAME} PRIVATE ${ST_FLAGS})

    # Build include dir list from subdirectory names
    set(_inc_dirs)
    foreach(_dir ${ST_DIRS})
        list(APPEND _inc_dirs "${CMAKE_CURRENT_SOURCE_DIR}/../${_dir}")
    endforeach()
    # Always include parent (for fft_radix8.h) and scalar (always needed as fallback)
    list(APPEND _inc_dirs "${CMAKE_CURRENT_SOURCE_DIR}/..")
    list(APPEND _inc_dirs "${CMAKE_CURRENT_SOURCE_DIR}/../scalar")

    target_include_directories(${ST_NAME} PRIVATE ${_inc_dirs})

    if(TEST_MATH_LIB)
        target_link_libraries(${ST_NAME} PRIVATE ${TEST_MATH_LIB})
    endif()

    add_test(NAME ${ST_NAME} COMMAND ${ST_NAME})
endfunction()

# ── Standalone: Scalar (always built) ───────────────────────────────────────
if(MSVC)
    set(_scalar_flags "")
else()
    set(_scalar_flags "-mfma")
endif()

standalone_test(
    NAME   test_radix8_scalar
    SOURCE test_radix8_scalar.c
    FLAGS  ${_scalar_flags}
    DIRS   scalar
)

# ── Standalone: AVX2 (built when AVX2 or higher available) ──────────────────
if(RADIX8_RESOLVED_ISA STREQUAL "AVX2" OR RADIX8_RESOLVED_ISA STREQUAL "AVX512")
    standalone_test(
        NAME   test_radix8_avx2
        SOURCE test_radix8_avx2.c
        FLAGS  ${FLAGS_AVX2}
        DIRS   avx2
    )

    standalone_test(
        NAME   test_radix8_n1
        SOURCE test_radix8_n1.c
        FLAGS  ${FLAGS_AVX2}
        DIRS   avx2
    )
endif()

# ── Standalone: AVX-512 (built when AVX-512 available) ──────────────────────
if(RADIX8_RESOLVED_ISA STREQUAL "AVX512")
    standalone_test(
        NAME   test_radix8_avx512
        SOURCE test_radix8_avx512.c
        FLAGS  ${FLAGS_AVX512}
        DIRS   avx512 avx2
    )
endif()

# ── Driver integration test (always built, uses resolved ISA) ───────────────
add_executable(test_radix8_drivers test_radix8_drivers.c)
target_link_libraries(test_radix8_drivers PRIVATE radix8)
target_compile_options(test_radix8_drivers PRIVATE ${RADIX8_C_FLAGS})
target_include_directories(test_radix8_drivers PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

if(TEST_MATH_LIB)
    target_link_libraries(test_radix8_drivers PRIVATE ${TEST_MATH_LIB})
endif()

add_test(NAME test_radix8_drivers COMMAND test_radix8_drivers)

# ── Summary ─────────────────────────────────────────────────────────────────
message(STATUS "Radix-8 tests enabled (ISA: ${RADIX8_RESOLVED_ISA})")

# Count tests
get_property(_tests DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY TESTS)
list(LENGTH _tests _n_tests)
message(STATUS "  ${_n_tests} test executables registered")
