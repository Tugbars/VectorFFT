##############################################################################
# radix3/tests/ CMakeLists.txt
#
# Test executables for radix-3 FFT stages.
#
#   1. STANDALONE test — includes all ISA headers directly (test_radix3.c)
#   2. DRIVER test     — links against libradix3 (test_radix3_drivers.c)
#
##############################################################################
enable_testing()

if(NOT MSVC)
    find_library(R3_TEST_MATH_LIB m)
endif()

# ── Standalone header test (always built, uses resolved ISA) ────────────────
add_executable(test_radix3 test_radix3.c)
target_compile_options(test_radix3 PRIVATE ${R3_C_FLAGS})
target_include_directories(test_radix3 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
if(R3_TEST_MATH_LIB)
    target_link_libraries(test_radix3 PRIVATE ${R3_TEST_MATH_LIB})
endif()
add_test(NAME test_radix3 COMMAND test_radix3)

# ── Driver integration test (links against library) ────────────────────────
add_executable(test_radix3_drivers test_radix3_drivers.c)
target_link_libraries(test_radix3_drivers PRIVATE radix3)
target_compile_options(test_radix3_drivers PRIVATE ${R3_C_FLAGS})
target_include_directories(test_radix3_drivers PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
if(R3_TEST_MATH_LIB)
    target_link_libraries(test_radix3_drivers PRIVATE ${R3_TEST_MATH_LIB})
endif()
add_test(NAME test_radix3_drivers COMMAND test_radix3_drivers)

message(STATUS "Radix-3 tests enabled (ISA: ${R3_RESOLVED_ISA})")
