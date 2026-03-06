# ============================================================================
# 3-execute/CMakeLists.txt
#
# Planner, execution engine, Bluestein
# Headers: vfft_planner.h, vfft_register_codelets.h, fft_bluestein.h
#
# Tests:
#   test_planner     — 57-point correctness (no FFTW, no codelet deps)
#   test_integrated  — optimized codelets + planner vs FFTW
#   bench_bluestein  — Bluestein correctness + bench vs FFTW
# ============================================================================

# ── Find FFTW ──────────────────────────────────────────────────────────────
find_package(FFTW3 CONFIG QUIET)
if(FFTW3_FOUND OR TARGET FFTW3::fftw3)
    set(FFTW3_FOUND TRUE)
    if(TARGET FFTW3::fftw3)
        set(FFTW3_LIBRARIES FFTW3::fftw3)
    endif()
    message(STATUS "3-execute: FFTW3 found via CMake config")
else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFTW3 QUIET fftw3)
        if(FFTW3_FOUND)
            message(STATUS "3-execute: FFTW3 found via pkg-config")
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
        message(STATUS "3-execute: FFTW3 found via manual search")
    endif()
endif()

# ── Highest ISA flags ─────────────────────────────────────────────────────
set(EXEC_TEST_ISA_FLAGS "")
if(VFFT_HAS_AVX512)
    set(EXEC_TEST_ISA_FLAGS ${VFFT_FLAGS_AVX512})
elseif(VFFT_HAS_AVX2)
    set(EXEC_TEST_ISA_FLAGS ${VFFT_FLAGS_AVX2})
elseif(VFFT_HAS_SSE2)
    set(EXEC_TEST_ISA_FLAGS ${VFFT_FLAGS_SSE2})
endif()

# ── Include paths for codelet headers ─────────────────────────────────────
# test_integrated.c includes fft_radix11_genfft.h etc.
# These live in sibling directories under src/
set(EXEC_CODELET_INCLUDES
    ${PROJECT_SOURCE_DIR}/src/radix11
    ${PROJECT_SOURCE_DIR}/src/radix13
    ${PROJECT_SOURCE_DIR}/src/radix17
    ${PROJECT_SOURCE_DIR}/src/radix19
    ${PROJECT_SOURCE_DIR}/src/radix23
)

# ── Helper ────────────────────────────────────────────────────────────────
function(exec_add_target NAME SOURCE)
    add_executable(${NAME} ${SOURCE})

    target_compile_options(${NAME} PRIVATE
        ${VFFT_BASE_FLAGS}
        ${EXEC_TEST_ISA_FLAGS}
    )
    set_target_properties(${NAME} PROPERTIES C_STANDARD ${VFFT_C_STANDARD})

    if(VFFT_BASE_DEFINITIONS)
        target_compile_definitions(${NAME} PRIVATE ${VFFT_BASE_DEFINITIONS})
    endif()

    target_include_directories(${NAME} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/src/common
        ${EXEC_CODELET_INCLUDES}
    )

    if(NOT WIN32)
        target_link_libraries(${NAME} PRIVATE m)
    endif()
endfunction()

# ── Planner correctness (no FFTW, no codelet deps) ───────────────────────
exec_add_target(test_planner test_planner.c)
add_test(NAME execute_planner COMMAND test_planner)
set_tests_properties(execute_planner PROPERTIES LABELS "execute")
message(STATUS "3-execute: test_planner [ISA: ${EXEC_TEST_ISA_FLAGS}]")

# ── FFTW-dependent targets ───────────────────────────────────────────────
if(FFTW3_FOUND)

    # Integrated test
    exec_add_target(test_integrated test_integrated.c)
    if(TARGET FFTW3::fftw3)
        target_link_libraries(test_integrated PRIVATE FFTW3::fftw3)
    else()
        target_include_directories(test_integrated PRIVATE ${FFTW3_INCLUDE_DIRS})
        target_link_libraries(test_integrated PRIVATE ${FFTW3_LIBRARIES})
    endif()
    add_test(NAME execute_integrated COMMAND test_integrated)
    set_tests_properties(execute_integrated PROPERTIES LABELS "execute")

    # Bluestein benchmark
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench_bluestein.c)
        exec_add_target(bench_bluestein bench_bluestein.c)
        if(TARGET FFTW3::fftw3)
            target_link_libraries(bench_bluestein PRIVATE FFTW3::fftw3)
        else()
            target_include_directories(bench_bluestein PRIVATE ${FFTW3_INCLUDE_DIRS})
            target_link_libraries(bench_bluestein PRIVATE ${FFTW3_LIBRARIES})
        endif()
    endif()

    message(STATUS "3-execute: FFTW3 benchmarks enabled")
else()
    message(WARNING "3-execute: FFTW3 not found — integrated test + benchmarks disabled")
endif()

# ── Install ───────────────────────────────────────────────────────────────
install(FILES
    vfft_planner.h
    vfft_register_codelets.h
    fft_bluestein.h
    DESTINATION include/vectorfft/execute
)