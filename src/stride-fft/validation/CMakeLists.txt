cmake_minimum_required(VERSION 3.16)

# =====================================================================
# VectorFFT Validation Tests
#
# Internal development tests for codelet correctness, log3 twiddle
# derivation, blocked factorization experiments, and standalone
# algorithm tests (Bluestein, Rader).
#
# These are NOT production benchmarks. For VectorFFT vs FFTW/MKL
# comparisons, see the bench/ directory.
#
# Build:
#   cmake --build build --target vfft_validation
#   (or individually: vfft_test_log3, vfft_val_bluestein, etc.)
#
# Enable from parent CMakeLists.txt:
#   set(VFFT_BUILD_VALIDATION ON)
#   add_subdirectory(src/stride-fft/validation)
# =====================================================================

if(NOT TARGET vfft_stride)
    message(WARNING "vfft_stride target not found — validation requires parent build")
    return()
endif()

if(NOT FFTW3_LIB OR NOT FFTW3_INC)
    message(WARNING "FFTW3 not found — validation tests disabled")
    return()
endif()

# Helper: define a validation target with common settings
macro(add_validation_target name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE vfft_stride ${FFTW3_LIB})
    target_include_directories(${name} PRIVATE ${FFTW3_INC})
    target_compile_definitions(${name} PRIVATE _CRT_SECURE_NO_WARNINGS)
    set_target_properties(${name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/validation"
    )
endmacro()

# -- Log3 correctness (220 exhaustive tests) --
add_validation_target(vfft_test_log3 test_log3_all.c)

# -- Log3 calibration bench (per-radix threshold sweep) --
# Disabled: references removed calibration functions from planner.h
# add_validation_target(vfft_val_log3_calib bench_log3_calib.c)

# -- R=4 log3 A/B test --
add_validation_target(vfft_val_r4_log3 bench_r4_log3.c)

# -- R=7 log3 isolation test --
add_validation_target(vfft_val_r7_log3 test_r7_log3.c)

# -- R=32 blocked/log3 test --
add_validation_target(vfft_val_r32_blocked bench_r32_blocked.c)

# -- Monolithic vs blocked decomposition --
add_validation_target(vfft_val_blocked bench_blocked.c)

# -- Bluestein standalone (bypasses planner) --
add_validation_target(vfft_val_bluestein bench_bluestein.c)

# -- Rader standalone (bypasses planner) --
add_validation_target(vfft_val_rader bench_rader.c)

# Convenience target: build all validation tests
add_custom_target(vfft_validation DEPENDS
    vfft_test_log3
    vfft_val_r4_log3
    vfft_val_r7_log3
    vfft_val_r32_blocked
    vfft_val_blocked
    vfft_val_bluestein
    vfft_val_rader
)
