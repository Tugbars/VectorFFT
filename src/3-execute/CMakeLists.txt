cmake_minimum_required(VERSION 3.16)
project(VectorFFT_Execute C)

set(CMAKE_C_STANDARD 11)

# ═══════════════════════════════════════════════════════════════
# ISA detection
# ═══════════════════════════════════════════════════════════════

include(CheckCCompilerFlag)
check_c_compiler_flag("-mavx512f"  HAS_AVX512F)
check_c_compiler_flag("-mavx512dq" HAS_AVX512DQ)
check_c_compiler_flag("-mavx2"     HAS_AVX2)
check_c_compiler_flag("-mfma"      HAS_FMA)

set(VFFT_EXEC_FLAGS "")
if(HAS_AVX512F AND HAS_AVX512DQ)
    list(APPEND VFFT_EXEC_FLAGS -mavx512f -mavx512dq)
endif()
if(HAS_AVX2)
    list(APPEND VFFT_EXEC_FLAGS -mavx2)
endif()
if(HAS_FMA)
    list(APPEND VFFT_EXEC_FLAGS -mfma)
endif()

# ═══════════════════════════════════════════════════════════════
# Header-only interface library
# ═══════════════════════════════════════════════════════════════

add_library(vfft_execute INTERFACE)
target_include_directories(vfft_execute INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../codelets/prime>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../codelets/composite>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../codelets/small>
    $<INSTALL_INTERFACE:include/vectorfft>
)
target_compile_options(vfft_execute INTERFACE ${VFFT_EXEC_FLAGS})
target_link_libraries(vfft_execute INTERFACE m)

# ═══════════════════════════════════════════════════════════════
# Tests & benchmarks
# ═══════════════════════════════════════════════════════════════

enable_testing()

find_library(FFTW3_LIB fftw3)
find_path(FFTW3_INCLUDE fftw3.h)

# Planner correctness — no FFTW needed
add_executable(test_planner test_planner.c)
target_link_libraries(test_planner vfft_execute)
target_compile_options(test_planner PRIVATE -O2 -Wno-unused-function)
add_test(NAME execute_planner COMMAND test_planner)

# FFTW-dependent targets
if(FFTW3_LIB)
    # Integrated test (optimized primes + planner vs FFTW)
    add_executable(test_integrated test_integrated.c)
    target_link_libraries(test_integrated vfft_execute ${FFTW3_LIB})
    target_compile_options(test_integrated PRIVATE -O2 -Wno-unused-function)
    if(FFTW3_INCLUDE)
        target_include_directories(test_integrated PRIVATE ${FFTW3_INCLUDE})
    endif()
    add_test(NAME execute_integrated COMMAND test_integrated)

    # Bluestein benchmark
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/bench_bluestein.c)
        add_executable(bench_bluestein bench_bluestein.c)
        target_link_libraries(bench_bluestein vfft_execute ${FFTW3_LIB})
        target_compile_options(bench_bluestein PRIVATE -O2 -Wno-unused-function)
        if(FFTW3_INCLUDE)
            target_include_directories(bench_bluestein PRIVATE ${FFTW3_INCLUDE})
        endif()
    endif()

    message(STATUS "execute: FFTW3 found — benchmarks enabled")
else()
    message(WARNING "execute: FFTW3 not found — benchmarks disabled")
endif()

# ═══════════════════════════════════════════════════════════════
# Install
# ═══════════════════════════════════════════════════════════════

install(FILES
    vfft_planner.h
    vfft_register_codelets.h
    fft_bluestein.h
    DESTINATION include/vectorfft/execute
)
