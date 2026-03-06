cmake_minimum_required(VERSION 3.16)
project(VectorFFT_Radix11 C)

set(CMAKE_C_STANDARD 11)

# ═══════════════════════════════════════════════════════════════
# ISA DETECTION
# ═══════════════════════════════════════════════════════════════

include(CheckCCompilerFlag)

check_c_compiler_flag("-mavx512f" HAS_AVX512F)
check_c_compiler_flag("-mavx2"    HAS_AVX2)
check_c_compiler_flag("-mfma"     HAS_FMA)

set(RADIX11_C_FLAGS "")
if(HAS_AVX512F)
    list(APPEND RADIX11_C_FLAGS -mavx512f -mavx512dq)
    message(STATUS "Radix-11: AVX-512 enabled")
endif()
if(HAS_AVX2)
    list(APPEND RADIX11_C_FLAGS -mavx2)
    message(STATUS "Radix-11: AVX2 enabled")
endif()
if(HAS_FMA)
    list(APPEND RADIX11_C_FLAGS -mfma)
    message(STATUS "Radix-11: FMA enabled")
endif()

# ═══════════════════════════════════════════════════════════════
# HEADER-ONLY LIBRARY (interface target)
# ═══════════════════════════════════════════════════════════════

add_library(radix11 INTERFACE)
target_include_directories(radix11 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(radix11 INTERFACE ${RADIX11_C_FLAGS})

# ═══════════════════════════════════════════════════════════════
# TEST SUITE
# ═══════════════════════════════════════════════════════════════

enable_testing()

# Full correctness + benchmark test
add_executable(test_radix11 bench_final_r11.c)
target_link_libraries(test_radix11 radix11 m)
target_compile_options(test_radix11 PRIVATE -O2 -Wno-unused-function)

# Need FFTW for benchmarking
find_library(FFTW3_LIB fftw3)
if(FFTW3_LIB)
    target_link_libraries(test_radix11 ${FFTW3_LIB})
    message(STATUS "Radix-11: FFTW3 found for benchmarking")
else()
    message(WARNING "Radix-11: FFTW3 not found, benchmark will fail to link")
endif()

add_test(NAME radix11_correctness COMMAND test_radix11)

# ═══════════════════════════════════════════════════════════════
# INSTALL
# ═══════════════════════════════════════════════════════════════

install(FILES
    fft_radix11_genfft.h
    fft_radix11_dispatch.h
    DESTINATION include/vectorfft
)
