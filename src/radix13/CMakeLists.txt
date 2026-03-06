cmake_minimum_required(VERSION 3.16)
project(VectorFFT_Radix13 C)
set(CMAKE_C_STANDARD 11)

include(CheckCCompilerFlag)
check_c_compiler_flag("-mavx512f" HAS_AVX512F)
check_c_compiler_flag("-mavx2" HAS_AVX2)
check_c_compiler_flag("-mfma" HAS_FMA)

set(RADIX13_C_FLAGS "")
if(HAS_AVX512F) list(APPEND RADIX13_C_FLAGS -mavx512f -mavx512dq) endif()
if(HAS_AVX2) list(APPEND RADIX13_C_FLAGS -mavx2) endif()
if(HAS_FMA) list(APPEND RADIX13_C_FLAGS -mfma) endif()

add_library(radix13 INTERFACE)
target_include_directories(radix13 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(radix13 INTERFACE ${RADIX13_C_FLAGS})

enable_testing()
add_executable(test_radix13 bench_final_r13.c)
target_link_libraries(test_radix13 radix13 m)
target_compile_options(test_radix13 PRIVATE -O2 -Wno-unused-function)
find_library(FFTW3_LIB fftw3)
if(FFTW3_LIB)
    target_link_libraries(test_radix13 ${FFTW3_LIB})
endif()
add_test(NAME radix13_correctness COMMAND test_radix13)

install(FILES fft_radix13_genfft.h fft_radix13_dispatch.h DESTINATION include/vectorfft)
